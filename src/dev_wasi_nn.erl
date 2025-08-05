%%% @doc WASI-NN device implementation for HyperBEAM
%%% Implements wasi_nn API functions as imported functions by WASM modules
-module(dev_wasi_nn).
-export([info/1, info/3, infer/3, infer_sec/3]).
-include("include/hb.hrl").

%% @doc Exported function for getting device info, controls which functions are
%% exposed via the device API.
info(_) -> 
    #{ exports => [info, infer] }.

%% @doc HTTP info response providing information about this device
info(_Msg1, _Msg2, _Opts) ->
    InfoBody = #{
        <<"description">> => <<"AI device for handling Inference">>,
        <<"version">> => <<"1.0">>,
        <<"api">> => #{
            <<"infer">> => #{
                <<"description">> => <<"AI Inference">>,
                <<"method">> => <<"GET or POST">>,
                <<"required_params">> => #{
                    <<"prompt">> => <<"Prompt for Infer">>,
                    <<"model-id">> => <<"Arweave TX ID of the model file">>
                }
            },
            <<"infer_sec">> => #{
                <<"description">> => <<"AI Inference with Attestation Token">>,
                <<"method">> => <<"GET or POST">>,
                <<"required_params">> => #{
                    <<"prompt">> => <<"Prompt for Infer">>,
                    <<"model-id">> => <<"Arweave TX ID of the model file">>,
                    <<"config">> => <<"Attestation token configuration">>
                }
            }
        }
    },
    {ok, InfoBody}.

infer(M1, M2, Opts) ->
    TxID = maps:get(<<"model_id">>, M2, undefined),
    DefaultModel = <<"qwen2.5-14b-instruct-q2_k.gguf">>,

    ModelPath = case TxID of
        undefined -> DefaultModel;
        _ ->
            case download_and_store_model(TxID) of
                {ok, LocalModelPath} -> LocalModelPath;
                {error, Reason} ->
                    ?event(dev_wasi_nn, {model_download_failed, TxID, Reason}),
                    DefaultModel
            end
    end,
    load_and_infer(M1, M2#{<<"model_path">> => <<"models/", ModelPath/binary>>}, Opts).

infer_sec(M1, M2, Opts) ->
    case dev_cc:generate(#{}, #{nonce => <<"da4a06c3604a5fac8aa0b4aaf5a6354cdd0dc7c193299bc3464f30b5cbfb931a">>}, Opts) of
        {ok, TokenJSON} ->
            case infer(M1, M2#{<<"config">> => TokenJSON}, Opts) of
                {ok, Result} ->
                    {ok, Result#{<<"X-Attestation">> => TokenJSON}};
                {error, Reason} ->
                    ?event(dev_wasi_nn, {infer_sec_failed, Reason}),
                    {error, {infer_sec_failed, Reason}}
            end;
        {error, Reason} ->
            ?event(dev_wasi_nn, {infer_sec_failed, Reason}),
            {error, {infer_sec_failed, Reason}}
    end.

%% @doc Download model from Arweave and store it locally
download_and_store_model(TxID) ->
    % Configure local storage
    StoreConfig = #{
        <<"store-module">> => hb_store_fs,
        <<"name">> => <<"./models">>  % Directory where models will be stored
    },
    % Use TX ID as filename with .gguf extension
    ModelFileName = <<TxID/binary, ".gguf">>,
    LocalPath = <<"./models/", ModelFileName/binary>>,
    
    % First try to read the model from local storage
    case hb_store:read(StoreConfig, binary_to_list(ModelFileName)) of
        {ok, _ExistingData} ->
            % File already exists locally, no need to download
            ?event(dev_wasi_nn, {model_already_exists, TxID, LocalPath}),
            {ok, binary_to_list(LocalPath)};
        not_found ->
            % File doesn't exist, proceed with download
            try
                % Download data from Arweave using the TX ID
                case hb_gateway_client:data(TxID, #{
                    http_connect_timeout => 10 * 60 * 1000 % 10 minutes for large model downloads
                }) of
                    {ok, ModelData} ->
                        % Store the model file locally
                        case hb_store:write(StoreConfig, ModelFileName, ModelData) of
                            ok ->
                                % Return the local file path
                                ?event(dev_wasi_nn, {model_downloaded, TxID, LocalPath}),
                                {ok, binary_to_list(LocalPath)};
                            StoreError ->
                                ?event(dev_wasi_nn, {model_store_failed, TxID, StoreError}),
                                {error, {store_failed, StoreError}}
                        end;
                    {error, DownloadError} ->
                        ?event(dev_wasi_nn, {model_download_failed, TxID, DownloadError}),
                        {error, {download_failed, DownloadError}}
                end
            catch
                Error:Reason ->
                    ?event(dev_wasi_nn, {model_download_exception, TxID, Error, Reason}),
                    {error, {exception, Error, Reason}}
            end;
        {error, ReadError} ->
            % Error reading from storage, log and proceed with download
            ?event(dev_wasi_nn, {model_read_error, TxID, ReadError}),
            try
                % Download data from Arweave using the TX ID
                case hb_gateway_client:data(TxID, #{
                    http_connect_timeout => 10 * 60 * 1000 % 10 minutes for large model downloads
                }) of
                    {ok, ModelData} ->
                        % Store the model file locally
                        case hb_store:write(StoreConfig, ModelFileName, ModelData) of
                            ok ->
                                % Return the local file path
                                ?event(dev_wasi_nn, {model_downloaded, TxID, LocalPath}),
                                {ok, binary_to_list(LocalPath)};
                            StoreError ->
                                ?event(dev_wasi_nn, {model_store_failed, TxID, StoreError}),
                                {error, {store_failed, StoreError}}
                        end;
                    {error, DownloadError} ->
                        ?event(dev_wasi_nn, {model_download_failed, TxID, DownloadError}),
                        {error, {download_failed, DownloadError}}
                end
            catch
                Error:Reason ->
                    ?event(dev_wasi_nn, {model_download_exception, TxID, Error, Reason}),
                    {error, {exception, Error, Reason}}
            end
    end.

%% @doc Load model and perform inference using persistent context management with session support
load_and_infer(_M1, M2, Opts) ->
    Model = maps:get(<<"model_path">>, M2, <<"">>),
    
    % Minimal configuration for Gemma 3 model
    DefaultBaseConfig = #{
        <<"model">> => #{
            <<"n_ctx">> => 8192,               % 8K context window (Gemma 3 sweet spot)
            <<"n_batch">> => 512,              % Optimal batch size for Gemma 3
            <<"n_gpu_layers">> => 99           % Offload all layers to GPU
            %% Most other model params use llama.cpp defaults:
            %% - threads: auto-detected
            %% - use_mmap: true (default)
            %% - use_mlock: false (default)
            %% - numa: auto-detected
        },
        
        <<"sampling">> => #{
            <<"temperature">> => 0.7,          % Balanced creativity/coherence for Gemma 3
            <<"top_p">> => 0.9,                % Nucleus sampling 
            <<"repeat_penalty">> => 1.1        % Mild repetition penalty for Gemma 3
            
            %% Advanced sampling - commented out, using llama.cpp defaults:
            %% <<"top_k">> => 40,                 % Default: 40
            %% <<"min_p">> => 0.05,               % Default: 0.05
            %% <<"typical_p">> => 1.0,            % Default: 1.0 (disabled)
            %% <<"presence_penalty">> => 0.0,     % Default: 0.0
            %% <<"frequency_penalty">> => 0.0,    % Default: 0.0
            %% <<"penalty_last_n">> => 64,        % Default: 64
            
            %% DRY Sampling - using defaults:
            %% <<"dry_multiplier">> => 0.8,       % Default handles this
            %% <<"dry_base">> => 1.75,            % Default handles this
            %% <<"dry_allowed_length">> => 2,     % Default handles this
            %% <<"dry_penalty_last_n">> => -1,    % Default handles this
            %% <<"dry_sequence_breakers">> => [<<"\n">>, <<":">>, <<"\"">>, <<"*">>],
            
            %% Mirostat - using defaults (disabled):
            %% <<"mirostat">> => 0,               % Default: 0 (disabled)
            %% <<"mirostat_tau">> => 5.0,         % Default: 5.0
            %% <<"mirostat_eta">> => 0.1,         % Default: 0.1
            
            %% Other sampling - using defaults:
            %% <<"seed">> => -1,                  % Default: -1 (random)
            %% <<"n_probs">> => 0,                % Default: 0
            %% <<"min_keep">> => 1,               % Default: 1
            %% <<"ignore_eos">> => false,         % Default: false
            %% <<"grammar">> => <<"">>            % Default: empty
        },
        
        <<"stopping">> => #{
            <<"max_tokens">> => 512            % Reasonable default for most use cases
            %% Other stopping params use defaults:
            %% <<"stop">> => [],                  % Default: empty
            %% <<"ignore_eos">> => false          % Default: false
        },
        
        <<"backend">> => #{
            <<"max_sessions">> => 2
        %%     <<"idle_timeout_ms">> => 300000,   % Backend handles defaults
        %%     <<"auto_cleanup">> => true,        % Backend handles defaults
        %%     <<"queue_size">> => 50,            % Backend handles defaults
        %%     <<"default_task_timeout_ms">> => 30000,  % Backend handles defaults
        %%     <<"priority_scheduling_enabled">> => true,
        %%     <<"fair_scheduling_enabled">> => true
        }
        
        %% <<"memory">> => #{
        %%     <<"context_shifting">> => true,    % Backend handles defaults
        %%     <<"cache_strategy">> => <<"smart">>,    % Backend handles defaults
        %%     <<"max_cache_tokens">> => 100000,  % Backend handles defaults
        %%     <<"memory_pressure_threshold">> => 0.85,  % Backend handles defaults
        %%     <<"n_keep_tokens">> => 256,        % Backend handles defaults
        %%     <<"n_discard_tokens">> => 512,     % Backend handles defaults
        %%     <<"enable_partial_cache_deletion">> => true,
        %%     <<"enable_token_cache_reuse">> => true,
        %%     <<"cache_deletion_strategy">> => <<"smart">>,
        %%     <<"max_memory_mb">> => 0           % Backend handles defaults
        %% },
        
        %% <<"logging">> => #{
        %%     <<"level">> => <<"info">>,         % Backend handles defaults
        %%     <<"enable_debug">> => false,       % Backend handles defaults
        %%     <<"timestamps">> => true,          % Backend handles defaults
        %%     <<"colors">> => false,             % Backend handles defaults
        %%     <<"file">> => <<"">>               % Backend handles defaults
        %% },
        
        %% <<"performance">> => #{
        %%     <<"batch_processing">> => true,    % Backend handles defaults
        %%     <<"batch_size">> => 512,           % Backend handles defaults
        %%     <<"batch_timeout_ms">> => 100      % Backend handles defaults
        %% }
    },
    ModelConfig = hb_json:encode(DefaultBaseConfig),
    UserConfig = maps:get(<<"config">>, M2, ""),
    UserConfigStr = hb_util:list(UserConfig),
    Prompt = maps:get(<<"prompt">>, M2),
    UserSessionId = maps:get(<<"session_id">>, M2, undefined),
    Reference = maps:get(<<"reference">>, M2, undefined),
    Worker = maps:get(<<"worker">>, M2, undefined),

    % Use provided session ID or generate a new one
    SessionId = case UserSessionId of
        undefined -> generate_session_id(Opts);
        _ -> binary_to_list(UserSessionId)
    end,
    
    try
        % Use persistent context management (fast if model already loaded)
        case dev_wasi_nn_nif:switch_model(binary_to_list(Model), ModelConfig) of
            {ok, Context} ->
                % Create or reuse session-specific execution context
                case dev_wasi_nn_nif:init_execution_context_once(Context, SessionId) of
                    {ok, ExecContextId} ->
                        % Run inference with session-specific context
                        case dev_wasi_nn_nif:run_inference(Context, ExecContextId, binary_to_list(Prompt), UserConfigStr) of
                            {ok, Output} ->
                                ?event(dev_wasi_nn, {inference_success, Reference}),
                                {ok, #{
                                    <<"body">> => hb_json:encode(#{
                                        <<"result">> => Output
                                    }),
                                    <<"X-Session">> => list_to_binary(SessionId),
                                    <<"X-Reference">> => Reference,
                                    <<"X-Worker">> => Worker,
                                    <<"action">> => <<"Infer-Response">>,
                                    <<"status">> => 200
                                }};
                            {error, Reason} ->
                                ?event(dev_wasi_nn, {inference_failed, SessionId, Reason}),
                                {error, Reason}
                        end;
                    {error, Reason2} ->
                        ?event(dev_wasi_nn, {session_init_failed, SessionId, Reason2}),
                        {error, Reason2}
                end;
            {error, Reason3} ->
                ?event(dev_wasi_nn, {model_load_failed, SessionId, Model, Reason3}),
                {error, Reason3}
        end
    catch
        Error:Exception ->
            ?event(dev_wasi_nn, {inference_exception, SessionId, Error, Exception}),
            {error, {exception, Error, Exception}}
    end.

%% @doc Generate a unique session ID for each request
generate_session_id(_Opts) ->
    % Use a combination of timestamp and random number for uniqueness
    Timestamp = erlang:system_time(microsecond),
    Random = rand:uniform(999),
    SessionId = io_lib:format("req_~p_~p", [Timestamp, Random]),
    lists:flatten(SessionId).