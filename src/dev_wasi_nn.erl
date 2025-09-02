%%% @doc WASI-NN device implementation for HyperBEAM
%%% Implements wasi_nn API functions as imported functions by WASM modules
-module(dev_wasi_nn).

-export([info/1, info/3, infer/3, infer_sec/3]).

-include("include/hb.hrl").
-include_lib("eunit/include/eunit.hrl").

-define(SESSIONS_DIR, "sessions").

%% @doc Exported function for getting device info, controls which functions are
%% exposed via the device API.
info(_) ->
    #{exports => [info, infer, infer_sec]}.

%% @doc HTTP info response providing information about this device
info(_Msg1, _Msg2, _Opts) ->
    InfoBody =
        #{<<"description">> => <<"AI device for handling Inference">>,
          <<"version">> => <<"1.0">>,
          <<"api">> =>
              #{<<"infer">> =>
                    #{<<"description">> => <<"AI Inference">>,
                      <<"method">> => <<"GET or POST">>,
                      <<"required_params">> =>
                          #{<<"prompt">> => <<"Prompt for Infer">>,
                            <<"model-id">> => <<"Arweave TX ID of the model file">>}},
                <<"infer_sec">> =>
                    #{<<"description">> => <<"AI Inference with Attestation Token">>,
                      <<"method">> => <<"GET or POST">>,
                      <<"required_params">> =>
                          #{<<"prompt">> => <<"Prompt for Infer">>,
                            <<"model-id">> => <<"Arweave TX ID of the model file">>,
                            <<"config">> => <<"Attestation token configuration">>}}}},
    {ok, InfoBody}.

infer(M1, M2, Opts) ->
    TxID = maps:get(<<"model-id">>, M2, undefined),
    DefaultModel = <<"qwen2.5-14b-instruct-q2_k.gguf">>,

    ModelPath =
        case TxID of
            undefined ->
                DefaultModel;
            _ ->
                case download_and_store_model(TxID) of
                    {ok, LocalModelPath} ->
                        LocalModelPath;
                    {error, Reason} ->
                        ?event(dev_wasi_nn, {model_download_failed, TxID, Reason}),
                        DefaultModel
                end
        end,
    load_and_infer(M1, M2#{<<"model_path">> => <<"models/", ModelPath/binary>>}, Opts).

infer_sec(M1, M2, Opts) ->
    case dev_cc:generate(#{},
                         #{nonce =>
                               <<"da4a06c3604a5fac8aa0b4aaf5a6354cdd0dc7c193299bc3464f30b5cbfb931a">>},
                         Opts)
    of
        {ok, TokenJSON} ->
            case infer(M1, M2, Opts) of
                {ok, Result} ->
                    ExistingBody = maps:get(<<"body">>, Result, <<"{}">>),
                    ExistingData = hb_json:decode(ExistingBody),
                    UpdatedData = ExistingData#{<<"attestation">> => hb_json:decode(TokenJSON)},
                    UpdatedResult = Result#{<<"body">> => hb_json:encode(UpdatedData)},
                    {ok, UpdatedResult};
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
    StoreConfig =
        #{<<"store-module">> => hb_store_fs,
          <<"name">> => <<"./models">>},  % Directory where models will be stored
    % Use TX ID as filename with .gguf extension
    ModelFileName = <<TxID/binary, ".gguf">>,
    LocalPath = <<"./models/", ModelFileName/binary>>,

    % First try to read the model from local storage
    case hb_store:read(StoreConfig, binary_to_list(ModelFileName)) of
        {ok, _ExistingData} ->
            % File already exists locally, no need to download
            ?event(dev_wasi_nn, {model_already_exists, TxID, LocalPath}),
            {ok, ModelFileName};
        not_found ->
            % File doesn't exist, proceed with download
            try
                % Download data from Arweave using the TX ID
                case hb_gateway_client:data(TxID,
                                            #{http_connect_timeout =>
                                                  10
                                                  * 60
                                                  * 1000}) % 10 minutes for large model downloads
                of
                    {ok, ModelData} ->
                        % Store the model file locally
                        case hb_store:write(StoreConfig, ModelFileName, ModelData) of
                            ok ->
                                % Return the local file path
                                ?event(dev_wasi_nn, {model_downloaded, TxID, LocalPath}),
                                {ok, ModelFileName};
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
                case hb_gateway_client:data(TxID,
                                            #{http_connect_timeout =>
                                                  10
                                                  * 60
                                                  * 1000}) % 10 minutes for large model downloads
                of
                    {ok, ModelData} ->
                        % Store the model file locally
                        case hb_store:write(StoreConfig, ModelFileName, ModelData) of
                            ok ->
                                % Return the local file path
                                ?event(dev_wasi_nn, {model_downloaded, TxID, LocalPath}),
                                {ok, ModelFileName};
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
load_and_infer(M1, M2, Opts) ->
    Model = maps:get(<<"model_path">>, M2, <<"">>),
    ModelPathStr = binary_to_list(Model),
    % Minimal configuration for Gemma 3 model
    DefaultBaseConfig =
        #{<<"model">> =>
              #{<<"n_ctx">> => 8192,               % 8K context window (Gemma 3 sweet spot)
                <<"n_batch">> => 512,              % Optimal batch size for Gemma 3
                <<"n_gpu_layers">> => 99},           % Offload all layers to GPU
          %% Most other model params use llama.cpp defaults:
          %% - threads: auto-detected
          %% - use_mmap: true (default)
          %% - use_mlock: false (default)
          %% - numa: auto-detected
          <<"sampling">> =>
              #{<<"temperature">> => 0.7,          % Balanced creativity/coherence for Gemma 3
                <<"top_p">> => 0.9,                % Nucleus sampling
                <<"repeat_penalty">> => 1.1},        % Mild repetition penalty for Gemma 3
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
          <<"stopping">> =>
              #{<<"max_tokens">> => 512},            % Reasonable default for most use cases
          %% Other stopping params use defaults:
          %% <<"stop">> => [],                  % Default: empty
          %% <<"ignore_eos">> => false          % Default: false
          <<"backend">> => #{<<"max_sessions">> => 2}},
    %%     <<"idle_timeout_ms">> => 300000,   % Backend handles defaults
    %%     <<"auto_cleanup">> => true,        % Backend handles defaults
    %%     <<"queue_size">> => 50,            % Backend handles defaults
    %%     <<"default_task_timeout_ms">> => 30000,  % Backend handles defaults
    %%     <<"priority_scheduling_enabled">> => true,
    %%     <<"fair_scheduling_enabled">> => true
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
    ModelConfig = hb_json:encode(DefaultBaseConfig),
    ModelConfigStr = binary_to_list(ModelConfig),
    UserConfig = maps:get(<<"config">>, M2, ""),
    UserConfigStr = hb_util:list(UserConfig),
    
    Prompt = maps:get(<<"prompt">>, M2, <<>>),
    Body = maps:get(<<"body">>, M1, <<>>),
    DecodedBody = try hb_json:decode(Body) catch _:_ -> #{} end,

    PipedTranscript = maps:get(<<"transcription">>, DecodedBody, undefined),
    
    EffectivePrompt =
        case PipedTranscript of
            undefined -> Prompt;
            _ when is_binary(PipedTranscript) ->
                case Prompt of
                    <<>> -> PipedTranscript;
                    _ -> <<Prompt/binary, " ", PipedTranscript/binary>>
                end
        end,
    EffectivePromptStr = binary_to_list(EffectivePrompt),

    UserSessionId = maps:get(<<"session_id">>, M2, maps:get(<<"session_id">>, DecodedBody, undefined)),
    Reference = maps:get(<<"reference">>, M2, undefined),
    Worker = maps:get(<<"worker">>, M2, undefined),

    SessionIdBin =
        case UserSessionId of
            undefined -> generate_session_id(Opts);
            Bin when is_binary(Bin) -> Bin
        end,
    SessionIdList = binary_to_list(SessionIdBin),


    try
        case dev_wasi_nn_nif:ensure_model_loaded(ModelPathStr, ModelConfigStr) of
            ok ->
                case dev_wasi_nn_nif:init_session(SessionIdList) of
                    {ok, SessionResource} ->
                        case dev_wasi_nn_nif:run_inference(SessionResource, EffectivePromptStr, UserConfigStr) of
                            {ok, Output} ->
                                ?event(dev_wasi_nn, {inference_success, Reference}),
                                save_llm_response(SessionIdBin, Output),
                                ResponseBody = #{
                                    <<"result">> => Output,
                                    <<"transcription">> => PipedTranscript,
                                    <<"session_id">> => SessionIdBin
                                },
                                {ok,
                                 #{<<"body">> => hb_json:encode(ResponseBody),
                                   <<"X-Session">> => SessionIdBin,
                                   <<"X-Reference">> => Reference,
                                   <<"X-Worker">> => Worker,
                                   <<"action">> => <<"Infer-Response">>,
                                   <<"status">> => 200}};
                            {error, Reason} ->
                                ?event(dev_wasi_nn, {inference_failed, SessionIdList, Reason}),
                                {error, Reason}
                        end;
                    {error, Reason2} ->
                        ?event(dev_wasi_nn, {session_init_failed, SessionIdList, Reason2}),
                        {error, Reason2}
                end;
            {error, Reason3} ->
                ?event(dev_wasi_nn, {model_load_failed, SessionIdList, Model, Reason3}),
                {error, Reason3}
        end
    catch
        Error:Exception ->
            ?event(dev_wasi_nn, {inference_exception, SessionIdList, Error, Exception}),
            {error, {exception, Error, Exception}}
    end.

%% @doc Saves the LLM's text response to the session's response transcript file.
save_llm_response(SessionID, LLMResponse) ->
    ResponseAudiosPath = filename:join([?SESSIONS_DIR, binary_to_list(SessionID), "response-audios"]),
    ok = filelib:ensure_dir(filename:join(ResponseAudiosPath, "dummy.txt")),
    JsonPath = filename:join(ResponseAudiosPath, "string-list.json"),
    
    CurrentList = case file:read_file(JsonPath) of
        {ok, JsonBinary} ->
            try hb_json:decode(JsonBinary) of
                Decoded when is_list(Decoded) -> Decoded;
                _ -> []
            catch
                _:_ -> []
            end;
        {error, enoent} -> []
    end,
    
    NewList = CurrentList ++ [LLMResponse],
    file:write_file(JsonPath, hb_json:encode(NewList)).

%% @doc Generate a unique session ID for each request
generate_session_id(_Opts) ->
    Timestamp = erlang:system_time(microsecond),
    Random = rand:uniform(999),
    SessionId = io_lib:format("req_~p_~p", [Timestamp, Random]),
    list_to_binary(lists:flatten(SessionId)).

%% ===================================================================
%% EUnit Tests
%% ===================================================================
-ifdef(TEST).

-define(TEST_SESSION_ID, <<"wasi_nn_session_test">>).
-define(TEST_SESSION_PATH, filename:join(?SESSIONS_DIR, ?TEST_SESSION_ID)).

setup() ->
    % Clean up before test and create the required structure
    file:del_dir_r(?TEST_SESSION_PATH),
    ResponseAudioPath = filename:join([?TEST_SESSION_PATH, "response-audios"]),
    ok = filelib:ensure_dir(filename:join(ResponseAudioPath, "dummy.txt")),
    ok.

teardown(_) ->
    % dont Clean up the session directory after the test

    ok.

session_based_inference_test_() ->
    {foreach, fun setup/0, fun teardown/1, [fun test_infer_and_save_response/0]}.

test_infer_and_save_response() ->
    Transcript = <<"Who are you?">>,
    % M1 simulates the output from the speech-to-text device
    M1 = #{<<"body">> => hb_json:encode(#{
        <<"transcription">> => Transcript,
        <<"session_id">> => ?TEST_SESSION_ID
    })},
    % M2 is the direct request to the wasi-nn device
    M2 = #{<<"model_path">> => <<"models/gemma.gguf">>, % Assumes a test model exists
           <<"prompt">> => <<"In one sentence, respond to this question:">>},
    
    {ok, Response} = infer(M1, M2, #{}),
    
    ?assertEqual(200, maps:get(<<"status">>, Response)),
    
    DecodedBody = hb_json:decode(maps:get(<<"body">>, Response)),
    ?assertMatch(#{<<"result">> := _,
                   <<"transcription">> := Transcript,
                   <<"session_id">> := ?TEST_SESSION_ID}, DecodedBody),
    
    LLMResult = maps:get(<<"result">>, DecodedBody),
    ?assert(is_binary(LLMResult) andalso size(LLMResult) > 0),
    
    % Verify side-effect: check if the response was saved correctly
    JsonPath = filename:join([?TEST_SESSION_PATH, "response-audios", "string-list.json"]),
    ?assert(filelib:is_regular(JsonPath)),
    {ok, JsonBinary} = file:read_file(JsonPath),
    [SavedResult] = hb_json:decode(JsonBinary),
    ?assertEqual(LLMResult, SavedResult).

-endif.