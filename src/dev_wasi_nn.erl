%%% @doc WASI-NN device implementation for HyperBEAM
%%% Implements wasi_nn API functions as imported functions by WASM modules
-module(dev_wasi_nn).
-export([info/1, info/3, infer/3]).
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
    ?event(dev_wasi_nn, {infer, M1, M2, Opts, ModelPath}),
    load_and_infer(M1, M2#{<<"model_path">> => <<"models/", ModelPath/binary>>}, Opts).

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
    ModelConfig = maps:get(<<"config">>, M2, "{\"n_gpu_layers\":96,\"ctx_size\":64000,\"batch_size\":64000}"),
    BackendConfig = maps:get(<<"backend_config">>, M2, undefined),
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
        case dev_wasi_nn_nif:switch_model(binary_to_list(Model), ModelConfig, BackendConfig) of
            {ok, Context} ->
                % Create or reuse session-specific execution context
                case dev_wasi_nn_nif:init_execution_context_once(Context, SessionId) of
                    {ok, ExecContextId} ->
                        % Run inference with session-specific context
                        case dev_wasi_nn_nif:run_inference(Context, ExecContextId, binary_to_list(Prompt)) of
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