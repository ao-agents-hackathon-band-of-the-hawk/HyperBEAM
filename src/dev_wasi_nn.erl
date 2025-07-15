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

infer(_M1, M2, Opts) ->
    % Extract parameters
    TxID = hb_ao:get(<<"model-id">>, M2, undefined, Opts),
    ModelConfig = hb_ao:get(<<"config">>, M2, "{\"n_gpu_layers\":48,\"ctx_size\":64000}", Opts),
    Prompt = hb_ao:get(<<"prompt">>, M2, Opts),
    SessionId = hb_ao:get(<<"session-id">>, M2, undefined, Opts),
    
    case TxID of
        undefined ->
            % Fallback to original behavior if no TX ID provided
            load_and_infer("model/qwen2.5-14b-instruct-q2_k.gguf", ModelConfig, Prompt, SessionId, Opts);
        _ ->
            % Download model from Arweave using TX ID
            case download_and_store_model(TxID, Opts) of
                {ok, LocalModelPath} ->
                    load_and_infer(LocalModelPath, ModelConfig, Prompt, SessionId, Opts);
                {error, Reason} ->
                    {error, {model_download_failed, Reason}}
            end
    end.

%% @doc Download model from Arweave and store it locally
download_and_store_model(TxID, Opts) ->
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
load_and_infer(ModelPath, ModelConfig, Prompt, ProvidedSessionId, Opts) ->
    % Use provided session ID or generate a new one
    SessionId = case ProvidedSessionId of
        undefined -> generate_session_id(Opts);
        _ -> binary_to_list(ProvidedSessionId)
    end,
    
    try
        % Use persistent context management (fast if model already loaded)
        case dev_wasi_nn_nif:switch_model(ModelPath, ModelConfig) of
            {ok, Context} ->
                % Create or reuse session-specific execution context
                case dev_wasi_nn_nif:init_execution_context_once(Context, SessionId) of
                    {ok, ExecContextId} ->
                        % Run inference with session-specific context
                        case dev_wasi_nn_nif:run_inference(Context, ExecContextId, binary_to_list(Prompt)) of
                            {ok, Output} ->
                                {ok, #{
                                    <<"result">> => hb_util:encode(Output),
                                    <<"session-id">> => list_to_binary(SessionId)
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
                ?event(dev_wasi_nn, {model_load_failed, SessionId, ModelPath, Reason3}),
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
    Random = rand:uniform(999999),
    % Include process info to help with debugging
    Pid = self(),
    SessionId = io_lib:format("req_~p_~p_~p", [Timestamp, Random, Pid]),
    lists:flatten(SessionId).