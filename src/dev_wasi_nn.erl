%%% @doc A WASI-NN device implementation for HyperBEAM that provides AI inference
%%% capabilities. This device supports loading models from Arweave transactions
%%% and performing inference with session management for optimal performance.
%%% Models are cached locally to avoid repeated downloads.
-module(dev_wasi_nn).
-export([info/1, info/3, infer/3]).
-include("include/hb.hrl").
-include_lib("eunit/include/eunit.hrl").
-hb_debug(print).
%% @doc Get device information and exported functions.
%% Returns the list of functions that are exposed via the device API.
%% This is used by the HyperBEAM runtime to determine which endpoints
%% are available for this device.
%%
%% @param _ Ignored parameter.
%% @returns A map containing the list of exported functions.
info(_) -> 
    #{ exports => [info, infer] }.

%% @doc Provide HTTP info response about this device.
%% Returns comprehensive information about the WASI-NN device including
%% its capabilities, version, and API documentation. This endpoint helps
%% users understand how to interact with the AI inference functionality.
%%
%% @param _Msg1 Ignored parameter.
%% @param _Msg2 Ignored parameter.
%% @param _Opts Ignored parameter.
%% @returns {ok, InfoBody} containing device information and API documentation.
info(_Msg1, _Msg2, _Opts) ->
    InfoBody = #{
        <<"description">> => <<"GPU device for handling LLM Inference">>,
        <<"version">> => <<"1.0">>,
        <<"api">> => #{
            <<"infer">> => #{
                <<"description">> => <<"LLM Inference">>,
                <<"method">> => <<"GET or POST">>,
                <<"required_params">> => #{
                    <<"prompt">> => <<"Prompt for Infer">>,
                    <<"model-id">> => <<"Arweave TX ID of the model file">>
                }
            }
        }
    },
    {ok, InfoBody}.

%% @doc Perform AI inference using a specified model and prompt.
%% This function handles the complete inference workflow including model
%% retrieval (either from local cache or Arweave), session management,
%% and inference execution. Models are automatically downloaded and cached
%% locally for improved performance on subsequent requests.
%%
%% @param _M1 Ignored parameter.
%% @param M2 The request message containing inference parameters:
%%           - <<"model-id">>: Arweave transaction ID of the model file
%%           - <<"config">>: JSON configuration for the model (optional)
%%           - <<"prompt">>: The input prompt for inference
%%           - <<"session-id">>: Session identifier for context reuse (optional)
%% @param Opts A map of configuration options.
%% @returns {ok, #{<<"result">> := Result, <<"session-id">> := SessionId}} on success,
%%          {error, Reason} on failure.
infer(_M1, M2, Opts) ->
    TxID = hb_ao:get(<<"model-id">>, M2, undefined, Opts),
    ModelConfig = hb_ao:get(<<"config">>, M2, 
        "{\"n_gpu_layers\":96,\"ctx_size\":64000,\"batch_size\":64000}", Opts),
    Prompt = hb_ao:get(<<"prompt">>, M2, Opts),
    SessionId = hb_ao:get(<<"session-id">>, M2, undefined, Opts),
    
    ?event(dev_wasi_nn, {infer, {tx_id, TxID}, {session_id, SessionId}}),
    
    case TxID of
        ?event(dev_wasi_nn, {infer, {downloading_model, TxID}}),
        case download_and_store_model(TxID, Opts) of
            {ok, LocalModelPath} ->
                ?event(dev_wasi_nn, {infer, {model_ready, LocalModelPath}}),
                load_and_infer(LocalModelPath, ModelConfig, Prompt, SessionId, Opts);
            {error, Reason} ->
                ?event(dev_wasi_nn, {infer, {model_download_failed, Reason}}),
                {error, {model_download_failed, Reason}}
        end
    end.

%%%--------------------------------------------------------------------
%%% Helper Functions
%%%--------------------------------------------------------------------



%% @doc Load model and perform inference using persistent context management.
%% Handles the complete inference workflow including model loading, session
%% management, and inference execution. Uses session IDs to maintain context
%% across multiple requests for improved performance.
%%
%% @param ModelPath The local file path to the model.
%% @param ModelConfig JSON configuration string for the model.
%% @param Prompt The input prompt for inference.
%% @param ProvidedSessionId Optional session ID for context reuse.
%% @param Opts A map of configuration options.
%% @returns {ok, #{<<"result">> := Result, <<"session-id">> := SessionId}} on success,
%%          {error, Reason} on failure.
load_and_infer(ModelPath, ModelConfig, Prompt, ProvidedSessionId, Opts) ->
    SessionId = case ProvidedSessionId of
        undefined -> generate_session_id(Opts);
        _ -> binary_to_list(ProvidedSessionId)
    end,
    ?event(dev_wasi_nn, {load_and_infer, {model_path, ModelPath}, {session_id, SessionId}}),
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
    end.

%% @doc Download model data from Arweave and store it locally.
%% Helper function that handles the actual download and storage process
%% when a model is not found in the local cache.
%%
%% @param TxID The Arweave transaction ID of the model.
%% @param StoreConfig Configuration for the local storage.
%% @param ModelFileName The filename to use for storing the model.
%% @param LocalPath The full local path where the model will be stored.
%% @returns {ok, LocalPath} on success, {error, Reason} on failure.
download_model_from_arweave(TxID, StoreConfig, ModelFileName, LocalPath) ->
    try
        case hb_gateway_client:data(TxID, #{
            http_connect_timeout => 10 * 60 * 1000
        }) of
            {ok, ModelData} ->
                ?event(dev_wasi_nn, {download_model_from_arweave, {data_received, TxID}}),
                case hb_store:write(StoreConfig, ModelFileName, ModelData) of
                    ok ->
                        ?event(dev_wasi_nn, {download_model_from_arweave, {model_stored, TxID, LocalPath}}),
                        {ok, binary_to_list(LocalPath)};
                    StoreError ->
                        ?event(dev_wasi_nn, {download_model_from_arweave, {store_failed, TxID, StoreError}}),
                        {error, {store_failed, StoreError}}
                end;
            {error, DownloadError} ->
                ?event(dev_wasi_nn, {download_model_from_arweave, {download_failed, TxID, DownloadError}}),
                {error, {download_failed, DownloadError}}
        end
    catch
        Error:Reason ->
            ?event(dev_wasi_nn, {download_model_from_arweave, {exception, TxID, Error, Reason}}),
            {error, {exception, Error, Reason}}
    end.
%% @doc Download model from Arweave and store it locally.
%% Attempts to retrieve a model file from local storage first, and if not found,
%% downloads it from Arweave using the provided transaction ID. The model is
%% stored locally with a .gguf extension for future use.
%%
%% @param TxID The Arweave transaction ID of the model file.
%% @param Opts A map of configuration options (currently unused).
%% @returns {ok, LocalPath} on success where LocalPath is the local file path,
%%          {error, Reason} on failure.
download_and_store_model(TxID, _Opts) ->
    StoreConfig = #{
        <<"store-module">> => hb_store_fs,
        <<"name">> => <<"./models">>
    },
    ModelFileName = <<TxID/binary, ".gguf">>,
    LocalPath = <<"./models/", ModelFileName/binary>>,
    ?event(dev_wasi_nn, {download_and_store_model, {tx_id, TxID}, {local_path, LocalPath}}),
    
    case hb_store:read(StoreConfig, binary_to_list(ModelFileName)) of
        {ok, _ExistingData} ->
            ?event(dev_wasi_nn, {download_and_store_model, {model_already_exists, TxID}}),
            {ok, binary_to_list(LocalPath)};
        not_found ->
            ?event(dev_wasi_nn, {download_and_store_model, {downloading_from_arweave, TxID}}),
            download_model_from_arweave(TxID, StoreConfig, ModelFileName, LocalPath);
        {error, ReadError} ->
            ?event(dev_wasi_nn, {download_and_store_model, {read_error, TxID, ReadError}}),
            download_model_from_arweave(TxID, StoreConfig, ModelFileName, LocalPath)
    end.

%% @doc Generate a unique session ID for inference requests.
%% Creates a unique identifier combining timestamp, random number, and process
%% information to ensure session uniqueness across concurrent requests.
%%
%% @param _Opts Configuration options (currently unused).
%% @returns A string representing the unique session ID.
generate_session_id(_Opts) ->
    Timestamp = erlang:system_time(microsecond),
    Random = rand:uniform(999999),
    Pid = self(),
    SessionId = io_lib:format("req_~p_~p_~p", [Timestamp, Random, Pid]),
    lists:flatten(SessionId).
