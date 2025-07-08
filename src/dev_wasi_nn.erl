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
    % Get Arweave TX ID for the model
    TxID = hb_ao:get(<<"model-id">>, M2, undefined, Opts),
    ModelConfig = hb_ao:get(<<"config">>, M2, "{\"n_gpu_layers\":48,\"ctx_size\":64000}", Opts),
    Prompt = hb_ao:get(<<"prompt">>, M2, Opts),
    
    case TxID of
        undefined ->
            % Fallback to original behavior if no TX ID provided
            load_and_infer("test/qwen2.5-14b-instruct-q2_k.gguf", ModelConfig, Prompt, Opts);
        _ ->
            % Download model from Arweave using TX ID
            case download_and_store_model(TxID, Opts) of
                {ok, LocalModelPath} ->
                    load_and_infer(LocalModelPath, ModelConfig, Prompt, Opts);
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
            ?event({model_already_exists, TxID, LocalPath}, Opts),
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
                                ?event({model_downloaded, TxID, LocalPath}, Opts),
                                {ok, binary_to_list(LocalPath)};
                            StoreError ->
                                ?event({model_store_failed, TxID, StoreError}, Opts),
                                {error, {store_failed, StoreError}}
                        end;
                    {error, DownloadError} ->
                        ?event({model_download_failed, TxID, DownloadError}, Opts),
                        {error, {download_failed, DownloadError}}
                end
            catch
                Error:Reason ->
                    ?event({model_download_exception, TxID, Error, Reason}, Opts),
                    {error, {exception, Error, Reason}}
            end;
        {error, ReadError} ->
            % Error reading from storage, log and proceed with download
            ?event({model_read_error, TxID, ReadError}, Opts),
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
                                ?event({model_downloaded, TxID, LocalPath}, Opts),
                                {ok, binary_to_list(LocalPath)};
                            StoreError ->
                                ?event({model_store_failed, TxID, StoreError}, Opts),
                                {error, {store_failed, StoreError}}
                        end;
                    {error, DownloadError} ->
                        ?event({model_download_failed, TxID, DownloadError}, Opts),
                        {error, {download_failed, DownloadError}}
                end
            catch
                Error:Reason ->
                    ?event({model_download_exception, TxID, Error, Reason}, Opts),
                    {error, {exception, Error, Reason}}
            end
    end.

%% @doc Load model and perform inference
load_and_infer(ModelPath, ModelConfig, Prompt, Opts) ->
    case dev_wasi_nn_nif:load_by_name_with_config_once(undefined, ModelPath, ModelConfig) of
        {ok, Context} ->
            case dev_wasi_nn_nif:init_execution_context_once(Context) of
                ok ->
                    case dev_wasi_nn_nif:run_inference(Context, binary_to_list(Prompt)) of
                        {ok, Output} ->
                            {ok, Output};
                        {error, Reason} ->
                            {error, Reason}
                    end;
                {error, Reason2} ->
                    {error, Reason2}
            end;
        {error, Reason3} ->
            {error, Reason3}
    end.