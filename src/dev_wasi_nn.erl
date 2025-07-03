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
                    <<"prompt">> => <<"Prompt for Infer">>
                }
            }
        }
    },
    {ok, InfoBody}.

infer(_M1, M2, Opts) ->
    ModelID = hb_ao:get(<<"model-id">>, M2, "test/qwen2.5-14b-instruct-q2_k.gguf", Opts),
    ModelConfig = hb_ao:get(<<"config">>, M2, "{\"n_gpu_layers\":48,\"ctx_size\":64000}", Opts),
    Prompt = hb_ao:get(<<"prompt">>, M2, Opts),
    case dev_wasi_nn_nif:load_by_name_with_config_once(undefined, ModelID, ModelConfig) of
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