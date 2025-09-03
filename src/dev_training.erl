-module(dev_training).
-export([train/3, convert/3, train_and_convert/3, info/1, info/3]).
-include("include/hb.hrl").
-include_lib("eunit/include/eunit.hrl").

-define(DEFAULT_BASE_MODEL, <<"Qwen/Qwen1.5-0.5B-Chat">>).
-define(DEFAULT_GGUF_PRECISION, <<"q8_0">>).

%% @doc Declares the functions this device exports.
info(_) ->
    #{exports => [train, convert, train_and_convert, info]}.

%% @doc Provides API information for all functions.
info(_M1, _M2, _Opts) ->
    InfoBody = #{
        <<"description">> => <<"A device for fine-tuning LoRA adapters and converting them to GGUF format.">>,
        <<"version">> => <<"1.0">>,
        <<"api">> => #{
            <<"train">> => #{
                <<"description">> => <<"1) Fine-tunes a LoRA adapter and saves the result as safetensors in a session directory.">>,
                <<"method">> => <<"POST">>,
                <<"parameters">> => get_training_params_doc(<<"Required.">>)
            },
            <<"convert">> => #{
                <<"description">> => <<"2) Converts an existing LoRA adapter (safetensors) from a session into GGUF format.">>,
                <<"method">> => <<"POST">>,
                <<"parameters">> => get_conversion_params_doc()
            },
            <<"train_and_convert">> => #{
                <<"description">> => <<"3) A full pipeline that fine-tunes a LoRA adapter and then immediately converts it to GGUF format.">>,
                <<"method">> => <<"POST">>,
                <<"parameters">> => maps:merge(
                    get_training_params_doc(<<"Required.">>),
                    get_conversion_params_doc()
                )
            }
        }
    },
    {ok, InfoBody}.

%% @doc 1) Fine-tunes a LoRA adapter (safetensors).
train(_M1, M2, _Opts) ->
    case parse_training_params(M2) of
        {ok, {SessionID, ModelID, LoraParams}} ->
            case dev_training_nif:finetune_lora_nif(SessionID, ModelID, LoraParams) of
                {ok, AdapterPath} ->
                    ResponseBody = hb_json:encode(#{
                        <<"status">> => <<"success">>,
                        <<"adapter_path">> => AdapterPath,
                        <<"session_id">> => SessionID
                    }),
                    {ok, #{ <<"body">> => ResponseBody, <<"status">> => 200 }};
                {error, Reason} ->
                    error_response(500, <<"LoRA training failed">>, Reason)
            end;
        {error, Reason} ->
            error_response(400, <<"Invalid parameters">>, Reason)
    end.

%% @doc 2) Converts an existing safetensor adapter to GGUF.
convert(_M1, M2, _Opts) ->
    case maps:get(<<"session_id">>, M2, undefined) of
        undefined ->
            error_response(400, <<"Missing required parameter">>, <<"session_id">>);
        SessionID ->
            AdapterPath = filename:join(["runs", binary_to_list(SessionID), "finetune_lora"]),
            case filelib:is_dir(AdapterPath) of
                true ->
                    {BaseModelID, GGUFParams} = parse_conversion_params(M2),
                    case dev_training_nif:convert_lora_to_gguf_nif(SessionID, BaseModelID, list_to_binary(AdapterPath), GGUFParams) of
                        {ok, GGUFPath} ->
                             ResponseBody = hb_json:encode(#{
                                <<"status">> => <<"success">>,
                                <<"gguf_path">> => GGUFPath,
                                <<"session_id">> => SessionID
                            }),
                            {ok, #{ <<"body">> => ResponseBody, <<"status">> => 200 }};
                        {error, Reason} ->
                            error_response(500, <<"GGUF conversion failed">>, Reason)
                    end;
                false ->
                    error_response(404, <<"LoRA adapter directory not found for session">>, #{
                        <<"session_id">> => SessionID,
                        <<"expected_path">> => list_to_binary(AdapterPath)
                    })
            end
    end.


%% @doc 3) Fine-tunes and then converts to GGUF.
train_and_convert(_M1, M2, _Opts) ->
    case parse_training_params(M2) of
        {ok, {SessionID, ModelID, LoraParams}} ->
            case dev_training_nif:finetune_lora_nif(SessionID, ModelID, LoraParams) of
                {ok, AdapterPath} ->
                    io:format("LoRA training successful, adapter at: ~s~n", [AdapterPath]),
                    {BaseModelID, GGUFParams} = parse_conversion_params(M2),
                    case dev_training_nif:convert_lora_to_gguf_nif(SessionID, BaseModelID, AdapterPath, GGUFParams) of
                        {ok, GGUFPath} ->
                            ResponseBody = hb_json:encode(#{
                                <<"status">> => <<"success">>,
                                <<"adapter_path">> => AdapterPath,
                                <<"gguf_path">> => GGUFPath,
                                <<"session_id">> => SessionID
                            }),
                            {ok, #{ <<"body">> => ResponseBody, <<"status">> => 200 }};
                        {error, ConvReason} ->
                             error_response(500, <<"GGUF conversion failed after training">>, ConvReason)
                    end;
                {error, TrainReason} ->
                    error_response(500, <<"LoRA training failed">>, TrainReason)
            end;
        {error, Reason} ->
            error_response(400, <<"Invalid parameters">>, Reason)
    end.

%% --- Private Helper Functions ---

get_training_params_doc(SessionIDRequirement) ->
    #{
        <<"session_id">> => SessionIDRequirement,
        <<"model_id">> => iolist_to_binary(io_lib:format(<<"Optional. The base model ID from Hugging Face. Defaults to ~s.">>, [?DEFAULT_BASE_MODEL])),
        <<"dataset_path">> => <<"Required. Path to the training data JSON file.">>,
        <<"num_epochs">> => <<"Optional. Number of training epochs.">>,
        <<"batch_size">> => <<"Optional. Training batch size.">>,
        <<"lora_rank">> => <<"Optional. The rank for the LoRA matrices.">>,
        <<"lora_alpha">> => <<"Optional. The alpha parameter for LoRA.">>,
        <<"lora_dropout">> => <<"Optional. Dropout probability for LoRA layers.">>
    }.

get_conversion_params_doc() ->
    #{
        <<"session_id">> => <<"Required. The session containing the 'finetune_lora' output.">>,
        <<"base_model_id">> => iolist_to_binary(io_lib:format(<<"Optional. The base model ID used for training. Defaults to ~s.">>, [?DEFAULT_BASE_MODEL])),
        <<"gguf_precision">> => iolist_to_binary(io_lib:format(<<"Optional. The quantization for the GGUF file. Defaults to ~s.">>, [?DEFAULT_GGUF_PRECISION]))
    }.

parse_training_params(M) ->
    case maps:get(<<"session_id">>, M, undefined) of
        undefined -> {error, <<"Missing required parameter: session_id">>};
        SessionID ->
            case maps:get(<<"dataset_path">>, M, undefined) of
                undefined -> {error, <<"Missing required parameter: dataset_path">>};
                DatasetPath ->
                    ModelID = maps:get(<<"model_id">>, M, ?DEFAULT_BASE_MODEL),
                    % Collect all optional params, only including them if they exist in M.
                    LoraParams = maps:from_list([
                        {K, maps:get(atom_to_binary(K, utf8), M)}
                        || K <- [num_epochs, batch_size, lora_rank, lora_alpha, lora_dropout],
                           maps:is_key(atom_to_binary(K, utf8), M)
                    ]),
                    FinalParams = LoraParams#{dataset_path => DatasetPath},
                    {ok, {SessionID, ModelID, FinalParams}}
            end
    end.

parse_conversion_params(M) ->
    BaseModelID = maps:get(<<"base_model_id">>, M, ?DEFAULT_BASE_MODEL),
    GGUFPrecision = maps:get(<<"gguf_precision">>, M, ?DEFAULT_GGUF_PRECISION),
    GGUFParams = #{gguf_precision => GGUFPrecision},
    {BaseModelID, GGUFParams}.

error_response(Status, Error, Reason) when is_binary(Reason) ->
    Body = hb_json:encode(#{<<"error">> => Error, <<"reason">> => Reason}),
    {ok, #{ <<"body">> => Body, <<"status">> => Status }};
error_response(Status, Error, Reason) ->
    ReasonBin = iolist_to_binary(io_lib:format("~p", [Reason])),
    Body = hb_json:encode(#{<<"error">> => Error, <<"reason">> => ReasonBin}),
    {ok, #{ <<"body">> => Body, <<"status">> => Status }}.

%% ===================================================================
%% EUnit Tests
%% ===================================================================
-ifdef(TEST).

-define(TEST_SESSION, <<"pipeline_test_session">>).
-define(TEST_DATA, <<"native/pyrust_nn/data.json">>).

setup() ->
    file:del_dir_r(filename:join("runs", binary_to_list(?TEST_SESSION))),
    ok.

teardown(_) ->
    ok.

full_pipeline_test_() ->
    {setup, fun setup/0, fun teardown/1, fun(_) -> ?_test(
        begin
            M2 = #{
                <<"session_id">> => ?TEST_SESSION,
                <<"dataset_path">> => ?TEST_DATA,
                <<"num_epochs">> => 1,
                <<"lora_rank">> => 2,
                % FIX: Use a supported quantization type.
                <<"gguf_precision">> => <<"q8_0">>
            },

            {ok, Response} = train_and_convert(#{}, M2, #{}),
            ?assertEqual(200, maps:get(<<"status">>, Response)),
            Body = hb_json:decode(maps:get(<<"body">>, Response)),
            ?assert(maps:is_key(<<"adapter_path">>, Body)),
            ?assert(maps:is_key(<<"gguf_path">>, Body)),

            AdapterPath = maps:get(<<"adapter_path">>, Body),
            GGUFPath = maps:get(<<"gguf_path">>, Body),
            ?assert(filelib:is_dir(binary_to_list(AdapterPath))),
            ?assert(filelib:is_regular(binary_to_list(GGUFPath)))
        end
    ) end}.

train_only_test_() ->
    {setup, fun setup/0, fun teardown/1, fun(_) -> ?_test(
        begin
            M2 = #{
                <<"session_id">> => ?TEST_SESSION,
                <<"dataset_path">> => ?TEST_DATA,
                <<"num_epochs">> => 1
            },
            {ok, Response} = train(#{}, M2, #{}),
            ?assertEqual(200, maps:get(<<"status">>, Response)),
            Body = hb_json:decode(maps:get(<<"body">>, Response)),
            ?assert(maps:is_key(<<"adapter_path">>, Body)),
            AdapterPath = maps:get(<<"adapter_path">>, Body),
            ?assert(filelib:is_dir(binary_to_list(AdapterPath)))
        end
    ) end}.

convert_only_after_train_test_() ->
    {setup, fun setup/0, fun teardown/1, fun(_) -> ?_test(
        begin
            % Step 1: Train something to have a file to convert
            TrainM2 = #{
                <<"session_id">> => ?TEST_SESSION,
                <<"dataset_path">> => ?TEST_DATA,
                <<"num_epochs">> => 1
            },
            {ok, _} = train(#{}, TrainM2, #{}),

            % Step 2: Now convert it
            ConvertM2 = #{
                <<"session_id">> => ?TEST_SESSION,
                <<"gguf_precision">> => <<"f16">>
            },
            {ok, Response} = convert(#{}, ConvertM2, #{}),
            ?assertEqual(200, maps:get(<<"status">>, Response)),
            Body = hb_json:decode(maps:get(<<"body">>, Response)),
            ?assert(maps:is_key(<<"gguf_path">>, Body)),
            GGUFPath = maps:get(<<"gguf_path">>, Body),
            ?assert(filelib:is_regular(binary_to_list(GGUFPath)))
        end
    ) end}.

missing_params_test() ->
    ?assertMatch({ok, #{<<"status">> := 400}}, train(#{}, #{}, #{})),
    ?assertMatch({ok, #{<<"status">> := 400}}, train(#{}, #{<<"session_id">> => <<"s">>}, #{})),
    ?assertMatch({ok, #{<<"status">> := 400}}, convert(#{}, #{}, #{})),
    ?assertMatch({ok, #{<<"status">> := 404}}, convert(#{}, #{<<"session_id">> => <<"non-existent">>}, #{})) .

-endif.