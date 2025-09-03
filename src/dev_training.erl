-module(dev_training).
-export([finetune_lora/3, check_python_env/3, info/1, info/3]).
-include("include/hb.hrl").
-include_lib("eunit/include/eunit.hrl").

%% @doc Declares the functions this device exports.
info(_) ->
    #{exports => [finetune_lora, check_python_env, info]}.

%% @doc Provides API information for the training device.
info(_M1, _M2, _Opts) ->
    InfoBody = #{
        <<"description">> => <<"A device for orchestrating AI model training pipelines, such as LoRA fine-tuning.">>,
        <<"version">> => <<"1.0">>,
        <<"api">> => #{
            <<"finetune_lora">> => #{
                <<"description">> => <<"Initiates a LoRA fine-tuning job. All parameters are passed in the request body. Returns the path to the trained LoRA adapter upon completion.">>,
                <<"method">> => <<"POST">>,
                <<"parameters">> => #{
                    <<"session_id">> => <<"Required. A unique identifier for the training session. All artifacts will be stored in a directory named after this ID.">>,
                    <<"model_id">> => <<"Required. The Hugging Face model identifier for the base model to be fine-tuned (e.g., 'Qwen/Qwen1.5-0.5B-Chat').">>,
                    <<"dataset_path">> => <<"Required. The path to the training data file (e.g., 'native/pyrust_nn/data.json').">>,
                    <<"num_epochs">> => <<"Optional. Number of training epochs.">>,
                    <<"batch_size">> => <<"Optional. Training batch size.">>,
                    <<"learning_rate">> => <<"Optional. The learning rate for the optimizer.">>,
                    <<"lora_rank">> => <<"Optional. The rank of the LoRA update matrices.">>,
                    <<"lora_alpha">> => <<"Optional. The alpha parameter for LoRA scaling.">>,
                    <<"lora_dropout">> => <<"Optional. Dropout probability for LoRA layers.">>,
                    <<"checkpoint_lora">> => <<"Optional. Path to an existing LoRA adapter to continue training from.">>
                }
            },
            <<"check_python_env">> => #{
                <<"description">> => <<"Verifies that the Python environment is set up correctly and essential libraries like 'torch' and 'transformers' can be imported.">>,
                <<"method">> => <<"POST">>,
                <<"parameters">> => #{
                    <<"session_id">> => <<"Required. A unique session ID to create a log file for the check.">>
                }
            }
        }
    },
    {ok, InfoBody}.

%% @doc Initiates a LoRA fine-tuning job.
finetune_lora(_M1, M2, _Opts) ->
    case {maps:get(<<"session_id">>, M2, undefined), maps:get(<<"model_id">>, M2, undefined)} of
        {undefined, _} ->
            {ok, #{ <<"body">> => hb_json:encode(#{<<"error">> => <<"Missing required parameter: 'session_id'">>}), <<"status">> => 400 }};
        {_, undefined} ->
            {ok, #{ <<"body">> => hb_json:encode(#{<<"error">> => <<"Missing required parameter: 'model_id'">>}), <<"status">> => 400 }};
        {SessionID, ModelID} ->
            NifParams = to_atom_map(M2),
            case maps:is_key(dataset_path, NifParams) of
                true ->
                    case dev_training_nif:finetune_lora_nif(SessionID, ModelID, NifParams) of
                        {ok, OutputPath} ->
                            {ok, #{
                                <<"body">> => hb_json:encode(#{
                                    <<"status">> => <<"success">>,
                                    <<"message">> => <<"LoRA fine-tuning completed.">>,
                                    <<"adapter_path">> => OutputPath
                                }),
                                <<"status">> => 200
                            }};
                        {error, Reason} ->
                            Error = iolist_to_binary(io_lib:format("~p", [Reason])),
                            {ok, #{ <<"body">> => hb_json:encode(#{<<"error">> => <<"LoRA fine-tuning failed">>, <<"reason">> => Error}), <<"status">> => 500 }}
                    end;
                false ->
                    {ok, #{ <<"body">> => hb_json:encode(#{<<"error">> => <<"Missing required parameter: 'dataset_path'">>}), <<"status">> => 400 }}
            end
    end.

%% @doc Checks the Python environment.
check_python_env(_M1, M2, _Opts) ->
    case maps:get(<<"session_id">>, M2, undefined) of
        undefined ->
            {ok, #{ <<"body">> => hb_json:encode(#{<<"error">> => <<"Missing required parameter: 'session_id'">>}), <<"status">> => 400 }};
        SessionID ->
            case dev_training_nif:check_python_env(SessionID) of
                {ok, Message} ->
                    {ok, #{
                        <<"body">> => hb_json:encode(#{
                            <<"status">> => <<"success">>,
                            <<"message">> => Message
                        }),
                        <<"status">> => 200
                    }};
                {error, Reason} ->
                    Error = iolist_to_binary(io_lib:format("~p", [Reason])),
                    {ok, #{ <<"body">> => hb_json:encode(#{<<"error">> => <<"Python environment check failed">>, <<"reason">> => Error}), <<"status">> => 500 }}
            end
    end.


%% --- Private Functions ---

%% @doc Converts a map with binary keys to a map with atom keys for the NIF.
%% Also converts numeric values passed as strings into number types.
to_atom_map(BinMap) ->
    ValidKeys = [
        <<"dataset_path">>, <<"num_epochs">>, <<"batch_size">>,
        <<"learning_rate">>, <<"lora_rank">>, <<"lora_alpha">>,
        <<"lora_dropout">>, <<"checkpoint_lora">>
    ],
    maps:fold(
        fun(K, V, Acc) ->
            case is_binary(K) andalso lists:member(K, ValidKeys) of
                true ->
                    Value = try
                        case is_binary(V) of
                            true ->
                                try binary_to_integer(V)
                                catch error:badarg ->
                                    try binary_to_float(V)
                                    catch error:badarg -> V
                                    end
                                end;
                            false -> V
                        end
                    catch
                        _:_ -> V
                    end,
                    Acc#{binary_to_atom(K, utf8) => Value};
                false ->
                    Acc % Ignore keys that are not valid or not binaries
            end
        end,
        #{},
        BinMap
    ).

%% ===================================================================
%% EUnit Tests
%% ===================================================================
-ifdef(TEST).

-define(TEST_SESSION_ID, <<"eunit_training_session">>).
-define(TEST_MODEL_ID, <<"Qwen/Qwen1.5-0.5B-Chat">>).
-define(TEST_DATA_FILE, <<"native/pyrust_nn/data.json">>).

setup() ->
    % Ensure the session directory does not exist before the test
    file:del_dir_r(filename:join(["runs", binary_to_list(?TEST_SESSION_ID)])).

teardown(_) ->
    % Clean up the session directory after the test
    ok.

check_python_env_device_test_() ->
    {setup,
     fun setup/0,
     fun teardown/1,
     fun(_) ->
        ?_test(
        begin
            M2 = #{ <<"session_id">> => ?TEST_SESSION_ID },
            {ok, Response} = check_python_env(#{}, M2, #{}),
            ?assertEqual(200, maps:get(<<"status">>, Response)),
            Body = hb_json:decode(maps:get(<<"body">>, Response)),
            ?assertEqual(<<"success">>, maps:get(<<"status">>, Body)),
            ?assertEqual(<<"Successfully imported torch and transformers.">>, maps:get(<<"message">>, Body))
        end)
     end}.

finetune_lora_device_success_test_() ->
    {setup,
     fun setup/0,
     fun teardown/1,
     fun(_) ->
        ?_test(
        begin
            M2 = #{
                <<"session_id">> => ?TEST_SESSION_ID,
                <<"model_id">> => ?TEST_MODEL_ID,
                <<"dataset_path">> => ?TEST_DATA_FILE,
                <<"num_epochs">> => 1,
                <<"batch_size">> => 1
            },
            {ok, Response} = finetune_lora(#{}, M2, #{}),
            ?assertEqual(200, maps:get(<<"status">>, Response)),
            Body = hb_json:decode(maps:get(<<"body">>, Response)),
            ?assertEqual(<<"success">>, maps:get(<<"status">>, Body)),
            ?assert(maps:is_key(<<"adapter_path">>, Body))
        end)
     end}.

finetune_lora_device_missing_param_test() ->
    ?_test(
    begin
        % Missing session_id
        M2_no_session = #{ <<"model_id">> => ?TEST_MODEL_ID, <<"dataset_path">> => ?TEST_DATA_FILE },
        {ok, R1} = finetune_lora(#{}, M2_no_session, #{}),
        ?assertEqual(400, maps:get(<<"status">>, R1)),
        ?assertMatch(#{<<"error">> := <<"Missing required parameter: 'session_id'">>}, hb_json:decode(maps:get(<<"body">>, R1))),

        % Missing model_id
        M2_no_model = #{ <<"session_id">> => ?TEST_SESSION_ID, <<"dataset_path">> => ?TEST_DATA_FILE },
        {ok, R2} = finetune_lora(#{}, M2_no_model, #{}),
        ?assertEqual(400, maps:get(<<"status">>, R2)),
        ?assertMatch(#{<<"error">> := <<"Missing required parameter: 'model_id'">>}, hb_json:decode(maps:get(<<"body">>, R2))),

        % Missing dataset_path
        M2_no_data = #{ <<"session_id">> => ?TEST_SESSION_ID, <<"model_id">> => ?TEST_MODEL_ID },
        {ok, R3} = finetune_lora(#{}, M2_no_data, #{}),
        ?assertEqual(400, maps:get(<<"status">>, R3)),
        ?assertMatch(#{<<"error">> := <<"Missing required parameter: 'dataset_path'">>}, hb_json:decode(maps:get(<<"body">>, R3)))
    end).

to_atom_map_helper_test() ->
    ?_test(
    begin
        InputMap = #{
            <<"dataset_path">> => <<"data.json">>,
            <<"num_epochs">> => <<"5">>, % String number
            <<"learning_rate">> => 0.0002, % Float
            <<"unknown_param">> => <<"should be ignored">>
        },
        ExpectedMap = #{
            dataset_path => <<"data.json">>,
            num_epochs => 5,
            learning_rate => 0.0002
        },
        ?assertEqual(ExpectedMap, to_atom_map(InputMap))
    end).

-endif.