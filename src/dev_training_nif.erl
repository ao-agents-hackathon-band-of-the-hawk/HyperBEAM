-module(dev_training_nif).
-export([
    finetune_lora_nif/3,
    check_python_env/1
]).
-on_load(init/0).

-include("include/cargo.hrl").

init() ->
    ?load_nif_from_crate(pyrust_nn, 0).

%% @doc NIF wrapper for finetune_lora.
-spec finetune_lora_nif(binary(), binary(), map()) -> {ok, binary()} | {error, binary()}.
finetune_lora_nif(_SessionID, _ModelID, _Params) ->
    erlang:nif_error(nif_not_loaded).

%% @doc NIF wrapper for a simple Python environment check.
-spec check_python_env(binary()) -> {ok, binary()} | {error, binary()}.
check_python_env(_SessionID) ->
    erlang:nif_error(nif_not_loaded).

%% ===================================================================
%% EUnit Tests
%% ===================================================================
-ifdef(TEST).
-include_lib("eunit/include/eunit.hrl").

check_python_env_test() ->
    SessionID = <<"eunit-check-env-session">>,
    ?assertMatch({ok, <<"Successfully imported torch and transformers.">>}, check_python_env(SessionID)),
    file:del_dir_r(filename:join(["runs", binary_to_list(SessionID)])).

%% @doc This is an integration test and requires a full Python environment.
finetune_lora_nif_success_test() ->
    SessionID = <<"eunit-lora-session">>,
    ModelID = <<"Qwen/Qwen1.5-0.5B-Chat">>,
    DataFile = <<"native/pyrust_nn/data.json">>,

    % FIX: Use atoms for the map keys to match the manual Rust decoder.
    Params = #{
        dataset_path => DataFile,
        num_epochs => 1,
        batch_size => 1,
        lora_rank => 4,
        lora_alpha => 8,
        lora_dropout => 0.05
    },

    Result = finetune_lora_nif(SessionID, ModelID, Params),

    ?assertMatch({ok, _OutputPath}, Result),
    case Result of
        {ok, Path} ->
            ?assert(is_binary(Path)),
            ?assert(filelib:is_dir(binary_to_list(Path))),
            ?debugFmt("LoRA adapter successfully saved to: ~s", [Path]);
        {error, Reason} ->
            ?debugFmt("LoRA fine-tuning failed unexpectedly: ~p", [Reason]),
            ?assert(false)
    end,
    file:del_dir_r(filename:join(["runs", binary_to_list(SessionID)])).


finetune_lora_nif_failure_test() ->
    SessionID = <<"eunit-lora-fail-session">>,
    ModelID = <<"non-existent-model-abc/123">>,
    DataFile = <<"non-existent-data.json">>,

    % FIX: Use atoms for the map keys.
    Params = #{
        dataset_path => DataFile,
        num_epochs => 1,
        batch_size => 1
    },
    ?assertMatch({error, _Reason}, finetune_lora_nif(SessionID, ModelID, Params)),
    file:del_dir_r(filename:join(["runs", binary_to_list(SessionID)])).


invalid_argument_type_test_() ->
    ParamsWithRequired = #{dataset_path => <<"path">>},
    ParamsWithoutRequired = #{num_epochs => 1},
    [
        {"SessionID as list", ?_assertError(badarg, finetune_lora_nif("not a binary", <<"m">>, ParamsWithRequired))},
        {"ModelID as integer", ?_assertError(badarg, finetune_lora_nif(<<"s">>, 123, ParamsWithRequired))},
        {"Params missing required field", ?_assertError(badarg, finetune_lora_nif(<<"s">>, <<"m">>, ParamsWithoutRequired))}
    ].

-endif.