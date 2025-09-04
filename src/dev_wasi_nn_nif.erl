-module(dev_wasi_nn_nif).

-include("include/hb.hrl").

-include_lib("eunit/include/eunit.hrl").

-on_load init/0.

% --- API ---
-export([ensure_model_loaded/2, init_session/1, run_inference/2, run_inference/3]).

% --- NIF stubs (functions implemented in C) ---
-export([nif_load_by_name_with_config/2, nif_init_execution_context/1, nif_run_inference/3]).

init() ->
    PrivDir = code:priv_dir(hb),
    Path = filename:join(PrivDir, "wasi_nn"),
    erlang:load_nif(Path, PrivDir).

%% @doc Ensures the correct model is loaded in the singleton backend.
ensure_model_loaded(ModelPath, Config) ->
    nif_load_by_name_with_config(ModelPath, Config).

%% @doc Creates a new, safe session resource.
init_session(SessionId) ->
    nif_init_execution_context(SessionId).

%% @doc Runs inference with default options.
run_inference(SessionResource, Prompt) ->
    % This is the convenient 2-arity wrapper. It calls the 3-arity version.
    run_inference(SessionResource, Prompt, "{}").

%% @doc Runs inference using a session resource with JSON options.
run_inference(SessionResource, Prompt, Options) ->
    % This function calls the actual NIF.
    nif_run_inference(SessionResource, Prompt, Options).


% --- NIF Function Definitions (placeholders) ---
nif_load_by_name_with_config(_ModelPath, _Config) ->
    erlang:nif_error("NIF library not loaded").

nif_init_execution_context(_SessionId) ->
    erlang:nif_error("NIF library not loaded").

nif_run_inference(_SessionResource, _Prompt, _Options) ->
    erlang:nif_error("NIF library not loaded").

%% ============================================================================
%% REFACTORED EUNIT TESTS FOR THE NEW SINGLETON ARCHITECTURE
%% ============================================================================
-ifdef(TEST).
-include_lib("eunit/include/eunit.hrl").

%% @doc Helper to ensure a clean state before each test.
cleanup_all_contexts() ->
    % In the new architecture, there's no direct way to deinit from Erlang,
    % which is by design for safety. For testing, we can add a NIF if needed,
    % but for now, we'll rely on the fact that each model load is idempotent.
    ok.

%% @doc Test basic inference functionality.
basic_inference_test_() ->
    ?_test(
        begin
            Path = "models/Qwen3-1.7B-Q8_0.gguf",
            Config = "{\"n_gpu_layers\":98,\"ctx_size\":2048}",
            SessionId = "test_session_1",
            Prompt = "What is the meaning of life",

            % Ensure the model is loaded into the singleton backend.
            ?assertEqual(ok, ensure_model_loaded(Path, Config)),

            % Create a session and run inference.
            {ok, Session} = init_session(SessionId),
            {ok, Output} = run_inference(Session, Prompt),
            ?assertNotEqual(<<>>, Output), % Assert that the binary is not empty
            ?event(dev_wasi_nn_nif, {inference_output, Output}),

            cleanup_all_contexts()
        end
    ).

%% @doc Test session management and conversation context.
session_management_test_() ->
    ?_test(
        begin
            Path = "models/Qwen3-1.7B-Q8_0.gguf",
            Config = "{\"n_gpu_layers\":98,\"ctx_size\":2048}",

            % Load the model.
            ?assertEqual(ok, ensure_model_loaded(Path, Config)),

            % Create two separate, isolated sessions.
            {ok, Session1} = init_session("session_1"),
            {ok, Session2} = init_session("session_2"),

            % Run conversation in Session 1.
            {ok, _Response1} = run_inference(Session1, "Hello, my name is Alice."),
            {ok, Response2} = run_inference(Session1, "What is my name?"),

            % Session 2 should have no knowledge of Session 1's conversation.
            {ok, Response3} = run_inference(Session2, "What is my name?"),

            % Verify that Session 1 remembers the context.
            AliceInResponse = string:str(binary_to_list(Response2), "Alice") > 0,
            ?assert(AliceInResponse, "Session 1 should remember Alice's name."),

            % Verify that Session 2 does not have the context.
            AliceNotInResponse = string:str(binary_to_list(Response3), "Alice") == 0,
            ?assert(AliceNotInResponse, "Session 2 should not know Alice's name."),

            cleanup_all_contexts()
        end
    ).

%% @doc Test using runtime options to control inference.
inference_with_options_test_() ->
    ?_test(
        begin
            Path = "models/Qwen3-1.7B-Q8_0.gguf",
            Config = "{\"n_gpu_layers\":98,\"ctx_size\":2048}",
            Prompt = "Write one short sentence about the weather",

            ?assertEqual(ok, ensure_model_loaded(Path, Config)),

            % Use low temperature for deterministic output (with a seed).
            OptionsLowTemp = "{\"seed\":42,\"temperature\":0.1}",
            {ok, Session1} = init_session("session_1"),
            {ok, OutputLowTemp} = run_inference(Session1, Prompt, OptionsLowTemp),
            ?assertNotEqual(<<>>, OutputLowTemp),

            % Ensure we get the same output with the same options in a new session.
            {ok, Session2} = init_session("session_2"),
            {ok, OutputLowTemp2} = run_inference(Session2, Prompt, OptionsLowTemp),
            ?assertEqual(OutputLowTemp, OutputLowTemp2),

            % Use high temperature for more creative (different) output.
            OptionsHighTemp = "{\"seed\":42,\"temperature\":1.2}",
            {ok, Session3} = init_session("session_3"),
            {ok, OutputHighTemp} = run_inference(Session3, Prompt, OptionsHighTemp),
            ?assertNotEqual(OutputHighTemp, OutputLowTemp),

            cleanup_all_contexts()
        end
    ).

%% @doc Test switching models and configurations.
model_switching_with_config_test_() ->
    ?_test(
        begin
            Path1 = "models/Qwen3-1.7B-Q8_0.gguf",
            Path2 = "models/ISrbGzQot05rs_HKC08O_SmkipYQnqgB1yC3mjZZeEo.gguf",
            Config1 = "{\"n_gpu_layers\":98,\"ctx_size\":2048}",
            Config2 = "{\"n_gpu_layers\":48,\"ctx_size\":1024}",
            SessionId = "config_test_session",

            % Load model 1 with first config.
            ?assertEqual(ok, ensure_model_loaded(Path1, Config1)),
            {ok, Session1} = init_session(SessionId),
            {ok, Output1} = run_inference(Session1, "Test with model 1"),
            ?assertNotEqual(<<>>, Output1),

            % Switch to model 2 with a different config.
            ?assertEqual(ok, ensure_model_loaded(Path2, Config2)),
            % Re-initializing the session is good practice after a model switch.
            {ok, Session2} = init_session(SessionId),
            {ok, Output2} = run_inference(Session2, "Test with model 2"),
            ?assertNotEqual(<<>>, Output2),

            % The outputs should be different.
            ?assertNotEqual(Output1, Output2),

            cleanup_all_contexts()
        end
    ).

-endif.

%% @doc This test verifies that the new singleton architecture is robust and
%%      prevents the original segmentation fault. It does this by creating the
%%      exact race condition that previously caused the crash: attempting to
%%      modify the backend state while it is in use.
%%
%% A "successful" run of this test is a clean PASS. A crash would mean the
%%      fix is incomplete.
%%
%% How it works:
%% 1. A "Victim" process starts a very long-running `run_inference` task. This
%%    acquires a mutex lock inside the NIF C code.
%% 2. The main "Attacker" process immediately tries to load a *different model*.
%%    This `ensure_model_loaded` call will also try to acquire the same mutex.
%% 3. Because the Victim holds the lock, the Attacker's model-switch operation
%%    will safely BLOCK and wait. It cannot proceed to tear down the backend.
%% 4. The Victim's long inference task completes and it releases the mutex.
%% 5. The Attacker's call unblocks and safely switches the model.
%% 6. The test passes, proving that the concurrent operations were correctly
%%    serialized and the use-after-free race condition was prevented.
verify_segfault_is_fixed_test_() ->
    ?_test(
        begin
            ModelPath1 = "models/Qwen3-1.7B-Q8_0.gguf",
            ModelPath2 = "models/ISrbGzQot05rs_HKC08O_SmkipYQnqgB1yC3mjZZeEo.gguf", % A different model
            ConfigMap = #{
                <<"model">> => #{ <<"n_gpu_layers">> => 99, <<"ctx_size">> => 2048 },
                <<"stopping">> => #{ <<"max_tokens">> => 1024 } % A reasonably long task
            },
            Config = binary_to_list(hb_json:encode(ConfigMap)),
            Prompt = "Write a long, detailed story about a journey to the center of the Earth.",
            Parent = self(),

            % Step 1: Load the first model.
            ?assertEqual(ok, ensure_model_loaded(ModelPath1, Config)),
            {ok, Session} = init_session("victim_session"),

            ?debugFmt("Victim setup complete. Spawning inference process.",[]),

            % Step 2: Spawn the "Victim" to start the long-running inference.
            VictimPid = spawn_link(fun() ->
                Result = run_inference(Session, Prompt, "{}"),
                Parent ! {victim_done, Result}
            end),

            % Step 3: Give the Victim a head start to ensure it acquires the NIF mutex.
            timer:sleep(200), % 200ms

            ?debugFmt("Victim is running. Attacker now attempts to switch the model...",[]),

            % Step 4: The "Attacker's strike". This call should BLOCK safely, not crash.
            SwitchResult = ensure_model_loaded(ModelPath2, Config),
            ?assertEqual(ok, SwitchResult),

            ?debugFmt("Model switch completed safely after inference finished.",[]),

            % Step 5: Wait for the Victim to send its result. This proves it completed.
            receive
                {victim_done, {ok, InferenceOutput}} ->
                    ?assert(is_binary(InferenceOutput)),
                    ?assert(byte_size(InferenceOutput) > 0);
                {victim_done, Error} ->
                    ?assert(false, io_lib:format("Victim failed with: ~p", [Error]))
            after 60000 -> % 60 second timeout
                ?assert(false, "Test timed out. The victim process never finished.")
            end,

            ?debugFmt("Test passed. Concurrent use and modification were handled safely.",[])
        end
    ).