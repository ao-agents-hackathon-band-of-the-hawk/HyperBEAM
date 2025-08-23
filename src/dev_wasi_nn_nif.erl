-module(dev_wasi_nn_nif).

-include("include/hb.hrl").

-include_lib("eunit/include/eunit.hrl").

-on_load init/0.

-export([init_backend/0, init_backend_with_config/1, load_by_name_with_config/3,
         init_execution_context/2, close_execution_context/2, deinit_backend/1, run_inference/3,
         run_inference/4, run_inference_with_options/4, set_input/3, compute/2, get_output/3]).
-export([init_execution_context_once/2, switch_model/2]).
-export([cleanup_model_contexts/1, cleanup_all_contexts/0, get_current_model_info/0]).

%% Module-level cache
-define(CACHE_TAB, wasi_nn_cache).
-define(SINGLETON_KEY, global_cache).
-define(CACHE_OWNER_NAME, wasi_nn_cache_owner).  % Registered name for cache owner process

%% Function to start the dedicated ETS table owner process
start_cache_owner() ->
    case whereis(?CACHE_OWNER_NAME) of
        undefined ->
            % No owner process exists, create one
            Pid = spawn(fun() ->
                           % Create the table if it doesn't exist
                           case ets:info(?CACHE_TAB) of
                               undefined ->
                                   ?event(dev_wasi_nn_nif,
                                          {cache_owner_creating_table, ?CACHE_TAB}),
                                   ets:new(?CACHE_TAB, [set, named_table, public]);
                               _ ->
                                   ?event(dev_wasi_nn_nif, {cache_table_already_exists, ?CACHE_TAB})
                           end,
                           % Register the process with a name for easy lookup
                           register(?CACHE_OWNER_NAME, self()),
                           cache_owner_loop()
                        end),
            {ok, Pid};
        Pid ->
            % Owner process already exists
            {ok, Pid}
    end.

%% Loop function for the cache owner process - keeps the process alive
cache_owner_loop() ->
    receive
        stop ->
            ?event(dev_wasi_nn_nif, {cache_owner_stopping}),
            ok;
        {From, ping} ->
            From ! {self(), pong},
            cache_owner_loop();
        _ ->
            cache_owner_loop()
    after 3600000 -> % Stay alive for a long time (1 hour), then check again
        cache_owner_loop()
    end.

%% Create ETS table in a persistent process if it doesn't exist
init() ->
    PrivDir = code:priv_dir(hb),
    Path = filename:join(PrivDir, "wasi_nn"),
    ?event(dev_wasi_nn_nif, {loading_nif_from, Path}),

    % Start the dedicated cache owner process
    start_cache_owner(),

    % No need to create the table here, the owner process handles this
    case erlang:load_nif(Path, 0) of
        ok ->
            ?event(dev_wasi_nn_nif, {nif_loaded_successfully}),
            ok;
        {error, {load_failed, Reason}} ->
            ?event(dev_wasi_nn_nif, {failed_to_load_nif, Reason}),
            exit({load_failed, {load_failed, Reason}});
        {error, Reason} ->
            ?event(dev_wasi_nn_nif, {failed_to_load_nif_with_error, Reason}),
            exit({load_failed, Reason})
    end.

init_backend() ->
    erlang:nif_error("NIF library not loaded").

init_backend_with_config(_Config) ->
    erlang:nif_error("NIF library not loaded").

load_by_name_with_config(_Context, _Path, _Config) ->
    erlang:nif_error("NIF library not loaded").

init_execution_context(_Context, _SessionId) ->
    erlang:nif_error("NIF library not loaded").

close_execution_context(_Context, _ExecContextId) ->
    erlang:nif_error("NIF library not loaded").

set_input(_Context, _ExecContextId, _Input) ->
    erlang:nif_error("NIF library not loaded").

compute(_Context, _ExecContextId) ->
    erlang:nif_error("NIF library not loaded").

get_output(_Context, _ExecContextId, _Index) ->
    erlang:nif_error("NIF library not loaded").

deinit_backend(_Context) ->
    erlang:nif_error("NIF library not loaded").

run_inference(_Context, _ExecContextId, _Prompt) ->
    erlang:nif_error("NIF library not loaded").

run_inference(_Context, _ExecContextId, _Prompt, _Options) ->
    erlang:nif_error("NIF library not loaded").

run_inference_with_options(_Context, _ExecContextId, _Prompt, _Options) ->
    run_inference(_Context, _ExecContextId, _Prompt, _Options).

%% ============================================================================
%% GLOBAL PERSISTENT CONTEXT MANAGEMENT
%% ============================================================================

%% Switch to a different model, creating a new context for each model
switch_model(ModelPath, Config) ->
    ensure_cache_table(),
    ModelKey = {?SINGLETON_KEY, model_context, ModelPath},

    case ets:lookup(?CACHE_TAB, ModelKey) of
        [{_, {ok, Context, CachedConfig}}] when CachedConfig =:= Config ->
            ?event(dev_wasi_nn_nif, {model_already_loaded, ModelPath, reusing_context}),
            % Update current model reference
            ets:insert(?CACHE_TAB, {{?SINGLETON_KEY, current_model}, {ModelPath, Config, Context}}),
            {ok, Context};
        [{_, {ok, OldContext, _OldConfig}}] ->
            ?event(dev_wasi_nn_nif, {model_different_config, ModelPath, reinitializing}),
            % Cleanup old context for this model
            deinit_backend(OldContext),
            % Create new context for this model
            create_model_context(ModelPath, Config);
        [] ->
            ?event(dev_wasi_nn_nif, {model_not_loaded, ModelPath, creating_new_context}),
            create_model_context(ModelPath, Config)
    end.

%% Helper function to create a new model context
create_model_context(ModelPath, Config) ->
    ensure_cache_table(),
    ModelKey = {?SINGLETON_KEY, global_backend, ModelPath},

    % Get or create the global backend context for this model
    case ets:lookup(?CACHE_TAB, ModelKey) of
        [{_, {ok, Context}}] ->
            ?event(dev_wasi_nn_nif, {using_existing_global_backend, ModelPath}),
            load_model_with_context(Context, ModelPath, Config);
        [] ->
            ?event(dev_wasi_nn_nif, {creating_new_global_backend, ModelPath}),
            case init_backend() of
                {ok, Context} ->
                    ets:insert(?CACHE_TAB, {ModelKey, {ok, Context}}),
                    ?event(dev_wasi_nn_nif, {global_backend_created, ModelPath}),
                    load_model_with_context(Context, ModelPath, Config);
                Error ->
                    ?event(dev_wasi_nn_nif, {failed_to_create_global_backend, ModelPath, Error}),
                    Error
            end
    end.

%% Helper function to load model with existing context
load_model_with_context(Context, ModelPath, Config) ->
    ConfigStr = hb_util:list(Config),
    case load_by_name_with_config(Context, ModelPath, ConfigStr) of
        ok ->
            ModelKey = {?SINGLETON_KEY, model_context, ModelPath},
            ets:insert(?CACHE_TAB, {ModelKey, {ok, Context, Config}}),
            ets:insert(?CACHE_TAB, {{?SINGLETON_KEY, current_model}, {ModelPath, Config, Context}}),
            ?event(dev_wasi_nn_nif, {model_context_created, ModelPath}),
            {ok, Context};
        Error ->
            ?event(dev_wasi_nn_nif, {failed_to_load_model, ModelPath, Error}),
            % Cleanup the backend context since model loading failed
            deinit_backend(Context),
            ets:delete(?CACHE_TAB, {?SINGLETON_KEY, global_backend, ModelPath}),
            {error, {model_load_failed, Error}}
    end.

%% Get information about the currently loaded model
get_current_model_info() ->
    ensure_cache_table(),
    case ets:lookup(?CACHE_TAB, {?SINGLETON_KEY, current_model}) of
        [{_, {ModelPath, Config, Context}}] ->
            {ok, {ModelPath, Config, Context}};
        [] ->
            {error, no_model_loaded}
    end.

%% Clean up all contexts for a specific model
cleanup_model_contexts(ModelPath) ->
    ensure_cache_table(),
    % Clean up all execution contexts for this model
    ets:match_delete(?CACHE_TAB,
                     {{?SINGLETON_KEY, context_initialized, ModelPath, '_'}, '_'}),
    % Clean up the model context
    case ets:lookup(?CACHE_TAB, {?SINGLETON_KEY, model_context, ModelPath}) of
        [{_, {ok, Context, _Config}}] ->
            deinit_backend(Context),
            ets:delete(?CACHE_TAB, {?SINGLETON_KEY, model_context, ModelPath}),
            ets:delete(?CACHE_TAB, {?SINGLETON_KEY, global_backend, ModelPath}),
            ?event(dev_wasi_nn_nif, {cleaned_up_contexts, ModelPath}),
            ok;
        [] ->
            ?event(dev_wasi_nn_nif, {no_context_to_cleanup, ModelPath}),
            ok
    end.

%% Clean up all cached contexts (useful for testing or memory management)
cleanup_all_contexts() ->
    ensure_cache_table(),
    % Get all model contexts and clean them up
    ModelContexts =
        ets:match(?CACHE_TAB, {{?SINGLETON_KEY, model_context, '$1'}, {ok, '$2', '$3'}}),

    % Clear the entire cache
    ets:delete_all_objects(?CACHE_TAB),
    ?event(dev_wasi_nn_nif, {all_contexts_cleaned_up}),
    ok.

%% Helper function to safely access the ETS table
ensure_cache_table() ->
    case ets:info(?CACHE_TAB) of
        undefined ->
            % Start the cache owner which will create the table
            ?event(dev_wasi_nn_nif, {table_doesnt_exist, starting_cache_owner}),
            start_cache_owner();
        _ ->
            % Table exists, ensure owner process is running
            case whereis(?CACHE_OWNER_NAME) of
                undefined ->
                    % Strange case: table exists but no owner - restart owner
                    ?event(dev_wasi_nn_nif, {table_exists_no_owner, restarting_owner}),
                    start_cache_owner();
                _ ->
                    % All good, table exists and owner is running
                    ok
            end
    end.

%% Function to ensure execution context is only initialized once per session and model
init_execution_context_once(Context, SessionId) ->
    ensure_cache_table(),
    % Get current model info to create a unique session key per model
    case get_current_model_info() of
        {ok, {ModelPath, _Config, _Context}} ->
            SessionKey = {?SINGLETON_KEY, context_initialized, ModelPath, SessionId},
            case ets:lookup(?CACHE_TAB, SessionKey) of
                [{_, {ok, ExecContextId}}] ->
                    ?event(dev_wasi_nn_nif,
                           {execution_context_already_initialized, SessionId, ModelPath}),
                    {ok, ExecContextId};
                [] ->
                    Result = init_execution_context(Context, SessionId),
                    case Result of
                        {ok, ExecContextId} ->
                            ets:insert(?CACHE_TAB, {SessionKey, {ok, ExecContextId}}),
                            ?event(dev_wasi_nn_nif,
                                   {execution_context_initialized, SessionId, ModelPath}),
                            {ok, ExecContextId};
                        Error ->
                            ?event(dev_wasi_nn_nif,
                                   {failed_to_initialize_execution_context,
                                    SessionId,
                                    ModelPath,
                                    Error}),
                            Error
                    end
            end;
        {error, no_model_loaded} ->
            ?event(dev_wasi_nn_nif, {no_model_loaded, cannot_initialize_execution_context}),
            {error, no_model_loaded}
    end.

%% Test basic inference functionality
basic_inference_test() ->
    Path = "models/qwen2.5-14b-instruct-q2_k.gguf",
    Config =
        "{\"n_gpu_layers\":98,\"ctx_size\":2048,\"stream-stdout\":true,\"enab"
        "le_debug_log\":true}",
    SessionId = "test_session_1",
    Prompt1 = "What is the meaning of life",

    % Test 1: Load first model (should create new context)
    {ok, Context1} = switch_model(Path, Config),
    ?event(dev_wasi_nn_nif, {model_loaded, Context1, Path, Config}),

    % Test 2: Create session and run inference
    {ok, ExecContextId1} = init_execution_context_once(Context1, SessionId),
    {ok, Output1} = run_inference(Context1, ExecContextId1, Prompt1),
    ?event(dev_wasi_nn_nif, {run_inference, Context1, ExecContextId1, Prompt1, Output1}),
    ?assertNotEqual(Output1, ""),

    % Cleanup all contexts
    cleanup_all_contexts(),
    ok.

%% Test session management capabilities with more detailed conversation
session_management_test() ->
    Path = "models/qwen2.5-14b-instruct-q2_k.gguf",
    Config =
        "{\"n_gpu_layers\":98,\"ctx_size\":2048,\"stream-stdout\":true,\"enab"
        "le_debug_log\":true}",

    % Load model
    {ok, Context} = switch_model(Path, Config),

    % Create multiple sessions
    {ok, Session1} = init_execution_context_once(Context, "session_1"),
    {ok, Session2} = init_execution_context_once(Context, "session_2"),

    % Verify sessions are different
    ?assertNotEqual(Session1, Session2),

    % Run inference in both sessions
    {ok, Response1} = run_inference(Context, Session1, "Hello, my name is Alice."),
    ?event(dev_wasi_nn_nif, {session1_response1, Response1}),

    % This should remember Alice's name
    {ok, Response2} = run_inference(Context, Session1, "What is my name?"),
    ?event(dev_wasi_nn_nif, {session1_response2, Response2}),

    % Verify that the response mentions Alice
    AliceInResponse =
        case string:find(Response2, "Alice") of
            nomatch ->
                false;
            _ ->
                true
        end,

    % Session 2 should not know Alice's name
    {ok, Response3} = run_inference(Context, Session2, "What is my name?"),
    ?event(dev_wasi_nn_nif, {session2_response1, Response3}),

    % Verify the responses
    ?assert(AliceInResponse, "Session 1 should remember Alice's name"),

    % Cleanup
    cleanup_all_contexts(),
    ok.

% {
%   "temperature": 0.8,
%   "top_p": 0.95,
%   "top_k": 40,
%   "max_tokens": 512,
%   "repeat_penalty": 1.1,
%   "stop": ["\n\n", ""]
% }
%% Test inference with options
inference_with_options_test() ->
    Path = "models/qwen2.5-14b-instruct-q2_k.gguf",
    Config =
        "{\"n_gpu_layers\":98,\"ctx_size\":2048,\"stream-stdout\":true,\"enab"
        "le_debug_log\":true}",

    {ok, Context} = switch_model(Path, Config),

    Prompt = "Write one short sentence about the weather",

    % Use low temperature for more deterministic output
    OptionsLowTemp = "{\"seed\":42,\"temperature\":0.1}",
    {ok, ExecContextId1} = init_execution_context_once(Context, "session_1"),
    {ok, OutputLowTemp} =
        run_inference_with_options(Context, ExecContextId1, Prompt, OptionsLowTemp),
    ?assertNotEqual(OutputLowTemp, ""),

    % Ensure we get same output with same options
    {ok, ExecContextId2} = init_execution_context_once(Context, "session_2"),
    {ok, OutputLowTemp2} =
        run_inference_with_options(Context, ExecContextId2, Prompt, OptionsLowTemp),
    ?assertEqual(OutputLowTemp2, OutputLowTemp),

    % Use high temperature for more creative output
    {ok, ExecContextId3} = init_execution_context_once(Context, "session_3"),
    OptionsHighTemp = "{\"seed\":42,\"temperature\":1.0}",
    {ok, OutputHighTemp} =
        run_inference_with_options(Context, ExecContextId3, Prompt, OptionsHighTemp),
    ?assertNotEqual(OutputHighTemp, OutputLowTemp),

    % Cleanup
    cleanup_all_contexts(),
    ok.

%% Test model switching with different configurations
model_switching_with_config_test() ->
    Path = "models/qwen2.5-14b-instruct-q2_k.gguf",
    Path2 = "models/ISrbGzQot05rs_HKC08O_SmkipYQnqgB1yC3mjZZeEo.gguf",
    Config1 = "{\"n_gpu_layers\":98,\"ctx_size\":2048}",
    Config2 = "{\"n_gpu_layers\":48,\"ctx_size\":1024}",
    SessionId = "config_test_session",

    % Load model with first config
    {ok, Context1} = switch_model(Path, Config1),
    {ok, ExecContextId1} = init_execution_context_once(Context1, SessionId),
    {ok, Output1} = run_inference(Context1, ExecContextId1, "Test with config 1"),
    ?assertNotEqual(Output1, ""),

    % Switch to same model with different config
    {ok, Context2} = switch_model(Path2, Config2),
    {ok, ExecContextId2} = init_execution_context_once(Context2, SessionId),
    {ok, Output2} = run_inference(Context2, ExecContextId2, "Test with config 2"),
    ?assertNotEqual(Output2, ""),

    % Cleanup
    cleanup_all_contexts(),
    ok.
