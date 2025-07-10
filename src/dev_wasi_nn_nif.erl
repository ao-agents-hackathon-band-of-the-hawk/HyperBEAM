-module(dev_wasi_nn_nif).
-include("include/hb.hrl").
-include_lib("eunit/include/eunit.hrl").
-on_load(init/0).
-export([
    init_backend/0,
    load_by_name_with_config/3,
    init_execution_context/2,
    close_execution_context/2,
    deinit_backend/1,
    run_inference/3
]).
-export([init_execution_context_once/2, switch_model/2]).

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
                        io:format("Cache owner creating table ~p~n", [?CACHE_TAB]),
                        ets:new(?CACHE_TAB, [set, named_table, public]);
                    _ ->
                        io:format("Cache table ~p already exists, taking ownership~n", [?CACHE_TAB])
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
            io:format("Cache owner stopping~n"),
            ok;
        {From, ping} ->
            From ! {self(), pong},
            cache_owner_loop();
        _ -> 
            cache_owner_loop()
    after 
        3600000 -> % Stay alive for a long time (1 hour), then check again
            cache_owner_loop()
    end.

%% Create ETS table in a persistent process if it doesn't exist
init() ->
    PrivDir = code:priv_dir(hb),
    Path = filename:join(PrivDir, "wasi_nn"),
    io:format("Loading NIF from: ~p~n", [Path]),
    
    % Start the dedicated cache owner process
    start_cache_owner(),
    
    % No need to create the table here, the owner process handles this
    case erlang:load_nif(Path, 0) of
        ok ->
            io:format("NIF loaded successfully~n"),
            ok;
        {error, {load_failed, Reason}} ->
            io:format("Failed to load NIF: ~p~n", [Reason]),
            exit({load_failed, {load_failed, Reason}});
        {error, Reason} ->
            io:format("Failed to load NIF with error: ~p~n", [Reason]),
            exit({load_failed, Reason})
    end.

init_backend() ->
    erlang:nif_error("NIF library not loaded").

load_by_name_with_config(_Context, _Path, _Config) ->
    erlang:nif_error("NIF library not loaded").

init_execution_context(_Context, _SessionId) ->
    erlang:nif_error("NIF library not loaded").

close_execution_context(_Context, _ExecContextId) ->
    erlang:nif_error("NIF library not loaded").

% set_input(_Context, _Prompt) ->
%     erlang:nif_error("NIF library not loaded").

% compute(_Context) ->
%     erlang:nif_error("NIF library not loaded").
% get_output(_Context) ->
%     erlang:nif_error("NIF library not loaded").
deinit_backend(_Context) ->
    erlang:nif_error("NIF library not loaded").
run_inference(_Context, _ExecContextId, _Prompt) ->
    erlang:nif_error("NIF library not loaded").

%% ============================================================================
%% GLOBAL PERSISTENT CONTEXT MANAGEMENT
%% ============================================================================

%% Get or create the global llama.cpp backend context (persistent across processes)
get_or_create_global_context() ->
    ensure_cache_table(),
    case ets:lookup(?CACHE_TAB, {?SINGLETON_KEY, global_backend}) of
        [{_, {ok, Context}}] ->
            io:format("Using existing global backend context~n"),
            {ok, Context};
        [] ->
            io:format("Creating new global backend context~n"),
            case init_backend() of
                {ok, Context} ->
                    ets:insert(?CACHE_TAB, {{?SINGLETON_KEY, global_backend}, {ok, Context}}),
                    io:format("Global backend context created and cached~n"),
                    {ok, Context};
                Error ->
                    io:format("Failed to create global backend: ~p~n", [Error]),
                    Error
            end
    end.

%% Switch to a different model while keeping the same backend
switch_model(ModelPath, Config) ->
    ensure_cache_table(),
    case get_or_create_global_context() of
        {ok, Context} ->
            ModelKey = {?SINGLETON_KEY, current_model},
            CurrentModel = case ets:lookup(?CACHE_TAB, ModelKey) of
                [{_, ModelInfo}] -> ModelInfo;
                [] -> undefined
            end,
            
            NewModelInfo = {ModelPath, Config},
            
            case CurrentModel of
                NewModelInfo ->
                    io:format("Model ~p already loaded, skipping~n", [ModelPath]),
                    {ok, Context};
                _ ->
                    io:format("Switching from ~p to ~p~n", [CurrentModel, NewModelInfo]),
                    case load_by_name_with_config(Context, ModelPath, Config) of
                        ok ->
                            ets:insert(?CACHE_TAB, {ModelKey, NewModelInfo}),
                            io:format("Model switched successfully to ~p~n", [ModelPath]),
                            {ok, Context};
                        Error ->
                            io:format("Failed to switch model: ~p~n", [Error]),
                            {error, {model_switch_failed, Error}}
                    end
            end;
        Error ->
            Error
    end.

%% Get information about the currently loaded model
get_current_model_info() ->
    ensure_cache_table(),
    case ets:lookup(?CACHE_TAB, {?SINGLETON_KEY, current_model}) of
        [{_, ModelInfo}] -> {ok, ModelInfo};
        [] -> {error, no_model_loaded}
    end.

%% Helper function to safely access the ETS table
ensure_cache_table() ->
    case ets:info(?CACHE_TAB) of
        undefined ->
            % Start the cache owner which will create the table
            io:format("Table doesn't exist, starting cache owner process~n"),
            start_cache_owner();
        _ ->
            % Table exists, ensure owner process is running
            case whereis(?CACHE_OWNER_NAME) of
                undefined ->
                    % Strange case: table exists but no owner - restart owner
                    io:format("Table exists but no owner, restarting owner process~n"),
                    start_cache_owner();
                _ ->
                    % All good, table exists and owner is running
                    ok
            end
    end.

%% Function to ensure execution context is only initialized once per session
init_execution_context_once(Context, SessionId) ->
    ensure_cache_table(),
    SessionKey = {?SINGLETON_KEY, context_initialized, SessionId},
    case ets:lookup(?CACHE_TAB, SessionKey) of
        [{_, {ok, ExecContextId}}] ->
            io:format("Execution context already initialized for session ~p~n", [SessionId]),
            {ok, ExecContextId};
        [] ->
            ModelContext =
                case ets:lookup(?CACHE_TAB, {?SINGLETON_KEY, model_loaded}) of
                    [{_, {ok, StoredContext}}] ->
                        StoredContext;
                    [] ->
                        % If we don't have a cached context, use the provided one
                        % This is a fallback but ideally switch_model should be called first
                        Context
                end,

            Result = init_execution_context(ModelContext, SessionId),
            case Result of
                {ok, ExecContextId} ->
                    ets:insert(?CACHE_TAB, {SessionKey, {ok, ExecContextId}}),
                    {ok, ExecContextId};
                Error ->
                    Error
            end
    end.

run_inference_test() ->
    Path = "test/qwen2.5-14b-instruct-q2_k.gguf",
    Config =
        "{\"n_gpu_layers\":98,\"ctx_size\":2048,\"stream-stdout\":true,\"enable_debug_log\":true}",
    SessionId = "test_session_1",
    Prompt1 = "What is the meaning of life",
    
    % Test 1: Use persistent context management
    {ok, Context} = switch_model(Path, Config),
    ?event(dev_wasi_nn_nif, {persistent_load, Context, Path, Config}),
    
    % Test 2: Create session and run inference
    {ok, ExecContextId} = init_execution_context(Context, SessionId),
    {ok, Output1} = run_inference(Context, ExecContextId, Prompt1),
	?event(dev_wasi_nn_nif, {run_inference, Context, ExecContextId, Prompt1, Output1}),
    ?assertNotEqual(Output1, ""),
    
    % Test 3: Multiple inference calls on same session
    Prompt2 = "Who are you",
    {ok, Output2} = run_inference(Context, ExecContextId, Prompt2),
	?event(dev_wasi_nn_nif, {run_inference, Context, ExecContextId, Prompt2, Output2}),
    ?assertNotEqual(Output2, ""),
    
    % Test 4: Test that context is reused (should be very fast)
    StartTime = erlang:system_time(millisecond),
    {ok, Context2} = switch_model(Path, Config),
    EndTime = erlang:system_time(millisecond),
    ReuseDuration = EndTime - StartTime,
    ?event(dev_wasi_nn_nif, {context_reuse, ReuseDuration}),
    ?assert(ReuseDuration < 100), % Should be nearly instant
    ?assertEqual(Context, Context2), % Should be the same context
    
    % Test 5: Check current model info
    {ok, ModelInfo} = get_current_model_info(),
    ?assertEqual({Path, Config}, ModelInfo),
    
    % Cleanup
    close_execution_context(Context, ExecContextId).
