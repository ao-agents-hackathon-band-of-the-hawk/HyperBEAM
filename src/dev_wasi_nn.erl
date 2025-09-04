%%% @doc WASI-NN device implementation for HyperBEAM
%%% Implements wasi_nn API functions as imported functions by WASM modules
-module(dev_wasi_nn).

-export([info/1, info/3, infer/3, infer_sec/3, save_llm_response/2]).

-include("include/hb.hrl").
-include_lib("eunit/include/eunit.hrl").

-define(SESSIONS_DIR, "sessions").

%% @doc Exported function for getting device info, controls which functions are
%% exposed via the device API.
info(_) ->
    #{exports => [info, infer, infer_sec]}.

%% @doc HTTP info response providing information about this device
info(_Msg1, _Msg2, _Opts) ->
    InfoBody =
        #{<<"description">> => <<"AI device for handling Inference with optional LoRA adapter loading.">>,
          <<"version">> => <<"1.1">>,
          <<"api">> =>
              #{<<"infer">> =>
                    #{<<"description">> => <<"AI Inference. Can dynamically load a LoRA adapter.">>,
                      <<"method">> => <<"GET or POST">>,
                      <<"params">> =>
                          #{<<"prompt">> => #{<<"required">> => true, <<"description">> => <<"Prompt for Infer">>},
                            <<"model-id">> => #{<<"required">> => false, <<"description">> => <<"Arweave TX ID of the model file. Defaults to a preloaded model.">>},
                            <<"lora_id">> => #{<<"required">> => false, <<"description">> => <<"ID of a LoRA adapter. Can be a session ID from a local training run (e.g., 'pipeline_test_session') or an Arweave TX ID of a .gguf LoRA file.">>},
                            <<"lora_scale">> => #{<<"required">> => false, <<"description">> => <<"The weight of the LoRA adapter. Defaults to 1.0.">>}
                           }},
                <<"infer_sec">> =>
                    #{<<"description">> => <<"AI Inference with Attestation Token and optional LoRA.">>,
                      <<"method">> => <<"GET or POST">>,
                      <<"params">> =>
                          #{<<"prompt">> => #{<<"required">> => true, <<"description">> => <<"Prompt for Infer">>},
                            <<"model-id">> => #{<<"required">> => false, <<"description">> => <<"Arweave TX ID of the model file.">>},
                            <<"lora_id">> => #{<<"required">> => false, <<"description">> => <<"ID of a LoRA adapter.">>},
                            <<"lora_scale">> => #{<<"required">> => false, <<"description">> => <<"The weight of the LoRA adapter. Defaults to 1.0.">>},
                            <<"config">> => #{<<"required">> => false, <<"description">> => <<"Attestation token configuration">>}}}}},
    {ok, InfoBody}.

infer(M1, M2, Opts) ->
    TxID = maps:get(<<"model-id">>, M2, undefined),
    DefaultModel = <<"Qwen3-1.7B-Q8_0.gguf">>,

    ModelPath =
        case TxID of
            undefined ->
                DefaultModel;
            _ ->
                case download_and_store_model(TxID) of
                    {ok, LocalModelPath} ->
                        LocalModelPath;
                    {error, Reason} ->
                        ?event(dev_wasi_nn, {model_download_failed, TxID, Reason}),
                        DefaultModel
                end
        end,
    
    UpdatedM2 = M2#{<<"model_path">> => <<"models/", ModelPath/binary>>},

    % --- New LoRA loading logic ---
    case maps:get(<<"lora_id">>, M2, undefined) of
        undefined ->
            % No lora_id, proceed as before
            load_and_infer(M1, UpdatedM2, Opts);
        LoraID ->
            % A lora_id is present, resolve it to a file path
            case resolve_lora_path(LoraID) of
                {ok, LoraPath} ->
                    load_and_infer(M1, UpdatedM2#{<<"lora_path">> => LoraPath}, Opts);
                % FIX: Use a unique variable name for the error reason.
                {error, LoraReason} ->
                    ErrorBody = hb_json:encode(#{
                        <<"error">> => <<"LoRA adapter not found">>,
                        <<"reason">> => LoraReason,
                        <<"lora_id">> => LoraID
                    }),
                    {ok, #{<<"body">> => ErrorBody, <<"status">> => 404}}
            end
    end.

infer_sec(M1, M2, Opts) ->
    case dev_cc:generate(#{},
                         #{nonce =>
                               <<"da4a06c3604a5fac8aa0b4aaf5a6354cdd0dc7c193299bc3464f30b5cbfb931a">>},
                         Opts)
    of
        {ok, TokenJSON} ->
            case infer(M1, M2, Opts) of
                {ok, Result} ->
                    ExistingBody = maps:get(<<"body">>, Result, <<"{}">>),
                    ExistingData = hb_json:decode(ExistingBody),
                    UpdatedData = ExistingData#{<<"attestation">> => hb_json:decode(TokenJSON)},
                    UpdatedResult = Result#{<<"body">> => hb_json:encode(UpdatedData)},
                    {ok, UpdatedResult};
                {error, Reason} ->
                    ?event(dev_wasi_nn, {infer_sec_failed, Reason}),
                    {error, {infer_sec_failed, Reason}}
            end;
        {error, Reason} ->
            ?event(dev_wasi_nn, {infer_sec_failed, Reason}),
            {error, {infer_sec_failed, Reason}}
    end.

%% @doc Download model from Arweave and store it locally
download_and_store_model(TxID) ->
    % Configure local storage
    StoreConfig =
        #{<<"store-module">> => hb_store_fs,
          <<"name">> => <<"./models">>},  % Directory where models will be stored
    % Use TX ID as filename with .gguf extension
    ModelFileName = <<TxID/binary, ".gguf">>,
    LocalPath = <<"./models/", ModelFileName/binary>>,

    % First try to read the model from local storage
    case hb_store:read(StoreConfig, binary_to_list(ModelFileName)) of
        {ok, _ExistingData} ->
            % File already exists locally, no need to download
            ?event(dev_wasi_nn, {model_already_exists, TxID, LocalPath}),
            {ok, ModelFileName};
        not_found ->
            % File doesn't exist, proceed with download
            try
                % Download data from Arweave using the TX ID
                case hb_gateway_client:data(TxID,
                                            #{http_connect_timeout =>
                                                  10
                                                  * 60
                                                  * 1000}) % 10 minutes for large model downloads
                of
                    {ok, ModelData} ->
                        % Store the model file locally
                        case hb_store:write(StoreConfig, ModelFileName, ModelData) of
                            ok ->
                                % Return the local file path
                                ?event(dev_wasi_nn, {model_downloaded, TxID, LocalPath}),
                                {ok, ModelFileName};
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
                case hb_gateway_client:data(TxID,
                                            #{http_connect_timeout =>
                                                  10
                                                  * 60
                                                  * 1000}) % 10 minutes for large model downloads
                of
                    {ok, ModelData} ->
                        % Store the model file locally
                        case hb_store:write(StoreConfig, ModelFileName, ModelData) of
                            ok ->
                                % Return the local file path
                                ?event(dev_wasi_nn, {model_downloaded, TxID, LocalPath}),
                                {ok, ModelFileName};
                            StoreError ->
                                ?event(dev_wasi_nn, {model_store_failed, TxID, StoreError}),
                                {error, {store_failed, StoreError}}
                        end;
                    {error, DownloadError} ->
                        ?event(dev_wasi_nn, {model_download_failed, TxID, DownloadError}),
                        {error, {download_failed, DownloadError}}
                end
            catch
                % FIX: Use different variable names to avoid unsafe variable error.
                Error2:Reason2 ->
                    ?event(dev_wasi_nn, {model_download_exception, TxID, Error2, Reason2}),
                    {error, {exception, Error2, Reason2}}
            end
    end.

%% @doc Resolves a LoRA ID to a local file path.
%% First checks for a local training run artifact, then falls back to Arweave.
-spec resolve_lora_path(binary()) -> {ok, binary()} | {error, any()}.
resolve_lora_path(LoraID) ->
    LocalPathStr = filename:join(["runs", binary_to_list(LoraID), "lora_to_gguf", "lora_adapter.gguf"]),
    case filelib:is_regular(LocalPathStr) of
        true ->
            ?event(dev_wasi_nn, {lora_found_locally, LoraID, LocalPathStr}),
            {ok, list_to_binary(LocalPathStr)};
        false ->
            ?event(dev_wasi_nn, {lora_not_found_locally, LoraID, "attempting_arweave_download"}),
            download_and_store_lora(LoraID)
    end.

%% @doc Download LoRA from Arweave and store it locally in the ./loras directory.
-spec download_and_store_lora(binary()) -> {ok, binary()} | {error, any()}.
download_and_store_lora(TxID) ->
    StoreConfig = #{<<"store-module">> => hb_store_fs, <<"name">> => <<"./loras">>},
    LoraFileName = <<TxID/binary, ".gguf">>,
    LocalPath = <<"./loras/", LoraFileName/binary>>,

    case hb_store:read(StoreConfig, binary_to_list(LoraFileName)) of
        {ok, _ExistingData} ->
            ?event(dev_wasi_nn, {lora_already_exists, TxID, LocalPath}),
            {ok, LocalPath};
        not_found ->
            try
                case hb_gateway_client:data(TxID, #{http_connect_timeout => 5 * 60 * 1000}) of % 5 mins for LoRA
                    {ok, LoraData} ->
                        case hb_store:write(StoreConfig, LoraFileName, LoraData) of
                            ok ->
                                ?event(dev_wasi_nn, {lora_downloaded, TxID, LocalPath}),
                                {ok, LocalPath};
                            StoreError ->
                                ?event(dev_wasi_nn, {lora_store_failed, TxID, StoreError}),
                                {error, {store_failed, StoreError}}
                        end;
                    {error, DownloadError} ->
                        ?event(dev_wasi_nn, {lora_download_failed, TxID, DownloadError}),
                        {error, {download_failed, DownloadError}}
                end
            catch
                Error:Reason ->
                    ?event(dev_wasi_nn, {lora_download_exception, TxID, Error, Reason}),
                    {error, {exception, Error, Reason}}
            end;
        {error, ReadError} ->
            ?event(dev_wasi_nn, {lora_read_error, TxID, ReadError}),
            {error, {read_failed, ReadError}}
    end.

%% @doc Load model and perform inference using persistent context management with session support
load_and_infer(M1, M2, Opts) ->
    Model = maps:get(<<"model_path">>, M2, <<"">>),
    ModelPathStr = binary_to_list(Model),

    DefaultBaseConfig =
        #{<<"model">> =>
              #{<<"n_ctx">> => 8192,
                <<"n_batch">> => 512,
                <<"n_gpu_layers">> => 99},
          <<"sampling">> =>
              #{<<"temperature">> => 0.7,
                <<"top_p">> => 0.9,
                <<"repeat_penalty">> => 1.1},
          <<"stopping">> =>
              #{<<"max_tokens">> => 512},
          <<"backend">> => #{<<"max_sessions">> => 2}},

    % FIX: Move lora_path and lora_scale binding/usage closer together to resolve warnings.
    LoraPath = maps:get(<<"lora_path">>, M2, undefined),

    ConfigWithLora =
        case LoraPath of
            undefined ->
                DefaultBaseConfig;
            Path when is_binary(Path) ->
                LoraScale = maps:get(<<"lora_scale">>, M2, 1.0),
                LoraAdapterConfig = #{
                    <<"path">> => Path,
                    <<"scale">> => LoraScale
                },
                % The backend expects an array of adapters.
                maps:merge(DefaultBaseConfig, #{<<"lora_adapters">> => [LoraAdapterConfig]})
        end,

    ModelConfig = hb_json:encode(ConfigWithLora),
    ModelConfigStr = binary_to_list(ModelConfig),
    UserConfig = maps:get(<<"config">>, M2, ""),
    UserConfigStr = hb_util:list(UserConfig),
    
    Prompt = maps:get(<<"prompt">>, M2, <<>>),
    Body = maps:get(<<"body">>, M1, <<>>),
    DecodedBody = try hb_json:decode(Body) catch _:_ -> #{} end,

    PipedTranscript = maps:get(<<"transcription">>, DecodedBody, undefined),
    
    EffectivePrompt =
        case PipedTranscript of
            undefined -> Prompt;
            _ when is_binary(PipedTranscript) ->
                case Prompt of
                    <<>> -> PipedTranscript;
                    _ -> <<Prompt/binary, " ", PipedTranscript/binary>>
                end
        end,
    EffectivePromptStr = binary_to_list(EffectivePrompt),

    UserSessionId = maps:get(<<"session_id">>, M2, maps:get(<<"session_id">>, DecodedBody, undefined)),
    Reference = maps:get(<<"reference">>, M2, undefined),
    Worker = maps:get(<<"worker">>, M2, undefined),

    SessionIdBin =
        case UserSessionId of
            undefined -> generate_session_id(Opts);
            Bin when is_binary(Bin) -> Bin
        end,
    SessionIdList = binary_to_list(SessionIdBin),

    try
        case dev_wasi_nn_nif:ensure_model_loaded(ModelPathStr, ModelConfigStr) of
            ok ->
                case dev_wasi_nn_nif:init_session(SessionIdList) of
                    {ok, SessionResource} ->
                        case dev_wasi_nn_nif:run_inference(SessionResource, EffectivePromptStr, UserConfigStr) of
                            {ok, Output} ->
                                ?event(dev_wasi_nn, {inference_success, Reference}),
                                save_llm_response(SessionIdBin, Output),
                                ResponseBody = #{
                                    <<"result">> => Output,
                                    <<"transcription">> => PipedTranscript,
                                    <<"session_id">> => SessionIdBin
                                },
                                {ok,
                                 #{<<"body">> => hb_json:encode(ResponseBody),
                                   <<"X-Session">> => SessionIdBin,
                                   <<"X-Reference">> => Reference,
                                   <<"X-Worker">> => Worker,
                                   <<"action">> => <<"Infer-Response">>,
                                   <<"status">> => 200}};
                            {error, Reason} ->
                                ?event(dev_wasi_nn, {inference_failed, SessionIdList, Reason}),
                                {error, Reason}
                        end;
                    {error, Reason2} ->
                        ?event(dev_wasi_nn, {session_init_failed, SessionIdList, Reason2}),
                        {error, Reason2}
                end;
            {error, Reason3} ->
                ?event(dev_wasi_nn, {model_load_failed, SessionIdList, Model, Reason3}),
                {error, Reason3}
        end
    catch
        Error:Exception ->
            ?event(dev_wasi_nn, {inference_exception, SessionIdList, Error, Exception}),
            {error, {exception, Error, Exception}}
    end.

%% @doc Saves the LLM response to a JSON list in the session directory,
%% applying regex to filter out markdown and newlines.
save_llm_response(SessionID, LLMResponse) ->
    ResponseAudiosPath = filename:join([?SESSIONS_DIR, binary_to_list(SessionID), "response-audios"]),
    ok = filelib:ensure_dir(filename:join(ResponseAudiosPath, "dummy.txt")),
    JsonPath = filename:join(ResponseAudiosPath, "string-list.json"),
    
    CurrentList = case file:read_file(JsonPath) of
        {ok, JsonBinary} ->
            try hb_json:decode(JsonBinary) of
                Decoded when is_list(Decoded) -> Decoded;
                _ -> []
            catch
                _:_ -> []
            end;
        {error, enoent} -> []
    end,

    % New Stage: Remove <think> </think> tags and their contents.
    CleanedThink = re:replace(LLMResponse, "(?s)<think>.*?</think>", <<>>, [global, {return, binary}]),

    % Stage 1: Remove multi-line code blocks entirely.
    CleanedAfterCodeBlocks = re:replace(CleanedThink, "(?s)```.*?```", <<>>, [global, {return, binary}]),

    % Stage 2: Remove common inline formatting characters.
    CleanedAfterInline = re:replace(CleanedAfterCodeBlocks, "[\\*_`~]", <<>>, [global, {return, binary}]),

    % Stage 3: Remove block-level markers from the start of each line.
    BlockMarkersPattern = "^(#+\\s*|\\s*[-*]\\s+|\\s*\\d+\\.\\s+|>\\s*)",
    SanitizedResponse = re:replace(CleanedAfterInline, BlockMarkersPattern, <<>>, [global, multiline, {return, binary}]),
    
    % Stage 4: Replace one or more newline characters (CR/LF) with a single space.
    CleanedNewlines = re:replace(SanitizedResponse, "[\\r\\n]+", <<" ">>, [global, {return, binary}]),

    % Stage 5: Trim leading/trailing whitespace from the final result.
    TrimmedResponse = re:replace(CleanedNewlines, "^\\s+|\\s+$", <<>>, [{return, binary}]),

    % New Stage: Remove all non-alphabetic and non-space characters, keeping only A-Z, a-z, and spaces.
    FinalResponse = re:replace(TrimmedResponse, "[^a-zA-Z ]", <<>>, [global, {return, binary}]),

    NewList = CurrentList ++ [FinalResponse],
    file:write_file(JsonPath, hb_json:encode(NewList)).

%% @doc Generate a unique session ID for each request
generate_session_id(_Opts) ->
    Timestamp = erlang:system_time(microsecond),
    Random = rand:uniform(999),
    SessionId = io_lib:format("req_~p_~p", [Timestamp, Random]),
    list_to_binary(lists:flatten(SessionId)).

%% ===================================================================
%% EUnit Tests
%% ===================================================================
-ifdef(TEST).

-define(TEST_SESSION_ID, <<"wasi_nn_session_test">>).
-define(TEST_SESSION_PATH, filename:join(?SESSIONS_DIR, ?TEST_SESSION_ID)).
-define(TEST_LORA_SESSION_ID, <<"pipeline_test_session">>).
-define(TEST_LORA_PATH, "runs/pipeline_test_session/lora_to_gguf/lora_adapter.gguf").

setup() ->
    % Clean up before test and create the required structure
    file:del_dir_r(?TEST_SESSION_PATH),
    ResponseAudioPath = filename:join([?TEST_SESSION_PATH, "response-audios"]),
    ok = filelib:ensure_dir(filename:join(ResponseAudioPath, "dummy.txt")),
    ok.

teardown(_) ->
    % dont Clean up the session directory after the test
    ok.

session_based_inference_test_() ->
    {foreach, fun setup/0, fun teardown/1, [fun test_infer_and_save_response/0]}.

test_infer_and_save_response() ->
    Transcript = <<"Who are you?">>,
    % M1 simulates the output from the speech-to-text device
    M1 = #{<<"body">> => hb_json:encode(#{
        <<"transcription">> => Transcript,
        <<"session_id">> => ?TEST_SESSION_ID
    })},
    % M2 is the direct request to the wasi-nn device
    M2 = #{<<"model_path">> => <<"models/Qwen3-1.7B-Q8_0.gguf">>, % Assumes a test model exists
           <<"prompt">> => <<"In one sentence, respond to this question:">>},
    
    {ok, Response} = infer(M1, M2, #{}),
    
    ?assertEqual(200, maps:get(<<"status">>, Response)),
    
    DecodedBody = hb_json:decode(maps:get(<<"body">>, Response)),
    ?assertMatch(#{<<"result">> := _,
                   <<"transcription">> := Transcript,
                   <<"session_id">> := ?TEST_SESSION_ID}, DecodedBody),
    
    LLMResult = maps:get(<<"result">>, DecodedBody),
    ?assert(is_binary(LLMResult) andalso size(LLMResult) > 0),
    
    % Verify side-effect: check if the response was saved correctly
    JsonPath = filename:join([?TEST_SESSION_PATH, "response-audios", "string-list.json"]),
    ?assert(filelib:is_regular(JsonPath)),
    {ok, JsonBinary} = file:read_file(JsonPath),
    [SavedResult] = hb_json:decode(JsonBinary),
    % The saved result is sanitized, so we can't directly compare.
    % We just check that something was saved.
    ?assert(is_binary(SavedResult) andalso size(SavedResult) > 0).

lora_inference_test_() ->
    Precondition =
        fun() ->
            case filelib:is_regular(?TEST_LORA_PATH) of
                true ->
                    ok;
                false ->
                    Msg = io_lib:format(
                        "LoRA adapter not found at ~s. Run `rebar3 eunit --module dev_training` first to generate it.",
                        [?TEST_LORA_PATH]
                    ),
                    {skip, lists:flatten(Msg)}
            end
        end,
    {setup, Precondition, fun(_) -> ok end,
     fun(_) ->
         ?_test(
             begin
                 M2 = #{
                     % Use the same base model the LoRA was trained on.
                     <<"model_id">> => <<"Qwen1.5-0.5B-Chat">>,
                     <<"lora_id">> => ?TEST_LORA_SESSION_ID,
                     <<"prompt">> => <<"Hi! My name is Arpit.">>
                 },
                 {ok, Response} = infer(#{}, M2, #{}),
                 ?assertEqual(200, maps:get(<<"status">>, Response), "Inference with LoRA failed"),

                 Body = hb_json:decode(maps:get(<<"body">>, Response)),
                 ?assert(maps:is_key(<<"result">>, Body)),
                 Result = maps:get(<<"result">>, Body),
                 ?assert(size(Result) > 0),
                 io:format("~n--- LORA INFERENCE RESULT ---~n~s~n---------------------------~n", [Result])
             end
         )
     end}.

markdown_and_newline_filtering_test() ->
    TestDir = ?SESSIONS_DIR,
    SessionID = <<"test_session_with_markdown_and_newlines">>,

    LLMResponseWithMarkdown = <<
        "# Main Heading\n\nThis is a paragraph with **bold text**, *italic emphasis*, and some `inline code`.\n"
        "Here is a list:\n"
        "- First item\n"
        "1. Second item\n"
        "> This is a blockquote.\n"
        "And here is a code block to be removed:\n"
        "```erlang\n-module(test).\n```\n"
        "The text continues after the block."
    >>,

    % The expected output is now a single line of text with newlines replaced by spaces.
    ExpectedSanitizedResponse = <<
        "Main Heading This is a paragraph with bold text, italic emphasis, and some inline code. "
        "Here is a list: First item Second item This is a blockquote. "
        "And here is a code block to be removed: "
        "The text continues after the block."
    >>,

    try
        % 1. Execute the function under test
        ok = save_llm_response(SessionID, LLMResponseWithMarkdown),

        % 2. Verify the result
        JsonPath = filename:join([?SESSIONS_DIR, binary_to_list(SessionID), "response-audios", "string-list.json"]),
        {ok, JsonBinary} = file:read_file(JsonPath),
        DecodedList = hb_json:decode(JsonBinary),

        % 3. Assert that the list contains the correctly sanitized string
        ?assertMatch([ExpectedSanitizedResponse], DecodedList)

    after
        % 4. Cleanup
        file:del_dir_r(filename:join(TestDir, binary_to_list(SessionID)))
    end.

-endif.