-module(dev_text_to_speech).
-export([generate/3, info/1, info/3]).
-include("include/hb.hrl").
-include_lib("eunit/include/eunit.hrl").

-define(DEFAULT_SPEAKER, 1). % Default to speaker 1 (AI response)
-define(SESSIONS_DIR, "sessions").

%% @doc Declares the functions this device exports.
info(_) ->
    #{exports => [generate, info]}.

%% @doc Provides API information.
info(_M1, _M2, _Opts) ->
    InfoBody = #{
        <<"description">> => <<"A device for generating speech from text, with session-based context awareness.">>,
        <<"version">> => <<"1.1">>,
        <<"api">> => #{
            <<"generate">> => #{
                <<"description">> => <<"Generates WAV audio. Uses 'result' from a previous device (M1) and a 'session_id'. Saves audio to the session directory.">>,
                <<"method">> => <<"POST">>,
                <<"parameters">> => #{
                    <<"text">> => <<"Optional. Standalone text to generate. Will not be saved to a session unless a session_id is also provided.">>,
                    <<"speaker">> => <<"Optional. An integer for the speaker ID. Defaults to 1 (response).">>,
                    <<"session_id">> => <<"Optional. The session identifier for contextual generation and storage.">>
                }
            }
        }
    },
    {ok, InfoBody}.

%% @doc Generates speech, using session context if available, and saves the audio.
generate(M1, M2, _Opts) ->
    % Prioritize session_id from the previous device's body, then the current request.
    DecodedM1Body = try hb_json:decode(maps:get(<<"body">>, M1, <<>>)) catch _:_ -> #{} end,
    SessionID = maps:get(<<"session_id">>, M2, maps:get(<<"session_id">>, DecodedM1Body, undefined)),

    % Prioritize text from the previous device's 'result' key, then the current request's 'text' param.
    TextToSpeak = maps:get(<<"result">>, DecodedM1Body, maps:get(<<"text">>, M2, <<>>)),
    
    FinalText = string:trim(binary_to_list(TextToSpeak)),

    case length(FinalText) > 0 of
        true ->
            SpeakerStr = maps:get(<<"speaker">>, M2, integer_to_binary(?DEFAULT_SPEAKER)),
            try binary_to_integer(SpeakerStr) of
                Speaker when is_integer(Speaker) ->
                    SessionIDForNif = case SessionID of
                        undefined -> <<>>;
                        _ -> SessionID
                    end,
                    case dev_text_to_speech_nif:generate_audio(list_to_binary(FinalText), Speaker, SessionIDForNif) of
                        {ok, AudioData} ->
                            handle_audio_saving(AudioData, SessionID),
                            {ok, #{
                                <<"body">> => AudioData,
                                <<"status">> => 200,
                                <<"headers">> => #{ <<"content-type">> => <<"audio/wav">> }
                            }};
                        {error, Reason} ->
                            Error = iolist_to_binary(io_lib:format("~p", [Reason])),
                            {ok, #{ <<"body">> => hb_json:encode(#{<<"error">> => <<"Audio generation failed">>, <<"reason">> => Error}), <<"status">> => 500 }}
                    end
            catch
                error:badarg ->
                    {ok, #{ <<"body">> => hb_json:encode(#{<<"error">> => <<"'speaker' parameter must be an integer">>}), <<"status">> => 400 }}
            end;
        false ->
            {ok, #{ <<"body">> => hb_json:encode(#{<<"error">> => <<"No text provided from previous device or as a parameter">>}), <<"status">> => 400 }}
    end.

%% --- Private Functions ---

%% @doc If a session ID is provided, save the audio to the session directory.
handle_audio_saving(_AudioData, undefined) ->
    % No session ID, so don't save the file.
    ok;
handle_audio_saving(AudioData, SessionID) ->
    ResponseAudiosPath = filename:join([?SESSIONS_DIR, binary_to_list(SessionID), "response-audios"]),
    ok = filelib:ensure_dir(filename:join(ResponseAudiosPath, "dummy.txt")), % Ensure dir exists

    {ok, Files} = file:list_dir(ResponseAudiosPath),
    AudioFiles = [F || F <- Files, lists:member(filename:extension(F), [".wav", ".mp3"])],
    NextIndex = length(AudioFiles),
    
    AudioFilename = io_lib:format("~p.wav", [NextIndex]),
    AudioPath = filename:join(ResponseAudiosPath, AudioFilename),

    case file:write_file(AudioPath, AudioData) of
        ok ->
            ?event(dev_text_to_speech, {saved_session_audio, AudioPath});
        {error, Reason} ->
            ?event(dev_text_to_speech, {save_audio_failed, SessionID, Reason})
    end.


%% ===================================================================
%% EUnit Tests
%% ===================================================================
-ifdef(TEST).

-define(TEST_SESSION_ID, <<"tts_test_session">>).
-define(TEST_SESSION_PATH, filename:join(?SESSIONS_DIR, ?TEST_SESSION_ID)).

setup() ->
    % Clean up and create a fresh test session directory
    file:del_dir_r(?TEST_SESSION_PATH),
    ResponseAudioPath = filename:join([?TEST_SESSION_PATH, "response-audios"]),
    ok = filelib:ensure_dir(filename:join(ResponseAudioPath, "dummy.txt")),
    
    % Simulate the state after wasi-nn has run: create the transcript file
    Transcript = <<"This is a test response from the LLM.">>,
    JsonPath = filename:join(ResponseAudioPath, "string-list.json"),
    ok = file:write_file(JsonPath, hb_json:encode([Transcript])),
    Transcript. % Return transcript for use in the test case

teardown(_) ->
    % Per request, do not clean up so the generated files can be reviewed.
    ok.

session_generation_test_() ->
    {setup,
     fun setup/0,
     fun teardown/1,
     fun(Transcript) ->
        ?_test(
        begin
            % M1 simulates the output from the wasi-nn device
            M1 = #{<<"body">> => hb_json:encode(#{
                <<"result">> => Transcript,
                <<"session_id">> => ?TEST_SESSION_ID
            })},
            
            {ok, Response} = generate(M1, #{}, #{}),
            
            ?assertEqual(200, maps:get(<<"status">>, Response)),
            AudioData = maps:get(<<"body">>, Response),
            ?assert(is_binary(AudioData) andalso size(AudioData) > 100),

            % Verify side-effect: check that the audio file was created
            ResponseAudioPath = filename:join([?TEST_SESSION_PATH, "response-audios"]),
            AudioFilePath = filename:join(ResponseAudioPath, "0.wav"),
            ?assert(filelib:is_regular(AudioFilePath)),
            {ok, SavedData} = file:read_file(AudioFilePath),
            ?assertEqual(AudioData, SavedData)
        end)
     end}.

append_to_session_test_() ->
    {setup,
     fun setup/0,
     fun teardown/1,
     fun(Transcript1) ->
        ?_test(
        begin
            % --- First Call ---
            M1_1 = #{<<"body">> => hb_json:encode(#{<<"result">> => Transcript1, <<"session_id">> => ?TEST_SESSION_ID})},
            {ok, _Response1} = generate(M1_1, #{}, #{}),
            ?assert(filelib:is_regular(filename:join([?TEST_SESSION_PATH, "response-audios", "0.wav"]))),

            % --- CORRECT FIX: Manually simulate wasi-nn updating the transcript list ---
            Transcript2 = <<"This is a second response in the same session.">>,
            JsonPath = filename:join([?TEST_SESSION_PATH, "response-audios", "string-list.json"]),
            UpdatedTranscripts = [Transcript1, Transcript2],
            ok = file:write_file(JsonPath, hb_json:encode(UpdatedTranscripts)),
            
            % --- Second Call ---
            M1_2 = #{<<"body">> => hb_json:encode(#{<<"result">> => Transcript2, <<"session_id">> => ?TEST_SESSION_ID})},
            {ok, _Response2} = generate(M1_2, #{}, #{}),
            
            % Verify the second audio file was created
            ?assert(filelib:is_regular(filename:join([?TEST_SESSION_PATH, "response-audios", "1.wav"]))),

            % Verify the transcript file now has two entries
            {ok, FinalJson} = file:read_file(JsonPath),
            ?assertEqual(UpdatedTranscripts, hb_json:decode(FinalJson))
        end)
     end}.

no_text_provided_failure_test() ->
    {ok, Response} = generate(#{}, #{}, #{}),
    ?assertEqual(400, maps:get(<<"status">>, Response)),
    DecodedBody = hb_json:decode(maps:get(<<"body">>, Response)),
    ?assertEqual(<<"No text provided from previous device or as a parameter">>, maps:get(<<"error">>, DecodedBody)).

-endif.