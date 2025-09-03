-module(dev_text_to_speech).
-export([generate/3, info/1, info/3]).
-include("include/hb.hrl").
-include_lib("eunit/include/eunit.hrl").

-define(DEFAULT_SPEAKER, 1). % Default to speaker 1 (AI response)
-define(SESSIONS_DIR, "sessions").
-define(DEFAULT_REF_AUDIO_PATH, <<"native/text_to_speech/utterance_0.mp3">>).
-define(DEFAULT_REF_AUDIO_TEXT, <<"In a 1997 AI class at UT Austin, a neural net playing infinite board tic-tac-toe found an unbeatable strategy. Choose moves billions of squares away, causing your opponents to run out of memory and crash.">>).

%% @doc Declares the functions this device exports.
info(_) ->
    #{exports => [generate, info]}.

%% @doc Provides API information.
info(_M1, _M2, _Opts) ->
    InfoBody = #{
        <<"description">> => <<"A device for generating speech from text, with session-based context awareness.">>,
        <<"version">> => <<"1.3">>,
        <<"api">> => #{
            <<"generate">> => #{
                <<"description">> => <<"Generates WAV audio. Uses 'result' from a previous device (M1) and a 'session_id'. Saves audio to the session directory. Uses a default reference voice for speaker 1 unless overridden.">>,
                <<"method">> => <<"POST">>,
                <<"parameters">> => #{
                    <<"text">> => <<"Optional. Standalone text to generate. Will not be saved to a session unless a session_id is also provided.">>,
                    <<"speaker">> => <<"Optional. An integer for the speaker ID. Defaults to 1 (response).">>,
                    <<"session_id">> => <<"Optional. The session identifier for contextual generation and storage.">>,
                    <<"reference_audio_path">> => <<"Optional. Path to a WAV/MP3 file to override the default voice reference for the responder (speaker 1).">>,
                    <<"reference_audio_text">> => <<"Optional. The transcript of the reference audio file. Required if reference_audio_path is provided.">>
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
            ReferenceAudioPath = maps:get(<<"reference_audio_path">>, M2, ?DEFAULT_REF_AUDIO_PATH),
            ReferenceAudioText = maps:get(<<"reference_audio_text">>, M2, ?DEFAULT_REF_AUDIO_TEXT),
            try binary_to_integer(SpeakerStr) of
                Speaker when is_integer(Speaker) ->
                    SessionIDForNif = case SessionID of
                        undefined -> <<>>;
                        _ -> SessionID
                    end,
                    case dev_text_to_speech_nif:generate_audio(
                            list_to_binary(FinalText),
                            Speaker,
                            SessionIDForNif,
                            ReferenceAudioPath,
                            ReferenceAudioText
                        ) of
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

default_reference_audio_test() ->
    ?_test(
    begin
        TextToSpeak = <<"This should be generated with the default reference voice.">>,
        M1 = #{},
        M2 = #{ <<"text">> => TextToSpeak },

        {ok, Response} = generate(M1, M2, #{}),

        ?assertEqual(200, maps:get(<<"status">>, Response)),
        AudioData = maps:get(<<"body">>, Response),
        ?assert(is_binary(AudioData) andalso size(AudioData) > 100),
        ok = file:write_file("/tmp/eunit_tts_default_ref.wav", AudioData)
    end).

disable_reference_audio_test() ->
    ?_test(
    begin
        TextToSpeak = <<"This should be generated without any reference voice.">>,
        M1 = #{},
        M2 = #{
            <<"text">> => TextToSpeak,
            <<"reference_audio_path">> => <<>>, % Explicitly disable
            <<"reference_audio_text">> => <<>>  % Explicitly disable
        },

        {ok, Response} = generate(M1, M2, #{}),

        ?assertEqual(200, maps:get(<<"status">>, Response)),
        AudioData = maps:get(<<"body">>, Response),
        ?assert(is_binary(AudioData) andalso size(AudioData) > 100),
        ok = file:write_file("/tmp/eunit_tts_no_ref.wav", AudioData)
    end).

override_reference_audio_test_() ->
    {setup,
     fun setup/0,
     fun teardown/1,
     fun(_Transcript) ->
        ?_test(
        begin
            TextToSpeak = <<"This should sound like a different reference speaker.">>,
            M1 = #{},

            % FIX: The root cause of the badarg is that ?TEST_SESSION_PATH is a binary,
            % which "infects" the result of filename:join, making it also a binary.
            % We must first convert the base path to a list before joining.
            BasePathAsList = binary_to_list(?TEST_SESSION_PATH),
            OverrideRefAudioPathAsList = filename:join([BasePathAsList, "response-audios", "override.wav"]),
            OverrideRefAudioText = <<"This is an override voice.">>,
            
            % Read source file and write to the new path (using the list version of the path)
            {ok, SampleWav} = file:read_file(binary_to_list(?DEFAULT_REF_AUDIO_PATH)),
            ok = file:write_file(OverrideRefAudioPathAsList, SampleWav),

            M2 = #{
                <<"text">> => TextToSpeak,
                <<"reference_audio_path">> => list_to_binary(OverrideRefAudioPathAsList),
                <<"reference_audio_text">> => OverrideRefAudioText
            },

            {ok, Response} = generate(M1, M2, #{}),

            ?assertEqual(200, maps:get(<<"status">>, Response)),
            AudioData = maps:get(<<"body">>, Response),
            ?assert(is_binary(AudioData) andalso size(AudioData) > 100),
            ok = file:write_file("/tmp/eunit_tts_override_ref.wav", AudioData)
        end)
     end}.

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
            
            % M2 is empty, so this test will use the DEFAULT reference audio
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

no_text_provided_failure_test() ->
    {ok, Response} = generate(#{}, #{}, #{}),
    ?assertEqual(400, maps:get(<<"status">>, Response)),
    DecodedBody = hb_json:decode(maps:get(<<"body">>, Response)),
    ?assertEqual(<<"No text provided from previous device or as a parameter">>, maps:get(<<"error">>, DecodedBody)).

-endif.