-module(dev_speech_to_text).
-export([transcribe/3]).
-include("include/hb.hrl").
-include_lib("eunit/include/eunit.hrl").

-define(SESSIONS_DIR, "sessions").

transcribe(_M1, M2, _Opts) ->
    Data = maps:get(<<"body">>, M2, <<>>),
    case Data of
        <<>> ->
            {ok, #{<<"body">> => hb_json:encode(#{<<"error">> => <<"No audio data provided">>}),
                   <<"status">> => 400}};
        _ ->
            SessionID = maps:get(<<"session_id">>, M2, undefined),
            handle_transcription(Data, SessionID)
    end.

%% --- Private Functions ---

%% @doc Handles transcription when no session_id is provided, using a temporary file.
handle_transcription(Data, undefined) ->
    Timestamp = erlang:system_time(millisecond),
    TempFilename = io_lib:format("/tmp/hyperbeam_audio_~p.mp3", [Timestamp]),
    TempPath = lists:flatten(TempFilename),
    case file:write_file(TempPath, Data) of
        ok ->
            ?event(dev_speech_to_text, {saved_audio, TempPath}),
            Result = case dev_speech_to_text_nif:transcribe_audio(list_to_binary(TempPath)) of
                {ok, Transcript} ->
                    ?event(dev_speech_to_text, {transcription_success, Transcript}),
                    {ok, #{<<"body">> => hb_json:encode(#{<<"transcription">> => Transcript}),
                           <<"status">> => 200}};
                {error, Reason} ->
                    ?event(dev_speech_to_text, {transcription_failed, Reason}),
                    {ok, #{<<"body">> => hb_json:encode(#{<<"error">> => <<"Transcription failed">>, <<"reason">> => Reason}),
                           <<"status">> => 500}}
            end,
            % Always delete the temporary file after transcription attempt
            file:delete(TempPath),
            Result;
        {error, Reason} ->
            ?event(dev_speech_to_text, {save_audio_failed, Reason}),
            {ok, #{<<"body">> => hb_json:encode(#{<<"error">> => <<"Failed to save audio">>, <<"reason">> => list_to_binary(io_lib:format("~p", [Reason]))}),
                   <<"status">> => 500}}
    end;

%% @doc Handles transcription for a given session_id, saving audio and transcript permanently.
handle_transcription(Data, SessionID) when is_binary(SessionID) ->
    UserAudiosPathStr = filename:join([?SESSIONS_DIR, binary_to_list(SessionID), "user-audios"]),
    ok = filelib:ensure_dir(filename:join(UserAudiosPathStr, "dummy.txt")), % Ensure dir exists

    {ok, Files} = file:list_dir(UserAudiosPathStr),
    AudioFiles = [F || F <- Files, lists:member(filename:extension(F), [".wav", ".mp3"])],
    NextIndex = length(AudioFiles),
    
    AudioFilename = io_lib:format("~p.wav", [NextIndex]),
    AudioPathStr = filename:join(UserAudiosPathStr, AudioFilename),

    case file:write_file(AudioPathStr, Data) of
        ok ->
            ?event(dev_speech_to_text, {saved_session_audio, AudioPathStr}),
            case dev_speech_to_text_nif:transcribe_audio(list_to_binary(AudioPathStr)) of
                {ok, Transcript} ->
                    ?event(dev_speech_to_text, {transcription_success, Transcript}),
                    JsonPathStr = filename:join(UserAudiosPathStr, "string-list.json"),
                    update_transcript_file(Transcript, JsonPathStr),
                    {ok, #{<<"body">> => hb_json:encode(#{<<"transcription">> => Transcript, <<"session_id">> => SessionID}),
                           <<"status">> => 200}};
                {error, Reason} ->
                    ?event(dev_speech_to_text, {transcription_failed, Reason}),
                    % NOTE: We DO NOT delete the audio file on failure, as it's part of the session history.
                    {ok, #{<<"body">> => hb_json:encode(#{<<"error">> => <<"Transcription failed">>, <<"reason">> => Reason}),
                           <<"status">> => 500}}
            end;
        {error, Reason} ->
            ?event(dev_speech_to_text, {save_audio_failed, Reason}),
            {ok, #{<<"body">> => hb_json:encode(#{<<"error">> => <<"Failed to save audio">>, <<"reason">> => list_to_binary(io_lib:format("~p", [Reason]))}),
                   <<"status">> => 500}}
    end.

update_transcript_file(Transcript, JsonPath) ->
    CurrentList = case file:read_file(JsonPath) of
        {ok, JsonBinary} ->
            try hb_json:decode(JsonBinary) of
                Decoded when is_list(Decoded) -> Decoded;
                _ -> []
            catch
                _:_ -> [] % Handle cases where JSON is malformed
            end;
        {error, enoent} -> % File doesn't exist yet, start with an empty list
            []
    end,
    NewList = CurrentList ++ [Transcript],
    file:write_file(JsonPath, hb_json:encode(NewList)).


%% ===================================================================
%% EUnit Tests
%% ===================================================================
-ifdef(TEST).

-define(TEST_SESSION_ID, <<"speech_to_text_session_test">>).
-define(TEST_SESSION_PATH, filename:join(?SESSIONS_DIR, ?TEST_SESSION_ID)).

setup() ->
    % Clean up before test runs to ensure a fresh state
    cleanup_session_dir(),
    ok.

teardown(_) ->
    % Do nothing on teardown to preserve the directory for review.
    ok.

cleanup_session_dir() ->
    file:del_dir_r(?TEST_SESSION_PATH).

session_based_transcription_test_() ->
    % Use 'foreach' to run setup/teardown for EACH test function.
    {foreach,
     fun setup/0,
     fun teardown/1,
     [
      fun test_first_transcription_in_session/0,
      fun test_second_transcription_in_session/0
     ]
    }.

test_first_transcription_in_session() ->
    ?debugMsg("Testing first transcription in a new session..."),
    {ok, AudioData} = file:read_file("sessions/test-session/user-audios/0.wav"),
    M2 = #{<<"body">> => AudioData, <<"session_id">> => ?TEST_SESSION_ID},
    
    {ok, Response} = transcribe(#{}, M2, #{}),
    ?assertEqual(200, maps:get(<<"status">>, Response)),
    
    DecodedBody = hb_json:decode(maps:get(<<"body">>, Response)),
    Transcript = maps:get(<<"transcription">>, DecodedBody),
    ?assert(is_binary(Transcript) andalso size(Transcript) > 0),
    
    % Verify side effects
    UserAudiosPath = filename:join([?TEST_SESSION_PATH, "user-audios"]),
    ?assert(filelib:is_dir(UserAudiosPath)),
    ?assert(filelib:is_regular(filename:join(UserAudiosPath, "0.wav"))),
    
    JsonPath = filename:join(UserAudiosPath, "string-list.json"),
    ?assert(filelib:is_regular(JsonPath)),
    {ok, JsonBinary} = file:read_file(JsonPath),
    [TranscriptFromFile] = hb_json:decode(JsonBinary),
    ?assertEqual(Transcript, TranscriptFromFile).

test_second_transcription_in_session() ->
    ?debugMsg("Testing appending a second transcription to the session..."),
    
    % --- First call ---
    {ok, AudioData1} = file:read_file("sessions/test-session/user-audios/0.wav"),
    M2_1 = #{<<"body">> => AudioData1, <<"session_id">> => ?TEST_SESSION_ID},
    {ok, _Response1} = transcribe(#{}, M2_1, #{}),
    
    % --- Second call ---
    {ok, AudioData2} = file:read_file("sessions/test-session/user-audios/1.wav"),
    M2_2 = #{<<"body">> => AudioData2, <<"session_id">> => ?TEST_SESSION_ID},
    {ok, Response2} = transcribe(#{}, M2_2, #{}),
    ?assertEqual(200, maps:get(<<"status">>, Response2)),
    
    DecodedBody2 = hb_json:decode(maps:get(<<"body">>, Response2)),
    Transcript2 = maps:get(<<"transcription">>, DecodedBody2),
    
    % Verify side effects
    UserAudiosPath = filename:join([?TEST_SESSION_PATH, "user-audios"]),
    ?assert(filelib:is_regular(filename:join(UserAudiosPath, "1.wav"))), % Check for the new audio file
    
    JsonPath = filename:join(UserAudiosPath, "string-list.json"),
    {ok, JsonBinary} = file:read_file(JsonPath),
    [_, Transcript2FromFile] = hb_json:decode(JsonBinary), % Should now have two entries
    ?assertEqual(Transcript2, Transcript2FromFile).

no_audio_data_test() ->
    M2 = #{<<"body">> => <<>>},
    {ok, Response} = transcribe(#{}, M2, #{}),
    ?assertMatch(#{<<"body">> := _, <<"status">> := 400}, Response),
    Decoded = hb_json:decode(maps:get(<<"body">>, Response)),
    % CORRECTED: Assert against the correct error message.
    ?assertMatch(#{<<"error">> := <<"No audio data provided">>}, Decoded).

-endif.