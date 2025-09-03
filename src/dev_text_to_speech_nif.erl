%% @doc NIF wrapper for text_to_speech Rust crate.
-module(dev_text_to_speech_nif).
-export([generate_audio/5]).
-on_load(init/0).

-include("include/cargo.hrl").

init() ->
    ?load_nif_from_crate(text_to_speech, 0).

generate_audio(_Text, _Speaker, _SessionID, _ReferenceAudioPath, _ReferenceAudioText) ->
    erlang:nif_error(nif_not_loaded).

%% EUnit Tests
-ifdef(TEST).
-include_lib("eunit/include/eunit.hrl").

-define(REF_AUDIO_PATH, <<"native/text_to_speech/utterance_0.mp3">>).
-define(REF_AUDIO_TEXT, <<"In a 1997 AI class at UT Austin, a neural net playing infinite board tic-tac-toe found an unbeatable strategy. Choose moves billions of squares away, causing your opponents to run out of memory and crash.">>).


%% @doc Test successful audio generation with a valid session ID, using context.
contextual_generation_with_session_test() ->
    Text = <<"This is a new sentence, continuing the conversation.">>,
    Speaker = 1,
    SessionID = <<"test-session">>, % Use the session created by the python script
    Result = generate_audio(Text, Speaker, SessionID, <<>>, <<>>),
    ?assertMatch({ok, _AudioData}, Result),
    case Result of
        {ok, AudioData} ->
            ?assert(is_binary(AudioData)),
            ?assert(size(AudioData) > 100),
            % Optionally save the file to listen and verify the voice consistency
            Filename = "/tmp/eunit_tts_contextual_output.wav",
            ?assertEqual(ok, file:write_file(Filename, AudioData)),
            ?debugFmt("SUCCESS: Contextual audio file saved for playback at ~s", [Filename]);
        {error, Reason} ->
            ?debugFmt("Contextual audio generation failed unexpectedly: ~p", [Reason]),
            ?assert(false)
    end.

%% @doc Test audio generation without a session ID (no context).
no_context_generation_test() ->
    Text = <<"This is a standalone test without any context.">>,
    Speaker = 1,
    SessionID = <<>>, % Empty session ID
    Result = generate_audio(Text, Speaker, SessionID, <<>>, <<>>),
    ?assertMatch({ok, _AudioData}, Result),
    case Result of
        {ok, AudioData} ->
            ?assert(is_binary(AudioData)),
            ?assert(size(AudioData) > 100),
            Filename = "/tmp/eunit_tts_no_context_output.wav",
            ?assertEqual(ok, file:write_file(Filename, AudioData)),
            ?debugFmt("SUCCESS: No-context audio file saved for playback at ~s", [Filename]);
        {error, Reason} ->
            ?debugFmt("No-context audio generation failed unexpectedly: ~p", [Reason]),
            ?assert(false)
    end.

%% @doc Test audio generation with the default reference audio file.
default_reference_audio_generation_test() ->
    Text = <<"Does my voice sound like the reference audio?">>,
    Speaker = 1,
    SessionID = <<>>, % No session, just reference
    Result = generate_audio(Text, Speaker, SessionID, ?REF_AUDIO_PATH, ?REF_AUDIO_TEXT),
    ?assertMatch({ok, _AudioData}, Result),
    case Result of
        {ok, AudioData} ->
            ?assert(is_binary(AudioData)),
            ?assert(size(AudioData) > 100),
            Filename = "/tmp/eunit_tts_with_default_reference_audio.wav",
            ?assertEqual(ok, file:write_file(Filename, AudioData)),
            ?debugFmt("SUCCESS: Default reference audio test file saved for playback at ~s", [Filename]);
        {error, Reason} ->
            ?debugFmt("Default reference audio generation failed unexpectedly: ~p", [Reason]),
            ?assert(false)
    end.

%% @doc Test with a session ID that does not exist. Should run without context.
non_existent_session_test() ->
    Text = <<"Testing with a session that does not exist.">>,
    Speaker = 0,
    SessionID = <<"non-existent-session-12345">>,
    Result = generate_audio(Text, Speaker, SessionID, <<>>, <<>>),
    ?assertMatch({ok, _AudioData}, Result),
    case Result of
        {ok, AudioData} ->
            ?assert(is_binary(AudioData)),
            ?assert(size(AudioData) > 100);
        {error, Reason} ->
            ?debugFmt("Audio generation with non-existent session failed unexpectedly: ~p", [Reason]),
            ?assert(false)
    end.


%% @doc Test that passing incorrect types to the NIF results in a `badarg` error.
invalid_argument_type_test_() ->
    [
        {"Text as list", ?_assertError(badarg, generate_audio("not a binary", 0, <<"s">>, <<>>, <<>>))},
        {"Speaker as binary", ?_assertError(badarg, generate_audio(<<"t">>, <<"0">>, <<"s">>, <<>>, <<>>))},
        {"SessionID as integer", ?_assertError(badarg, generate_audio(<<"t">>, 0, 123, <<>>, <<>>))},
        {"Ref Path as integer", ?_assertError(badarg, generate_audio(<<"t">>, 0, <<"s">>, 123, <<>>))},
        {"Ref Text as integer", ?_assertError(badarg, generate_audio(<<"t">>, 0, <<"s">>, <<>>, 123))}
    ].

-endif.