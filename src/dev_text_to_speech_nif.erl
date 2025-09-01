%% @doc NIF wrapper for text_to_speech Rust crate.
-module(dev_text_to_speech_nif).
-export([generate_audio/2]).
-on_load(init/0).

-include("include/cargo.hrl").

init() ->
    ?load_nif_from_crate(text_to_speech, 0).

generate_audio(_Text, _Speaker) ->
    erlang:nif_error(nif_not_loaded).

%% EUnit Tests
-ifdef(TEST).
-include_lib("eunit/include/eunit.hrl").

%% @doc Test successful audio generation with valid inputs.
successful_generation_test() ->
    Text = <<"Hello from the EUnit test.">>,
    Speaker = 0,
    Result = generate_audio(Text, Speaker),
    ?assertMatch({ok, _AudioData}, Result),
    case Result of
        {ok, AudioData} ->
            ?assert(is_binary(AudioData)),
            ?assert(size(AudioData) > 100), % Assert that we got some meaningful data, not an empty binary. WAV headers alone are ~44 bytes.
            ok;
        {error, Reason} ->
            ?debugFmt("Audio generation failed unexpectedly: ~p", [Reason]),
            ?assert(false)
    end.

%% @doc Test generation with a different speaker ID.
different_speaker_test() ->
    Text = <<"Testing a different speaker.">>,
    Speaker = 1,
    Result = generate_audio(Text, Speaker),
    ?assertMatch({ok, _AudioData}, Result),
     case Result of
        {ok, AudioData} ->
            ?assert(is_binary(AudioData)),
            ?assert(size(AudioData) > 100),
            ok;
        {error, Reason} ->
            ?debugFmt("Audio generation failed unexpectedly: ~p", [Reason]),
            ?assert(false)
    end.

%% @doc Test how the NIF handles empty text.
empty_text_test() ->
    Text = <<>>,
    Speaker = 0,
    Result = generate_audio(Text, Speaker),
    % The python script might succeed and generate a short, silent WAV file.
    % We expect an {ok, binary} tuple.
    ?assertMatch({ok, _AudioData}, Result),
     case Result of
        {ok, AudioData} ->
            ?assert(is_binary(AudioData));
        {error, Reason} ->
            ?debugFmt("Audio generation with empty text failed unexpectedly: ~p", [Reason]),
            ?assert(false)
    end.

%% @doc Test that passing incorrect types to the NIF results in a `badarg` error.
invalid_argument_type_test_() ->
    [
        {"Text as list", ?_assertError(badarg, generate_audio("not a binary", 0))},
        {"Speaker as binary", ?_assertError(badarg, generate_audio(<<"test">>, <<"0">>))},
        {"Speaker as atom", ?_assertError(badarg, generate_audio(<<"test">>, speaker_zero))}
    ].

-endif.