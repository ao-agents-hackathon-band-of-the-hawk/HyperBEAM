%% @doc NIF wrapper for speech_to_text Rust crate.
-module(dev_speech_to_text_nif).
-export([transcribe_audio/1]).
-on_load(init/0).

-include("include/cargo.hrl").
-include("include/hb.hrl").
-include_lib("eunit/include/eunit.hrl").

init() ->
    ?load_nif_from_crate(speech_to_text, 0).

transcribe_audio(_) ->
    erlang:nif_error(nif_not_loaded).

%% EUnit Tests
-ifdef(TEST).

basic_transcription_test() ->
    Path = <<"native/speech_to_text/src/man.mp3">>,  % Use binary instead of list
    Result = transcribe_audio(Path),
    case Result of
        {ok, Transcription} ->
            ?assert(is_list(Transcription) orelse is_binary(Transcription)),
            ?event(dev_speech_to_text_nif, {transcription_result, Transcription});
        {error, Reason} ->
            ?event(dev_speech_to_text_nif, {transcription_failed, Reason}),
            ?assert(false)  % Fail if error
    end.

error_handling_test() ->
    InvalidPath = <<"/invalid/path.mp3">>,
    Result = transcribe_audio(InvalidPath),
    case Result of
        {ok, _} ->
            ?assert(false);  % Should not succeed
        {error, Reason} ->
            ?event(dev_speech_to_text_nif, {expected_error, Reason}),
            ok
    end.

-endif.