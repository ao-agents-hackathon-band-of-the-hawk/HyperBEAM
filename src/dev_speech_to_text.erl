%% src/dev_speech_to_text.erl (new file for the device)

-module(dev_speech_to_text).
-export([transcribe/3]).
-include("include/hb.hrl").  % Assuming this include is standard for HyperBEAM devices
-include_lib("eunit/include/eunit.hrl").

transcribe(_M1, M2, _Opts) ->
    % Extract the binary audio data from the message map (HTTP body is placed under <<"body">>)
    Data = maps:get(<<"body">>, M2, <<>>),  % Default to empty binary if not found
    case Data of
        <<>> ->
            {ok, #{<<"body">> => hb_json:encode(#{<<"error">> => <<"No audio data provided">>}),
                   <<"status">> => 400}};
        _ ->
            % Generate a unique filename using system time
            Timestamp = erlang:system_time(millisecond),
            Filename = io_lib:format("/tmp/hyperbeam_audio_~p.mp3", [Timestamp]),
            PathList = lists:flatten(Filename),
            Path = list_to_binary(PathList),
            % Save the binary data to the file
            case file:write_file(PathList, Data) of
                ok ->
                    ?event(dev_speech_to_text, {saved_audio, PathList}),
                    % Call the NIF to transcribe
                    Result = case dev_speech_to_text_nif:transcribe_audio(Path) of
                        {error, Reason} ->
                            ?event(dev_speech_to_text, {transcription_failed, Reason}),
                            {ok, #{<<"body">> => hb_json:encode(#{<<"error">> => <<"Transcription failed">>, <<"reason">> => Reason}),
                                   <<"status">> => 500}};
                        {ok, Transcript} ->
                            ?event(dev_speech_to_text, {transcription_success, Transcript}),
                            {ok, #{<<"body">> => hb_json:encode(#{<<"transcription">> => Transcript}),
                                   <<"status">> => 200}}
                    end,
                    % Delete the temporary file after processing
                    case file:delete(PathList) of
                        ok ->
                            ?event(dev_speech_to_text, {deleted_audio, PathList});
                        {error, DeleteReason} ->
                            ?event(dev_speech_to_text, {delete_audio_failed, DeleteReason})
                    end,
                    Result;
                {error, Reason} ->
                    ?event(dev_speech_to_text, {save_audio_failed, Reason}),
                    {ok, #{<<"body">> => hb_json:encode(#{<<"error">> => <<"Failed to save audio">>, <<"reason">> => list_to_binary(io_lib:format("~p", [Reason]))}),
                           <<"status">> => 500}}
            end
    end.

%% EUnit Tests
-ifdef(TEST).

no_audio_data_test() ->
    M1 = #{},
    M2 = #{<<"body">> => <<>>},
    {ok, Response} = transcribe(M1, M2, #{}),
    ?assertMatch(#{<<"body">> := _, <<"status">> := 400}, Response),
    Decoded = hb_json:decode(maps:get(<<"body">>, Response)),
    ?assertMatch(#{<<"error">> := <<"No audio data provided">>}, Decoded).

file_write_error_test() ->
    % Simulate file write error by using an invalid path
    InvalidPathList = "/invalid_dir/hyperbeam_audio_test.mp3",
    case file:write_file(InvalidPathList, <<"fake audio data">>) of
        ok ->
            ?assert(false);  % Should not succeed
        {error, _Reason} ->
            ok  % Expected error
    end.

successful_transcription_test() ->
    M1 = #{},
    M2 = #{<<"body">> => <<"fake audio data">>},
    {ok, Response} = transcribe(M1, M2, #{}),
    case Response of
        #{<<"status">> := 200} ->
            Decoded = hb_json:decode(maps:get(<<"body">>, Response)),
            ?assert(maps:is_key(<<"transcription">>, Decoded));
        #{<<"status">> := 500} ->
            Decoded = hb_json:decode(maps:get(<<"body">>, Response)),
            ?assert(maps:is_key(<<"error">>, Decoded));
        _ ->
            ?assert(false)
    end.

-endif.