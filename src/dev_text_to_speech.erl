-module(dev_text_to_speech).
-export([generate/3, info/1, info/3]).
-include("include/hb.hrl").

-define(DEFAULT_SPEAKER, 0).

%% @doc Declares the functions this device exports to the HyperBeam system.
info(_) ->
    #{exports => [generate, info]}.

%% @doc Provides helpful API information about the device's functions.
info(_M1, _M2, _Opts) ->
    InfoBody = #{
        <<"description">> => <<"A device for generating speech from text using the CSM model.">>,
        <<"version">> => <<"1.0">>,
        <<"api">> => #{
            <<"generate">> => #{
                <<"description">> => <<"Generates a WAV audio file from the given text.">>,
                <<"method">> => <<"GET or POST">>,
                <<"parameters">> => #{
                    <<"text">> => <<"Required. The text to be converted to speech.">>,
                    <<"speaker">> => <<"Optional. An integer for the speaker ID. Defaults to 0.">>
                }
            }
        }
    },
    {ok, InfoBody}.

%% @doc The primary function to generate speech.
%% HyperBeam places all URL query parameters directly into the M2 map.
generate(_M1, M2, _Opts) ->
    case maps:get(<<"text">>, M2, undefined) of
        Text when is_binary(Text) ->
            % The 'speaker' parameter is optional. Default to '0' if not present or invalid.
            SpeakerStr = maps:get(<<"speaker">>, M2, integer_to_binary(?DEFAULT_SPEAKER)),
            try binary_to_integer(SpeakerStr) of
                Speaker when is_integer(Speaker) ->
                    % Call the NIF to perform the text-to-speech generation.
                    case dev_text_to_speech_nif:generate_audio(Text, Speaker) of
                        {ok, AudioData} ->
                            {ok, #{
                                <<"body">> => AudioData,
                                <<"status">> => 200,
                                <<"headers">> => #{
                                    <<"content-type">> => <<"audio/wav">>
                                }
                            }};
                        {error, Reason} ->
                            ?event(dev_text_to_speech, {generation_failed, Reason}),
                            Error = iolist_to_binary(io_lib:format("~p", [Reason])),
                            {ok, #{
                                <<"body">> => hb_json:encode(#{<<"error">> => <<"Audio generation failed">>, <<"reason">> => Error}),
                                <<"status">> => 500
                            }}
                    end
            catch
                error:badarg ->
                    {ok, #{
                        <<"body">> => hb_json:encode(#{<<"error">> => <<"'speaker' parameter must be an integer">>}),
                        <<"status">> => 400
                    }}
            end;
        undefined ->
            % The required 'text' parameter was not found in the M2 map.
            {ok, #{
                <<"body">> => hb_json:encode(#{<<"error">> => <<"Missing required 'text' parameter">>}),
                <<"status">> => 400
            }}
    end.

%% ===================================================================
%% EUnit Tests
%% ===================================================================
-ifdef(TEST).
-include_lib("eunit/include/eunit.hrl").

%% @doc Test successful generation by providing params in M2, simulating the framework.
successful_generation_and_save_test() ->
    Text = <<"This is a test of the text to speech device.">>,
    Filename = "/tmp/eunit_tts_output_default_speaker.wav",
    % Simulate the HyperBeam framework by placing parameters directly in M2.
    M2 = #{<<"text">> => Text},

    {ok, Response} = generate(#{}, M2, #{}),

    ?assert(maps:is_key(<<"status">>, Response)),
    ?assertEqual(200, maps:get(<<"status">>, Response)),

    AudioData = maps:get(<<"body">>, Response),
    ?assert(is_binary(AudioData)),
    ?assert(size(AudioData) > 44),

    ?assertEqual(ok, file:write_file(Filename, AudioData)),
    ?debugFmt("SUCCESS: Audio file saved for playback at ~s", [Filename]).

%% @doc Test with a specific speaker ID provided in M2.
specific_speaker_and_save_test() ->
    Text = <<"This audio should be generated with speaker one.">>,
    Filename = "/tmp/eunit_tts_output_speaker_1.wav",
    % M2 contains all necessary parameters.
    M2 = #{<<"text">> => Text, <<"speaker">> => <<"1">>},

    {ok, Response} = generate(#{}, M2, #{}),

    ?assert(maps:is_key(<<"status">>, Response)),
    ?assertEqual(200, maps:get(<<"status">>, Response)),

    AudioData = maps:get(<<"body">>, Response),
    ?assert(is_binary(AudioData)),

    ?assertEqual(ok, file:write_file(Filename, AudioData)),
    ?debugFmt("SUCCESS: Audio file saved for playback at ~s", [Filename]).

%% @doc Test failure when the 'text' parameter is missing from M2.
missing_text_parameter_test() ->
    M2 = #{}, % M2 is empty, no 'text' parameter.
    {ok, Response} = generate(#{}, M2, #{}),

    ?assertEqual(400, maps:get(<<"status">>, Response)),
    DecodedBody = hb_json:decode(maps:get(<<"body">>, Response)),
    ?assertEqual(<<"Missing required 'text' parameter">>, maps:get(<<"error">>, DecodedBody)).

%% @doc Test failure when 'speaker' is not a valid integer.
invalid_speaker_parameter_test() ->
    Text = <<"This should fail.">>,
    M2 = #{<<"text">> => Text, <<"speaker">> => <<"not_a_number">>},
    {ok, Response} = generate(#{}, M2, #{}),

    ?assertEqual(400, maps:get(<<"status">>, Response)),
    DecodedBody = hb_json:decode(maps:get(<<"body">>, Response)),
    ?assertEqual(<<"'speaker' parameter must be an integer">>, maps:get(<<"error">>, DecodedBody)).

-endif.