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
        <<"description">> => <<"A composable device for generating speech from text.">>,
        <<"version">> => <<"1.0">>,
        <<"api">> => #{
            <<"generate">> => #{
                <<"description">> => <<"Generates a WAV audio file. It combines text from a wasi-nn device's 'result' key (in M1) with new text provided as a parameter (in M2).">>,
                <<"method">> => <<"GET or POST">>,
                <<"parameters">> => #{
                    <<"text">> => <<"Optional. New text to append to any text received from a previous device.">>,
                    <<"speaker">> => <<"Optional. An integer for the speaker ID. Defaults to 0.">>
                }
            }
        }
    },
    {ok, InfoBody}.

%% @doc Generates speech, combining text from M1 (previous device) and M2 (current request).
generate(M1, M2, _Opts) ->
    % Extract the 'result' text from the previous device's output (M1).
    TextFromM1 = get_result_from_m1(M1),

    % Extract new text from the current request's parameters (M2).
    TextFromM2 = maps:get(<<"text">>, M2, <<>>),

    % --- THE CORRECT FIX ---
    % First, create a single combined binary. This is the most robust way
    % to join the parts before trimming.
    CombinedBinary = <<TextFromM1/binary, " ", TextFromM2/binary>>,

    % Then, convert to a list for trimming, which is compatible with all OTP versions.
    CombinedList = binary_to_list(CombinedBinary),
    TrimmedList = string:trim(CombinedList),
    FinalText = list_to_binary(TrimmedList),
    % --- END FIX ---

    case byte_size(FinalText) > 0 of
        true ->
            SpeakerStr = maps:get(<<"speaker">>, M2, integer_to_binary(?DEFAULT_SPEAKER)),
            try binary_to_integer(SpeakerStr) of
                Speaker when is_integer(Speaker) ->
                    case dev_text_to_speech_nif:generate_audio(FinalText, Speaker) of
                        {ok, AudioData} ->
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
            % No text was found in either M1's result or M2's parameters.
            {ok, #{ <<"body">> => hb_json:encode(#{<<"error">> => <<"No text provided from previous device or as a parameter">>}), <<"status">> => 400 }}
    end.

%% --- Private Functions ---

%% @doc Safely extracts the 'result' text from M1's JSON body.
get_result_from_m1(M1) ->
    Body = maps:get(<<"body">>, M1, <<>>),
    try hb_json:decode(Body) of
        #{<<"result">> := Result} when is_binary(Result) ->
            Result;
        _ ->
            <<>> % Body is not a JSON map with a 'result' key.
    catch
        error:_ ->
            <<>> % Body is not valid JSON, ignore.
    end.

%% ===================================================================
%% EUnit Tests
%% ===================================================================
-ifdef(TEST).
-include_lib("eunit/include/eunit.hrl").

%% @doc Test chaining: text from M1's 'result' key is combined with text from M2.
chaining_with_m1_and_m2_test() ->
    LLMResult = <<"The capital of France is Paris.">>,
    Filename = "/tmp/eunit_tts_chained_output.wav",
    % Simulate M1 from a wasi-nn device.
    M1 = #{<<"body">> => hb_json:encode(#{<<"result">> => LLMResult})},
    % M2 has additional text to be appended.
    M2 = #{<<"text">> => <<"And it is a beautiful city.">>, <<"speaker">> => <<"1">>},

    {ok, Response} = generate(M1, M2, #{}),

    ?assertEqual(200, maps:get(<<"status">>, Response)),
    AudioData = maps:get(<<"body">>, Response),
    ?assertEqual(ok, file:write_file(Filename, AudioData)),
    ?debugFmt("SUCCESS: Chained audio file saved for playback at ~s", [Filename]).

%% @doc Test chaining: only M1 provides text from its 'result' key.
chaining_with_m1_only_test() ->
    LLMResult = <<"Testing one, two, three.">>,
    Filename = "/tmp/eunit_tts_m1_only_output.wav",
    % Simulate M1 from a wasi-nn device.
    M1 = #{<<"body">> => hb_json:encode(#{<<"result">> => LLMResult})},
    M2 = #{}, % No additional text in M2.

    {ok, Response} = generate(M1, M2, #{}),

    ?assertEqual(200, maps:get(<<"status">>, Response)),
    ?assertEqual(ok, file:write_file(Filename, maps:get(<<"body">>, Response))),
    ?debugFmt("SUCCESS: M1-only audio file saved for playback at ~s", [Filename]).

%% @doc Test standalone usage: only M2 provides text.
standalone_with_m2_only_test() ->
    Text = <<"This is a standalone test.">>,
    Filename = "/tmp/eunit_tts_m2_only_output.wav",
    M1 = #{}, % No previous device output.
    M2 = #{<<"text">> => Text},

    {ok, Response} = generate(M1, M2, #{}),

    ?assertEqual(200, maps:get(<<"status">>, Response)),
    ?assertEqual(ok, file:write_file(Filename, maps:get(<<"body">>, Response))),
    ?debugFmt("SUCCESS: M2-only audio file saved for playback at ~s", [Filename]).

%% @doc Test failure when no text is provided in M1 or M2.
no_text_provided_failure_test() ->
    {ok, Response} = generate(#{}, #{}, #{}),
    ?assertEqual(400, maps:get(<<"status">>, Response)),
    DecodedBody = hb_json:decode(maps:get(<<"body">>, Response)),
    ?assertEqual(<<"No text provided from previous device or as a parameter">>, maps:get(<<"error">>, DecodedBody)).

%% @doc Test failure when M1's body does not contain a 'result' key.
no_result_key_in_m1_test() ->
    M1 = #{<<"body">> => hb_json:encode(#{<<"transcription">> => <<"some text">>})},
    M2 = #{},
    {ok, Response} = generate(M1, M2, #{}),
    ?assertEqual(400, maps:get(<<"status">>, Response)),
    DecodedBody = hb_json:decode(maps:get(<<"body">>, Response)),
    ?assertEqual(<<"No text provided from previous device or as a parameter">>, maps:get(<<"error">>, DecodedBody)).

-endif.