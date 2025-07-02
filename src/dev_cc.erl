%%% @doc NVIDIA GPU TEE Attestation Device
%%%
%%% This device provides GPU attestation token generation and verification
%%% using NVIDIA GPU TEE (Trusted Execution Environment) technology.
-module(dev_cc).
-export([generate/3, verify/3]).
-include("include/hb.hrl").
-include_lib("eunit/include/eunit.hrl").

%% Python script timeout (milliseconds)
-define(PYTHON_TIMEOUT, 30000).

%% Python environment check timeout (milliseconds)
-define(PYTHON_ENV_TIMEOUT, 10000).

%% Test constants
-define(TEST_MOCK_NONCE, <<"da4a06c3604a5fac8aa0b4aaf5a6354cdd0dc7c193299bc3464f30b5cbfb931a">>).

%% @doc Generate an NVIDIA GPU TEE attestation token.
-spec generate(M1 :: term(), M2 :: term(), Opts :: map()) ->
    {ok, map()} | {error, term()}.
generate(_M1, _M2, Opts) ->
    maybe
        LoadedOpts = hb_cache:ensure_all_loaded(Opts, Opts),
        % Ensure Python environment is ready
        {ok, _} ?= ensure_python_environment(),
        % Validate wallet availability
        {ok, ValidWallet} ?= 
            case hb_opts:get(priv_wallet, no_viable_wallet, LoadedOpts) of
                no_viable_wallet -> {error, no_wallet_available};
                Wallet -> {ok, Wallet}
            end,
        % Generate address and node message components
        Address = hb_util:human_id(ar_wallet:to_address(ValidWallet)),
        NodeMsg = hb_private:reset(LoadedOpts),
        {ok, PublicNodeMsgID} ?= dev_message:id(
            NodeMsg,
            #{ <<"committers">> => <<"none">> },
            LoadedOpts
        ),
        RawPublicNodeMsgID = hb_util:native_id(PublicNodeMsgID),
        % Get nonce from options or generate it
        Nonce = case hb_opts:get(nonce, undefined, LoadedOpts) of
            undefined ->
                generate_nonce(Address, RawPublicNodeMsgID);
            ProvidedNonce ->
                ProvidedNonce
        end,
        % Generate the GPU attestation token using Python
        {ok, TokenJSON} ?= call_python_attestation(generate, #{
            <<"nonce">> => Nonce,
            <<"name">> => <<"hyperbeam-node">>,
            <<"claims_version">> => <<"3.0">>,
            <<"device_type">> => <<"gpu">>,
            <<"environment">> => <<"local">>
        }),
        % Package the complete attestation message
        AttestationMsg = #{
            <<"nonce">> => hb_util:encode(Nonce),
            <<"address">> => Address,
            <<"node-message">> => NodeMsg,
            <<"token">> => TokenJSON
        },
        {ok, AttestationMsg}
    else
        {error, Reason} -> {error, Reason};
        Error -> {error, Error}
    end.

%% @doc Verify an NVIDIA GPU TEE attestation token.
-spec verify(M1 :: term(), M2 :: term(), NodeOpts :: map()) ->
    {ok, binary()} | {error, term()}.
verify(_M1, M2, NodeOpts) ->
    maybe
        % Ensure Python environment is ready
        {ok, _} ?= ensure_python_environment(),
        {ok, {Msg, Address, TokenJSON, MsgWithJSONToken}} 
            ?= extract_and_normalize_message(M2, NodeOpts),
        % Verify signature and address
        {ok, SigResult} ?= 
            verify_signature_and_address(MsgWithJSONToken, Address, NodeOpts),
        % Extract nonce for verification
        Nonce = hb_ao:get(<<"nonce">>, Msg, NodeOpts),
        % Verify the GPU attestation token
        {ok, TokenResult} ?= verify_token(TokenJSON, Nonce, NodeOpts),
        Valid = SigResult andalso TokenResult,
        {ok, hb_util:bin(Valid)}
    else
        {error, Reason} -> {error, Reason}
    end.

%% @doc Ensure Python environment and dependencies are ready.
%%
%% This function checks if Python environment is available and dependencies are installed.
%% If not, it attempts to install them automatically.
-spec ensure_python_environment() -> {ok, true} | {error, term()}.
ensure_python_environment() ->
    case get(python_env_checked) of
        true ->
            {ok, true};
        _ ->
            case check_python_environment() of
                true ->
                    put(python_env_checked, true),
                    {ok, true};
                false ->
                    case setup_python_environment() of
                        {ok, _} ->
                            put(python_env_checked, true),
                            {ok, true};
                        {error, Reason} ->
                            {error, {python_env_setup_failed, Reason}}
                    end
            end
    end.

%% @doc Check if Python environment and dependencies are available.
-spec check_python_environment() -> boolean().
check_python_environment() ->
    ScriptDir = get_python_script_dir(),
    % Check if Python and dependencies are available by running a simple test
    TestCmd = lists:flatten(io_lib:format(
        "cd ~s && python3 -c \"import nv_attestation_sdk; print('OK')\"",
        [ScriptDir]
    )),
    case os:cmd(TestCmd) of
        "OK\n" -> true;
        _ -> false
    end.

%% @doc Setup Python environment by installing dependencies.
-spec setup_python_environment() -> {ok, true} | {error, term()}.
setup_python_environment() ->
    ScriptDir = get_python_script_dir(),
    ?event({cc_python_setup, {dir, ScriptDir}}),
    
    % Check if we have uv or fall back to pip
    case check_uv_available() of
        true ->
            setup_with_uv(ScriptDir);
        false ->
            setup_with_pip(ScriptDir)
    end.

%% @doc Check if uv package manager is available.
-spec check_uv_available() -> boolean().
check_uv_available() ->
    case os:cmd("which uv") of
        "" -> false;
        _ -> true
    end.

%% @doc Setup Python environment using uv.
-spec setup_with_uv(ScriptDir :: string()) -> {ok, true} | {error, term()}.
setup_with_uv(ScriptDir) ->
    ?event({cc_python_setup, using_uv}),
    Commands = [
        "cd " ++ ScriptDir,
        "uv sync"
    ],
    Cmd = string:join(Commands, " && "),
    case os:cmd(Cmd ++ " 2>&1") of
        Output ->
            case string:find(Output, "error") of
                nomatch ->
                    ?event({cc_python_setup, {uv_success, Output}}),
                    {ok, true};
                _ ->
                    ?event({cc_python_setup, {uv_error, Output}}),
                    {error, {uv_failed, Output}}
            end
    end.

%% @doc Setup Python environment using pip.
-spec setup_with_pip(ScriptDir :: string()) -> {ok, true} | {error, term()}.
setup_with_pip(ScriptDir) ->
    ?event({cc_python_setup, using_pip}),
    Commands = [
        "cd " ++ ScriptDir,
        "python3 -m pip install nv-attestation-sdk>=2.5.0"
    ],
    Cmd = string:join(Commands, " && "),
    case os:cmd(Cmd ++ " 2>&1") of
        Output ->
            case string:find(Output, "Successfully installed") of
                nomatch ->
                    case string:find(Output, "already satisfied") of
                        nomatch ->
                            ?event({cc_python_setup, {pip_error, Output}}),
                            {error, {pip_failed, Output}};
                        _ ->
                            ?event({cc_python_setup, {pip_already_satisfied, Output}}),
                            {ok, true}
                    end;
                _ ->
                    ?event({cc_python_setup, {pip_success, Output}}),
                    {ok, true}
            end
    end.

%% @doc Extract and normalize the GPU attestation message.
-spec extract_and_normalize_message(M2 :: term(), NodeOpts :: map()) ->
    {ok, {map(), binary(), binary(), map()}} | {error, term()}.
extract_and_normalize_message(M2, NodeOpts) ->
    maybe
        RawMsg = hb_ao:get(<<"body">>, M2, M2, NodeOpts#{ hashpath => ignore }),
        MsgWithJSONToken =
            hb_util:ok(
                hb_message:with_only_committed(
                    hb_message:with_only_committers(
                        RawMsg,
                        hb_message:signers(RawMsg, NodeOpts),
                        NodeOpts
                    ),
                    NodeOpts
                )
            ),
        % Extract the token from the message
        TokenJSON = hb_ao:get(<<"token">>, MsgWithJSONToken, NodeOpts),
        Msg = maps:without([<<"token">>], MsgWithJSONToken),
        Address = hb_ao:get(<<"address">>, Msg, NodeOpts),
        {ok, {Msg, Address, TokenJSON, MsgWithJSONToken}}
    else
        {error, Reason} -> {error, Reason};
        Error -> {error, Error}
    end.

%% @doc Verify message signature and signing address.
-spec verify_signature_and_address(MsgWithJSONToken :: map(), 
    Address :: binary(), NodeOpts :: map()) ->
    {ok, true} | {error, signature_or_address_invalid}.
verify_signature_and_address(MsgWithJSONToken, Address, NodeOpts) ->
    Signers = hb_message:signers(MsgWithJSONToken, NodeOpts),
    SigIsValid = hb_message:verify(MsgWithJSONToken, Signers),
    AddressIsValid = lists:member(Address, Signers),
    case SigIsValid andalso AddressIsValid of
        true -> {ok, true};
        false -> {error, signature_or_address_invalid}
    end.

%% @doc Verify the GPU attestation token against policy.
-spec verify_token(TokenJSON :: binary(), Nonce :: binary(), NodeOpts :: map()) ->
    {ok, true} | {error, token_invalid}.
verify_token(TokenJSON, Nonce, _NodeOpts) ->
    case call_python_attestation(verify, #{
        <<"token">> => TokenJSON,
        <<"nonce">> => Nonce,
        <<"name">> => <<"hyperbeam-node">>,
        <<"device_type">> => <<"gpu">>,
        <<"environment">> => <<"local">>
    }) of
        {ok, VerifyResult} ->
            case hb_json:decode(VerifyResult) of
                #{<<"valid">> := true} ->
                    {ok, true};
                _ ->
                    {error, token_invalid}
            end;
        {error, _Error} ->
            {error, token_invalid}
    end.

%% @doc Call Python attestation script via Port.
-spec call_python_attestation(Action :: atom(), Data :: map()) ->
    {ok, binary()} | {error, term()}.
call_python_attestation(Action, Data) ->
    try
        Request = #{
            <<"action">> => atom_to_binary(Action),
            <<"data">> => Data
        },
        RequestJSON = hb_json:encode(Request),
        ScriptPath = get_python_script_path(),
        ScriptDir = get_python_script_dir(),
        
        % Use virtual environment if available, otherwise use system Python
        PythonCmd = case filelib:is_dir(filename:join(ScriptDir, ".venv")) of
            true ->
                % Use virtual environment
                case check_uv_available() of
                    true -> "uv run python";
                    false -> filename:join([ScriptDir, ".venv", "bin", "python"])
                end;
            false ->
                "python3"
        end,
        
        % Create temporary file for JSON data
        TempFile = "/tmp/dev_cc_" ++ integer_to_list(erlang:system_time()) ++ ".json",
        ok = file:write_file(TempFile, RequestJSON),
        % Use shell command with temp file
        ShellCmd = lists:flatten(io_lib:format("cat ~s | ~s ~s && rm ~s", 
            [TempFile, PythonCmd, ScriptPath, TempFile])),
        Port = open_port({spawn, ShellCmd}, 
            [binary, use_stdio, stderr_to_stdout, 
             {cd, ScriptDir}]),
        % Wait for response
        Result = receive
            {Port, {data, ResponseData}} ->
                case hb_json:decode(ResponseData) of
                    #{<<"status">> := <<"ok">>, <<"result">> := ResultData} ->
                        case ResultData of
                            #{<<"token">> := Token} -> 
                                {ok, Token};
                            #{<<"valid">> := _} ->
                                {ok, hb_json:encode(ResultData)};
                            _ -> 
                                {ok, hb_json:encode(ResultData)}
                        end;
                    #{<<"status">> := <<"error">>, <<"error">> := Error} ->
                        {error, {python_error, Error}};
                    _ ->
                        {error, {invalid_response, ResponseData}}
                end;
            {Port, {exit_status, Status}} when Status =/= 0 ->
                {error, {python_exit_error, Status}}
        after ?PYTHON_TIMEOUT ->
            {error, python_timeout}
        end,
        port_close(Port),
        Result
    catch
        _Type:Reason ->
            {error, {python_call_failed, Reason}}
    end.

%% @doc Get the path to the Python attestation script directory.
-spec get_python_script_dir() -> string().
get_python_script_dir() ->
    case code:priv_dir(hb) of
        {error, bad_name} ->
            filename:join(["..", "native", "dev_cc"]);
        PrivDir ->
            filename:join([PrivDir, "..", "native", "dev_cc"])
    end.

%% @doc Get the path to the Python attestation script.
-spec get_python_script_path() -> string().
get_python_script_path() ->
    filename:join([get_python_script_dir(), "main.py"]).

%% @doc Generate the nonce for GPU attestation token.
-spec generate_nonce(RawAddress :: binary(), RawNodeMsgID :: binary()) -> binary().
generate_nonce(RawAddress, RawNodeMsgID) ->
    Address = hb_util:native_id(RawAddress),
    NodeMsgID = hb_util:native_id(RawNodeMsgID),
    << Address/binary, NodeMsgID/binary >>.

%% Unit tests

%% @doc Test token generation with valid configuration.
generate_success_test() ->
    TestWallet = ar_wallet:new(),
    TestOpts = #{
        priv_wallet => TestWallet,
        nonce => ?TEST_MOCK_NONCE
    },
    case generate(#{}, #{}, TestOpts) of
        {ok, Result} ->
            ?assert(is_map(Result)),
            ?assert(maps:is_key(<<"nonce">>, Result)),
            ?assert(maps:is_key(<<"address">>, Result)),
            ?assert(maps:is_key(<<"node-message">>, Result)),
            ?assert(maps:is_key(<<"token">>, Result)),
            Token = maps:get(<<"token">>, Result),
            ?assert(is_binary(Token)),
            ?assert(byte_size(Token) > 0);
        {error, {python_error, <<"No evidence available for attestation">>}} ->
            ?assert(true);
        {error, {python_env_setup_failed, _}} ->
            % Python environment setup failed - this is expected in some environments
            ?assert(true);
        Other ->
            ?assertEqual({ok, result}, Other)
    end.

%% @doc Test successful round-trip: generate then verify.
verify_mock_generate_success_test_() ->
    { timeout, 30, fun verify_mock_generate_success/0 }.
verify_mock_generate_success() ->
    TestWallet = ar_wallet:new(),
    GenerateOpts = #{
        priv_wallet => TestWallet,
        nonce => ?TEST_MOCK_NONCE
    },
    case generate(#{}, #{}, GenerateOpts) of
        {ok, GeneratedMsg} ->
            ?assert(is_map(GeneratedMsg)),
            ?assert(maps:is_key(<<"token">>, GeneratedMsg)),
            ?assert(maps:is_key(<<"address">>, GeneratedMsg)),
            ?assert(maps:is_key(<<"nonce">>, GeneratedMsg)),
            {ok, VerifyResult} = 
                verify(
                    #{}, 
                    hb_message:commit(GeneratedMsg, GenerateOpts),
                    #{}
                ),
            ?assertEqual(<<"true">>, VerifyResult),
            TokenData = maps:get(<<"token">>, GeneratedMsg),
            ?assert(is_binary(TokenData)),
            ?assert(byte_size(TokenData) > 0);
        {error, {python_env_setup_failed, _}} ->
            % Python environment setup failed - skip verification test
            ?assert(true);
        {error, {python_error, <<"No evidence available for attestation">>}} ->
            % No GPU hardware available - this is expected in some environments
            ?assert(true)
    end. 