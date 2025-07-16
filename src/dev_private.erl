%%% @doc Private conversation device for end-to-end encrypted messaging.
%%%
%%% This device provides session-based encrypted communication where:
%%% 1. Sessions are created with unique IDs and AES keys
%%% 2. Session keys are protected using RSA public key encryption
%%% 3. Messages are encrypted before being stored on-chain
%%% 4. Users can decrypt messages on their side using session keys
%%% 5. Each session has its own encryption key for isolation
-module(dev_private).
-export([info/1, info/3, init/3, create_session/3, set_key/3, encrypt/3, decrypt/3]).
-include("include/hb.hrl").
-include_lib("eunit/include/eunit.hrl").
-include_lib("public_key/include/public_key.hrl").

%% @doc Controls which functions are exposed via the device API.
%%
%% This function defines the security boundary for the private conversation device
%% by explicitly listing which functions are available through the API.
%%
%% @param _ Ignored parameter
%% @returns A map with the `exports' key containing a list of allowed functions
info(_) -> 
    #{ exports => [info, init, create_session, set_key, encrypt, decrypt] }.

%% @doc Provides information about the private conversation device and its API.
%%
%% This function returns detailed documentation about the device, including:
%% 1. A high-level description of the device's purpose
%% 2. Version information
%% 3. Available API endpoints with their parameters and descriptions
%%
%% @param _Msg1 Ignored parameter
%% @param _Msg2 Ignored parameter
%% @param _Opts A map of configuration options
%% @returns {ok, Map} containing the device information and documentation
info(_Msg1, _Msg2, _Opts) ->
    InfoBody = #{
        <<"description">> => 
            <<"Private conversation device for end-to-end encrypted messaging with RSA key protection">>,
        <<"version">> => <<"1.0">>,
        <<"api">> => #{
            <<"info">> => #{
                <<"description">> => <<"Get device info">>
            },
            <<"init">> => #{
                <<"description">> => <<"Initialize the private conversation device with RSA key pair">>,
                <<"parameters">> => #{},
                <<"returns">> => #{
                    <<"message">> => <<"Initialization status">>,
                    <<"public_key">> => <<"Server's RSA public key in PEM format">>
                },
                <<"security">> => <<"Generates a new RSA key pair for the server if not already initialized">>
            },
            <<"create_session">> => #{
                <<"description">> => <<"Create a new private conversation session">>,
                <<"parameters">> => #{
                    <<"session_name">> => <<"Optional human-readable name for the session">>,
                    <<"client_public_key">> => <<"Client's RSA public key for secure key exchange">>,
                    <<"encrypted_session_key">> => <<"Client-generated AES key encrypted with server's public key">>
                },
                <<"returns">> => #{
                    <<"session_id">> => <<"Unique identifier for the session">>
                }
            },
            <<"set_key">> => #{
                <<"description">> => <<"Store the client's encrypted session key (TLS-like handshake)">>,
                <<"parameters">> => #{
                    <<"session_id">> => <<"Session identifier from create_session">>,
                    <<"encrypted_session_key">> => <<"Client-generated AES key encrypted with server's public key">>
                },
                <<"returns">> => #{
                    <<"status">> => <<"Key successfully stored">>
                },
                <<"security">> => <<"Client generates the session key, server only stores it after decryption">>
            },
            <<"encrypt">> => #{
                <<"description">> => <<"Encrypt a message for a session">>,
                <<"parameters">> => #{
                    <<"session_id">> => <<"Session identifier">>,
                    <<"message">> => <<"Message to encrypt (string or binary)">>
                },
                <<"returns">> => #{
                    <<"encrypted_message">> => <<"Base64-encoded encrypted message">>,
                    <<"iv">> => <<"Initialization vector used for encryption">>
                }
            },
            <<"decrypt">> => #{
                <<"description">> => <<"Decrypt a message for a session">>,
                <<"parameters">> => #{
                    <<"session_id">> => <<"Session identifier">>,
                    <<"encrypted_message">> => <<"Base64-encoded encrypted message">>,
                    <<"iv">> => <<"Initialization vector used during encryption">>
                },
                <<"returns">> => #{
                    <<"decrypted_message">> => <<"Original plaintext message">>
                }
            }
        }
    },
    {ok, #{<<"status">> => 200, <<"body">> => InfoBody}}.

%% @doc Initialize the private conversation device with RSA key pair.
%%
%% This function performs the following operations:
%% 1. Checks if the device is already initialized
%% 2. Generates or retrieves an RSA key pair for the server
%% 3. Stores the key pair in the node's configuration
%% 4. Returns confirmation with the public key
%%
%% This follows the same pattern as dev_green_zone initialization.
%%
%% @param _M1 Ignored parameter
%% @param _M2 May contain initialization parameters
%% @param Opts A map of configuration options
%% @returns {ok, Map} containing the public key and confirmation, or
%% {error, Binary} on failure
-spec init(M1 :: term(), M2 :: term(), Opts :: map()) -> {ok, map()} | {error, binary()}.
init(_M1, _M2, Opts) ->
    ?event(priv_device, {init, start}),
    case get_server_public_key_pem(Opts) of
        {ok, PublicKeyPEM} ->
            % Already initialized, return existing public key
            {ok, #{
                <<"status">> => 200,
                <<"body">> => #{
                    <<"message">> => <<"Private device already initialized">>,
                    <<"public_key">> => PublicKeyPEM,
                    <<"initialized">> => true
                }
            }};
        {error, _} ->
            % Initialize new device
            ?event(priv_device, {init, generating_keypair}),
            
            % Generate RSA key pair (2048-bit)
            PrivateKey = public_key:generate_key({rsa, 2048, 65537}),
            PublicKey = extract_public_key(PrivateKey),
            
            % Convert to PEM format
            PrivateKeyPEM = public_key:pem_encode([
                public_key:pem_entry_encode('RSAPrivateKey', PrivateKey)
            ]),
            PublicKeyPEM = public_key:pem_encode([
                public_key:pem_entry_encode('RSAPublicKey', PublicKey)
            ]),
            
            % Store in node configuration
            hb_http_server:set_opts(Opts#{
                priv_device_private_key => PrivateKeyPEM,
                priv_device_public_key => PublicKeyPEM,
                priv_sessions => #{}
            }),
            
            ?event(priv_device, {init, complete}),
            
            {ok, #{
                <<"status">> => 200,
                <<"body">> => #{
                    <<"message">> => <<"Private device initialized successfully">>
                }
            }}
    end.

%% @doc Creates a new private conversation session with TLS-like handshake.
%%
%% This function performs the following operations:
%% 1. Checks if the device is initialized
%% 2. Generates a unique session ID using crypto random bytes
%% 3. Optionally accepts a client public key and encrypted session key
%% 4. If encrypted key is provided, stores it for later use in set_key
%% 5. Optionally accepts a human-readable session name
%% 6. Stores the session information in the node's configuration
%% 7. Returns the session ID, creation timestamp, and server's public key
%%
%% @param _M1 Ignored parameter
%% @param M2 May contain session configuration like name, client_public_key, encrypted_session_key
%% @param Opts A map of configuration options
%% @returns {ok, Map} containing session_id, creation details, and server's public key, or
%% {error, Binary} on failure
-spec create_session(M1 :: term(), M2 :: term(), Opts :: map()) -> 
    {ok, map()} | {error, binary()}.
create_session(_M1, M2, Opts) ->
    ?event(priv_session, {create_session, start}),
    
    % Get server's public key (also checks if device is initialized)
    case get_server_public_key_pem(Opts) of
        {error, Reason} ->
            {error, Reason};
        {ok, ServerPublicKeyPEM} ->
            % Generate a unique session ID
            SessionID = base64:encode(crypto:strong_rand_bytes(16)),
                    
                    % Get optional client public key
                    ClientPublicKey = case M2 of
                        #{<<"client_public_key">> := PubKey} -> PubKey;
                        _ -> undefined
                    end,
                    
                    % Get optional encrypted session key
                    EncryptedSessionKey = case M2 of
                        #{<<"encrypted_session_key">> := EncKey} -> EncKey;
                        _ -> undefined
                    end,
                    
                    % Create session metadata
                    SessionData = #{
                        session_id => SessionID,
                        client_public_key => ClientPublicKey,
                        encrypted_session_key => EncryptedSessionKey,
                        session_key => undefined  % Will be set by set_key/3
                    },
                    
                    % Store the session in the node's configuration
                    PrivSessions = hb_opts:get(priv_sessions, #{}, Opts),
                    UpdatedSessions = maps:put(SessionID, SessionData, PrivSessions),
                    
                    hb_http_server:set_opts(Opts#{
                        priv_sessions => UpdatedSessions
                    }),
                    
                    ?event(priv_session, {create_session, complete, SessionID}),
                    
                    {ok, #{
                        <<"status">> => 200,
                        <<"body">> => #{
                            <<"session_id">> => base64:encode(SessionID),
                            <<"server_public_key">> => base64:encode(ServerPublicKeyPEM)
                    }
                }}
    end.

%% @doc Stores the client's encrypted session key (TLS-like handshake).
%%
%% This function performs the following operations:
%% 1. Validates that the session ID exists
%% 2. Receives the client's encrypted session key
%% 3. Decrypts the session key using the server's private key
%% 4. Stores the decrypted session key for future encryption/decryption
%% 5. Returns confirmation of successful key storage
%%
%% SECURITY: This implements a TLS-like handshake where:
%% - Client generates the session key
%% - Client encrypts it with server's public key
%% - Server decrypts and stores it
%% - Only the client knows the original session key
%%
%% @param _M1 Ignored parameter
%% @param M2 Map containing session_id and encrypted_session_key
%% @param Opts A map of configuration options containing server's private key
%% @returns {ok, Map} containing confirmation, or
%% {error, Binary} if session not found or decryption fails
-spec set_key(M1 :: term(), M2 :: term(), Opts :: map()) -> 
    {ok, map()} | {error, binary()}.
set_key(_M1, M2, Opts) ->
    ?event(priv_session, {set_key, start}),
    
    SessionID = case M2 of
        #{<<"session_id">> := ID} -> ID;
        _ -> hb_opts:get(<<"session_id">>, undefined, Opts)
    end,
    
    EncryptedSessionKey = case M2 of
        #{<<"encrypted_session_key">> := EncKey} -> EncKey;
        _ -> hb_opts:get(<<"encrypted_session_key">>, undefined, Opts)
    end,
    
    % Get server's private key from device configuration
    case get_server_private_key_pem(Opts) of
        {error, Reason} ->
            {error, Reason};
        {ok, ServerPrivateKeyPEM} ->
            case {SessionID, EncryptedSessionKey} of
                {undefined, _} ->
                    {error, <<"Session ID is required">>};
                {_, undefined} ->
                    {error, <<"Encrypted session key is required">>};
                {_, _} ->
                    PrivSessions = hb_opts:get(priv_sessions, #{}, Opts),
                    case maps:find(SessionID, PrivSessions) of
                        {ok, SessionData} ->
                            try
                                % Decrypt the client's session key
                                EncryptedKeyBin = base64:decode(EncryptedSessionKey),
                                DecryptedSessionKey = decrypt_with_private_key(EncryptedKeyBin, ServerPrivateKeyPEM),
                                
                                % Update session with the decrypted key
                                UpdatedSessionData = SessionData#{
                                    session_key => DecryptedSessionKey,
                                    key_set_at => erlang:system_time(second)
                                },
                                UpdatedSessions = maps:put(SessionID, UpdatedSessionData, PrivSessions),
                                
                                hb_http_server:set_opts(Opts#{
                                    priv_sessions => UpdatedSessions
                                }),
                                
                                ?event(priv_session, {set_key, success, SessionID}),
                                
                                {ok, #{
                                    <<"status">> => 200,
                                    <<"body">> => #{
                                        <<"session_id">> => SessionID,
                                        <<"message">> => <<"Session key successfully stored">>,
                                        <<"key_set_at">> => erlang:system_time(second)
                                    }
                                }}
                            catch
                                Error:Reason ->
                                    ?event(priv_session, {set_key, decrypt_error, SessionID, Error, Reason}),
                                    {error, <<"Failed to decrypt session key - invalid key or ciphertext">>}
                            end;
                        error ->
                            ?event(priv_session, {set_key, not_found, SessionID}),
                            {error, <<"Session not found">>}
                    end
            end
    end.

%% @doc Encrypts a message for a specific session.
%%
%% @param _M1 Ignored parameter
%% @param M2 Map containing session_id and message
%% @param Opts A map of configuration options
%% @returns {ok, Map} containing encrypted message and IV, or
%% {error, Binary} if session not found or encryption fails
-spec encrypt(M1 :: term(), M2 :: term(), Opts :: map()) -> 
    {ok, map()} | {error, binary()}.
encrypt(_M1, M2, Opts) ->
    ?event(priv_session, {encrypt, start}),
    
    SessionID = case M2 of
        #{<<"session_id">> := ID} -> ID;
        _ -> hb_opts:get(<<"session_id">>, undefined, Opts)
    end,
    
    Message = case M2 of
        #{<<"message">> := Msg} -> Msg;
        _ -> hb_opts:get(<<"message">>, undefined, Opts)
    end,
    
    case {SessionID, Message} of
        {undefined, _} ->
            {error, <<"Session ID is required">>};
        {_, undefined} ->
            {error, <<"Message is required">>};
        {_, _} ->
            PrivSessions = hb_opts:get(priv_sessions, #{}, Opts),
            case maps:find(SessionID, PrivSessions) of
                {ok, #{session_key := SessionKey}} when SessionKey =/= undefined ->
                    % Generate a fresh IV for this encryption
                    IV = crypto:strong_rand_bytes(16),
                    
                    % Convert message to binary if it's not already
                    MessageBin = case is_binary(Message) of
                        true -> Message;
                        false -> hb_util:bin(Message)
                    end,
                    
                    % Encrypt using AES-256-GCM
                    {EncryptedMessage, Tag} = crypto:crypto_one_time_aead(
                        aes_256_gcm,
                        SessionKey,
                        IV,
                        MessageBin,
                        <<>>,
                        true
                    ),
                    
                    ?event(priv_session, {encrypt, success, SessionID}),
                    
                    {ok, #{
                        <<"status">> => 200,
                        <<"body">> => #{
                            <<"session_id">> => SessionID,
                            <<"encrypted_message">> => 
                                base64:encode(<<EncryptedMessage/binary, Tag/binary>>),
                            <<"iv">> => base64:encode(IV)
                        }
                    }};
                {ok, _} ->
                    ?event(priv_session, {encrypt, key_not_set, SessionID}),
                    {error, <<"Session key not set - call set_key first">>};
                error ->
                    ?event(priv_session, {encrypt, not_found, SessionID}),
                    {error, <<"Session not found">>}
            end
    end.

%% @doc Decrypts a message for a specific session.
%%
%% @param _M1 Ignored parameter
%% @param M2 Map containing session_id, encrypted_message, and iv
%% @param Opts A map of configuration options
%% @returns {ok, Map} containing the decrypted message, or
%% {error, Binary} if session not found or decryption fails
-spec decrypt(M1 :: term(), M2 :: term(), Opts :: map()) -> 
    {ok, map()} | {error, binary()}.
decrypt(_M1, M2, Opts) ->
    ?event(priv_session, {decrypt, start}),
    
    SessionID = case M2 of
        #{<<"session_id">> := ID} -> ID;
        _ -> hb_opts:get(<<"session_id">>, undefined, Opts)
    end,
    
    EncryptedMessage = case M2 of
        #{<<"encrypted_message">> := Msg} -> Msg;
        _ -> hb_opts:get(<<"encrypted_message">>, undefined, Opts)
    end,
    
    IV = case M2 of
        #{<<"iv">> := IVVal} -> IVVal;
        _ -> hb_opts:get(<<"iv">>, undefined, Opts)
    end,
    
    case {SessionID, EncryptedMessage, IV} of
        {undefined, _, _} ->
            {error, <<"Session ID is required">>};
        {_, undefined, _} ->
            {error, <<"Encrypted message is required">>};
        {_, _, undefined} ->
            {error, <<"IV is required">>};
        {_, _, _} ->
            PrivSessions = hb_opts:get(priv_sessions, #{}, Opts),
            case maps:find(SessionID, PrivSessions) of
                {ok, #{session_key := SessionKey}} when SessionKey =/= undefined ->
                    try
                        % Decode the encrypted message and IV
                        Combined = base64:decode(EncryptedMessage),
                        IVBin = base64:decode(IV),
                        
                        % Separate ciphertext and authentication tag
                        CipherLen = byte_size(Combined) - 16,
                        <<Ciphertext:CipherLen/binary, Tag:16/binary>> = Combined,
                        
                        % Decrypt using AES-256-GCM
                        DecryptedMessage = crypto:crypto_one_time_aead(
                            aes_256_gcm,
                            SessionKey,
                            IVBin,
                            Ciphertext,
                            <<>>,
                            Tag,
                            false
                        ),
                        
                        ?event(priv_session, {decrypt, success, SessionID}),
                        
                        {ok, #{
                            <<"status">> => 200,
                            <<"body">> => #{
                                <<"session_id">> => SessionID,
                                <<"decrypted_message">> => DecryptedMessage
                            }
                        }}
                    catch
                        Error:Reason ->
                            ?event(priv_session, {decrypt, error, SessionID, Error, Reason}),
                            {error, <<"Decryption failed - invalid ciphertext or key">>}
                    end;
                {ok, _} ->
                    ?event(priv_session, {decrypt, key_not_set, SessionID}),
                    {error, <<"Session key not set - call set_key first">>};
                error ->
                    ?event(priv_session, {decrypt, not_found, SessionID}),
                    {error, <<"Session not found">>}
            end
    end.

%% @doc Encrypts data with an RSA public key.
%%
%% This function securely encrypts data for transmission using RSA-OAEP:
%% 1. Parses the PEM-encoded public key
%% 2. Performs RSA public key encryption
%% 3. Returns the encrypted data
%%
%% @param Data The data to encrypt (binary)
%% @param PublicKeyPEM The RSA public key in PEM format
%% @returns The encrypted data (binary)
-spec encrypt_with_public_key(Data :: binary(), PublicKeyPEM :: binary()) -> binary().
encrypt_with_public_key(Data, PublicKeyPEM) ->
    ?event(priv_session, {encrypt_with_public_key, start}),
    
    % Parse the PEM-encoded public key
    [PublicKeyEntry] = public_key:pem_decode(PublicKeyPEM),
    PublicKey = public_key:pem_entry_decode(PublicKeyEntry),
    
    % Encrypt using RSA-OAEP
    EncryptedData = public_key:encrypt_public(Data, PublicKey),
    
    ?event(priv_session, {encrypt_with_public_key, complete}),
    EncryptedData.

%% @doc Decrypts data with an RSA private key.
%%
%% This function decrypts RSA-encrypted data:
%% 1. Parses the PEM-encoded private key
%% 2. Performs RSA private key decryption
%% 3. Returns the decrypted data
%%
%% @param EncryptedData The encrypted data (binary)
%% @param PrivateKeyPEM The RSA private key in PEM format
%% @returns The decrypted data (binary)
-spec decrypt_with_private_key(EncryptedData :: binary(), PrivateKeyPEM :: binary()) -> binary().
decrypt_with_private_key(EncryptedData, PrivateKeyPEM) ->
    ?event(priv_session, {decrypt_with_private_key, start}),
    
    % Parse the PEM-encoded private key
    [PrivateKeyEntry] = public_key:pem_decode(PrivateKeyPEM),
    PrivateKey = public_key:pem_entry_decode(PrivateKeyEntry),
    
    % Decrypt using RSA
    DecryptedData = public_key:decrypt_private(EncryptedData, PrivateKey),
    
    ?event(priv_session, {decrypt_with_private_key, complete}),
    DecryptedData.

%% @doc Helper function to extract public key from RSA private key.
-spec extract_public_key(#'RSAPrivateKey'{}) -> #'RSAPublicKey'{}.
extract_public_key(#'RSAPrivateKey'{modulus = N, publicExponent = E}) ->
    #'RSAPublicKey'{modulus = N, publicExponent = E}.

%% @doc Get server's private key in PEM format from device configuration.
-spec get_server_private_key_pem(Opts :: map()) -> {ok, binary()} | {error, binary()}.
get_server_private_key_pem(Opts) ->
    case hb_opts:get(priv_device_private_key, undefined, Opts) of
        undefined ->
            {error, <<"Private device not initialized - call init first">>};
        PrivateKeyPEM ->
            {ok, PrivateKeyPEM}
    end.

%% @doc Get server's public key in PEM format from device configuration.
-spec get_server_public_key_pem(Opts :: map()) -> {ok, binary()} | {error, binary()}.
get_server_public_key_pem(Opts) ->
    case hb_opts:get(priv_device_public_key, undefined, Opts) of
        undefined ->
            {error, <<"Private device not initialized - call init first">>};
        PublicKeyPEM ->
            {ok, PublicKeyPEM}
    end.

%% @doc Simple test for the complete workflow
secure_workflow_test() ->
    % 1. 调用init - 生成密钥对
    PrivateKey = public_key:generate_key({rsa, 2048, 65537}),
    PublicKey = extract_public_key(PrivateKey),
    PrivateKeyPEM = public_key:pem_encode([public_key:pem_entry_encode('RSAPrivateKey', PrivateKey)]),
    PublicKeyPEM = public_key:pem_encode([public_key:pem_entry_encode('RSAPublicKey', PublicKey)]),
    
    % 模拟初始化后的状态
    InitOpts = #{
        priv_device_private_key => PrivateKeyPEM,
        priv_device_public_key => PublicKeyPEM,
        priv_sessions => #{}
    },
    
    % 2. 调用create_session
    {ok, #{<<"body">> := #{<<"session_id">> := SessionID}}} = 
        create_session(undefined, #{}, InitOpts),
    
    % 3. 生成一个随机密钥
    SessionKey = crypto:strong_rand_bytes(32),
    EncryptedSessionKey = encrypt_with_public_key(SessionKey, PublicKeyPEM),
    
    % 4. 调用set_key  
    SessionData = #{session_key => undefined},
    OptsWithSession = InitOpts#{priv_sessions => #{SessionID => SessionData}},
    
    {ok, _} = set_key(undefined, #{
        <<"session_id">> => SessionID,
        <<"encrypted_session_key">> => base64:encode(EncryptedSessionKey)
    }, OptsWithSession),
    
    % 5. 生成一个随机内容
    TestMessage = <<"Hello, World!">>,
    
    % 模拟set_key成功后的状态
    UpdatedSessionData = #{session_key => SessionKey},
    FinalOpts = InitOpts#{priv_sessions => #{SessionID => UpdatedSessionData}},
    
    % 6. 调用加密
    {ok, #{<<"body">> := #{
        <<"encrypted_message">> := EncMsg,
        <<"iv">> := IV
    }}} = encrypt(undefined, #{
        <<"session_id">> => SessionID,
        <<"message">> => TestMessage
    }, FinalOpts),
    
    % 调用解密
    {ok, #{<<"body">> := #{<<"decrypted_message">> := DecryptedMessage}}} = 
        decrypt(undefined, #{
            <<"session_id">> => SessionID,
            <<"encrypted_message">> => EncMsg,
            <<"iv">> => IV
        }, FinalOpts),
    
    % 7. 验证内容是否一致
    ?assertEqual(TestMessage, DecryptedMessage).

%% @doc Test RSA encryption/decryption
rsa_encrypt_decrypt_test() ->
    % 生成RSA密钥对
    PrivateKey = public_key:generate_key({rsa, 2048, 65537}),
    PublicKey = extract_public_key(PrivateKey),
    PrivateKeyPEM = public_key:pem_encode([public_key:pem_entry_encode('RSAPrivateKey', PrivateKey)]),
    PublicKeyPEM = public_key:pem_encode([public_key:pem_entry_encode('RSAPublicKey', PublicKey)]),
    
    % 测试数据
    TestData = crypto:strong_rand_bytes(32),
    
    % 加密
    EncryptedData = encrypt_with_public_key(TestData, PublicKeyPEM),
    
    % 解密
    DecryptedData = decrypt_with_private_key(EncryptedData, PrivateKeyPEM),
    
    % 验证
    ?assertEqual(TestData, DecryptedData).

%% @doc Test initialization behavior
init_test() ->
    % Test first initialization
    EmptyOpts = #{},
    {ok, #{<<"body">> := #{
        <<"public_key">> := PublicKeyPEM1,
        <<"initialized">> := true
    }}} = init(undefined, #{}, EmptyOpts),
    
    % Test second initialization (should return existing key)
    OptsWithKey = #{priv_device_public_key => PublicKeyPEM1},
    {ok, #{<<"body">> := #{
        <<"public_key">> := PublicKeyPEM2,
        <<"message">> := <<"Private device already initialized">>
    }}} = init(undefined, #{}, OptsWithKey),
    
    % Should return the same key
    ?assertEqual(PublicKeyPEM1, PublicKeyPEM2).

%% @doc Test session creation without initialization
create_session_not_initialized_test() ->
    EmptyOpts = #{},
    {error, <<"Private device not initialized - call init first">>} = 
        create_session(undefined, #{}, EmptyOpts).

%% @doc Test key setting without session
set_key_no_session_test() ->
    % Initialize device first
    EmptyOpts = #{},
    {ok, #{<<"body">> := #{
        <<"public_key">> := _PublicKeyPEM
    }}} = init(undefined, #{}, EmptyOpts),
    
    % Try to set key for non-existent session
    TestOpts = EmptyOpts#{
        priv_device_private_key => <<"dummy_key">>,
        priv_sessions => #{}
    },
    {error, <<"Session not found">>} = set_key(undefined, #{
        <<"session_id">> => <<"fake_session">>,
        <<"encrypted_session_key">> => <<"fake_key">>
    }, TestOpts).
