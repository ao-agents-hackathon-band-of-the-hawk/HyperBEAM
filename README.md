#  Astral Plane HyperBEAM

## How to Run

### Prerequisites
- Ubuntu 22.04
- NVIDIA GPU (CUDA ARCH to be specified in the Makefile before compiling)

### Setup Instructions

1. **Set up the development environment**  
   Run the following command to install NVIDIA libraries and configure the environment:
   ```bash
   ./setup.sh
   ```

2. **Activate the Python virtual environment**  
   Load environment variables and activate the virtual environment with:
   ```bash
   source .venv/bin/activate
   ```

3. **Compile the `restart` module**  
   Navigate to the `restart` directory and compile the module:
   ```bash
   cd restart && cargo install --path .
   ```

4. **Run the application**  
   Execute the following command to start the application:
   ```bash
   rebar-shell-restart . "rebar3 as genesis_wasm shell"
   ```


For refernce audios
modify src/dev_text_to_speech.erl
```erlang
-define(DEFAULT_SPEAKER, 1). % Default to speaker 1 (AI response)
-define(SESSIONS_DIR, "sessions").
-define(DEFAULT_REF_AUDIO_PATH, <<"native/text_to_speech/utterance_0.mp3">>).
-define(DEFAULT_REF_AUDIO_TEXT, <<"In a 1997 AI class at UT Austin, a neural net playing infinite board tic-tac-toe found an unbeatable strategy. Choose moves billions of squares away, causing your opponents to run out of memory and crash.">>).
```


[Technical Documentation](https://github.com/ao-agents-hackathon-band-of-the-hawk/technical_doc)