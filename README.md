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