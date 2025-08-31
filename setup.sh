#!/bin/bash

sudo apt-get update && sudo apt-get install curl -y --no-install-recommends

curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub | sudo gpg --dearmor --yes -o /usr/share/keyrings/nvidia-cuda-keyring.gpg

echo "deb [signed-by=/usr/share/keyrings/nvidia-cuda-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /" | sudo tee /etc/apt/sources.list.d/cuda-repository.list

sudo apt-get update && sudo apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    pkg-config \
    libncurses-dev \
    libssl-dev \
    sudo \
    curl \
    ca-certificates python3.12 python3.12-venv python3.12-dev libpython3.12-dev zsh zlib1g-dev cudnn9-cuda-12 cuda-toolkit-12-4

# Build and Install Erlang/OTP
if [ -z "$(erl -eval 'erlang:display(erlang:system_info(otp_release)), halt().' 2>/dev/null)" ]; then \
    git clone --depth=1 --branch maint-27 https://github.com/erlang/otp.git && \
    cd otp && \
    ./configure --without-wx --without-debugger --without-observer --without-et && \
    make -j$(nproc) && \
    sudo make install && \
    cd .. && rm -rf otp; \
else
    echo "Erlang/OTP is already installed"
fi

# Build and Install Rebar3
if [ -z "$(rebar3 --version 2>/dev/null)" ]; then \
    git clone --depth=1 https://github.com/erlang/rebar3.git && \
    cd rebar3 && \
    ./bootstrap && \
    sudo mv rebar3 /usr/local/bin/ && \
    cd .. && rm -rf rebar3; \
else
    echo "Rebar3 is already installed"
fi

# Install Rust and Cargo
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
    sh -s -- -y --default-toolchain stable

#Install Node.js (includes npm and npx)
curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash - && \
    sudo apt-get install -y nodejs=22.16.0-1nodesource1 && \
    node -v && npm -v

#install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
#source uv
source $HOME/.local/bin/env
uv --version

export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libpython3.12.so  
python3.12 -m venv .venv
uv pip install -r requirements.txt
source .venv/bin/activate

if [ -d "/usr/local/cuda-12.4/bin" ]; then
    echo "CUDA bin directory exists."
    if [[ ":$PATH:" != *":/usr/local/cuda-12.4/bin:"* ]]; then
        echo "CUDA bin not in PATH."
        if ! grep -q 'export PATH="/usr/local/cuda-12.4/bin:$PATH"' .venv/bin/activate; then
            echo "Adding CUDA bin to PATH in activate script."
            echo 'export PATH="/usr/local/cuda-12.4/bin:$PATH"' >> .venv/bin/activate
        else
            echo "CUDA bin export already in activate script."
        fi
    else
        echo "CUDA bin already in PATH."
    fi
else
    echo "CUDA bin directory does not exist."
fi
if [ -d "/usr/local/cuda-12.4/lib64" ]; then
    echo "CUDA lib64 directory exists."
    if [[ ":$LD_LIBRARY_PATH:" != *":/usr/local/cuda-12.4/lib64:"* ]]; then
        echo "CUDA lib64 not in LD_LIBRARY_PATH."
        if ! grep -q 'export LD_LIBRARY_PATH="/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH"' .venv/bin/activate; then
            echo "Adding CUDA lib64 to LD_LIBRARY_PATH in activate script."
            echo 'export LD_LIBRARY_PATH="/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH"' >> .venv/bin/activate
        else
            echo "CUDA lib64 export already in activate script."
        fi
    else
        echo "CUDA lib64 already in LD_LIBRARY_PATH."
    fi
else
    echo "CUDA lib64 directory does not exist."
fi