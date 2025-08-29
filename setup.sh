#!/bin/bash

curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-cuda-keyring.gpg

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
    ca-certificates python3.12 python3.12-venv python3.12-dev libpython3.12-dev zsh uv zlib cudnn9-cuda-12
    
# Build and Install Erlang/OTP
git clone --depth=1 --branch maint-27 https://github.com/erlang/otp.git && \
    cd otp && \
    ./configure --without-wx --without-debugger --without-observer --without-et && \
    make -j$(nproc) && \
    sudo make install && \
    cd .. && rm -rf otp

# Build and Install Rebar3
git clone --depth=1 https://github.com/erlang/rebar3.git && \
    cd rebar3 && \
    ./bootstrap && \
    sudo mv rebar3 /usr/local/bin/ && \
    cd .. && rm -rf rebar3

# Install Rust and Cargo
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
    sh -s -- -y --default-toolchain stable

#Install Node.js (includes npm and npx)
curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash - && \
    sudo apt-get install -y nodejs=22.16.0-1nodesource1 && \
    node -v && npm -v

export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libpython3.12.so  
python3.12 -m venv .venv
uv pip install -r requirements.txt
source .venv/bin/activate
