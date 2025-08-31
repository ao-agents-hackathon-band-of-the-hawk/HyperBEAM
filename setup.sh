#!/bin/bash

sudo apt-get update && sudo apt-get install curl -y --no-install-recommends

curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub | sudo gpg --dearmor --yes -o /usr/share/keyrings/nvidia-cuda-keyring.gpg

echo "deb [signed-by=/usr/share/keyrings/nvidia-cuda-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /" | sudo tee /etc/apt/sources.list.d/cuda-repository.list

# for python 3.12
sudo add-apt-repository -y ppa:deadsnakes/ppa

sudo apt-get update && sudo apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    pkg-config \
    libncurses-dev \
    libssl-dev \
    sudo \
    curl \
    ca-certificates python3.12 python3.12-venv python3.12-dev libpython3.12-dev zsh zlib1g-dev cudnn9-cuda-12

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
