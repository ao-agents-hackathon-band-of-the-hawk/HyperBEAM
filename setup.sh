#!/bin/bash

sudo apt-get update && sudo apt-get install curl -y --no-install-recommends

curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub | sudo gpg --dearmor --yes -o /usr/share/keyrings/nvidia-cuda-keyring.gpg

echo "deb [signed-by=/usr/share/keyrings/nvidia-cuda-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /" | sudo tee /etc/apt/sources.list.d/cuda-repository.list

# Add DeadSnakes PPA for Python 3.12
sudo apt-get install -y software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa -y

# Update again after adding repos
sudo apt-get update

# Install autoconf for Erlang build
sudo apt-get install -y autoconf m4

# Install main dependencies
sudo apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    pkg-config \
    libssl-dev \
    sudo \
    curl \
    ca-certificates \
    zsh \
    zlib1g-dev \
    cuda-toolkit-12-4 \
    cudnn9-cuda-12 \
    libcudnn9-cuda-12 \
    g++-12

# Install ncurses packages for Erlang
sudo apt-get install -y --no-install-recommends \
    libncurses-dev \
    libncurses5-dev \
    libncursesw5-dev \
    libtinfo5

# Install Python 3.12 packages
sudo apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    libpython3.12-dev

# Build and Install Erlang/OTP
if [ -z "$(erl -eval 'erlang:display(erlang:system_info(otp_release)), halt().' 2>/dev/null)" ]; then
    rm -rf otp
    git clone --depth=1 --branch maint-27 https://github.com/erlang/otp.git
    cd otp
    ./otp_build autoconf
    ./configure --without-wx --without-debugger --without-observer --without-et
    make -j$(nproc)
    sudo make install
    cd ..
    rm -rf otp
else
    echo "Erlang/OTP is already installed"
fi

# Build and Install Rebar3
if [ -z "$(rebar3 --version 2>/dev/null)" ]; then \
    rm -rf rebar3
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

python3.12 -m venv .venv
uv pip install -r requirements.txt
source .venv/bin/activate

git clone https://github.com/ggml-org/llama.cpp.git --branch b6337 --depth=1 native/pyrust_nn/llama.cpp

if [ -f "/usr/lib/x86_64-linux-gnu/libpython3.12.so" ]; then
    echo "LD_PRELOAD library file exists."
    if ! grep -q 'export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libpython3.12.so"' .venv/bin/activate; then
        echo "Adding LD_PRELOAD export to activate script."
        echo 'export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libpython3.12.so"' >> .venv/bin/activate
    else
        echo "LD_PRELOAD export already in activate script."
    fi
else
    echo "LD_PRELOAD library file does not exist."
fi

if [ -d "/usr/local/cuda-12.4/bin" ]; then
    echo "CUDA bin directory exists."
    if ! grep -q 'export PATH="/usr/local/cuda-12.4/bin:$PATH"' .venv/bin/activate; then
        echo "Adding CUDA bin to PATH in activate script."
        echo 'export PATH="/usr/local/cuda-12.4/bin:$PATH"' >> .venv/bin/activate
    else
        echo "CUDA bin export already in activate script."
    fi
else
    echo "CUDA bin directory does not exist."
fi

if [ -d "/usr/local/cuda-12.4/lib64" ]; then
    echo "CUDA lib64 directory exists."
    if ! grep -q 'export LD_LIBRARY_PATH="/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH"' .venv/bin/activate; then
        echo "Adding CUDA lib64 to LD_LIBRARY_PATH in activate script."
        echo 'export LD_LIBRARY_PATH="/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH"' >> .venv/bin/activate
    else
        echo "CUDA lib64 export already in activate script."
    fi
else
    echo "CUDA lib64 directory does not exist."
fi