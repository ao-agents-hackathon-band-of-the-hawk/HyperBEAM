{ pkgs ? import (fetchTarball {
    url = "https://github.com/NixOS/nixpkgs/archive/refs/heads/nixos-24.11.tar.gz";
    sha256 = "1s2gr5rcyqvpr58vxdcb095mdhblij9bfzaximrva2243aal3dgx";  # Add this sha256 for reproducibility (prefetched from URL)
  }) {} }:

let
  erlang = pkgs.beam.interpreters.erlang_27;
  beamPackages = pkgs.beam.packagesWith erlang;
  rebar3 = beamPackages.rebar3;
in


pkgs.mkShell {
  buildInputs = with pkgs; [
    erlang
    rebar3
    rustc
    cargo
    cmake
    pkg-config
    git
    ncurses
    openssl
    nodejs_22
    rocksdb
    python3
    lua
    gnumake
    curl
    cacert
    ninja  # Ensure Nix's ninja is used for the build
    gcc-unwrapped.lib  # Provides libatomic.so
    numactl  # Provides libnuma.so
];
shellHook = ''
    export CMAKE_LIBRARY_PATH="${pkgs.gcc-unwrapped.lib}/lib64:${pkgs.numactl}/lib:''${CMAKE_LIBRARY_PATH}"
    echo "Setting up Python virtual environment..."
    if [ ! -d "venv" ]; then
      python3 -m venv venv
      source venv/bin/activate
      pip install mkdocs mkdocs-material mkdocs-git-revision-date-localized-plugin
      echo "Virtual environment created and packages installed."
    else
      source venv/bin/activate
      echo "Activated existing virtual environment."
    fi
  '';
}