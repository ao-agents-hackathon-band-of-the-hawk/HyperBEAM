{
  description = "Development shell for HyperBEAM";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.11";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }: 
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; config.allowUnfree = true; config.cudaSupport = true;};
        erlang = pkgs.beam.interpreters.erlang_27;
        beamPackages = pkgs.beam.packagesWith erlang;
        rebar3 = beamPackages.rebar3;
        
        # Common build inputs
        commonInputs = with pkgs; [
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
          python3
          lua
          gnumake
          curl
          cacert
          ninja
          gcc-unwrapped.lib
          gawk
          zlib
          ffmpeg
          stdenv.cc.cc.lib  # For libstdc++.so.6; handles /lib64
          uv
          ccache  # Added for compatible ccache
          tbb  # Added for compatible oneTBB/libtbb
        ];

        # Platform-specific inputs
        linuxInputs = with pkgs; [ rocksdb numactl cudaPackages.cudatoolkit cudaPackages.cudnn cudaPackages.libcublas];
        darwinInputs = [ /* Add Darwin-specific if needed */ ];

        totalInputs = commonInputs
          ++ pkgs.lib.optionals pkgs.stdenv.isLinux linuxInputs
          ++ pkgs.lib.optionals pkgs.stdenv.isDarwin darwinInputs;
        
      in {
        devShells.default = pkgs.mkShell {
         buildInputs = totalInputs;
            
          shellHook = ''
              export LD_PRELOAD="${pkgs.python3}/lib/libpython3.12.so"
              # Set platform-specific environment
              case "$OSTYPE" in
                linux*)
                    export CMAKE_LIBRARY_PATH="${pkgs.lib.makeLibraryPath totalInputs}:$CMAKE_LIBRARY_PATH"
                    unset LD_LIBRARY_PATH  # Clear any inherited paths
                    export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath totalInputs}"  # Auto-generated clean path + stubs subdir
                    preferred_shell=$(awk -F: -v user="$USER" '$1 == user {print $7}' /etc/passwd)
                    ;;
                darwin*)
                    export CMAKE_LIBRARY_PATH="${pkgs.lib.makeLibraryPath totalInputs}:$CMAKE_LIBRARY_PATH"
                    unset LD_LIBRARY_PATH
                    export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath totalInputs}"
                    preferred_shell=$(dscl . -read /Users/$USER UserShell 2>/dev/null | cut -d: -f2 | sed 's/^ //')
                    ;;
            esac
            if [ -z "$preferred_shell" ]; then
              preferred_shell="${pkgs.bashInteractive}/bin/bash"
            fi

            echo "Setting up Python virtual environment..."
            if [ ! -d ".venv" ]; then
              ${pkgs.python3}/bin/python3 -m venv .venv
              ${pkgs.uv}/bin/uv pip install -r requirements.txt
              echo "Virtual environment created and packages installed."
            fi            

            # Determine activation script based on shell
            shell_base=$(basename "$preferred_shell")
            case "$shell_base" in
              zsh|bash|sh) activate="venv/bin/activate" ;;
              fish) activate="venv/bin/activate.fish" ;;
              csh|tcsh) activate="venv/bin/activate.csh" ;;
              *) activate="" ;;
            esac

            if [ -n "$activate_script" ]; then
              exec "$preferred_shell" -c ". $activate_script; exec $preferred_shell -i"
            else
              export VIRTUAL_ENV="$(pwd)/venv"
              export PATH="$(pwd)/venv/bin:$PATH"
              exec "$preferred_shell" -i
            fi
          '';
        };
      }
    );
}