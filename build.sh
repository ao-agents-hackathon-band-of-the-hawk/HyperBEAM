#!/bin/bash
set -e

build_nif() {
    local nif_name=$1
    local nif_name_clean=${nif_name//-/_}  # Replace hyphens with underscores
    echo "Building ${nif_name}..."
    cd native/${nif_name}
    cargo build --release
    mkdir -p ../../priv/crates/${nif_name_clean}
    cp target/release/lib${nif_name_clean}.so ../../priv/crates/${nif_name_clean}/${nif_name_clean}.so
    cd ../..
}

build_nif "speech-to-text"