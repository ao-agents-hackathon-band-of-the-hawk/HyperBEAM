#!/bin/bash
set -e

build_nif() {
    local nif_name=$1
    echo "Building ${nif_name}..."
    cd native/${nif_name}
    cargo build --release
    mkdir -p ../../priv/crates/${nif_name}
    cp target/release/lib${nif_name}.so ../../priv/crates/${nif_name}/${nif_name}.so
    cd ../..
}

build_nif "speech_to_text"
build_nif "text_to_speech"