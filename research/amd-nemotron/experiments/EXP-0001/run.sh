#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)
MODEL="$REPO_ROOT/models/nemotron-speech-streaming-en-0.6b/nemotron-speech-streaming-en-0.6b-Q8_0.gguf"

run_one() {
    local arch=$1
    local visible=$2
    local mode=$3
    local out="$SCRIPT_DIR/raw/$arch-$mode.json"
    local bin="$REPO_ROOT/build/hip-$arch/bin/transcribe-stream-bench"
    if [[ "$mode" == disabled ]]; then
        HIP_VISIBLE_DEVICES="$visible" GGML_CUDA_DISABLE_GRAPHS=1 "$bin" --model "$MODEL" \
            --sample "$REPO_ROOT/samples/jfk.wav" --backend rocm --device 0 --language en \
            --feed-ms 80 --att-right 1 --warmup 2 --iters 30 --quiet --json-out "$out"
    else
        HIP_VISIBLE_DEVICES="$visible" "$bin" --model "$MODEL" --sample "$REPO_ROOT/samples/jfk.wav" \
            --backend rocm --device 0 --language en --feed-ms 80 --att-right 1 --warmup 2 --iters 30 \
            --quiet --json-out "$out"
    fi
}

mkdir -p "$SCRIPT_DIR/raw"
run_one gfx1100 0 enabled
run_one gfx1100 0 disabled
run_one gfx1201 1 enabled
run_one gfx1201 1 disabled
