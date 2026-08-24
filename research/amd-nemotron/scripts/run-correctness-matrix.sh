#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

run_target() {
    local target=$1
    local backend=$2
    local device=$3
    "$SCRIPT_DIR/run-correctness.sh" "$target" "$backend" nemotron-speech-streaming-en-0.6b en "$device"
    "$SCRIPT_DIR/run-correctness.sh" "$target" "$backend" nemotron-3.5-asr-streaming-0.6b en-US "$device"
}

run_target vulkan vulkan 0
run_target vulkan vulkan 1
HIP_VISIBLE_DEVICES=0 run_target hip-gfx1100 rocm 0
HIP_VISIBLE_DEVICES=1 run_target hip-gfx1201 rocm 0
