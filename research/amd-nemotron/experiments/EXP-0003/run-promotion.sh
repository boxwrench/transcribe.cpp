#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)
MODEL="$REPO_ROOT/models/nemotron-speech-streaming-en-0.6b/nemotron-speech-streaming-en-0.6b-Q8_0.gguf"
OUT_DIR=${1:-"$SCRIPT_DIR/attempts/$(date -u +%Y%m%dT%H%M%SZ)-promotion"}

if [[ -e "$OUT_DIR" ]]; then
    echo "refusing to overwrite existing attempt: $OUT_DIR" >&2
    exit 1
fi
mkdir -p "$OUT_DIR"

run_block() {
    local arch=$1
    local visible=$2
    local condition=$3
    local block=$4
    local build_name
    if [[ "$condition" == stock ]]; then
        build_name="exp0003-stock-$arch"
    else
        build_name="hip-$arch"
    fi
    HIP_VISIBLE_DEVICES="$visible" "$REPO_ROOT/build/$build_name/bin/transcribe-stream-bench" \
        --model "$MODEL" --sample "$REPO_ROOT/samples/jfk.wav" --backend rocm --device 0 \
        --language en --feed-ms 80 --att-right 1 --warmup 2 --iters 30 --quiet \
        --json-out "$OUT_DIR/$arch-$block-$condition.json"
}

run_arch() {
    local arch=$1
    local visible=$2
    local block=0
    for condition in stock candidate candidate stock stock candidate candidate stock; do
        block=$((block + 1))
        run_block "$arch" "$visible" "$condition" "$block"
    done
}

run_arch gfx1100 0
run_arch gfx1201 1
(
    cd "$OUT_DIR"
    sha256sum ./*.json > MANIFEST.sha256
)
echo "$OUT_DIR"
