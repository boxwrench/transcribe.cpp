#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
    echo "usage: $0 BUILD_NAME PHYSICAL_GPU TARGET OUTPUT_DIR" >&2
    exit 2
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)
BUILD_NAME=$1
PHYSICAL_GPU=$2
TARGET=$3
OUT_DIR=$4
CLI="$REPO_ROOT/build/$BUILD_NAME/bin/transcribe-cli"
EN_MODEL="$REPO_ROOT/models/nemotron-speech-streaming-en-0.6b/nemotron-speech-streaming-en-0.6b-Q8_0.gguf"
MULTI_MODEL="$REPO_ROOT/models/nemotron-3.5-asr-streaming-0.6b/nemotron-3.5-asr-streaming-0.6b-Q8_0.gguf"

[[ -x "$CLI" && -f "$EN_MODEL" && -f "$MULTI_MODEL" ]] || {
    echo "error: CLI or model missing" >&2
    exit 1
}
[[ ! -e "$OUT_DIR" ]] || { echo "refusing to overwrite $OUT_DIR" >&2; exit 1; }

mkdir -p "$OUT_DIR/transcripts" "$OUT_DIR/logs"
export HIP_VISIBLE_DEVICES=$PHYSICAL_GPU
export LD_LIBRARY_PATH="/opt/rocm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

run_case() {
    local id=$1 model=$2 language=$3 right=$4 sample=$5
    "$CLI" -q -m "$model" --backend rocm --device 0 --language "$language" \
        --stream-chunk-ms 80 --stream-att-right "$right" \
        -o "$OUT_DIR/transcripts/$id.txt" "$REPO_ROOT/samples/$sample.wav" \
        >"$OUT_DIR/logs/$id.stdout" 2>"$OUT_DIR/logs/$id.stderr"
}

run_case english-short-r0 "$EN_MODEL" en 0 jfk
run_case english-short-r1 "$EN_MODEL" en 1 jfk
run_case english-short-r6 "$EN_MODEL" en 6 jfk
run_case english-conversation-r1 "$EN_MODEL" en 1 love-loss
run_case english-noise-r1 "$EN_MODEL" en 1 noise
run_case english-long-r1 "$EN_MODEL" en 1 dots-full
run_case multilingual-de-r0 "$MULTI_MODEL" de-DE 0 german
run_case multilingual-de-r3 "$MULTI_MODEL" de-DE 3 german
run_case multilingual-de-r13 "$MULTI_MODEL" de-DE 13 german

sha256sum "$OUT_DIR"/transcripts/*.txt >"$OUT_DIR/transcripts.sha256"
jq -n --arg target "$TARGET" --arg commit "$(git -C "$REPO_ROOT" rev-parse HEAD)" \
    --arg en_sha "$(sha256sum "$EN_MODEL" | cut -d' ' -f1)" \
    --arg multi_sha "$(sha256sum "$MULTI_MODEL" | cut -d' ' -f1)" \
    '{schema:"amd-nemotron-promoted-quality-v1",target:$target,runtime_commit:$commit,models:{english_sha256:$en_sha,multilingual_sha256:$multi_sha},stream_chunk_ms:80,cases:["english-short-r0","english-short-r1","english-short-r6","english-conversation-r1","english-noise-r1","english-long-r1","multilingual-de-r0","multilingual-de-r3","multilingual-de-r13"]}' \
    >"$OUT_DIR/run.json"
manifest_tmp=$(mktemp)
trap 'rm -f "$manifest_tmp"' EXIT
(
    cd "$OUT_DIR"
    find . -type f ! -name MANIFEST.sha256 -print0 | sort -z | xargs -0 sha256sum >"$manifest_tmp"
)
mv "$manifest_tmp" "$OUT_DIR/MANIFEST.sha256"
trap - EXIT
echo "$OUT_DIR"
