#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/common.sh"

if [[ $# -ne 4 ]]; then
    echo "usage: $0 profile-gfx1100|profile-gfx1201 PHYSICAL_GPU TARGET_NAME SERVED_BASELINE_MS" >&2
    exit 2
fi
build_name=$1
physical_gpu=$2
target_name=$3
served_baseline_ms=$4
profiler=/opt/rocm/bin/rocprofv3
cli="$REPO_ROOT/build/$build_name/bin/transcribe-cli"
model="$REPO_ROOT/models/nemotron-speech-streaming-en-0.6b/nemotron-speech-streaming-en-0.6b-Q8_0.gguf"
[[ -x "$profiler" && -x "$cli" && -f "$model" ]] || { echo "error: profiler, diagnostic build, or model missing" >&2; exit 1; }

OUT=$(new_artifact_dir profiles "phase-map-$target_name")
mkdir -p "$OUT/rocprof"
export HIP_VISIBLE_DEVICES=$physical_gpu
export GGML_CUDA_DISABLE_GRAPHS=1
export LD_LIBRARY_PATH="/opt/rocm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

"$profiler" --kernel-trace --marker-trace --stats --kernel-rename \
    --output-format csv --output-directory "$OUT/rocprof" --output-file phase-map -- \
    "$cli" -q -m "$model" --backend rocm --device 0 --language en \
    --stream-chunk-ms 80 --stream-att-right 1 "$REPO_ROOT/samples/jfk.wav" \
    >"$OUT/stdout.txt" 2>"$OUT/stderr.txt"

uv run "$SCRIPT_DIR/analyze-phase-map.py" --target "$target_name" \
    --kernel-trace "$OUT/rocprof/phase-map_kernel_trace.csv" \
    --marker-trace "$OUT/rocprof/phase-map_marker_api_trace.csv" \
    --served-baseline-ms "$served_baseline_ms" --output "$OUT/result.json"
write_sha256_manifest "$OUT"
echo "$OUT"
