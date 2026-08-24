#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 || $# -gt 6 ]]; then
    echo "usage: $0 exp0004-gfx1100|exp0004-gfx1201 PHYSICAL_GPU TARGET enabled|disabled OUTPUT_DIR [full|device]" >&2
    exit 2
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)
BUILD_NAME=$1
PHYSICAL_GPU=$2
TARGET=$3
CONDITION=$4
OUT_DIR=$5
TRACE_MODE=${6:-full}
PROFILER=/opt/rocm/bin/rocprofv3
AMD_SMI=/opt/rocm/bin/amd-smi
BENCH="$REPO_ROOT/build/$BUILD_NAME/bin/transcribe-stream-bench"
MODEL="$REPO_ROOT/models/nemotron-speech-streaming-en-0.6b/nemotron-speech-streaming-en-0.6b-Q8_0.gguf"

if [[ "$CONDITION" != enabled && "$CONDITION" != disabled ]]; then
    echo "error: condition must be enabled or disabled" >&2
    exit 2
fi
if [[ "$TRACE_MODE" != full && "$TRACE_MODE" != device ]]; then
    echo "error: trace mode must be full or device" >&2
    exit 2
fi
if [[ -e "$OUT_DIR" ]]; then
    echo "refusing to overwrite existing capture: $OUT_DIR" >&2
    exit 1
fi
[[ -x "$PROFILER" && -x "$BENCH" && -f "$MODEL" ]] || {
    echo "error: profiler, benchmark, or model missing" >&2
    exit 1
}

mkdir -p "$OUT_DIR/rocprof"
export HIP_VISIBLE_DEVICES=$PHYSICAL_GPU
export LD_LIBRARY_PATH="/opt/rocm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if [[ "$CONDITION" == disabled ]]; then
    export GGML_CUDA_DISABLE_GRAPHS=1
else
    unset GGML_CUDA_DISABLE_GRAPHS
fi

"$AMD_SMI" metric -g "$PHYSICAL_GPU" -u -p -c -t --json >"$OUT_DIR/sensor-before.json" 2>"$OUT_DIR/sensor-before.stderr" || true
TRACE_ARGS=(--kernel-trace --memory-copy-trace --marker-trace --stats)
ANALYZE_ARGS=()
if [[ "$TRACE_MODE" == full ]]; then
    TRACE_ARGS+=(--hip-runtime-trace)
else
    ANALYZE_ARGS+=(--device-only)
fi
"$PROFILER" "${TRACE_ARGS[@]}" \
    --output-format csv --output-directory "$OUT_DIR/rocprof" --output-file timeline -- \
    "$BENCH" --model "$MODEL" --sample "$REPO_ROOT/samples/jfk.wav" \
    --backend rocm --device 0 --language en --feed-ms 80 --att-right 1 \
    --warmup 2 --iters 3 --quiet --json-out "$OUT_DIR/bench.json" \
    >"$OUT_DIR/stdout.txt" 2>"$OUT_DIR/stderr.txt"
"$AMD_SMI" metric -g "$PHYSICAL_GPU" -u -p -c -t --json >"$OUT_DIR/sensor-after.json" 2>"$OUT_DIR/sensor-after.stderr" || true

uv run "$SCRIPT_DIR/analyze-timeline.py" --trace-root "$OUT_DIR/rocprof" \
    --bench-json "$OUT_DIR/bench.json" --target "$TARGET" --condition "$CONDITION" \
    --output "$OUT_DIR/result.json" "${ANALYZE_ARGS[@]}" >"$OUT_DIR/analyzer.stdout"
(
    cd "$OUT_DIR"
    find . -type f ! -name MANIFEST.sha256 -print0 | sort -z | xargs -0 sha256sum >MANIFEST.sha256
)
echo "$OUT_DIR"
