#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/common.sh"

if [[ $# -lt 4 || $# -gt 5 ]]; then
    echo "usage: $0 BUILD_TARGET BACKEND MODEL_VARIANT LANGUAGE [DEVICE_INDEX]" >&2
    exit 2
fi
target=$1
backend=$2
variant=$3
language=$4
device=${5:--1}
bin_dir="$REPO_ROOT/build/$target/bin"
model="$REPO_ROOT/models/$variant/$variant-Q8_0.gguf"
[[ -x "$bin_dir/transcribe-bench" ]] || { echo "error: missing benchmark binary" >&2; exit 1; }
[[ -x "$bin_dir/transcribe-stream-bench" ]] || { echo "error: missing streaming benchmark binary" >&2; exit 1; }
[[ -f "$model" ]] || { echo "error: missing $model" >&2; exit 1; }

OUT=$(new_artifact_dir baselines "$target-$variant-q8_0")
device_args=()
[[ "$device" -ge 0 ]] && device_args=(--device "$device")
for sample in jfk dots; do
    "$bin_dir/transcribe-bench" --model "$model" --sample "$REPO_ROOT/samples/$sample.wav" \
        --backend "$backend" "${device_args[@]}" --warmup 2 --iters 10 --quiet \
        --json-out "$OUT/offline-$sample.json"
done

if [[ "$variant" == nemotron-speech-streaming-en-0.6b ]]; then
    profiles=(0 1 6 13)
else
    profiles=(0 3 6 13)
fi
for right in "${profiles[@]}"; do
    "$bin_dir/transcribe-stream-bench" --model "$model" --sample "$REPO_ROOT/samples/jfk.wav" \
        --backend "$backend" "${device_args[@]}" --language "$language" --feed-ms 80 --att-right "$right" \
        --warmup 2 --iters 30 --quiet --json-out "$OUT/stream-r$right-jfk.json"
done

jq -n --arg run_id "$(basename "$OUT")" --arg runtime_commit "$(runtime_commit)" \
    --arg model_sha256 "$(sha256sum "$model" | cut -d' ' -f1)" --arg target "$target" \
    --arg backend "$backend" --arg variant "$variant" --arg language "$language" --argjson device "$device" \
    --arg disable_graphs "${GGML_CUDA_DISABLE_GRAPHS-}" --arg hip_visible_devices "${HIP_VISIBLE_DEVICES-}" \
    '{schema:"amd-nemotron-baseline-run-v1",run_id:$run_id,runtime_commit:$runtime_commit,model_sha256:$model_sha256,target:$target,backend:$backend,variant:$variant,language:$language,device:$device,warmup:2,ordinary_iters:10,tail_iters:30,environment_overrides:{GGML_CUDA_DISABLE_GRAPHS:$disable_graphs,HIP_VISIBLE_DEVICES:$hip_visible_devices}}' \
    >"$OUT/run.json"
write_sha256_manifest "$OUT"
echo "$OUT"
