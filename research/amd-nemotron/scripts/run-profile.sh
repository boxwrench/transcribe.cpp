#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/common.sh"

if [[ $# -lt 3 || $# -gt 4 ]]; then
    echo "usage: $0 hip-gfx1100|hip-gfx1201 MODEL_VARIANT LANGUAGE [DEVICE_INDEX]" >&2
    exit 2
fi
target=$1
variant=$2
language=$3
device=${4:--1}
profiler=/opt/rocm/bin/rocprofv3
cli="$REPO_ROOT/build/$target/bin/transcribe-cli"
model="$REPO_ROOT/models/$variant/$variant-Q8_0.gguf"
[[ -x "$profiler" ]] || { echo "error: missing $profiler" >&2; exit 1; }
[[ -x "$cli" && -f "$model" ]] || { echo "error: build or model missing" >&2; exit 1; }

OUT=$(new_artifact_dir profiles "$target-$variant")
command=("$cli" -q -m "$model" --backend rocm --language "$language" --stream-chunk-ms 80)
[[ "$device" -ge 0 ]] && command+=(--device "$device")
if [[ "$variant" == nemotron-speech-streaming-en-0.6b ]]; then
    command+=(--stream-att-right 1)
else
    command+=(--stream-att-right 3)
fi
command+=("$REPO_ROOT/samples/jfk.wav")
printf '%q ' "${command[@]}" >"$OUT/command.txt"
printf '\n' >>"$OUT/command.txt"
"$profiler" --runtime-trace --stats --output-format csv --output-directory "$OUT/rocprof" -- "${command[@]}" \
    >"$OUT/stdout.txt" 2>"$OUT/stderr.txt"
jq -n --arg run_id "$(basename "$OUT")" --arg runtime_commit "$(runtime_commit)" \
    --arg model_sha256 "$(sha256sum "$model" | cut -d' ' -f1)" --arg target "$target" \
    --arg variant "$variant" --argjson device "$device" --arg disable_graphs "${GGML_CUDA_DISABLE_GRAPHS-}" \
    --arg hip_visible_devices "${HIP_VISIBLE_DEVICES-}" \
    '{schema:"amd-nemotron-profile-run-v1",run_id:$run_id,runtime_commit:$runtime_commit,model_sha256:$model_sha256,target:$target,variant:$variant,device:$device,environment_overrides:{GGML_CUDA_DISABLE_GRAPHS:$disable_graphs,HIP_VISIBLE_DEVICES:$hip_visible_devices}}' \
    >"$OUT/run.json"
write_sha256_manifest "$OUT"
echo "$OUT"
