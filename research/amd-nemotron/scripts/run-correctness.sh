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
cli="$REPO_ROOT/build/$target/bin/transcribe-cli"
model="$REPO_ROOT/models/$variant/$variant-Q8_0.gguf"
[[ -x "$cli" ]] || { echo "error: missing $cli" >&2; exit 1; }
[[ -f "$model" ]] || { echo "error: missing $model" >&2; exit 1; }

OUT=$(new_artifact_dir runs "correctness-$target-$variant")
model_sha=$(sha256sum "$model" | cut -d' ' -f1)
for sample in jfk dots noise; do
    command=("$cli" -q -m "$model" --backend "$backend" --language "$language" --stream-chunk-ms 80)
    [[ "$device" -ge 0 ]] && command+=(--device "$device")
    if [[ "$variant" == nemotron-speech-streaming-en-0.6b ]]; then
        command+=(--stream-att-right 1)
    else
        command+=(--stream-att-right 3)
    fi
    command+=(-o "$OUT/$sample.txt" "$REPO_ROOT/samples/$sample.wav")
    printf '%q ' "${command[@]}" >"$OUT/$sample.command"
    printf '\n' >>"$OUT/$sample.command"
    "${command[@]}" >"$OUT/$sample.stdout" 2>"$OUT/$sample.stderr"
done
sha256sum "$OUT"/*.txt >"$OUT/transcripts.sha256"
jq -n --arg run_id "$(basename "$OUT")" --arg runtime_commit "$(runtime_commit)" \
    --arg model_sha256 "$model_sha" --arg target "$target" --arg backend "$backend" \
    --arg variant "$variant" --arg language "$language" --argjson device "$device" \
    --arg disable_graphs "${GGML_CUDA_DISABLE_GRAPHS-}" --arg hip_visible_devices "${HIP_VISIBLE_DEVICES-}" \
    '{schema:"amd-nemotron-correctness-run-v1",run_id:$run_id,runtime_commit:$runtime_commit,model_sha256:$model_sha256,target:$target,backend:$backend,variant:$variant,language:$language,device:$device,environment_overrides:{GGML_CUDA_DISABLE_GRAPHS:$disable_graphs,HIP_VISIBLE_DEVICES:$hip_visible_devices}}' \
    >"$OUT/run.json"
write_sha256_manifest "$OUT"
echo "$OUT"
