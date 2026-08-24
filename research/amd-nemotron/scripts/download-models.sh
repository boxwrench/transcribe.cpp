#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/common.sh"
require_command hf
require_command jq

MODEL_SET="$CAMPAIGN_ROOT/manifests/model/q8_0-models.json"
OUT=$(new_artifact_dir runs model-download)
manifest="$OUT/models.jsonl"

while IFS=$'\t' read -r variant repo filename; do
    local_dir="$REPO_ROOT/models/$variant"
    mkdir -p "$local_dir"
    hf download "$repo" "$filename" --local-dir "$local_dir"
    path="$local_dir/$filename"
    sha=$(sha256sum "$path" | cut -d' ' -f1)
    bytes=$(stat -c %s "$path")
    jq -cn --arg variant "$variant" --arg repo "$repo" --arg filename "$filename" \
        --arg path "$path" --arg sha256 "$sha" --argjson bytes "$bytes" \
        '{variant:$variant,repo:$repo,filename:$filename,path:$path,sha256:$sha256,bytes:$bytes}' >>"$manifest"
done < <(jq -r '.models[] | [.variant,.repo,.filename] | @tsv' "$MODEL_SET")

jq -s --arg run_id "$(basename "$OUT")" --arg runtime_commit "$(runtime_commit)" \
    '{schema:"amd-nemotron-downloaded-models-v1",run_id:$run_id,runtime_commit:$runtime_commit,models:.}' \
    "$manifest" >"$OUT/run.json"
write_sha256_manifest "$OUT"
echo "$OUT"
