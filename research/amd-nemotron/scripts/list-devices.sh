#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/common.sh"

OUT=$(new_artifact_dir runs devices)
for target in cpu vulkan hip-gfx1100 hip-gfx1201; do
    cli="$REPO_ROOT/build/$target/bin/transcribe-cli"
    if [[ ! -x "$cli" ]]; then
        echo "missing: $cli" >"$OUT/$target.txt"
        continue
    fi
    "$cli" --list-devices >"$OUT/$target.txt" 2>&1
done
jq -n --arg run_id "$(basename "$OUT")" --arg runtime_commit "$(runtime_commit)" \
    '{schema:"amd-nemotron-device-run-v1",run_id:$run_id,runtime_commit:$runtime_commit}' >"$OUT/run.json"
write_sha256_manifest "$OUT"
echo "$OUT"
