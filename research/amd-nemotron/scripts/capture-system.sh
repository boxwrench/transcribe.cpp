#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/common.sh"

for command in uname lsb_release lscpu free cmake ninja gcc hipcc rocminfo rocm-smi vulkaninfo jq; do
    require_command "$command"
done

OUT=$(new_artifact_dir runs system)
capture() {
    local name=$1
    shift
    "$@" >"$OUT/$name.txt" 2>&1
}

capture uname uname -a
capture lsb_release lsb_release -a
capture lscpu lscpu
capture memory free -h
capture cmake cmake --version
capture ninja ninja --version
capture clang bash -c 'clang --version || true'
capture gcc gcc --version
capture hipcc hipcc --version
capture rocminfo rocminfo
capture rocm-smi rocm-smi --showallinfo
capture vulkan vulkaninfo --summary
capture profiler bash -c '/opt/rocm/bin/rocprofv3 --version || /opt/rocm/bin/rocprofv3 --help'
capture packages dpkg-query -W
capture git git -C "$REPO_ROOT" status --short --branch
capture remotes git -C "$REPO_ROOT" remote -v
capture ggml-upstream cat "$REPO_ROOT/ggml/UPSTREAM"

jq -n \
    --arg run_id "$(basename "$OUT")" \
    --arg timestamp "$(date -u --iso-8601=seconds)" \
    --arg runtime_commit "$(runtime_commit)" \
    --arg kernel "$(uname -r)" \
    --arg rocm "$(readlink -f /opt/rocm)" \
    '{schema:"amd-nemotron-system-run-v1",run_id:$run_id,timestamp:$timestamp,runtime_commit:$runtime_commit,kernel:$kernel,rocm_path:$rocm}' \
    >"$OUT/run.json"
write_sha256_manifest "$OUT"
echo "$OUT"
