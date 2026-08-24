#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/common.sh"

usage() {
    echo "usage: $0 [cpu|vulkan|hip-gfx1100|hip-gfx1201 ...]" >&2
}

targets=("$@")
if [[ ${#targets[@]} -eq 0 ]]; then
    targets=(cpu vulkan hip-gfx1100 hip-gfx1201)
fi

OUT=$(new_artifact_dir runs builds)
export CCACHE_DIR="$REPO_ROOT/.cache/ccache"
mkdir -p "$CCACHE_DIR"
for target in "${targets[@]}"; do
    case "$target" in
        cpu)
            args=(-DTRANSCRIBE_BUILD_TOOLS=ON)
            ;;
        vulkan)
            args=(-DTRANSCRIBE_VULKAN=ON -DTRANSCRIBE_BUILD_TOOLS=ON)
            ;;
        hip-gfx1100)
            args=(-DTRANSCRIBE_HIP=ON -DAMDGPU_TARGETS=gfx1100 -DTRANSCRIBE_BUILD_TOOLS=ON)
            ;;
        hip-gfx1201)
            args=(-DTRANSCRIBE_HIP=ON -DAMDGPU_TARGETS=gfx1201 -DTRANSCRIBE_BUILD_TOOLS=ON)
            ;;
        *)
            usage
            exit 2
            ;;
    esac
    build_dir="$REPO_ROOT/build/$target"
    cmake -S "$REPO_ROOT" -B "$build_dir" -G Ninja -DCMAKE_BUILD_TYPE=Release "${args[@]}" \
        2>&1 | tee "$OUT/$target-configure.log"
    cmake --build "$build_dir" 2>&1 | tee "$OUT/$target-build.log"
    ctest --test-dir "$build_dir" --output-on-failure 2>&1 | tee "$OUT/$target-ctest.log"
done

jq -n --arg run_id "$(basename "$OUT")" --arg runtime_commit "$(runtime_commit)" \
    --argjson targets "$(printf '%s\n' "${targets[@]}" | jq -R . | jq -s .)" \
    '{schema:"amd-nemotron-build-run-v1",run_id:$run_id,runtime_commit:$runtime_commit,targets:$targets}' >"$OUT/run.json"
write_sha256_manifest "$OUT"
echo "$OUT"
