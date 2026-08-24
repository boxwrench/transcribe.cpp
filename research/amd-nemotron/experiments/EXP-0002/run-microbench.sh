#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)
LABEL=${1:?usage: run-microbench.sh <label>}
OUT_DIR="$SCRIPT_DIR/attempts/$LABEL"

if [[ -e "$OUT_DIR" ]]; then
    echo "refusing to overwrite existing attempt: $OUT_DIR" >&2
    exit 1
fi
mkdir -p "$OUT_DIR"

run_target() {
    local arch=$1
    local visible=$2
    local bin="$REPO_ROOT/build/hip-$arch/bin/q8-matvec-bench"
    local shape
    for shape in 1024:1024:1 1024:4096:1 4096:1024:1; do
        IFS=: read -r k m n <<< "$shape"
        HIP_VISIBLE_DEVICES="$visible" GGML_CUDA_DISABLE_GRAPHS=1 "$bin" \
            --k "$k" --m "$m" --n "$n" --warmup 20 --iters 500 \
            --json-out "$OUT_DIR/$arch-k${k}-m${m}-n${n}.json"
    done
}

run_target gfx1100 0
run_target gfx1201 1
(
    cd "$OUT_DIR"
    sha256sum ./*.json > MANIFEST.sha256
)
printf 'EXP-0002 evidence written to %s\n' "$OUT_DIR"
