#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(git rev-parse --show-toplevel)
CAMPAIGN_ROOT="$REPO_ROOT/research/amd-nemotron"

utc_timestamp() {
    date -u +%Y%m%dT%H%M%SZ
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || {
        echo "error: required command not found: $1" >&2
        return 1
    }
}

new_artifact_dir() {
    local class=$1
    local label=$2
    local root="$CAMPAIGN_ROOT/$class"
    local path="$root/$(utc_timestamp)-$label"
    mkdir -p "$root"
    if [[ -e "$path" ]]; then
        echo "error: refusing to overwrite immutable artifact: $path" >&2
        return 1
    fi
    mkdir "$path"
    printf '%s\n' "$path"
}

write_sha256_manifest() {
    local dir=$1
    (
        cd "$dir"
        find . -type f ! -name MANIFEST.sha256 -print0 | sort -z | xargs -0 -r sha256sum > MANIFEST.sha256
    )
}

runtime_commit() {
    git -C "$REPO_ROOT" rev-parse HEAD
}
