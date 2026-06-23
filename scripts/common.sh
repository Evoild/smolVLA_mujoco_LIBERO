#!/usr/bin/env bash

set -euo pipefail

COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$COMMON_DIR/.." && pwd)"

die() {
  printf 'error: %s\n' "$*" >&2
  exit 2
}

require_value() {
  [[ $# -ge 2 && -n "${2:-}" ]] || die "missing value for $1"
}

run_lerobot() {
  local command="$1"
  shift
  if command -v "$command" >/dev/null 2>&1; then
    "$command" "$@"
    return
  fi
  local module="lerobot.scripts.${command//-/_}"
  local lerobot_src="${LEROBOT_SRC:-$REPO_ROOT/lerobot/src}"
  if [[ -d "$lerobot_src" ]]; then
    PYTHONPATH="$lerobot_src${PYTHONPATH:+:$PYTHONPATH}" python3 -m "$module" "$@"
    return
  fi
  die "$command is not installed; activate the LeRobot environment or set LEROBOT_SRC"
}

set_reproducible_env() {
  local seed="$1"
  export PYTHONHASHSEED="$seed"
  export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
  export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
}
