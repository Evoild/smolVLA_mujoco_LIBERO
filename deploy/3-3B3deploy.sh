#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

ENGINE_PATH="$REPO_ROOT/runs/deploy/3-3B3/smolvla_sample_actions_core_w8a8_calibrated.plan"
OUTPUT_ROOT="$REPO_ROOT/runs/deploy/3-3B3/deploy"

bash "$SCRIPT_DIR/3-3B1deploy.sh" \
  --engine-path "$ENGINE_PATH" \
  --output-root "$OUTPUT_ROOT" \
  --episodes 1
  "$@"
