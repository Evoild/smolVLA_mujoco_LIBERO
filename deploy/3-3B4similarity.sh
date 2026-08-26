#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$REPO_ROOT/scripts/common.sh"

POLICY_PATH="$REPO_ROOT/smolvla_libero"
ENGINE_PATH="$REPO_ROOT/runs/deploy/3-3B4/smolvla_sample_actions_core_vlm_w8a8.plan"
OUTPUT_DIR="$REPO_ROOT/runs/deploy/3-3B4/similarity"
DEVICE=cuda
CAPTURE=leaf
PYTHON="${PYTHON_BIN:-python3}"

usage() {
  printf '%s\n' "Usage: $0 [options]" \
    "  --policy-path PATH       default: $POLICY_PATH" \
    "  --engine-path PATH       default: $ENGINE_PATH" \
    "  --output-dir DIR         default: $OUTPUT_DIR" \
    "  --device DEVICE          default: $DEVICE" \
    "  --capture core|leaf      default: $CAPTURE" \
    "  PYTHON_BIN=/path/python optional Python with torch/lerobot/tensorrt dependencies"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --policy-path) require_value "$@"; POLICY_PATH="$2"; shift 2 ;;
    --engine-path) require_value "$@"; ENGINE_PATH="$2"; shift 2 ;;
    --output-dir) require_value "$@"; OUTPUT_DIR="$2"; shift 2 ;;
    --device) require_value "$@"; DEVICE="$2"; shift 2 ;;
    --capture) require_value "$@"; CAPTURE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ -f "$ENGINE_PATH" ]] || die "$ENGINE_PATH not found; run deploy/3-3B4export.sh first"

export LEROBOT_SRC="${LEROBOT_SRC:-$REPO_ROOT/lerobot/src}"
export PYTHONPATH="$LEROBOT_SRC${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON" "$SCRIPT_DIR/test.py" \
  --policy-path "$POLICY_PATH" \
  --engine-path "$ENGINE_PATH" \
  --device "$DEVICE" \
  --capture "$CAPTURE" \
  --quantize-module-prefix model.vlm_with_expert.vlm.model.text_model \
  --output-dir "$OUTPUT_DIR"

printf '%s\n' "step 3-3B4 similarity outputs:" \
  "  summary: $OUTPUT_DIR/summary.json" \
  "  module details: $OUTPUT_DIR/module_similarity.csv"
