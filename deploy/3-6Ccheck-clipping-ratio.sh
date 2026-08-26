#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$REPO_ROOT/scripts/common.sh"

POLICY_PATH="$REPO_ROOT/smolvla_libero"
SCALES_JSON="$REPO_ROOT/runs/deploy/3-3B4/activation_scales.json"
OUTPUT_ROOT="$REPO_ROOT/runs/deploy/3-6/C-mlp-only-clipping-ratio"
DEVICE=cuda
SEED=1000
NUM_SEEDS=1
BATCH_SIZE=1
TOKEN_LENGTH=48
INPUT_DTYPE=fp32
MODULE_REGEX="^model\\.vlm_with_expert\\.vlm\\.model\\.text_model\\.layers\\.[0-9]+\\.mlp\\.(gate_proj|up_proj|down_proj)$"
PYTHON="${PYTHON_BIN:-python3}"

usage() {
  printf '%s\n' "Usage: $0 [options]" \
    "  --policy-path PATH       default: $POLICY_PATH" \
    "  --scales-json PATH      default: $SCALES_JSON" \
    "  --output-root DIR       default: $OUTPUT_ROOT" \
    "  --device DEVICE         default: $DEVICE" \
    "  --seed N                default: $SEED" \
    "  --num-seeds N           default: $NUM_SEEDS" \
    "  --batch-size N          default: $BATCH_SIZE" \
    "  --token-length N        default: $TOKEN_LENGTH" \
    "  --input-dtype fp32|bf16 default: $INPUT_DTYPE" \
    "  --module-regex RE       default: $MODULE_REGEX" \
    "  PYTHON_BIN=/path/python optional Python with torch/lerobot dependencies"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --policy-path) require_value "$@"; POLICY_PATH="$2"; shift 2 ;;
    --scales-json) require_value "$@"; SCALES_JSON="$2"; shift 2 ;;
    --output-root) require_value "$@"; OUTPUT_ROOT="$2"; shift 2 ;;
    --device) require_value "$@"; DEVICE="$2"; shift 2 ;;
    --seed) require_value "$@"; SEED="$2"; shift 2 ;;
    --num-seeds) require_value "$@"; NUM_SEEDS="$2"; shift 2 ;;
    --batch-size) require_value "$@"; BATCH_SIZE="$2"; shift 2 ;;
    --token-length) require_value "$@"; TOKEN_LENGTH="$2"; shift 2 ;;
    --input-dtype) require_value "$@"; INPUT_DTYPE="$2"; shift 2 ;;
    --module-regex) require_value "$@"; MODULE_REGEX="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

case "$INPUT_DTYPE" in fp32|bf16) ;; *) die "--input-dtype must be fp32 or bf16" ;; esac

mkdir -p "$OUTPUT_ROOT"

export LEROBOT_SRC="${LEROBOT_SRC:-$REPO_ROOT/lerobot/src}"
export PYTHONPATH="$SCRIPT_DIR:$LEROBOT_SRC${PYTHONPATH:+:$PYTHONPATH}"

[[ -f "$SCALES_JSON" ]] || die "$SCALES_JSON not found; run deploy/3-3B4export.sh first or pass --scales-json"

"$PYTHON" "$SCRIPT_DIR/check_static_scale_clipping.py" \
  --policy-path "$POLICY_PATH" \
  --activation-scales-json "$SCALES_JSON" \
  --output-dir "$OUTPUT_ROOT" \
  --device "$DEVICE" \
  --seed "$SEED" \
  --num-seeds "$NUM_SEEDS" \
  --batch-size "$BATCH_SIZE" \
  --token-length "$TOKEN_LENGTH" \
  --input-dtype "$INPUT_DTYPE" \
  --module-regex "$MODULE_REGEX"

printf '%s\n' "step 3-6C clipping ratio outputs:" \
  "  clipping summary: $OUTPUT_ROOT/clipping_summary.json" \
  "  clipping rows: $OUTPUT_ROOT/clipping_rows.csv" \
  "  module regex: $MODULE_REGEX"
