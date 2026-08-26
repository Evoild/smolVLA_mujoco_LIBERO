#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$REPO_ROOT/scripts/common.sh"

POLICY_PATH="$REPO_ROOT/smolvla_libero"
OUTPUT_ROOT="$REPO_ROOT/runs/deploy/4/C-mlp-w8a8-best-static-fake-quant"
GATE_UP_SCALES="$REPO_ROOT/runs/deploy/3-7/B-formal-quant-deploy/gate_up_activation_channel_scales.json"
DOWN_SCALES="$REPO_ROOT/runs/deploy/4/B-down-recalibration-heldout/down_recalibration/activation_scales_512.json"
MISMATCH_CSV="$REPO_ROOT/runs/deploy/4/B-down-recalibration-heldout/down_recalibration/scale_mismatch_channels.csv"
DEVICE=cuda
SEED=1000
TASK_SUITE=libero_spatial
BATCH_SIZE=1
TOKEN_LENGTH=48
INPUT_DTYPE=fp32
TARGETED_FACTOR=1.1
PYTHON="${PYTHON_BIN:-python3}"

MLP_REGEX="^model\\.vlm_with_expert\\.vlm\\.model\\.text_model\\.layers\\.[0-9]+\\.mlp\\.(gate_proj|up_proj|down_proj)$"
LAYER_MLP_REGEX="^model\\.vlm_with_expert\\.vlm\\.model\\.text_model\\.layers\\.([0-9]+)\\.mlp\\.(gate_proj|up_proj|down_proj)$"

usage() {
  printf '%s\n' "Usage: $0 [options]" \
    "  --policy-path PATH        default: $POLICY_PATH" \
    "  --output-root DIR        default: $OUTPUT_ROOT" \
    "  --gate-up-scales PATH    default: $GATE_UP_SCALES" \
    "  --down-scales PATH       default: $DOWN_SCALES" \
    "  --mismatch-csv PATH      default: $MISMATCH_CSV" \
    "  --device DEVICE          default: $DEVICE" \
    "  --seed N                 default: $SEED" \
    "  --tasks NAME             default: $TASK_SUITE" \
    "  --batch-size N           default: $BATCH_SIZE" \
    "  --token-length N         default: $TOKEN_LENGTH" \
    "  --input-dtype fp32|bf16  default: $INPUT_DTYPE" \
    "  --targeted-factor F      default: $TARGETED_FACTOR" \
    "  PYTHON_BIN=/path/python  optional Python with torch/lerobot dependencies"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --policy-path) require_value "$@"; POLICY_PATH="$2"; shift 2 ;;
    --output-root) require_value "$@"; OUTPUT_ROOT="$2"; shift 2 ;;
    --gate-up-scales) require_value "$@"; GATE_UP_SCALES="$2"; shift 2 ;;
    --down-scales) require_value "$@"; DOWN_SCALES="$2"; shift 2 ;;
    --mismatch-csv) require_value "$@"; MISMATCH_CSV="$2"; shift 2 ;;
    --device) require_value "$@"; DEVICE="$2"; shift 2 ;;
    --seed) require_value "$@"; SEED="$2"; shift 2 ;;
    --tasks) require_value "$@"; TASK_SUITE="$2"; shift 2 ;;
    --batch-size) require_value "$@"; BATCH_SIZE="$2"; shift 2 ;;
    --token-length) require_value "$@"; TOKEN_LENGTH="$2"; shift 2 ;;
    --input-dtype) require_value "$@"; INPUT_DTYPE="$2"; shift 2 ;;
    --targeted-factor) require_value "$@"; TARGETED_FACTOR="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

case "$INPUT_DTYPE" in fp32|bf16) ;; *) die "--input-dtype must be fp32 or bf16" ;; esac

mkdir -p "$OUTPUT_ROOT"
COMBINED_SCALES="$OUTPUT_ROOT/gate_up_down_activation_scales_best_static.json"

export LEROBOT_SRC="${LEROBOT_SRC:-$REPO_ROOT/lerobot/src}"
export PYTHONPATH="$SCRIPT_DIR:$LEROBOT_SRC${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON" "$SCRIPT_DIR/build_4c_mlp_w8a8_scales.py" \
  --gate-up-scales "$GATE_UP_SCALES" \
  --down-scales "$DOWN_SCALES" \
  --mismatch-csv "$MISMATCH_CSV" \
  --targeted-factor "$TARGETED_FACTOR" \
  --output "$COMBINED_SCALES"

"$PYTHON" "$SCRIPT_DIR/diagnose_3_4_numeric_baseline.py" compare \
  --policy-path "$POLICY_PATH" \
  --device "$DEVICE" \
  --seed "$SEED" \
  --task "$TASK_SUITE step 4C gate/up/down W8A8 best static fake quant" \
  --batch-size "$BATCH_SIZE" \
  --token-length "$TOKEN_LENGTH" \
  --input-dtype "$INPUT_DTYPE" \
  --quantize-module-regex "$MLP_REGEX" \
  --fake-quant-kind w8a8 \
  --fake-quant-activation-scale-mode calibrated \
  --fake-quant-activation-scales-json "$COMBINED_SCALES" \
  --output-dir "$OUTPUT_ROOT/numeric_baseline"

"$PYTHON" "$SCRIPT_DIR/compare_4c_mlp_layer_outputs.py" \
  --policy-path "$POLICY_PATH" \
  --device "$DEVICE" \
  --seed "$SEED" \
  --task "$TASK_SUITE step 4C gate/up/down W8A8 best static fake quant" \
  --batch-size "$BATCH_SIZE" \
  --token-length "$TOKEN_LENGTH" \
  --input-dtype "$INPUT_DTYPE" \
  --activation-scales-json "$COMBINED_SCALES" \
  --module-regex "$LAYER_MLP_REGEX" \
  --output-dir "$OUTPUT_ROOT/mlp_layer_outputs"

printf '%s\n' "step 4C VLM MLP gate/up/down W8A8 best static fake quant outputs:" \
  "  combined scales: $COMBINED_SCALES" \
  "  numeric baseline: $OUTPUT_ROOT/numeric_baseline" \
  "  summary: $OUTPUT_ROOT/numeric_baseline/numeric_baseline_summary.json" \
  "  layer output compare: $OUTPUT_ROOT/mlp_layer_outputs" \
  "  layer output plot: $OUTPUT_ROOT/mlp_layer_outputs/mlp_layer_output_relative_l2.png"
