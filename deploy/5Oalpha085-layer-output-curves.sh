#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$REPO_ROOT/scripts/common.sh"

POLICY_PATH="$REPO_ROOT/smolvla_libero"
SCALES_JSON="$REPO_ROOT/runs/deploy/5/O-smoothquant-alpha-sweep-high/alpha_0_85/smoothquant_alpha_0.85_activation_scales.json"
OUTPUT_DIR="$REPO_ROOT/runs/deploy/5/O-smoothquant-alpha-sweep-high/alpha_0_85/layer_outputs"
TASK_SUITE=libero_spatial
DEVICE=cuda
SEED=1000
BATCH_SIZE=1
TOKEN_LENGTH=48
INPUT_SOURCE=rollout
COMPARE_SAMPLES=50
SAMPLE_STRIDE=5
PYTHON="${PYTHON_BIN:-python3}"

QUANT_MODULE_REGEX="^model\\.vlm_with_expert\\.vlm\\.model\\.text_model\\.layers\\.[0-9]+\\.(mlp\\.(gate_proj|up_proj|down_proj)|self_attn\\.(q_proj|k_proj|v_proj|o_proj))$"

usage() {
  printf '%s\n' "Usage: $0 [options]" \
    "  --policy-path PATH          default: $POLICY_PATH" \
    "  --scales-json PATH          default: $SCALES_JSON" \
    "  --output-dir DIR           default: $OUTPUT_DIR" \
    "  --tasks NAME               default: $TASK_SUITE" \
    "  --device DEVICE            default: $DEVICE" \
    "  --seed N                   default: $SEED" \
    "  --batch-size N             default: $BATCH_SIZE" \
    "  --token-length N           default: $TOKEN_LENGTH" \
    "  --input-source synthetic|rollout default: $INPUT_SOURCE" \
    "  --compare-samples N        default: $COMPARE_SAMPLES" \
    "  --sample-stride N          default: $SAMPLE_STRIDE" \
    "  PYTHON_BIN=/path/python    optional Python with torch/matplotlib dependencies"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --policy-path) require_value "$@"; POLICY_PATH="$2"; shift 2 ;;
    --scales-json) require_value "$@"; SCALES_JSON="$2"; shift 2 ;;
    --output-dir) require_value "$@"; OUTPUT_DIR="$2"; shift 2 ;;
    --tasks) require_value "$@"; TASK_SUITE="$2"; shift 2 ;;
    --device) require_value "$@"; DEVICE="$2"; shift 2 ;;
    --seed) require_value "$@"; SEED="$2"; shift 2 ;;
    --batch-size) require_value "$@"; BATCH_SIZE="$2"; shift 2 ;;
    --token-length) require_value "$@"; TOKEN_LENGTH="$2"; shift 2 ;;
    --input-source) require_value "$@"; INPUT_SOURCE="$2"; shift 2 ;;
    --compare-samples) require_value "$@"; COMPARE_SAMPLES="$2"; shift 2 ;;
    --sample-stride) require_value "$@"; SAMPLE_STRIDE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

case "$INPUT_SOURCE" in synthetic|rollout) ;; *) die "--input-source must be synthetic or rollout" ;; esac
[[ -f "$SCALES_JSON" ]] || die "$SCALES_JSON not found"

mkdir -p "$OUTPUT_DIR"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-smolvla}"
export LEROBOT_SRC="${LEROBOT_SRC:-$REPO_ROOT/lerobot/src}"
export PYTHONPATH="$SCRIPT_DIR:$LEROBOT_SRC${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON" "$SCRIPT_DIR/compare_5h_text_vlm_layer_outputs.py" \
  --policy-path "$POLICY_PATH" \
  --activation-scales-json "$SCALES_JSON" \
  --output-dir "$OUTPUT_DIR" \
  --device "$DEVICE" \
  --seed "$SEED" \
  --task "$TASK_SUITE" \
  --batch-size "$BATCH_SIZE" \
  --token-length "$TOKEN_LENGTH" \
  --model-dtype bf16 \
  --input-dtype bf16 \
  --input-source "$INPUT_SOURCE" \
  --compare-samples "$COMPARE_SAMPLES" \
  --sample-stride "$SAMPLE_STRIDE" \
  --max-parallel-tasks 1 \
  --quantize-module-regex "$QUANT_MODULE_REGEX"

printf '%s\n' "step 5O SmoothQuant alpha=0.85 layer output curves:" \
  "  scales: $SCALES_JSON" \
  "  output dir: $OUTPUT_DIR" \
  "  summary: $OUTPUT_DIR/text_vlm_layer_output_l2_summary.json" \
  "  all curves: $OUTPUT_DIR/text_vlm_layer_output_relative_l2.png" \
  "  attention curves: $OUTPUT_DIR/attention_layer_output_relative_l2.png" \
  "  mlp curves: $OUTPUT_DIR/mlp_layer_output_relative_l2.png"
