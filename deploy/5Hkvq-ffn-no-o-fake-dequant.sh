#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$REPO_ROOT/scripts/common.sh"

POLICY_PATH="$REPO_ROOT/smolvla_libero"
OUTPUT_ROOT="$REPO_ROOT/runs/deploy/5/H-bf16-int8-w8a8/kvq_ffn_no_o"
SCALAR_SCALES="$REPO_ROOT/runs/deploy/5/H-bf16-int8-w8a8/percentile_sweep/p99_999/activation_scales_p99.999.json"
TASK_SUITE=libero_spatial
DEVICE=cuda
SEED=1000
BATCH_SIZE=1
TOKEN_LENGTH=48
COMPARE_SAMPLES=50
SAMPLE_STRIDE=5
INPUT_SOURCE=rollout
PYTHON="${PYTHON_BIN:-python3}"

HOOK_MODULE_REGEX="^model\\.vlm_with_expert\\.vlm\\.model\\.text_model\\.layers\\.([0-9]+)\\.(?:self_attn\\.(q_proj|k_proj|v_proj|o_proj)|mlp\\.(gate_proj|up_proj|down_proj))$"

# W8A8:
#   q/k/v: all layers
#   FFN gate/up/down: all layers
# BF16:
#   o_proj: all layers
QUANT_MODULE_REGEX="^model\\.vlm_with_expert\\.vlm\\.model\\.text_model\\.layers\\.[0-9]+\\.(self_attn\\.(q_proj|k_proj|v_proj)|mlp\\.(gate_proj|up_proj|down_proj))$"

usage() {
  printf '%s\n' "Usage: $0 [options]" \
    "  --policy-path PATH          default: $POLICY_PATH" \
    "  --output-root DIR          default: $OUTPUT_ROOT" \
    "  --scalar-scales PATH       default: $SCALAR_SCALES" \
    "  --tasks NAME               default: $TASK_SUITE" \
    "  --device DEVICE            default: $DEVICE" \
    "  --seed N                   default: $SEED" \
    "  --batch-size N             default: $BATCH_SIZE" \
    "  --token-length N           default: $TOKEN_LENGTH" \
    "  --input-source synthetic|rollout default: $INPUT_SOURCE" \
    "  --compare-samples N        default: $COMPARE_SAMPLES" \
    "  --sample-stride N          default: $SAMPLE_STRIDE" \
    "  PYTHON_BIN=/path/python    optional Python with torch/lerobot dependencies"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --policy-path) require_value "$@"; POLICY_PATH="$2"; shift 2 ;;
    --output-root) require_value "$@"; OUTPUT_ROOT="$2"; shift 2 ;;
    --scalar-scales) require_value "$@"; SCALAR_SCALES="$2"; shift 2 ;;
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
[[ -f "$SCALAR_SCALES" ]] || die "missing scalar scales: $SCALAR_SCALES"

mkdir -p "$OUTPUT_ROOT"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-smolvla}"
export LEROBOT_SRC="${LEROBOT_SRC:-$REPO_ROOT/lerobot/src}"
export PYTHONPATH="$SCRIPT_DIR:$LEROBOT_SRC${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON" "$SCRIPT_DIR/diagnose_3_4_numeric_baseline.py" compare \
  --policy-path "$POLICY_PATH" \
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
  --quantize-module-regex "$QUANT_MODULE_REGEX" \
  --fake-quant-kind w8a8 \
  --fake-quant-activation-scale-mode calibrated \
  --fake-quant-activation-scales-json "$SCALAR_SCALES" \
  --output-dir "$OUTPUT_ROOT/fake_dequant_numeric"

"$PYTHON" "$SCRIPT_DIR/compare_5h_text_vlm_layer_outputs.py" \
  --policy-path "$POLICY_PATH" \
  --activation-scales-json "$SCALAR_SCALES" \
  --output-dir "$OUTPUT_ROOT/layer_outputs" \
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
  --module-regex "$HOOK_MODULE_REGEX" \
  --quantize-module-regex "$QUANT_MODULE_REGEX"

printf '%s\n' "step 5H.6 KVQ+FFN no-O fake-dequant outputs:" \
  "  output root: $OUTPUT_ROOT" \
  "  numeric summary: $OUTPUT_ROOT/fake_dequant_numeric/numeric_baseline_summary.json" \
  "  layer summary: $OUTPUT_ROOT/layer_outputs/text_vlm_layer_output_l2_summary.json" \
  "  all curves: $OUTPUT_ROOT/layer_outputs/text_vlm_layer_output_relative_l2.png" \
  "  attention curves: $OUTPUT_ROOT/layer_outputs/attention_layer_output_relative_l2.png" \
  "  mlp curves: $OUTPUT_ROOT/layer_outputs/mlp_layer_output_relative_l2.png" \
  "  quantized module regex: $QUANT_MODULE_REGEX"
