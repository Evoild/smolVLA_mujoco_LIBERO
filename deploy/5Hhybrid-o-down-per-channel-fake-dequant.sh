#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$REPO_ROOT/scripts/common.sh"

POLICY_PATH="$REPO_ROOT/smolvla_libero"
OUTPUT_ROOT="$REPO_ROOT/runs/deploy/5/H-bf16-int8-w8a8/hybrid_o_down_per_channel"
SCALAR_SCALES="$REPO_ROOT/runs/deploy/5/H-bf16-int8-w8a8/percentile_sweep/p99_999/activation_scales_p99.999.json"
CHANNEL_SCALES="$REPO_ROOT/runs/deploy/5/H-bf16-int8-w8a8/per_channel_fake_dequant/activation_channel_scales_512_p99.995.json"
TASK_SUITE=libero_spatial
DEVICE=cuda
SEED=1000
BATCH_SIZE=1
TOKEN_LENGTH=48
COMPARE_SAMPLES=50
SAMPLE_STRIDE=5
PYTHON="${PYTHON_BIN:-python3}"

QUANT_MODULE_REGEX="^model\\.vlm_with_expert\\.vlm\\.model\\.text_model\\.layers\\.[0-9]+\\.(mlp\\.(gate_proj|up_proj|down_proj)|self_attn\\.(q_proj|k_proj|v_proj|o_proj))$"
HYBRID_CHANNEL_REGEX="^model\\.vlm_with_expert\\.vlm\\.model\\.text_model\\.layers\\.[0-9]+\\.(mlp\\.down_proj|self_attn\\.o_proj)$"

usage() {
  printf '%s\n' "Usage: $0 [options]" \
    "  --policy-path PATH       default: $POLICY_PATH" \
    "  --output-root DIR       default: $OUTPUT_ROOT" \
    "  --scalar-scales PATH    default: $SCALAR_SCALES" \
    "  --channel-scales PATH   default: $CHANNEL_SCALES" \
    "  --tasks NAME            default: $TASK_SUITE" \
    "  --device DEVICE         default: $DEVICE" \
    "  --seed N                default: $SEED" \
    "  --batch-size N          default: $BATCH_SIZE" \
    "  --token-length N        default: $TOKEN_LENGTH" \
    "  --compare-samples N     default: $COMPARE_SAMPLES" \
    "  --sample-stride N       default: $SAMPLE_STRIDE" \
    "  PYTHON_BIN=/path/python optional Python with torch/lerobot dependencies"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --policy-path) require_value "$@"; POLICY_PATH="$2"; shift 2 ;;
    --output-root) require_value "$@"; OUTPUT_ROOT="$2"; shift 2 ;;
    --scalar-scales) require_value "$@"; SCALAR_SCALES="$2"; shift 2 ;;
    --channel-scales) require_value "$@"; CHANNEL_SCALES="$2"; shift 2 ;;
    --tasks) require_value "$@"; TASK_SUITE="$2"; shift 2 ;;
    --device) require_value "$@"; DEVICE="$2"; shift 2 ;;
    --seed) require_value "$@"; SEED="$2"; shift 2 ;;
    --batch-size) require_value "$@"; BATCH_SIZE="$2"; shift 2 ;;
    --token-length) require_value "$@"; TOKEN_LENGTH="$2"; shift 2 ;;
    --compare-samples) require_value "$@"; COMPARE_SAMPLES="$2"; shift 2 ;;
    --sample-stride) require_value "$@"; SAMPLE_STRIDE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ -f "$SCALAR_SCALES" ]] || die "missing scalar scales: $SCALAR_SCALES"
[[ -f "$CHANNEL_SCALES" ]] || die "missing channel scales: $CHANNEL_SCALES"

mkdir -p "$OUTPUT_ROOT"
export LEROBOT_SRC="${LEROBOT_SRC:-$REPO_ROOT/lerobot/src}"
export PYTHONPATH="$SCRIPT_DIR:$LEROBOT_SRC${PYTHONPATH:+:$PYTHONPATH}"

HYBRID_SCALES="$OUTPUT_ROOT/activation_scales_scalar_p99.999_o_down_per_channel_p99.995.json"

"$PYTHON" "$SCRIPT_DIR/make_hybrid_activation_scales.py" \
  --scalar-scales "$SCALAR_SCALES" \
  --channel-scales "$CHANNEL_SCALES" \
  --output "$HYBRID_SCALES" \
  --per-channel-module-regex "$HYBRID_CHANNEL_REGEX"

"$PYTHON" "$SCRIPT_DIR/diagnose_3_4_numeric_baseline.py" compare \
  --policy-path "$POLICY_PATH" \
  --device "$DEVICE" \
  --seed "$SEED" \
  --task "$TASK_SUITE" \
  --batch-size "$BATCH_SIZE" \
  --token-length "$TOKEN_LENGTH" \
  --model-dtype bf16 \
  --input-dtype bf16 \
  --input-source rollout \
  --compare-samples "$COMPARE_SAMPLES" \
  --sample-stride "$SAMPLE_STRIDE" \
  --max-parallel-tasks 1 \
  --quantize-module-regex "$QUANT_MODULE_REGEX" \
  --fake-quant-kind w8a8 \
  --fake-quant-activation-scale-mode calibrated \
  --fake-quant-activation-scales-json "$HYBRID_SCALES" \
  --output-dir "$OUTPUT_ROOT/fake_dequant_numeric"

printf '%s\n' "step 5H hybrid o/down per-channel fake-dequant outputs:" \
  "  hybrid scales: $HYBRID_SCALES" \
  "  numeric summary: $OUTPUT_ROOT/fake_dequant_numeric/numeric_baseline_summary.json" \
  "  aggregate rows: $OUTPUT_ROOT/fake_dequant_numeric/numeric_baseline_aggregate_rows.csv" \
  "  per-channel override regex: $HYBRID_CHANNEL_REGEX"
