#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$REPO_ROOT/scripts/common.sh"

POLICY_PATH="$REPO_ROOT/smolvla_libero"
OUTPUT_ROOT="$REPO_ROOT/runs/deploy/5/H-bf16-int8-w8a8/layer_outputs"
DEVICE=cuda
SEED=1000
TASK_SUITE=libero_spatial
BATCH_SIZE=1
TOKEN_LENGTH=48
INPUT_SOURCE=synthetic
COMPARE_SAMPLES=1
SAMPLE_STRIDE=5
PYTHON="${PYTHON_BIN:-python3}"

SCALAR_SCALES="$REPO_ROOT/runs/deploy/5/H-bf16-int8-w8a8/percentile_sweep/p99_999/activation_scales_p99.999.json"
CHANNEL_SCALES="$REPO_ROOT/runs/deploy/5/H-bf16-int8-w8a8/per_channel_fake_dequant/activation_channel_scales_512_p99.995.json"
HYBRID_SCALES="$REPO_ROOT/runs/deploy/5/H-bf16-int8-w8a8/hybrid_o_down_per_channel/activation_scales_scalar_p99.999_o_down_per_channel_p99.995.json"

usage() {
  printf '%s\n' "Usage: $0 [options]" \
    "  --policy-path PATH          default: $POLICY_PATH" \
    "  --output-root DIR          default: $OUTPUT_ROOT" \
    "  --scalar-scales PATH       default: $SCALAR_SCALES" \
    "  --channel-scales PATH      default: $CHANNEL_SCALES" \
    "  --hybrid-scales PATH       default: $HYBRID_SCALES" \
    "  --device DEVICE            default: $DEVICE" \
    "  --seed N                   default: $SEED" \
    "  --tasks NAME               default: $TASK_SUITE" \
    "  --input-source synthetic|rollout default: $INPUT_SOURCE" \
    "  --compare-samples N        default: $COMPARE_SAMPLES" \
    "  --sample-stride N          default: $SAMPLE_STRIDE" \
    "  PYTHON_BIN=/path/python    optional Python with torch/matplotlib dependencies"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --policy-path) require_value "$@"; POLICY_PATH="$2"; shift 2 ;;
    --output-root) require_value "$@"; OUTPUT_ROOT="$2"; shift 2 ;;
    --scalar-scales) require_value "$@"; SCALAR_SCALES="$2"; shift 2 ;;
    --channel-scales) require_value "$@"; CHANNEL_SCALES="$2"; shift 2 ;;
    --hybrid-scales) require_value "$@"; HYBRID_SCALES="$2"; shift 2 ;;
    --device) require_value "$@"; DEVICE="$2"; shift 2 ;;
    --seed) require_value "$@"; SEED="$2"; shift 2 ;;
    --tasks) require_value "$@"; TASK_SUITE="$2"; shift 2 ;;
    --input-source) require_value "$@"; INPUT_SOURCE="$2"; shift 2 ;;
    --compare-samples) require_value "$@"; COMPARE_SAMPLES="$2"; shift 2 ;;
    --sample-stride) require_value "$@"; SAMPLE_STRIDE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

case "$INPUT_SOURCE" in synthetic|rollout) ;; *) die "--input-source must be synthetic or rollout" ;; esac
[[ -f "$SCALAR_SCALES" ]] || die "$SCALAR_SCALES not found"
[[ -f "$CHANNEL_SCALES" ]] || die "$CHANNEL_SCALES not found"

mkdir -p "$OUTPUT_ROOT"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-smolvla}"
export LEROBOT_SRC="${LEROBOT_SRC:-$REPO_ROOT/lerobot/src}"
export PYTHONPATH="$SCRIPT_DIR:$LEROBOT_SRC${PYTHONPATH:+:$PYTHONPATH}"

declare -a names=("per_tensor_p99_999" "per_channel_p99_995")
declare -a scales_list=("$SCALAR_SCALES" "$CHANNEL_SCALES")
if [[ -f "$HYBRID_SCALES" ]]; then
  names+=("hybrid_o_down_per_channel")
  scales_list+=("$HYBRID_SCALES")
fi

for i in "${!names[@]}"; do
  name="${names[$i]}"
  scales="${scales_list[$i]}"
  "$PYTHON" "$SCRIPT_DIR/compare_5h_text_vlm_layer_outputs.py" \
    --policy-path "$POLICY_PATH" \
    --activation-scales-json "$scales" \
    --output-dir "$OUTPUT_ROOT/$name" \
    --device "$DEVICE" \
    --seed "$SEED" \
    --task "$TASK_SUITE" \
    --batch-size "$BATCH_SIZE" \
    --token-length "$TOKEN_LENGTH" \
    --model-dtype bf16 \
    --input-dtype bf16 \
    --input-source "$INPUT_SOURCE" \
    --compare-samples "$COMPARE_SAMPLES" \
    --sample-stride "$SAMPLE_STRIDE"
done

printf '%s\n' "step 5H layer output curves:" \
  "  output root: $OUTPUT_ROOT" \
  "  scalar summary: $OUTPUT_ROOT/per_tensor_p99_999/text_vlm_layer_output_l2_summary.json" \
  "  channel summary: $OUTPUT_ROOT/per_channel_p99_995/text_vlm_layer_output_l2_summary.json" \
  "  hybrid summary: $OUTPUT_ROOT/hybrid_o_down_per_channel/text_vlm_layer_output_l2_summary.json"
