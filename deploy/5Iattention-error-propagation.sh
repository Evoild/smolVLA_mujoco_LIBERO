#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$REPO_ROOT/scripts/common.sh"

POLICY_PATH="$REPO_ROOT/smolvla_libero"
OUTPUT_ROOT="$REPO_ROOT/runs/deploy/5/I-attention-error-propagation"
SCALES_JSON="$REPO_ROOT/runs/deploy/5/H-bf16-int8-w8a8/percentile_sweep/p99_999/activation_scales_p99.999.json"
TASK_SUITE=libero_spatial
DEVICE=cuda
SEED=1000
BATCH_SIZE=1
TOKEN_LENGTH=48
COMPARE_SAMPLES=20
SAMPLE_STRIDE=5
INPUT_SOURCE=rollout
LAYERS="10,18,24,26,27,28,29,30,31"
PYTHON="${PYTHON_BIN:-python3}"

usage() {
  printf '%s\n' "Usage: $0 [options]" \
    "  --policy-path PATH          default: $POLICY_PATH" \
    "  --output-root DIR          default: $OUTPUT_ROOT" \
    "  --scales-json PATH         default: $SCALES_JSON" \
    "  --tasks NAME               default: $TASK_SUITE" \
    "  --device DEVICE            default: $DEVICE" \
    "  --seed N                   default: $SEED" \
    "  --batch-size N             default: $BATCH_SIZE" \
    "  --token-length N           default: $TOKEN_LENGTH" \
    "  --input-source synthetic|rollout default: $INPUT_SOURCE" \
    "  --compare-samples N        default: $COMPARE_SAMPLES" \
    "  --sample-stride N          default: $SAMPLE_STRIDE" \
    "  --layers CSV               default: $LAYERS" \
    "  PYTHON_BIN=/path/python    optional Python with torch/lerobot dependencies"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --policy-path) require_value "$@"; POLICY_PATH="$2"; shift 2 ;;
    --output-root) require_value "$@"; OUTPUT_ROOT="$2"; shift 2 ;;
    --scales-json) require_value "$@"; SCALES_JSON="$2"; shift 2 ;;
    --tasks) require_value "$@"; TASK_SUITE="$2"; shift 2 ;;
    --device) require_value "$@"; DEVICE="$2"; shift 2 ;;
    --seed) require_value "$@"; SEED="$2"; shift 2 ;;
    --batch-size) require_value "$@"; BATCH_SIZE="$2"; shift 2 ;;
    --token-length) require_value "$@"; TOKEN_LENGTH="$2"; shift 2 ;;
    --input-source) require_value "$@"; INPUT_SOURCE="$2"; shift 2 ;;
    --compare-samples) require_value "$@"; COMPARE_SAMPLES="$2"; shift 2 ;;
    --sample-stride) require_value "$@"; SAMPLE_STRIDE="$2"; shift 2 ;;
    --layers) require_value "$@"; LAYERS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

case "$INPUT_SOURCE" in synthetic|rollout) ;; *) die "--input-source must be synthetic or rollout" ;; esac
[[ -f "$SCALES_JSON" ]] || die "missing activation scales: $SCALES_JSON"

mkdir -p "$OUTPUT_ROOT"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-smolvla}"
export LEROBOT_SRC="${LEROBOT_SRC:-$REPO_ROOT/lerobot/src}"
export PYTHONPATH="$SCRIPT_DIR:$LEROBOT_SRC${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON" "$SCRIPT_DIR/diagnose_5i_attention_error_propagation.py" \
  --policy-path "$POLICY_PATH" \
  --activation-scales-json "$SCALES_JSON" \
  --output-dir "$OUTPUT_ROOT" \
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
  --layers "$LAYERS"

printf '%s\n' "step 5I attention error propagation outputs:" \
  "  output root: $OUTPUT_ROOT" \
  "  summary: $OUTPUT_ROOT/attention_error_propagation_summary.json" \
  "  stage csv: $OUTPUT_ROOT/attention_error_propagation.csv" \
  "  sample rows csv: $OUTPUT_ROOT/attention_error_propagation_rows.csv" \
  "  amplification csv: $OUTPUT_ROOT/attention_error_amplification.csv" \
  "  mean plot: $OUTPUT_ROOT/attention_error_propagation.png"
