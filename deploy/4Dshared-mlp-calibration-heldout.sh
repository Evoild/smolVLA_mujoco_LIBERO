#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$REPO_ROOT/scripts/common.sh"

POLICY_PATH="$REPO_ROOT/smolvla_libero"
OUTPUT_ROOT="$REPO_ROOT/runs/deploy/4/D-shared-mlp-calibration-heldout"
TASK_SUITE=libero_spatial
DEVICE=cuda
SEED=1000
BATCH_SIZE=1
TOKEN_LENGTH=48
INPUT_DTYPE=fp32
CALIBRATION_SAMPLES=512
HELDOUT_SAMPLES=50
SAMPLE_STRIDE=5
PERCENTILES="99.0,99.5,99.9,99.95,99.99,99.995,max"
DEFAULT_PERCENTILE=99.99
PYTHON="${PYTHON_BIN:-python3}"

usage() {
  printf '%s\n' "Usage: $0 [options]" \
    "  --policy-path PATH             default: $POLICY_PATH" \
    "  --output-root DIR             default: $OUTPUT_ROOT" \
    "  --tasks NAME                  default: $TASK_SUITE" \
    "  --device DEVICE               default: $DEVICE" \
    "  --seed N                      default: $SEED" \
    "  --batch-size N                default: $BATCH_SIZE" \
    "  --token-length N              default: $TOKEN_LENGTH" \
    "  --input-dtype fp32|bf16       default: $INPUT_DTYPE" \
    "  --calibration-samples N       default: $CALIBRATION_SAMPLES" \
    "  --heldout-samples N           default: $HELDOUT_SAMPLES" \
    "  --sample-stride N             default: $SAMPLE_STRIDE" \
    "  --percentiles LIST            default: $PERCENTILES" \
    "  --default-percentile P        default: $DEFAULT_PERCENTILE" \
    "  PYTHON_BIN=/path/python       optional Python with torch/lerobot dependencies"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --policy-path) require_value "$@"; POLICY_PATH="$2"; shift 2 ;;
    --output-root) require_value "$@"; OUTPUT_ROOT="$2"; shift 2 ;;
    --tasks) require_value "$@"; TASK_SUITE="$2"; shift 2 ;;
    --device) require_value "$@"; DEVICE="$2"; shift 2 ;;
    --seed) require_value "$@"; SEED="$2"; shift 2 ;;
    --batch-size) require_value "$@"; BATCH_SIZE="$2"; shift 2 ;;
    --token-length) require_value "$@"; TOKEN_LENGTH="$2"; shift 2 ;;
    --input-dtype) require_value "$@"; INPUT_DTYPE="$2"; shift 2 ;;
    --calibration-samples) require_value "$@"; CALIBRATION_SAMPLES="$2"; shift 2 ;;
    --heldout-samples) require_value "$@"; HELDOUT_SAMPLES="$2"; shift 2 ;;
    --sample-stride) require_value "$@"; SAMPLE_STRIDE="$2"; shift 2 ;;
    --percentiles) require_value "$@"; PERCENTILES="$2"; shift 2 ;;
    --default-percentile) require_value "$@"; DEFAULT_PERCENTILE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

case "$INPUT_DTYPE" in fp32|bf16) ;; *) die "--input-dtype must be fp32 or bf16" ;; esac

mkdir -p "$OUTPUT_ROOT"

export LEROBOT_SRC="${LEROBOT_SRC:-$REPO_ROOT/lerobot/src}"
export PYTHONPATH="$SCRIPT_DIR:$LEROBOT_SRC${PYTHONPATH:+:$PYTHONPATH}"
export NUMBA_DISABLE_JIT="${NUMBA_DISABLE_JIT:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-smolvla}"

"$PYTHON" "$SCRIPT_DIR/recalibrate_mlp_gate_up_down_heldout.py" \
  --policy-path "$POLICY_PATH" \
  --output-dir "$OUTPUT_ROOT" \
  --device "$DEVICE" \
  --tasks "$TASK_SUITE" \
  --seed "$SEED" \
  --batch-size "$BATCH_SIZE" \
  --token-length "$TOKEN_LENGTH" \
  --input-dtype "$INPUT_DTYPE" \
  --calibration-samples "$CALIBRATION_SAMPLES" \
  --heldout-samples "$HELDOUT_SAMPLES" \
  --sample-stride "$SAMPLE_STRIDE" \
  --percentiles "$PERCENTILES" \
  --default-percentile "$DEFAULT_PERCENTILE"

printf '%s\n' "step 4D shared gate/up/down calibration held-out outputs:" \
  "  output root: $OUTPUT_ROOT" \
  "  summary: $OUTPUT_ROOT/summary.json" \
  "  percentile sweep: $OUTPUT_ROOT/percentile_sweep.csv" \
  "  best scales: $OUTPUT_ROOT/activation_scales_best.json" \
  "  plot: $OUTPUT_ROOT/percentile_sweep_relative_l2.png"
