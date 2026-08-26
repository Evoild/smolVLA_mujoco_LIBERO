#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$REPO_ROOT/scripts/common.sh"

POLICY_PATH="$REPO_ROOT/smolvla_libero"
OUTPUT_ROOT="$REPO_ROOT/runs/deploy/4/A-down-proj-calibration-test"
TASK_SUITE=libero_spatial
DEVICE=cuda
SEED=1000
CALIBRATION_SAMPLES=32
BATCH_SIZE=1
TOKEN_LENGTH=48
INPUT_DTYPE=fp32
PERCENTILE=99.99
RANGE_PERCENTILES="99.9,99.95,99.99,99.995,99.999,max"
RUN_CALIBRATION=true
PYTHON="${PYTHON_BIN:-python3}"

DOWN_MODULE_REGEX="^model\\.vlm_with_expert\\.vlm\\.model\\.text_model\\.layers\\.[0-9]+\\.mlp\\.down_proj$"

usage() {
  printf '%s\n' "Usage: $0 [options]" \
    "  --policy-path PATH              default: $POLICY_PATH" \
    "  --output-root DIR              default: $OUTPUT_ROOT" \
    "  --tasks NAME                   default: $TASK_SUITE" \
    "  --device DEVICE                default: $DEVICE" \
    "  --seed N                       default: $SEED" \
    "  --calibration-samples N        default: $CALIBRATION_SAMPLES" \
    "  --batch-size N                 default: $BATCH_SIZE" \
    "  --token-length N               default: $TOKEN_LENGTH" \
    "  --input-dtype fp32|bf16        default: $INPUT_DTYPE" \
    "  --percentile P                 default: $PERCENTILE" \
    "  --range-percentiles LIST       default: $RANGE_PERCENTILES" \
    "  --run-calibration true|false   default: $RUN_CALIBRATION" \
    "  PYTHON_BIN=/path/python        optional Python with torch/lerobot dependencies"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --policy-path) require_value "$@"; POLICY_PATH="$2"; shift 2 ;;
    --output-root) require_value "$@"; OUTPUT_ROOT="$2"; shift 2 ;;
    --tasks) require_value "$@"; TASK_SUITE="$2"; shift 2 ;;
    --device) require_value "$@"; DEVICE="$2"; shift 2 ;;
    --seed) require_value "$@"; SEED="$2"; shift 2 ;;
    --calibration-samples) require_value "$@"; CALIBRATION_SAMPLES="$2"; shift 2 ;;
    --batch-size) require_value "$@"; BATCH_SIZE="$2"; shift 2 ;;
    --token-length) require_value "$@"; TOKEN_LENGTH="$2"; shift 2 ;;
    --input-dtype) require_value "$@"; INPUT_DTYPE="$2"; shift 2 ;;
    --percentile) require_value "$@"; PERCENTILE="$2"; shift 2 ;;
    --range-percentiles) require_value "$@"; RANGE_PERCENTILES="$2"; shift 2 ;;
    --run-calibration) require_value "$@"; RUN_CALIBRATION="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

case "$INPUT_DTYPE" in fp32|bf16) ;; *) die "--input-dtype must be fp32 or bf16" ;; esac
case "$RUN_CALIBRATION" in true|false) ;; *) die "--run-calibration must be true or false" ;; esac

mkdir -p "$OUTPUT_ROOT"
SCALES_JSON="$OUTPUT_ROOT/down_activation_channel_scales_p${PERCENTILE}.json"
DIAG_DIR="$OUTPUT_ROOT/down_proj_diagnostics"

export LEROBOT_SRC="${LEROBOT_SRC:-$REPO_ROOT/lerobot/src}"
export PYTHONPATH="$SCRIPT_DIR:$LEROBOT_SRC${PYTHONPATH:+:$PYTHONPATH}"
export NUMBA_DISABLE_JIT="${NUMBA_DISABLE_JIT:-1}"

if [[ "$RUN_CALIBRATION" == "true" ]]; then
  "$PYTHON" "$SCRIPT_DIR/calibrate_smolvla_activation_channel_scales.py" \
    --policy-path "$POLICY_PATH" \
    --output "$SCALES_JSON" \
    --device "$DEVICE" \
    --tasks "$TASK_SUITE" \
    --seed "$SEED" \
    --samples "$CALIBRATION_SAMPLES" \
    --percentile "$PERCENTILE" \
    --batch-size "$BATCH_SIZE" \
    --token-length "$TOKEN_LENGTH" \
    --include-module-regex "$DOWN_MODULE_REGEX"
else
  [[ -f "$SCALES_JSON" ]] || die "$SCALES_JSON not found; enable --run-calibration true"
fi

"$PYTHON" "$SCRIPT_DIR/diagnose_down_proj_w8a8.py" \
  --policy-path "$POLICY_PATH" \
  --device "$DEVICE" \
  --seed "$SEED" \
  --task "$TASK_SUITE step 4A down_proj diagnosis" \
  --batch-size "$BATCH_SIZE" \
  --token-length "$TOKEN_LENGTH" \
  --input-dtype "$INPUT_DTYPE" \
  --activation-scales-json "$SCALES_JSON" \
  --down-module-regex "$DOWN_MODULE_REGEX" \
  --range-percentiles "$RANGE_PERCENTILES" \
  --output-dir "$DIAG_DIR"

printf '%s\n' "step 4A down_proj W8A8 calibration diagnostics outputs:" \
  "  output root: $OUTPUT_ROOT" \
  "  down activation scales: $SCALES_JSON" \
  "  diagnostics: $DIAG_DIR" \
  "  summary: $DIAG_DIR/summary.json"
