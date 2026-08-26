#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$REPO_ROOT/scripts/common.sh"

POLICY_PATH="$REPO_ROOT/smolvla_libero"
OUTPUT_ROOT="$REPO_ROOT/runs/deploy/5/L-kvqo-gate-up-w8a8-down-w8a16"
TASK_SUITE=libero_spatial
DEVICE=cuda
SEED=1000
BATCH_SIZE=1
TOKEN_LENGTH=48
CALIBRATION_SAMPLES=512
CALIBRATION_PERCENTILE=99.999
COMPARE_SAMPLES=50
SAMPLE_STRIDE=5
INPUT_SOURCE=rollout
RUN_CALIBRATION=false
PYTHON="${PYTHON_BIN:-python3}"

# W8A8:
#   q/k/v/o: all Text VLM layers
#   FFN gate/up: all Text VLM layers
# W8A16:
#   FFN down: all Text VLM layers
W8A8_MODULE_REGEX="^model\\.vlm_with_expert\\.vlm\\.model\\.text_model\\.layers\\.[0-9]+\\.(self_attn\\.(q_proj|k_proj|v_proj|o_proj)|mlp\\.(gate_proj|up_proj))$"
DOWN_W8A16_REGEX="^model\\.vlm_with_expert\\.vlm\\.model\\.text_model\\.layers\\.[0-9]+\\.mlp\\.down_proj$"

usage() {
  printf '%s\n' "Usage: $0 [options]" \
    "  --policy-path PATH              default: $POLICY_PATH" \
    "  --output-root DIR              default: $OUTPUT_ROOT" \
    "  --tasks NAME                   default: $TASK_SUITE" \
    "  --device DEVICE                default: $DEVICE" \
    "  --seed N                       default: $SEED" \
    "  --batch-size N                 default: $BATCH_SIZE" \
    "  --token-length N               default: $TOKEN_LENGTH" \
    "  --calibration-samples N        default: $CALIBRATION_SAMPLES" \
    "  --calibration-percentile P     default: $CALIBRATION_PERCENTILE" \
    "  --input-source synthetic|rollout default: $INPUT_SOURCE" \
    "  --compare-samples N            default: $COMPARE_SAMPLES" \
    "  --sample-stride N              default: $SAMPLE_STRIDE" \
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
    --batch-size) require_value "$@"; BATCH_SIZE="$2"; shift 2 ;;
    --token-length) require_value "$@"; TOKEN_LENGTH="$2"; shift 2 ;;
    --calibration-samples) require_value "$@"; CALIBRATION_SAMPLES="$2"; shift 2 ;;
    --calibration-percentile) require_value "$@"; CALIBRATION_PERCENTILE="$2"; shift 2 ;;
    --input-source) require_value "$@"; INPUT_SOURCE="$2"; shift 2 ;;
    --compare-samples) require_value "$@"; COMPARE_SAMPLES="$2"; shift 2 ;;
    --sample-stride) require_value "$@"; SAMPLE_STRIDE="$2"; shift 2 ;;
    --run-calibration) require_value "$@"; RUN_CALIBRATION="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

case "$INPUT_SOURCE" in synthetic|rollout) ;; *) die "--input-source must be synthetic or rollout" ;; esac
case "$RUN_CALIBRATION" in true|false) ;; *) die "--run-calibration must be true or false" ;; esac

mkdir -p "$OUTPUT_ROOT"
export NUMBA_DISABLE_JIT="${NUMBA_DISABLE_JIT:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-smolvla}"
export LEROBOT_SRC="${LEROBOT_SRC:-$REPO_ROOT/lerobot/src}"
export PYTHONPATH="$SCRIPT_DIR:$LEROBOT_SRC${PYTHONPATH:+:$PYTHONPATH}"

SCALES_JSON="$OUTPUT_ROOT/activation_scales_${CALIBRATION_SAMPLES}_p${CALIBRATION_PERCENTILE}_kvqo_gate_up.json"

if [[ "$RUN_CALIBRATION" == "true" ]]; then
  "$PYTHON" "$SCRIPT_DIR/calibrate_smolvla_activation_scales.py" \
    --policy-path "$POLICY_PATH" \
    --output "$SCALES_JSON" \
    --device "$DEVICE" \
    --tasks "$TASK_SUITE" \
    --seed "$SEED" \
    --samples "$CALIBRATION_SAMPLES" \
    --percentile "$CALIBRATION_PERCENTILE" \
    --batch-size "$BATCH_SIZE" \
    --max-parallel-tasks 1 \
    --token-length "$TOKEN_LENGTH" \
    --include-module-regex "$W8A8_MODULE_REGEX"
else
  SOURCE_SCALES="$REPO_ROOT/runs/deploy/5/H-bf16-int8-w8a8/kvqo_gate_up_recalibrated_no_down/activation_scales_${CALIBRATION_SAMPLES}_p${CALIBRATION_PERCENTILE}.json"
  [[ -f "$SOURCE_SCALES" ]] || die "missing source scales: $SOURCE_SCALES; use --run-calibration true"
  cp "$SOURCE_SCALES" "$SCALES_JSON"
fi

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
  --quantize-module-regex "$W8A8_MODULE_REGEX" \
  --fake-quant-kind w8a8 \
  --fake-quant-activation-scale-mode calibrated \
  --fake-quant-activation-scales-json "$SCALES_JSON" \
  --extra-w8a16-module-regex "$DOWN_W8A16_REGEX" \
  --output-dir "$OUTPUT_ROOT/fake_dequant_numeric"

printf '%s\n' "step 5L KVQO+gate/up W8A8 + down W8A16 fake-dequant outputs:" \
  "  output root: $OUTPUT_ROOT" \
  "  activation scales: $SCALES_JSON" \
  "  numeric summary: $OUTPUT_ROOT/fake_dequant_numeric/numeric_baseline_summary.json" \
  "  aggregate rows: $OUTPUT_ROOT/fake_dequant_numeric/numeric_baseline_aggregate_rows.csv" \
  "  W8A8 module regex: $W8A8_MODULE_REGEX" \
  "  W8A16 down regex: $DOWN_W8A16_REGEX"
