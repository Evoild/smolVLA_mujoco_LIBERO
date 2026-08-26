#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$REPO_ROOT/scripts/common.sh"

POLICY_PATH="$REPO_ROOT/smolvla_libero"
OUTPUT_ROOT="$REPO_ROOT/runs/deploy/5/H-bf16-int8-w8a8/down_asym_percentile_sweep"
BASE_SCALES="$REPO_ROOT/runs/deploy/5/H-bf16-int8-w8a8/all_asym_activation/activation_scales_512_p99.999_all_asym.json"
TASK_SUITE=libero_spatial
DEVICE=cuda
SEED=1000
BATCH_SIZE=1
TOKEN_LENGTH=48
CALIBRATION_SAMPLES=512
BASE_PERCENTILE=99.999
DOWN_PERCENTILES=(99.0 99.5 99.9 99.95 99.99 99.995 99.999)
COMPARE_SAMPLES=50
SAMPLE_STRIDE=5
INPUT_SOURCE=rollout
RUN_BASE_CALIBRATION=false
RUN_DOWN_CALIBRATION=true
PYTHON="${PYTHON_BIN:-python3}"

QUANT_MODULE_REGEX="^model\\.vlm_with_expert\\.vlm\\.model\\.text_model\\.layers\\.[0-9]+\\.(self_attn\\.(q_proj|k_proj|v_proj|o_proj)|mlp\\.(gate_proj|up_proj|down_proj))$"
DOWN_MODULE_REGEX="^model\\.vlm_with_expert\\.vlm\\.model\\.text_model\\.layers\\.[0-9]+\\.mlp\\.down_proj$"

usage() {
  printf '%s\n' "Usage: $0 [options]" \
    "  --policy-path PATH              default: $POLICY_PATH" \
    "  --output-root DIR              default: $OUTPUT_ROOT" \
    "  --base-scales PATH             default: $BASE_SCALES" \
    "  --tasks NAME                   default: $TASK_SUITE" \
    "  --device DEVICE                default: $DEVICE" \
    "  --seed N                       default: $SEED" \
    "  --batch-size N                 default: $BATCH_SIZE" \
    "  --token-length N               default: $TOKEN_LENGTH" \
    "  --calibration-samples N        default: $CALIBRATION_SAMPLES" \
    "  --base-percentile P            default: $BASE_PERCENTILE" \
    "  --down-percentiles LIST        default: ${DOWN_PERCENTILES[*]}" \
    "  --input-source synthetic|rollout default: $INPUT_SOURCE" \
    "  --compare-samples N            default: $COMPARE_SAMPLES" \
    "  --sample-stride N              default: $SAMPLE_STRIDE" \
    "  --run-base-calibration true|false default: $RUN_BASE_CALIBRATION" \
    "  --run-down-calibration true|false default: $RUN_DOWN_CALIBRATION" \
    "  PYTHON_BIN=/path/python        optional Python with torch/lerobot dependencies"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --policy-path) require_value "$@"; POLICY_PATH="$2"; shift 2 ;;
    --output-root) require_value "$@"; OUTPUT_ROOT="$2"; shift 2 ;;
    --base-scales) require_value "$@"; BASE_SCALES="$2"; shift 2 ;;
    --tasks) require_value "$@"; TASK_SUITE="$2"; shift 2 ;;
    --device) require_value "$@"; DEVICE="$2"; shift 2 ;;
    --seed) require_value "$@"; SEED="$2"; shift 2 ;;
    --batch-size) require_value "$@"; BATCH_SIZE="$2"; shift 2 ;;
    --token-length) require_value "$@"; TOKEN_LENGTH="$2"; shift 2 ;;
    --calibration-samples) require_value "$@"; CALIBRATION_SAMPLES="$2"; shift 2 ;;
    --base-percentile) require_value "$@"; BASE_PERCENTILE="$2"; shift 2 ;;
    --down-percentiles) require_value "$@"; read -r -a DOWN_PERCENTILES <<< "$2"; shift 2 ;;
    --input-source) require_value "$@"; INPUT_SOURCE="$2"; shift 2 ;;
    --compare-samples) require_value "$@"; COMPARE_SAMPLES="$2"; shift 2 ;;
    --sample-stride) require_value "$@"; SAMPLE_STRIDE="$2"; shift 2 ;;
    --run-base-calibration) require_value "$@"; RUN_BASE_CALIBRATION="$2"; shift 2 ;;
    --run-down-calibration) require_value "$@"; RUN_DOWN_CALIBRATION="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

case "$INPUT_SOURCE" in synthetic|rollout) ;; *) die "--input-source must be synthetic or rollout" ;; esac
case "$RUN_BASE_CALIBRATION" in true|false) ;; *) die "--run-base-calibration must be true or false" ;; esac
case "$RUN_DOWN_CALIBRATION" in true|false) ;; *) die "--run-down-calibration must be true or false" ;; esac

mkdir -p "$OUTPUT_ROOT"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-smolvla}"
export LEROBOT_SRC="${LEROBOT_SRC:-$REPO_ROOT/lerobot/src}"
export PYTHONPATH="$SCRIPT_DIR:$LEROBOT_SRC${PYTHONPATH:+:$PYTHONPATH}"

if [[ "$RUN_BASE_CALIBRATION" == "true" ]]; then
  BASE_SCALES="$OUTPUT_ROOT/base_activation_scales_${CALIBRATION_SAMPLES}_p${BASE_PERCENTILE}_all_asym.json"
  "$PYTHON" "$SCRIPT_DIR/calibrate_smolvla_activation_scales.py" \
    --policy-path "$POLICY_PATH" \
    --output "$BASE_SCALES" \
    --device "$DEVICE" \
    --tasks "$TASK_SUITE" \
    --seed "$SEED" \
    --samples "$CALIBRATION_SAMPLES" \
    --percentile "$BASE_PERCENTILE" \
    --batch-size "$BATCH_SIZE" \
    --max-parallel-tasks 1 \
    --token-length "$TOKEN_LENGTH" \
    --include-module-regex "$QUANT_MODULE_REGEX" \
    --asymmetric-module-regex "$QUANT_MODULE_REGEX"
else
  [[ -f "$BASE_SCALES" ]] || die "missing base scales: $BASE_SCALES"
fi

for percentile in "${DOWN_PERCENTILES[@]}"; do
  percentile_dir="p${percentile//./_}"
  run_dir="$OUTPUT_ROOT/$percentile_dir"
  down_scales="$run_dir/down_activation_scales_p${percentile}_asym.json"
  merged_scales="$run_dir/activation_scales_p${percentile}.json"
  mkdir -p "$run_dir"

  if [[ "$RUN_DOWN_CALIBRATION" == "true" ]]; then
    "$PYTHON" "$SCRIPT_DIR/calibrate_smolvla_activation_scales.py" \
      --policy-path "$POLICY_PATH" \
      --output "$down_scales" \
      --device "$DEVICE" \
      --tasks "$TASK_SUITE" \
      --seed "$SEED" \
      --samples "$CALIBRATION_SAMPLES" \
      --percentile "$percentile" \
      --batch-size "$BATCH_SIZE" \
      --max-parallel-tasks 1 \
      --token-length "$TOKEN_LENGTH" \
      --include-module-regex "$DOWN_MODULE_REGEX" \
      --asymmetric-module-regex "$DOWN_MODULE_REGEX"
  else
    [[ -f "$down_scales" ]] || die "$down_scales not found; enable --run-down-calibration true"
  fi

  "$PYTHON" "$SCRIPT_DIR/merge_activation_scale_overrides.py" \
    --base "$BASE_SCALES" \
    --override "$down_scales" \
    --output "$merged_scales" \
    --override-module-regex "$DOWN_MODULE_REGEX"

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
    --fake-quant-activation-scales-json "$merged_scales" \
    --output-dir "$run_dir/fake_dequant_numeric"
done

"$PYTHON" "$SCRIPT_DIR/summarize_5h_percentile_sweep.py" \
  --output-root "$OUTPUT_ROOT" \
  --percentiles "${DOWN_PERCENTILES[@]}"

printf '%s\n' "step 5H.12 down asymmetric percentile sweep outputs:" \
  "  output root: $OUTPUT_ROOT" \
  "  base scales: $BASE_SCALES" \
  "  summary json: $OUTPUT_ROOT/percentile_sweep_summary.json" \
  "  summary csv: $OUTPUT_ROOT/percentile_sweep_summary.csv" \
  "  quantized module regex: $QUANT_MODULE_REGEX" \
  "  down override regex: $DOWN_MODULE_REGEX"
