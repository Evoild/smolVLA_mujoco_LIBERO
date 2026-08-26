#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$REPO_ROOT/scripts/common.sh"

POLICY_PATH="$REPO_ROOT/smolvla_libero"
OUTPUT_ROOT="$REPO_ROOT/runs/deploy/5/O-smoothquant-alpha-sweep"
SOURCE_CHANNEL_SCALES="$REPO_ROOT/runs/deploy/5/O-smoothquant-fake-dequant/activation_channel_scales_512_p99.995.json"
TASK_SUITE=libero_spatial
DEVICE=cuda
SEED=1000
BATCH_SIZE=1
TOKEN_LENGTH=48
CALIBRATION_SAMPLES=512
CALIBRATION_PERCENTILE=99.995
ALPHAS=(0.2 0.3 0.4 0.5 0.6 0.7 0.8)
COMPARE_SAMPLES=50
SAMPLE_STRIDE=5
RUN_CALIBRATION=false
RUN_COMPARE=true
PYTHON="${PYTHON_BIN:-python3}"

QUANT_MODULE_REGEX="^model\\.vlm_with_expert\\.vlm\\.model\\.text_model\\.layers\\.[0-9]+\\.(mlp\\.(gate_proj|up_proj|down_proj)|self_attn\\.(q_proj|k_proj|v_proj|o_proj))$"

usage() {
  printf '%s\n' "Usage: $0 [options]" \
    "  --policy-path PATH              default: $POLICY_PATH" \
    "  --output-root DIR              default: $OUTPUT_ROOT" \
    "  --source-channel-scales PATH   default: $SOURCE_CHANNEL_SCALES" \
    "  --tasks NAME                   default: $TASK_SUITE" \
    "  --device DEVICE                default: $DEVICE" \
    "  --seed N                       default: $SEED" \
    "  --batch-size N                 default: $BATCH_SIZE" \
    "  --token-length N               default: $TOKEN_LENGTH" \
    "  --calibration-samples N        default: $CALIBRATION_SAMPLES" \
    "  --calibration-percentile P     default: $CALIBRATION_PERCENTILE" \
    "  --alphas LIST                  default: ${ALPHAS[*]}" \
    "  --compare-samples N            default: $COMPARE_SAMPLES" \
    "  --sample-stride N              default: $SAMPLE_STRIDE" \
    "  --run-calibration true|false   default: $RUN_CALIBRATION" \
    "  --run-compare true|false       default: $RUN_COMPARE" \
    "  PYTHON_BIN=/path/python        optional Python with torch/lerobot dependencies"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --policy-path) require_value "$@"; POLICY_PATH="$2"; shift 2 ;;
    --output-root) require_value "$@"; OUTPUT_ROOT="$2"; shift 2 ;;
    --source-channel-scales) require_value "$@"; SOURCE_CHANNEL_SCALES="$2"; shift 2 ;;
    --tasks) require_value "$@"; TASK_SUITE="$2"; shift 2 ;;
    --device) require_value "$@"; DEVICE="$2"; shift 2 ;;
    --seed) require_value "$@"; SEED="$2"; shift 2 ;;
    --batch-size) require_value "$@"; BATCH_SIZE="$2"; shift 2 ;;
    --token-length) require_value "$@"; TOKEN_LENGTH="$2"; shift 2 ;;
    --calibration-samples) require_value "$@"; CALIBRATION_SAMPLES="$2"; shift 2 ;;
    --calibration-percentile) require_value "$@"; CALIBRATION_PERCENTILE="$2"; shift 2 ;;
    --alphas) require_value "$@"; read -r -a ALPHAS <<< "$2"; shift 2 ;;
    --compare-samples) require_value "$@"; COMPARE_SAMPLES="$2"; shift 2 ;;
    --sample-stride) require_value "$@"; SAMPLE_STRIDE="$2"; shift 2 ;;
    --run-calibration) require_value "$@"; RUN_CALIBRATION="$2"; shift 2 ;;
    --run-compare) require_value "$@"; RUN_COMPARE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

case "$RUN_CALIBRATION" in true|false) ;; *) die "--run-calibration must be true or false" ;; esac
case "$RUN_COMPARE" in true|false) ;; *) die "--run-compare must be true or false" ;; esac

mkdir -p "$OUTPUT_ROOT"
export LEROBOT_SRC="${LEROBOT_SRC:-$REPO_ROOT/lerobot/src}"
export PYTHONPATH="$SCRIPT_DIR:$LEROBOT_SRC${PYTHONPATH:+:$PYTHONPATH}"

CHANNEL_SCALES_JSON="$SOURCE_CHANNEL_SCALES"
if [[ "$RUN_CALIBRATION" == "true" ]]; then
  CHANNEL_SCALES_JSON="$OUTPUT_ROOT/activation_channel_scales_${CALIBRATION_SAMPLES}_p${CALIBRATION_PERCENTILE}.json"
  "$PYTHON" "$SCRIPT_DIR/calibrate_smolvla_activation_channel_scales.py" \
    --policy-path "$POLICY_PATH" \
    --output "$CHANNEL_SCALES_JSON" \
    --device "$DEVICE" \
    --tasks "$TASK_SUITE" \
    --seed "$SEED" \
    --samples "$CALIBRATION_SAMPLES" \
    --percentile "$CALIBRATION_PERCENTILE" \
    --batch-size "$BATCH_SIZE" \
    --max-parallel-tasks 1 \
    --token-length "$TOKEN_LENGTH" \
    --include-module-regex "$QUANT_MODULE_REGEX"
else
  [[ -f "$CHANNEL_SCALES_JSON" ]] || die "$CHANNEL_SCALES_JSON not found; pass --run-calibration true or --source-channel-scales"
fi

for alpha in "${ALPHAS[@]}"; do
  alpha_dir="alpha_${alpha//./_}"
  run_dir="$OUTPUT_ROOT/$alpha_dir"
  smooth_scales_json="$run_dir/smoothquant_alpha_${alpha}_activation_scales.json"
  mkdir -p "$run_dir"

  "$PYTHON" "$SCRIPT_DIR/build_smoothquant_scales.py" \
    --policy-path "$POLICY_PATH" \
    --activation-channel-scales-json "$CHANNEL_SCALES_JSON" \
    --output "$smooth_scales_json" \
    --device "$DEVICE" \
    --alpha "$alpha" \
    --include-module-regex "$QUANT_MODULE_REGEX"

  if [[ "$RUN_COMPARE" == "true" ]]; then
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
      --fake-quant-activation-scales-json "$smooth_scales_json" \
      --output-dir "$run_dir/fake_dequant_numeric"
  fi
done

"$PYTHON" "$SCRIPT_DIR/summarize_smoothquant_alpha_sweep.py" \
  --output-root "$OUTPUT_ROOT" \
  --alphas "${ALPHAS[@]}"

printf '%s\n' "step 5O SmoothQuant alpha sweep outputs:" \
  "  output root: $OUTPUT_ROOT" \
  "  source per-channel scales: $CHANNEL_SCALES_JSON" \
  "  summary json: $OUTPUT_ROOT/smoothquant_alpha_sweep_summary.json" \
  "  summary csv: $OUTPUT_ROOT/smoothquant_alpha_sweep_summary.csv" \
  "  alphas: ${ALPHAS[*]}" \
  "  quantized module regex: $QUANT_MODULE_REGEX"
