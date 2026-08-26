#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$REPO_ROOT/scripts/common.sh"

POLICY_PATH="$REPO_ROOT/smolvla_libero"
OUTPUT_ROOT="$REPO_ROOT/runs/deploy/5/O-smoothquant-down-asym-alpha-sweep"
SOURCE_CHANNEL_SCALES="$REPO_ROOT/runs/deploy/5/O-smoothquant-fake-dequant/activation_channel_scales_512_p99.995.json"
TASK_SUITE=libero_spatial
DEVICE=cuda
SEED=1000
BATCH_SIZE=1
TOKEN_LENGTH=48
CALIBRATION_SAMPLES=512
CALIBRATION_PERCENTILE=99.995
ALPHAS=(0.7 0.8 0.85 0.9 0.95)
COMPARE_SAMPLES=50
SAMPLE_STRIDE=5
RUN_COMPARE=true
RUN_LAYER_CURVE=true
PYTHON="${PYTHON_BIN:-python3}"

QUANT_MODULE_REGEX="^model\\.vlm_with_expert\\.vlm\\.model\\.text_model\\.layers\\.[0-9]+\\.(mlp\\.(gate_proj|up_proj|down_proj)|self_attn\\.(q_proj|k_proj|v_proj|o_proj))$"
DOWN_MODULE_REGEX="^model\\.vlm_with_expert\\.vlm\\.model\\.text_model\\.layers\\.[0-9]+\\.mlp\\.down_proj$"

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
    "  --run-compare true|false       default: $RUN_COMPARE" \
    "  --run-layer-curve true|false   default: $RUN_LAYER_CURVE" \
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
    --run-compare) require_value "$@"; RUN_COMPARE="$2"; shift 2 ;;
    --run-layer-curve) require_value "$@"; RUN_LAYER_CURVE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

case "$RUN_COMPARE" in true|false) ;; *) die "--run-compare must be true or false" ;; esac
case "$RUN_LAYER_CURVE" in true|false) ;; *) die "--run-layer-curve must be true or false" ;; esac
[[ -f "$SOURCE_CHANNEL_SCALES" ]] || die "$SOURCE_CHANNEL_SCALES not found"

mkdir -p "$OUTPUT_ROOT"
export NUMBA_DISABLE_JIT="${NUMBA_DISABLE_JIT:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-smolvla}"
export LEROBOT_SRC="${LEROBOT_SRC:-$REPO_ROOT/lerobot/src}"
export PYTHONPATH="$SCRIPT_DIR:$LEROBOT_SRC${PYTHONPATH:+:$PYTHONPATH}"

for alpha in "${ALPHAS[@]}"; do
  alpha_dir="alpha_${alpha//./_}"
  run_dir="$OUTPUT_ROOT/$alpha_dir"
  smooth_scales_json="$run_dir/smoothquant_alpha_${alpha}_activation_scales.json"
  down_asym_scales_json="$run_dir/smoothquant_alpha_${alpha}_down_asym_activation_scales.json"
  mkdir -p "$run_dir"

  "$PYTHON" "$SCRIPT_DIR/build_smoothquant_scales.py" \
    --policy-path "$POLICY_PATH" \
    --activation-channel-scales-json "$SOURCE_CHANNEL_SCALES" \
    --output "$smooth_scales_json" \
    --device "$DEVICE" \
    --alpha "$alpha" \
    --include-module-regex "$QUANT_MODULE_REGEX"

  "$PYTHON" "$SCRIPT_DIR/calibrate_smoothquant_down_asym_scales.py" \
    --policy-path "$POLICY_PATH" \
    --smoothquant-scales-json "$smooth_scales_json" \
    --output "$down_asym_scales_json" \
    --device "$DEVICE" \
    --tasks "$TASK_SUITE" \
    --seed "$SEED" \
    --samples "$CALIBRATION_SAMPLES" \
    --percentile "$CALIBRATION_PERCENTILE" \
    --batch-size "$BATCH_SIZE" \
    --max-parallel-tasks 1 \
    --token-length "$TOKEN_LENGTH" \
    --down-module-regex "$DOWN_MODULE_REGEX"

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
      --fake-quant-activation-scales-json "$down_asym_scales_json" \
      --output-dir "$run_dir/fake_dequant_numeric"
  fi
done

if [[ "$RUN_COMPARE" == "true" ]]; then
  "$PYTHON" "$SCRIPT_DIR/summarize_smoothquant_alpha_sweep.py" \
    --output-root "$OUTPUT_ROOT" \
    --alphas "${ALPHAS[@]}"

best_alpha="$("$PYTHON" - "$OUTPUT_ROOT/smoothquant_alpha_sweep_summary.json" <<'PY'
import json
import sys

with open(sys.argv[1]) as f:
    rows = json.load(f)["rows"]
best = min(
    (row for row in rows if row.get("action_chunk_relative_l2_mean") is not None),
    key=lambda row: float(row["action_chunk_relative_l2_mean"]),
)
print(best["alpha"])
PY
)"

  if [[ "$RUN_LAYER_CURVE" == "true" ]]; then
    alpha_dir="alpha_${best_alpha//./_}"
    best_scales="$OUTPUT_ROOT/$alpha_dir/smoothquant_alpha_${best_alpha}_down_asym_activation_scales.json"
    "$PYTHON" "$SCRIPT_DIR/compare_5h_text_vlm_layer_outputs.py" \
      --policy-path "$POLICY_PATH" \
      --activation-scales-json "$best_scales" \
      --output-dir "$OUTPUT_ROOT/$alpha_dir/layer_outputs" \
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
      --quantize-module-regex "$QUANT_MODULE_REGEX"
  fi
fi

printf '%s\n' "step 5O SmoothQuant down-asymmetric alpha sweep outputs:" \
  "  output root: $OUTPUT_ROOT" \
  "  source per-channel scales: $SOURCE_CHANNEL_SCALES" \
  "  summary json: $OUTPUT_ROOT/smoothquant_alpha_sweep_summary.json" \
  "  summary csv: $OUTPUT_ROOT/smoothquant_alpha_sweep_summary.csv" \
  "  alphas: ${ALPHAS[*]}" \
  "  quantized module regex: $QUANT_MODULE_REGEX" \
  "  down asymmetric regex: $DOWN_MODULE_REGEX"
