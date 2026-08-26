#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$REPO_ROOT/scripts/common.sh"

POLICY_PATH="$REPO_ROOT/smolvla_libero"
OUTPUT_ROOT="$REPO_ROOT/runs/deploy/3-6/F-per-channel-fake-quant"
SCALES_JSON=""
DEVICE=cuda
SEED=1000
CALIBRATION_SAMPLES=32
BATCH_SIZE=1
TOKEN_LENGTH=48
INPUT_DTYPE=fp32
PERCENTILE=99.99
RUN_CALIBRATION=true
PYTHON="${PYTHON_BIN:-python3}"
CALIBRATE_MODULE_REGEX="^model\\.vlm_with_expert\\.vlm\\.model\\.text_model\\.layers\\.[0-9]+\\.mlp\\.(gate_proj|up_proj|down_proj)$"

usage() {
  printf '%s\n' "Usage: $0 [options]" \
    "  --policy-path PATH          default: $POLICY_PATH" \
    "  --output-root DIR          default: $OUTPUT_ROOT" \
    "  --scales-json PATH         default: OUTPUT_ROOT/activation_channel_scales.json" \
    "  --device DEVICE            default: $DEVICE" \
    "  --seed N                   default: $SEED" \
    "  --calibration-samples N    default: $CALIBRATION_SAMPLES" \
    "  --batch-size N             default: $BATCH_SIZE" \
    "  --token-length N           default: $TOKEN_LENGTH" \
    "  --input-dtype fp32|bf16    default: $INPUT_DTYPE" \
    "  --percentile P             default: $PERCENTILE" \
    "  --run-calibration true|false default: $RUN_CALIBRATION" \
    "  PYTHON_BIN=/path/python    optional Python with torch/lerobot dependencies"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --policy-path) require_value "$@"; POLICY_PATH="$2"; shift 2 ;;
    --output-root) require_value "$@"; OUTPUT_ROOT="$2"; shift 2 ;;
    --scales-json) require_value "$@"; SCALES_JSON="$2"; shift 2 ;;
    --device) require_value "$@"; DEVICE="$2"; shift 2 ;;
    --seed) require_value "$@"; SEED="$2"; shift 2 ;;
    --calibration-samples) require_value "$@"; CALIBRATION_SAMPLES="$2"; shift 2 ;;
    --batch-size) require_value "$@"; BATCH_SIZE="$2"; shift 2 ;;
    --token-length) require_value "$@"; TOKEN_LENGTH="$2"; shift 2 ;;
    --input-dtype) require_value "$@"; INPUT_DTYPE="$2"; shift 2 ;;
    --percentile) require_value "$@"; PERCENTILE="$2"; shift 2 ;;
    --run-calibration) require_value "$@"; RUN_CALIBRATION="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

case "$INPUT_DTYPE" in fp32|bf16) ;; *) die "--input-dtype must be fp32 or bf16" ;; esac
case "$RUN_CALIBRATION" in true|false) ;; *) die "--run-calibration must be true or false" ;; esac

mkdir -p "$OUTPUT_ROOT"
SCALES_JSON="${SCALES_JSON:-$OUTPUT_ROOT/activation_channel_scales.json}"

export LEROBOT_SRC="${LEROBOT_SRC:-$REPO_ROOT/lerobot/src}"
export PYTHONPATH="$SCRIPT_DIR:$LEROBOT_SRC${PYTHONPATH:+:$PYTHONPATH}"

if [[ "$RUN_CALIBRATION" == "true" ]]; then
  "$PYTHON" "$SCRIPT_DIR/calibrate_smolvla_activation_channel_scales.py" \
    --policy-path "$POLICY_PATH" \
    --output "$SCALES_JSON" \
    --device "$DEVICE" \
    --tasks libero_goal \
    --seed "$SEED" \
    --samples "$CALIBRATION_SAMPLES" \
    --percentile "$PERCENTILE" \
    --batch-size "$BATCH_SIZE" \
    --token-length "$TOKEN_LENGTH" \
    --include-module-regex "$CALIBRATE_MODULE_REGEX"
else
  [[ -f "$SCALES_JSON" ]] || die "$SCALES_JSON not found; enable --run-calibration true or pass --scales-json"
fi

VARIANTS=(gate_up all_mlp down_only)
declare -A REGEXES
REGEXES[gate_up]="^model\\.vlm_with_expert\\.vlm\\.model\\.text_model\\.layers\\.[0-9]+\\.mlp\\.(gate_proj|up_proj)$"
REGEXES[all_mlp]="^model\\.vlm_with_expert\\.vlm\\.model\\.text_model\\.layers\\.[0-9]+\\.mlp\\.(gate_proj|up_proj|down_proj)$"
REGEXES[down_only]="^model\\.vlm_with_expert\\.vlm\\.model\\.text_model\\.layers\\.[0-9]+\\.mlp\\.down_proj$"

for variant in "${VARIANTS[@]}"; do
  run_dir="$OUTPUT_ROOT/$variant"
  module_regex="${REGEXES[$variant]}"
  mkdir -p "$run_dir"
  "$PYTHON" "$SCRIPT_DIR/diagnose_3_4_numeric_baseline.py" compare \
    --policy-path "$POLICY_PATH" \
    --device "$DEVICE" \
    --seed "$SEED" \
    --batch-size "$BATCH_SIZE" \
    --token-length "$TOKEN_LENGTH" \
    --input-dtype "$INPUT_DTYPE" \
    --quantize-module-regex "$module_regex" \
    --fake-quant-activation-scale-mode calibrated \
    --fake-quant-activation-scales-json "$SCALES_JSON" \
    --output-dir "$run_dir/numeric_baseline"
done

"$PYTHON" "$SCRIPT_DIR/summarize_layer_mlp_compare.py" \
  --output-root "$OUTPUT_ROOT" \
  --variants "${VARIANTS[@]}"

printf '%s\n' "step 3-6F per-channel activation fake quant outputs:" \
  "  output root: $OUTPUT_ROOT" \
  "  per-channel activation scales: $SCALES_JSON" \
  "  summary json: $OUTPUT_ROOT/layer_mlp_compare_summary.json" \
  "  summary csv: $OUTPUT_ROOT/layer_mlp_compare_summary.csv"
