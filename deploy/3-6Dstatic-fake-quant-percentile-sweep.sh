#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$REPO_ROOT/scripts/common.sh"

POLICY_PATH="$REPO_ROOT/smolvla_libero"
OUTPUT_ROOT="$REPO_ROOT/runs/deploy/3-6/D-static-fake-quant-percentile-sweep"
DEVICE=cuda
SEED=1000
CALIBRATION_SAMPLES=32
BATCH_SIZE=1
TOKEN_LENGTH=48
INPUT_DTYPE=fp32
RUN_CALIBRATION=true
RUN_CLIPPING=true
PERCENTILES=(99.99 99.9 99.5 99 98)
FAKE_QUANT_MODULE_REGEX="^model\\.vlm_with_expert\\.vlm\\.model\\.text_model\\.layers\\.[0-9]+\\.mlp\\.(gate_proj|up_proj|down_proj)$"
PYTHON="${PYTHON_BIN:-python3}"

usage() {
  printf '%s\n' "Usage: $0 [options]" \
    "  --policy-path PATH          default: $POLICY_PATH" \
    "  --output-root DIR          default: $OUTPUT_ROOT" \
    "  --device DEVICE            default: $DEVICE" \
    "  --seed N                   default: $SEED" \
    "  --calibration-samples N    default: $CALIBRATION_SAMPLES" \
    "  --batch-size N             default: $BATCH_SIZE" \
    "  --token-length N           default: $TOKEN_LENGTH" \
    "  --input-dtype fp32|bf16    default: $INPUT_DTYPE" \
    "  --percentiles LIST         default: ${PERCENTILES[*]}" \
    "  --run-calibration true|false default: $RUN_CALIBRATION" \
    "  --run-clipping true|false  default: $RUN_CLIPPING" \
    "  --fake-quant-module-regex RE default: $FAKE_QUANT_MODULE_REGEX" \
    "  PYTHON_BIN=/path/python    optional Python with torch/lerobot dependencies"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --policy-path) require_value "$@"; POLICY_PATH="$2"; shift 2 ;;
    --output-root) require_value "$@"; OUTPUT_ROOT="$2"; shift 2 ;;
    --device) require_value "$@"; DEVICE="$2"; shift 2 ;;
    --seed) require_value "$@"; SEED="$2"; shift 2 ;;
    --calibration-samples) require_value "$@"; CALIBRATION_SAMPLES="$2"; shift 2 ;;
    --batch-size) require_value "$@"; BATCH_SIZE="$2"; shift 2 ;;
    --token-length) require_value "$@"; TOKEN_LENGTH="$2"; shift 2 ;;
    --input-dtype) require_value "$@"; INPUT_DTYPE="$2"; shift 2 ;;
    --percentiles) require_value "$@"; read -r -a PERCENTILES <<< "$2"; shift 2 ;;
    --run-calibration) require_value "$@"; RUN_CALIBRATION="$2"; shift 2 ;;
    --run-clipping) require_value "$@"; RUN_CLIPPING="$2"; shift 2 ;;
    --fake-quant-module-regex) require_value "$@"; FAKE_QUANT_MODULE_REGEX="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

case "$INPUT_DTYPE" in fp32|bf16) ;; *) die "--input-dtype must be fp32 or bf16" ;; esac
case "$RUN_CALIBRATION" in true|false) ;; *) die "--run-calibration must be true or false" ;; esac
case "$RUN_CLIPPING" in true|false) ;; *) die "--run-clipping must be true or false" ;; esac

mkdir -p "$OUTPUT_ROOT"

export LEROBOT_SRC="${LEROBOT_SRC:-$REPO_ROOT/lerobot/src}"
export PYTHONPATH="$SCRIPT_DIR:$LEROBOT_SRC${PYTHONPATH:+:$PYTHONPATH}"

for percentile in "${PERCENTILES[@]}"; do
  percentile_dir="p${percentile//./_}"
  run_dir="$OUTPUT_ROOT/$percentile_dir"
  scales_json="$run_dir/activation_scales_${percentile_dir}.json"
  mkdir -p "$run_dir"

  if [[ "$RUN_CALIBRATION" == "true" ]]; then
    "$PYTHON" "$SCRIPT_DIR/calibrate_smolvla_activation_scales.py" \
      --policy-path "$POLICY_PATH" \
      --output "$scales_json" \
      --device "$DEVICE" \
      --tasks libero_goal \
      --seed "$SEED" \
      --samples "$CALIBRATION_SAMPLES" \
      --percentile "$percentile" \
      --batch-size "$BATCH_SIZE" \
      --token-length "$TOKEN_LENGTH"
  else
    [[ -f "$scales_json" ]] || die "$scales_json not found; enable --run-calibration true"
  fi

  "$PYTHON" "$SCRIPT_DIR/diagnose_3_4_numeric_baseline.py" compare \
    --policy-path "$POLICY_PATH" \
    --device "$DEVICE" \
    --seed "$SEED" \
    --batch-size "$BATCH_SIZE" \
    --token-length "$TOKEN_LENGTH" \
    --input-dtype "$INPUT_DTYPE" \
    --quantize-module-regex "$FAKE_QUANT_MODULE_REGEX" \
    --fake-quant-activation-scale-mode calibrated \
    --fake-quant-activation-scales-json "$scales_json" \
    --output-dir "$run_dir/numeric_baseline"

  if [[ "$RUN_CLIPPING" == "true" ]]; then
    "$PYTHON" "$SCRIPT_DIR/check_static_scale_clipping.py" \
      --policy-path "$POLICY_PATH" \
      --activation-scales-json "$scales_json" \
      --output-dir "$run_dir/clipping" \
      --device "$DEVICE" \
      --seed "$SEED" \
      --num-seeds 1 \
      --batch-size "$BATCH_SIZE" \
      --token-length "$TOKEN_LENGTH" \
      --input-dtype "$INPUT_DTYPE" \
      --module-regex "$FAKE_QUANT_MODULE_REGEX"
  fi
done

"$PYTHON" "$SCRIPT_DIR/summarize_static_fake_quant_sweep.py" \
  --output-root "$OUTPUT_ROOT" \
  --percentiles "${PERCENTILES[@]}"

printf '%s\n' "step 3-6D static fake-quant percentile sweep outputs:" \
  "  output root: $OUTPUT_ROOT" \
  "  summary json: $OUTPUT_ROOT/static_fake_quant_sweep_summary.json" \
  "  summary csv: $OUTPUT_ROOT/static_fake_quant_sweep_summary.csv" \
  "  quantized module regex: $FAKE_QUANT_MODULE_REGEX"
