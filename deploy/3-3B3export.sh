#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$REPO_ROOT/scripts/common.sh"

POLICY_PATH="$REPO_ROOT/smolvla_libero"
OUTPUT_ROOT="$REPO_ROOT/runs/deploy/3-3B3"
SCALES_JSON=""
BF16_ONNX=""
INT8_ONNX=""
ENGINE_OUTPUT=""
DEVICE=cuda
BATCH_SIZE=1
TOKEN_LENGTH=48
OPSET=18
CALIBRATION_SAMPLES=32
CALIBRATION_PERCENTILE=99.99
CHECK=true
BUILD_ENGINE=true
PYTHON="${PYTHON_BIN:-python3}"

usage() {
  printf '%s\n' "Usage: $0 [options]" \
    "  --policy-path PATH          default: $POLICY_PATH" \
    "  --output-root DIR          default: $OUTPUT_ROOT" \
    "  --scales-json PATH         default: OUTPUT_ROOT/activation_scales.json" \
    "  --bf16-onnx PATH           default: OUTPUT_ROOT/smolvla_sample_actions_core_bf16.onnx" \
    "  --int8-onnx PATH           default: OUTPUT_ROOT/smolvla_sample_actions_core_w8a8_calibrated_qdq.onnx" \
    "  --engine-output PATH       default: OUTPUT_ROOT/smolvla_sample_actions_core_w8a8_calibrated.plan" \
    "  --device DEVICE            default: $DEVICE" \
    "  --batch-size N             default: $BATCH_SIZE" \
    "  --token-length N           default: $TOKEN_LENGTH" \
    "  --calibration-samples N    default: $CALIBRATION_SAMPLES" \
    "  --calibration-percentile P default: $CALIBRATION_PERCENTILE" \
    "  --opset N                  default: $OPSET" \
    "  --check true|false         default: $CHECK" \
    "  --build-engine true|false  default: $BUILD_ENGINE" \
    "  PYTHON_BIN=/path/python    optional Python with torch/lerobot/onnx dependencies"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --policy-path) require_value "$@"; POLICY_PATH="$2"; shift 2 ;;
    --output-root) require_value "$@"; OUTPUT_ROOT="$2"; shift 2 ;;
    --scales-json) require_value "$@"; SCALES_JSON="$2"; shift 2 ;;
    --bf16-onnx) require_value "$@"; BF16_ONNX="$2"; shift 2 ;;
    --int8-onnx) require_value "$@"; INT8_ONNX="$2"; shift 2 ;;
    --engine-output) require_value "$@"; ENGINE_OUTPUT="$2"; shift 2 ;;
    --device) require_value "$@"; DEVICE="$2"; shift 2 ;;
    --batch-size) require_value "$@"; BATCH_SIZE="$2"; shift 2 ;;
    --token-length) require_value "$@"; TOKEN_LENGTH="$2"; shift 2 ;;
    --calibration-samples) require_value "$@"; CALIBRATION_SAMPLES="$2"; shift 2 ;;
    --calibration-percentile) require_value "$@"; CALIBRATION_PERCENTILE="$2"; shift 2 ;;
    --opset) require_value "$@"; OPSET="$2"; shift 2 ;;
    --check) require_value "$@"; CHECK="$2"; shift 2 ;;
    --build-engine) require_value "$@"; BUILD_ENGINE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

case "$CHECK" in true|false) ;; *) die "--check must be true or false" ;; esac
case "$BUILD_ENGINE" in true|false) ;; *) die "--build-engine must be true or false" ;; esac

mkdir -p "$OUTPUT_ROOT"
SCALES_JSON="${SCALES_JSON:-$OUTPUT_ROOT/activation_scales.json}"
BF16_ONNX="${BF16_ONNX:-$OUTPUT_ROOT/smolvla_sample_actions_core_bf16.onnx}"
INT8_ONNX="${INT8_ONNX:-$OUTPUT_ROOT/smolvla_sample_actions_core_w8a8_calibrated_qdq.onnx}"
ENGINE_OUTPUT="${ENGINE_OUTPUT:-$OUTPUT_ROOT/smolvla_sample_actions_core_w8a8_calibrated.plan}"

export LEROBOT_SRC="${LEROBOT_SRC:-$REPO_ROOT/lerobot/src}"
export PYTHONPATH="$LEROBOT_SRC${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON" "$SCRIPT_DIR/calibrate_smolvla_activation_scales.py" \
  --policy-path "$POLICY_PATH" \
  --output "$SCALES_JSON" \
  --device "$DEVICE" \
  --tasks libero_goal \
  --seed 1000 \
  --samples "$CALIBRATION_SAMPLES" \
  --percentile "$CALIBRATION_PERCENTILE" \
  --batch-size "$BATCH_SIZE" \
  --token-length "$TOKEN_LENGTH"

export_args=(
  --policy-path "$POLICY_PATH"
  --output "$BF16_ONNX"
  --device "$DEVICE"
  --dtype bf16
  --input-mode image-embeds
  --batch-size "$BATCH_SIZE"
  --token-length "$TOKEN_LENGTH"
  --opset "$OPSET"
)
if [[ "$CHECK" == "true" ]]; then
  export_args+=(--check)
fi
"$PYTHON" "$SCRIPT_DIR/export_smolvla_sample_actions_onnx.py" "${export_args[@]}"

qdq_args=(
  --input "$BF16_ONNX"
  --output "$INT8_ONNX"
  --activation-scale-mode calibrated
  --activation-scales-json "$SCALES_JSON"
)
if [[ "$CHECK" == "true" ]]; then
  qdq_args+=(--check)
fi
"$PYTHON" "$SCRIPT_DIR/insert_linear_w8a8_qdq.py" "${qdq_args[@]}"

if [[ "$BUILD_ENGINE" == "true" ]]; then
  command -v trtexec >/dev/null 2>&1 || die "trtexec not found; rerun with --build-engine false or install TensorRT"
  trtexec \
    --onnx="$INT8_ONNX" \
    --saveEngine="$ENGINE_OUTPUT" \
    --int8 \
    --bf16 \
    --profilingVerbosity=detailed \
    --timingCacheFile="$OUTPUT_ROOT/smolvla_w8a8_calibrated.cache"
  [[ -s "$ENGINE_OUTPUT" ]] || die "TensorRT produced empty engine: $ENGINE_OUTPUT"
fi

printf '%s\n' "step 3-3B3 export outputs:" \
  "  activation scales: $SCALES_JSON" \
  "  bf16 source ONNX: $BF16_ONNX" \
  "  calibrated INT8 Q/DQ ONNX: $INT8_ONNX" \
  "  TensorRT engine: $ENGINE_OUTPUT"
