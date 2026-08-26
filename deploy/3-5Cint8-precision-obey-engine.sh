#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$REPO_ROOT/scripts/common.sh"

INT8_ONNX="$REPO_ROOT/runs/deploy/3-4/smolvla_debug_core_vlm_w8a8_qdq.onnx"
ORT_INT8_ONNX="$REPO_ROOT/runs/deploy/3-4/smolvla_debug_core_vlm_w8a8_qdq_ort_fp32.onnx"
BF16_ONNX="$REPO_ROOT/runs/deploy/3-4/smolvla_debug_core_bf16.onnx"
ORT_BF16_ONNX="$REPO_ROOT/runs/deploy/3-4/smolvla_debug_core_ort_fp32.onnx"
OUTPUT_ROOT="$REPO_ROOT/runs/deploy/3-5/C-int8-precision-obey"
ENGINE_OUTPUT=""
POLICY_PATH="$REPO_ROOT/smolvla_libero"
DEVICE=cuda
BATCH_SIZE=1
TOKEN_LENGTH=48
INPUT_DTYPE=fp32
BUILD_ENGINE=true
COMPARE=true
PYTHON="${PYTHON_BIN:-python3}"

usage() {
  printf '%s\n' "Usage: $0 [options]" \
    "  --policy-path PATH         default: $POLICY_PATH" \
    "  --int8-onnx PATH           default: $INT8_ONNX" \
    "  --ort-int8-onnx PATH       default: $ORT_INT8_ONNX" \
    "  --bf16-onnx PATH           default: $BF16_ONNX" \
    "  --ort-bf16-onnx PATH       default: $ORT_BF16_ONNX" \
    "  --output-root DIR          default: $OUTPUT_ROOT" \
    "  --engine-output PATH       default: OUTPUT_ROOT/smolvla_debug_core_vlm_w8a8_precision_obey.plan" \
    "  --device DEVICE            default: $DEVICE" \
    "  --batch-size N             default: $BATCH_SIZE" \
    "  --token-length N           default: $TOKEN_LENGTH" \
    "  --input-dtype fp32|bf16    default: $INPUT_DTYPE" \
    "  --build-engine true|false  default: $BUILD_ENGINE" \
    "  --compare true|false       default: $COMPARE" \
    "  PYTHON_BIN=/path/python    optional Python with torch/lerobot/onnxruntime/tensorrt dependencies" \
    "  Builds TensorRT engine from the 3-4 explicit Q/DQ INT8 ONNX with --precisionConstraints=obey and no explicit --bf16/--int8, then compares A vs D."
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --policy-path) require_value "$@"; POLICY_PATH="$2"; shift 2 ;;
    --int8-onnx) require_value "$@"; INT8_ONNX="$2"; shift 2 ;;
    --ort-int8-onnx) require_value "$@"; ORT_INT8_ONNX="$2"; shift 2 ;;
    --bf16-onnx) require_value "$@"; BF16_ONNX="$2"; shift 2 ;;
    --ort-bf16-onnx) require_value "$@"; ORT_BF16_ONNX="$2"; shift 2 ;;
    --output-root) require_value "$@"; OUTPUT_ROOT="$2"; shift 2 ;;
    --engine-output) require_value "$@"; ENGINE_OUTPUT="$2"; shift 2 ;;
    --device) require_value "$@"; DEVICE="$2"; shift 2 ;;
    --batch-size) require_value "$@"; BATCH_SIZE="$2"; shift 2 ;;
    --token-length) require_value "$@"; TOKEN_LENGTH="$2"; shift 2 ;;
    --input-dtype) require_value "$@"; INPUT_DTYPE="$2"; shift 2 ;;
    --build-engine) require_value "$@"; BUILD_ENGINE="$2"; shift 2 ;;
    --compare) require_value "$@"; COMPARE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

case "$BUILD_ENGINE" in true|false) ;; *) die "--build-engine must be true or false" ;; esac
case "$COMPARE" in true|false) ;; *) die "--compare must be true or false" ;; esac
case "$INPUT_DTYPE" in fp32|bf16) ;; *) die "--input-dtype must be fp32 or bf16" ;; esac

mkdir -p "$OUTPUT_ROOT"
ENGINE_OUTPUT="${ENGINE_OUTPUT:-$OUTPUT_ROOT/smolvla_debug_core_vlm_w8a8_precision_obey.plan}"

export LEROBOT_SRC="${LEROBOT_SRC:-$REPO_ROOT/lerobot/src}"
export PYTHONPATH="$LEROBOT_SRC${PYTHONPATH:+:$PYTHONPATH}"

[[ -f "$INT8_ONNX" ]] || die "$INT8_ONNX not found; run deploy/3-4diagnose.sh first"
[[ -f "$BF16_ONNX" ]] || die "$BF16_ONNX not found; run deploy/3-4diagnose.sh first"
[[ -f "$ORT_BF16_ONNX" ]] || die "$ORT_BF16_ONNX not found; run deploy/3-4diagnose.sh first"
[[ -f "$ORT_INT8_ONNX" ]] || die "$ORT_INT8_ONNX not found; run deploy/3-4diagnose.sh first"

if [[ "$BUILD_ENGINE" == "true" ]]; then
  command -v trtexec >/dev/null 2>&1 || die "trtexec not found; install TensorRT or rerun with --build-engine false"
  trtexec \
    --onnx="$INT8_ONNX" \
    --saveEngine="$ENGINE_OUTPUT" \
    --precisionConstraints=obey \
    --profilingVerbosity=detailed \
    --timingCacheFile="$OUTPUT_ROOT/smolvla_debug_vlm_w8a8_precision_obey.cache"
  [[ -s "$ENGINE_OUTPUT" ]] || die "TensorRT produced empty engine: $ENGINE_OUTPUT"
else
  [[ -f "$ENGINE_OUTPUT" ]] || die "$ENGINE_OUTPUT not found; enable --build-engine true or pass --engine-output"
fi

if [[ "$COMPARE" == "true" ]]; then
  "$PYTHON" "$SCRIPT_DIR/diagnose_3_4_numeric_baseline.py" compare \
    --policy-path "$POLICY_PATH" \
    --device "$DEVICE" \
    --batch-size "$BATCH_SIZE" \
    --token-length "$TOKEN_LENGTH" \
    --input-dtype "$INPUT_DTYPE" \
    --bf16-onnx "$BF16_ONNX" \
    --ort-bf16-onnx "$ORT_BF16_ONNX" \
    --int8-onnx "$INT8_ONNX" \
    --ort-int8-onnx "$ORT_INT8_ONNX" \
    --int8-engine "$ENGINE_OUTPUT" \
    --output-dir "$OUTPUT_ROOT/numeric_baseline"
fi

printf '%s\n' "step 3-5C INT8 precision-obey engine outputs:" \
  "  INT8 Q/DQ ONNX: $INT8_ONNX" \
  "  TensorRT engine: $ENGINE_OUTPUT" \
  "  timing cache: $OUTPUT_ROOT/smolvla_debug_vlm_w8a8_precision_obey.cache" \
  "  numeric baseline: $OUTPUT_ROOT/numeric_baseline" \
  "  key pairs: A_vs_D, E_vs_D, B_vs_D_ORT"
