#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$REPO_ROOT/scripts/common.sh"

POLICY_PATH="$REPO_ROOT/smolvla_libero"
OUTPUT_ROOT="$REPO_ROOT/runs/deploy/3-5/A-conservative-bf16"
BF16_ONNX="$REPO_ROOT/runs/deploy/3-4/smolvla_debug_core_bf16.onnx"
ORT_BF16_ONNX="$REPO_ROOT/runs/deploy/3-4/smolvla_debug_core_ort_fp32.onnx"
ENGINE_OUTPUT=""
DEVICE=cuda
BATCH_SIZE=1
TOKEN_LENGTH=48
INPUT_DTYPE=fp32
CHECK=true
MAKE_ORT_ONNX=true
BUILD_ENGINE=true
COMPARE=true
PYTHON="${PYTHON_BIN:-python3}"

usage() {
  printf '%s\n' "Usage: $0 [options]" \
    "  --policy-path PATH          default: $POLICY_PATH" \
    "  --output-root DIR          default: $OUTPUT_ROOT" \
    "  --bf16-onnx PATH           default: $BF16_ONNX" \
    "  --ort-bf16-onnx PATH       default: $ORT_BF16_ONNX" \
    "  --engine-output PATH       default: OUTPUT_ROOT/smolvla_debug_core_native_mixed_conservative.plan" \
    "  --device DEVICE            default: $DEVICE" \
    "  --batch-size N             default: $BATCH_SIZE" \
    "  --token-length N           default: $TOKEN_LENGTH" \
    "  --input-dtype fp32|bf16    default: $INPUT_DTYPE" \
    "  --check true|false         default: $CHECK" \
    "  --make-ort-onnx true|false default: $MAKE_ORT_ONNX" \
    "  --build-engine true|false  default: $BUILD_ENGINE" \
    "  --compare true|false       default: $COMPARE" \
    "  PYTHON_BIN=/path/python    optional Python with torch/lerobot/onnx/onnxruntime dependencies"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --policy-path) require_value "$@"; POLICY_PATH="$2"; shift 2 ;;
    --output-root) require_value "$@"; OUTPUT_ROOT="$2"; shift 2 ;;
    --bf16-onnx) require_value "$@"; BF16_ONNX="$2"; shift 2 ;;
    --ort-bf16-onnx) require_value "$@"; ORT_BF16_ONNX="$2"; shift 2 ;;
    --engine-output) require_value "$@"; ENGINE_OUTPUT="$2"; shift 2 ;;
    --device) require_value "$@"; DEVICE="$2"; shift 2 ;;
    --batch-size) require_value "$@"; BATCH_SIZE="$2"; shift 2 ;;
    --token-length) require_value "$@"; TOKEN_LENGTH="$2"; shift 2 ;;
    --input-dtype) require_value "$@"; INPUT_DTYPE="$2"; shift 2 ;;
    --check) require_value "$@"; CHECK="$2"; shift 2 ;;
    --make-ort-onnx) require_value "$@"; MAKE_ORT_ONNX="$2"; shift 2 ;;
    --build-engine) require_value "$@"; BUILD_ENGINE="$2"; shift 2 ;;
    --compare) require_value "$@"; COMPARE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

case "$CHECK" in true|false) ;; *) die "--check must be true or false" ;; esac
case "$MAKE_ORT_ONNX" in true|false) ;; *) die "--make-ort-onnx must be true or false" ;; esac
case "$BUILD_ENGINE" in true|false) ;; *) die "--build-engine must be true or false" ;; esac
case "$COMPARE" in true|false) ;; *) die "--compare must be true or false" ;; esac
case "$INPUT_DTYPE" in fp32|bf16) ;; *) die "--input-dtype must be fp32 or bf16" ;; esac

mkdir -p "$OUTPUT_ROOT"
ENGINE_OUTPUT="${ENGINE_OUTPUT:-$OUTPUT_ROOT/smolvla_debug_core_native_mixed_conservative.plan}"

export LEROBOT_SRC="${LEROBOT_SRC:-$REPO_ROOT/lerobot/src}"
export PYTHONPATH="$LEROBOT_SRC${PYTHONPATH:+:$PYTHONPATH}"

[[ -f "$BF16_ONNX" ]] || die "$BF16_ONNX not found; run deploy/3-4diagnose.sh first"

if [[ "$MAKE_ORT_ONNX" == "true" ]]; then
  ort_args=(
    make-ort-compatible
    --policy-path "$POLICY_PATH"
    --input "$BF16_ONNX"
    --output "$ORT_BF16_ONNX"
    --device "$DEVICE"
    --batch-size "$BATCH_SIZE"
    --token-length "$TOKEN_LENGTH"
    --input-dtype "$INPUT_DTYPE"
  )
  if [[ "$CHECK" == "true" ]]; then
    ort_args+=(--check)
  fi
  "$PYTHON" "$SCRIPT_DIR/diagnose_3_4_numeric_baseline.py" "${ort_args[@]}"
else
  [[ -f "$ORT_BF16_ONNX" ]] || die "$ORT_BF16_ONNX not found; enable --make-ort-onnx true or pass --ort-bf16-onnx"
fi

if [[ "$BUILD_ENGINE" == "true" ]]; then
  command -v trtexec >/dev/null 2>&1 || die "trtexec not found; rerun with --build-engine false or install TensorRT"
  trtexec \
    --onnx="$BF16_ONNX" \
    --saveEngine="$ENGINE_OUTPUT" \
    --noTF32 \
    --builderOptimizationLevel=0 \
    --precisionConstraints=obey \
    --profilingVerbosity=detailed
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
    --bf16-engine "$ENGINE_OUTPUT" \
    --output-dir "$OUTPUT_ROOT/numeric_baseline"
fi

printf '%s\n' "step 3-5A conservative BF16 outputs:" \
  "  BF16 source ONNX: $BF16_ONNX" \
  "  ORT-compatible BF16 shadow ONNX: $ORT_BF16_ONNX" \
  "  TensorRT conservative engine: $ENGINE_OUTPUT" \
  "  numeric baseline: $OUTPUT_ROOT/numeric_baseline"
