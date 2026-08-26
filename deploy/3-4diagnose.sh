#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$REPO_ROOT/scripts/common.sh"

POLICY_PATH="$REPO_ROOT/smolvla_libero"
OUTPUT_ROOT="$REPO_ROOT/runs/deploy/3-4"
SCALES_JSON="$REPO_ROOT/runs/deploy/3-3B4/activation_scales.json"
BF16_ONNX=""
INT8_ONNX=""
ORT_BF16_ONNX=""
ORT_INT8_ONNX=""
BF16_ENGINE=""
INT8_ENGINE=""
DEVICE=cuda
BATCH_SIZE=1
TOKEN_LENGTH=48
OPSET=18
INPUT_DTYPE=fp32
INCLUDE_NODE_REGEX="^/(q_proj|k_proj|v_proj|o_proj|mlp/(gate_proj|up_proj|down_proj))(_[0-9]+)?/MatMul$"
STOP_BEFORE_NODE_REGEX="^/action_in_proj/"
CHECK=true
EXPORT_ONNX=true
EXPORT_ORT_ONNX=true
INSERT_QDQ=true
BUILD_ENGINE=true
COMPARE=true
PYTHON="${PYTHON_BIN:-python3}"

usage() {
  printf '%s\n' "Usage: $0 [options]" \
    "  --policy-path PATH          default: $POLICY_PATH" \
    "  --output-root DIR          default: $OUTPUT_ROOT" \
    "  --scales-json PATH         default: $SCALES_JSON" \
    "  --bf16-onnx PATH           default: OUTPUT_ROOT/smolvla_debug_core_bf16.onnx" \
    "  --int8-onnx PATH           default: OUTPUT_ROOT/smolvla_debug_core_vlm_w8a8_qdq.onnx" \
    "  --ort-bf16-onnx PATH       default: OUTPUT_ROOT/smolvla_debug_core_ort_fp32.onnx" \
    "  --ort-int8-onnx PATH       default: OUTPUT_ROOT/smolvla_debug_core_vlm_w8a8_qdq_ort_fp32.onnx" \
    "  --bf16-engine PATH         default: OUTPUT_ROOT/smolvla_debug_core_bf16.plan" \
    "  --int8-engine PATH         default: OUTPUT_ROOT/smolvla_debug_core_vlm_w8a8.plan" \
    "  --device DEVICE            default: $DEVICE" \
    "  --batch-size N             default: $BATCH_SIZE" \
    "  --token-length N           default: $TOKEN_LENGTH" \
    "  --input-dtype fp32|bf16    default: $INPUT_DTYPE" \
    "  --opset N                  default: $OPSET" \
    "  --include-node-regex RE    default: $INCLUDE_NODE_REGEX" \
    "  --stop-before-node-regex RE default: $STOP_BEFORE_NODE_REGEX" \
    "  --check true|false         default: $CHECK" \
    "  --export-onnx true|false   default: $EXPORT_ONNX" \
    "  --export-ort-onnx true|false default: $EXPORT_ORT_ONNX" \
    "  --insert-qdq true|false    default: $INSERT_QDQ" \
    "  --build-engine true|false  default: $BUILD_ENGINE" \
    "  --compare true|false       default: $COMPARE" \
    "  PYTHON_BIN=/path/python    optional Python with torch/lerobot/onnx/onnxruntime dependencies"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --policy-path) require_value "$@"; POLICY_PATH="$2"; shift 2 ;;
    --output-root) require_value "$@"; OUTPUT_ROOT="$2"; shift 2 ;;
    --scales-json) require_value "$@"; SCALES_JSON="$2"; shift 2 ;;
    --bf16-onnx) require_value "$@"; BF16_ONNX="$2"; shift 2 ;;
    --int8-onnx) require_value "$@"; INT8_ONNX="$2"; shift 2 ;;
    --ort-bf16-onnx) require_value "$@"; ORT_BF16_ONNX="$2"; shift 2 ;;
    --ort-int8-onnx) require_value "$@"; ORT_INT8_ONNX="$2"; shift 2 ;;
    --bf16-engine) require_value "$@"; BF16_ENGINE="$2"; shift 2 ;;
    --int8-engine) require_value "$@"; INT8_ENGINE="$2"; shift 2 ;;
    --device) require_value "$@"; DEVICE="$2"; shift 2 ;;
    --batch-size) require_value "$@"; BATCH_SIZE="$2"; shift 2 ;;
    --token-length) require_value "$@"; TOKEN_LENGTH="$2"; shift 2 ;;
    --input-dtype) require_value "$@"; INPUT_DTYPE="$2"; shift 2 ;;
    --opset) require_value "$@"; OPSET="$2"; shift 2 ;;
    --include-node-regex) require_value "$@"; INCLUDE_NODE_REGEX="$2"; shift 2 ;;
    --stop-before-node-regex) require_value "$@"; STOP_BEFORE_NODE_REGEX="$2"; shift 2 ;;
    --check) require_value "$@"; CHECK="$2"; shift 2 ;;
    --export-onnx) require_value "$@"; EXPORT_ONNX="$2"; shift 2 ;;
    --export-ort-onnx) require_value "$@"; EXPORT_ORT_ONNX="$2"; shift 2 ;;
    --insert-qdq) require_value "$@"; INSERT_QDQ="$2"; shift 2 ;;
    --build-engine) require_value "$@"; BUILD_ENGINE="$2"; shift 2 ;;
    --compare) require_value "$@"; COMPARE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

case "$CHECK" in true|false) ;; *) die "--check must be true or false" ;; esac
case "$EXPORT_ONNX" in true|false) ;; *) die "--export-onnx must be true or false" ;; esac
case "$EXPORT_ORT_ONNX" in true|false) ;; *) die "--export-ort-onnx must be true or false" ;; esac
case "$INSERT_QDQ" in true|false) ;; *) die "--insert-qdq must be true or false" ;; esac
case "$BUILD_ENGINE" in true|false) ;; *) die "--build-engine must be true or false" ;; esac
case "$COMPARE" in true|false) ;; *) die "--compare must be true or false" ;; esac
case "$INPUT_DTYPE" in fp32|bf16) ;; *) die "--input-dtype must be fp32 or bf16" ;; esac

mkdir -p "$OUTPUT_ROOT"
BF16_ONNX="${BF16_ONNX:-$OUTPUT_ROOT/smolvla_debug_core_bf16.onnx}"
INT8_ONNX="${INT8_ONNX:-$OUTPUT_ROOT/smolvla_debug_core_vlm_w8a8_qdq.onnx}"
ORT_BF16_ONNX="${ORT_BF16_ONNX:-$OUTPUT_ROOT/smolvla_debug_core_ort_fp32.onnx}"
ORT_INT8_ONNX="${ORT_INT8_ONNX:-$OUTPUT_ROOT/smolvla_debug_core_vlm_w8a8_qdq_ort_fp32.onnx}"
BF16_ENGINE="${BF16_ENGINE:-$OUTPUT_ROOT/smolvla_debug_core_bf16.plan}"
INT8_ENGINE="${INT8_ENGINE:-$OUTPUT_ROOT/smolvla_debug_core_vlm_w8a8.plan}"

export LEROBOT_SRC="${LEROBOT_SRC:-$REPO_ROOT/lerobot/src}"
export PYTHONPATH="$LEROBOT_SRC${PYTHONPATH:+:$PYTHONPATH}"

if [[ "$EXPORT_ONNX" == "true" ]]; then
  export_args=(
    export
    --policy-path "$POLICY_PATH"
    --output "$BF16_ONNX"
    --device "$DEVICE"
    --batch-size "$BATCH_SIZE"
    --token-length "$TOKEN_LENGTH"
    --input-dtype "$INPUT_DTYPE"
    --opset "$OPSET"
  )
  if [[ "$CHECK" == "true" ]]; then
    export_args+=(--check)
  fi
  "$PYTHON" "$SCRIPT_DIR/diagnose_3_4_numeric_baseline.py" "${export_args[@]}"
else
  [[ -f "$BF16_ONNX" ]] || die "$BF16_ONNX not found; enable --export-onnx true or pass --bf16-onnx"
fi

if [[ "$EXPORT_ORT_ONNX" == "true" ]]; then
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
  [[ -f "$ORT_BF16_ONNX" ]] || die "$ORT_BF16_ONNX not found; enable --export-ort-onnx true or pass --ort-bf16-onnx"
fi

if [[ "$INSERT_QDQ" == "true" ]]; then
  [[ -f "$SCALES_JSON" ]] || die "$SCALES_JSON not found; run deploy/3-3B4export.sh first or pass --scales-json"
  qdq_args=(
    --input "$BF16_ONNX"
    --output "$INT8_ONNX"
    --activation-scale-mode calibrated
    --activation-scales-json "$SCALES_JSON"
    --include-node-regex "$INCLUDE_NODE_REGEX"
    --stop-before-node-regex "$STOP_BEFORE_NODE_REGEX"
  )
  if [[ "$CHECK" == "true" ]]; then
    qdq_args+=(--check)
  fi
  "$PYTHON" "$SCRIPT_DIR/insert_linear_w8a8_qdq.py" "${qdq_args[@]}"
else
  [[ -f "$INT8_ONNX" ]] || die "$INT8_ONNX not found; enable --insert-qdq true or pass --int8-onnx"
fi

if [[ "$EXPORT_ORT_ONNX" == "true" ]]; then
  ort_int8_args=(
    make-ort-compatible
    --policy-path "$POLICY_PATH"
    --input "$INT8_ONNX"
    --output "$ORT_INT8_ONNX"
    --device "$DEVICE"
    --batch-size "$BATCH_SIZE"
    --token-length "$TOKEN_LENGTH"
    --input-dtype "$INPUT_DTYPE"
  )
  if [[ "$CHECK" == "true" ]]; then
    ort_int8_args+=(--check)
  fi
  "$PYTHON" "$SCRIPT_DIR/diagnose_3_4_numeric_baseline.py" "${ort_int8_args[@]}"
else
  [[ -f "$ORT_INT8_ONNX" ]] || die "$ORT_INT8_ONNX not found; enable --export-ort-onnx true or pass --ort-int8-onnx"
fi

if [[ "$BUILD_ENGINE" == "true" ]]; then
  command -v trtexec >/dev/null 2>&1 || die "trtexec not found; rerun with --build-engine false or install TensorRT"
  trtexec \
    --onnx="$BF16_ONNX" \
    --saveEngine="$BF16_ENGINE" \
    --bf16 \
    --profilingVerbosity=detailed \
    --timingCacheFile="$OUTPUT_ROOT/smolvla_debug_bf16.cache"
  [[ -s "$BF16_ENGINE" ]] || die "TensorRT produced empty engine: $BF16_ENGINE"

  trtexec \
    --onnx="$INT8_ONNX" \
    --saveEngine="$INT8_ENGINE" \
    --int8 \
    --bf16 \
    --profilingVerbosity=detailed \
    --timingCacheFile="$OUTPUT_ROOT/smolvla_debug_vlm_w8a8.cache"
  [[ -s "$INT8_ENGINE" ]] || die "TensorRT produced empty engine: $INT8_ENGINE"
fi

if [[ "$COMPARE" == "true" ]]; then
  "$PYTHON" "$SCRIPT_DIR/diagnose_3_4_numeric_baseline.py" compare \
    --policy-path "$POLICY_PATH" \
    --device "$DEVICE" \
    --batch-size "$BATCH_SIZE" \
    --token-length "$TOKEN_LENGTH" \
    --input-dtype "$INPUT_DTYPE" \
    --bf16-onnx "$BF16_ONNX" \
    --int8-onnx "$INT8_ONNX" \
    --ort-bf16-onnx "$ORT_BF16_ONNX" \
    --ort-int8-onnx "$ORT_INT8_ONNX" \
    --bf16-engine "$BF16_ENGINE" \
    --int8-engine "$INT8_ENGINE" \
    --output-dir "$OUTPUT_ROOT/numeric_baseline"
fi

printf '%s\n' "step 3-4 diagnosis outputs:" \
  "  BF16 debug ONNX: $BF16_ONNX" \
  "  VLM-only W8A8 debug ONNX: $INT8_ONNX" \
  "  ORT-compatible BF16 shadow ONNX: $ORT_BF16_ONNX" \
  "  ORT-compatible INT8 shadow ONNX: $ORT_INT8_ONNX" \
  "  TensorRT BF16 debug engine: $BF16_ENGINE" \
  "  TensorRT INT8 Q/DQ debug engine: $INT8_ENGINE" \
  "  numeric baseline: $OUTPUT_ROOT/numeric_baseline"
