#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$REPO_ROOT/scripts/common.sh"

POLICY_PATH="$REPO_ROOT/smolvla_libero"
OUTPUT_ROOT="$REPO_ROOT/runs/deploy/3-6/A-mlp-only-int8-precision-obey"
SCALES_JSON="$REPO_ROOT/runs/deploy/3-3B4/activation_scales.json"
BF16_ONNX="$REPO_ROOT/runs/deploy/3-4/smolvla_debug_core_bf16.onnx"
ORT_BF16_ONNX="$REPO_ROOT/runs/deploy/3-4/smolvla_debug_core_ort_fp32.onnx"
INT8_ONNX=""
ORT_INT8_ONNX=""
ENGINE_OUTPUT=""
DEVICE=cuda
BATCH_SIZE=1
TOKEN_LENGTH=48
INPUT_DTYPE=fp32
INCLUDE_NODE_REGEX="^/mlp/(gate_proj|up_proj|down_proj)(_[0-9]+)?/MatMul$"
FAKE_QUANT_MODULE_REGEX="^model\\.vlm_with_expert\\.vlm\\.model\\.text_model\\.layers\\.[0-9]+\\.mlp\\.(gate_proj|up_proj|down_proj)$"
STOP_BEFORE_NODE_REGEX="^/action_in_proj/"
CHECK=true
INSERT_QDQ=true
MAKE_ORT_ONNX=true
BUILD_ENGINE=true
COMPARE=true
PYTHON="${PYTHON_BIN:-python3}"

usage() {
  printf '%s\n' "Usage: $0 [options]" \
    "  --policy-path PATH          default: $POLICY_PATH" \
    "  --output-root DIR          default: $OUTPUT_ROOT" \
    "  --scales-json PATH         default: $SCALES_JSON" \
    "  --bf16-onnx PATH           default: $BF16_ONNX" \
    "  --ort-bf16-onnx PATH       default: $ORT_BF16_ONNX" \
    "  --int8-onnx PATH           default: OUTPUT_ROOT/smolvla_debug_core_vlm_mlp_w8a8_qdq.onnx" \
    "  --ort-int8-onnx PATH       default: OUTPUT_ROOT/smolvla_debug_core_vlm_mlp_w8a8_qdq_ort_fp32.onnx" \
    "  --engine-output PATH       default: OUTPUT_ROOT/smolvla_debug_core_vlm_mlp_w8a8_precision_obey.plan" \
    "  --device DEVICE            default: $DEVICE" \
    "  --batch-size N             default: $BATCH_SIZE" \
    "  --token-length N           default: $TOKEN_LENGTH" \
    "  --input-dtype fp32|bf16    default: $INPUT_DTYPE" \
    "  --include-node-regex RE    default: $INCLUDE_NODE_REGEX" \
    "  --fake-quant-module-regex RE default: $FAKE_QUANT_MODULE_REGEX" \
    "  --stop-before-node-regex RE default: $STOP_BEFORE_NODE_REGEX" \
    "  --check true|false         default: $CHECK" \
    "  --insert-qdq true|false    default: $INSERT_QDQ" \
    "  --make-ort-onnx true|false default: $MAKE_ORT_ONNX" \
    "  --build-engine true|false  default: $BUILD_ENGINE" \
    "  --compare true|false       default: $COMPARE" \
    "  PYTHON_BIN=/path/python    optional Python with torch/lerobot/onnxruntime/tensorrt dependencies"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --policy-path) require_value "$@"; POLICY_PATH="$2"; shift 2 ;;
    --output-root) require_value "$@"; OUTPUT_ROOT="$2"; shift 2 ;;
    --scales-json) require_value "$@"; SCALES_JSON="$2"; shift 2 ;;
    --bf16-onnx) require_value "$@"; BF16_ONNX="$2"; shift 2 ;;
    --ort-bf16-onnx) require_value "$@"; ORT_BF16_ONNX="$2"; shift 2 ;;
    --int8-onnx) require_value "$@"; INT8_ONNX="$2"; shift 2 ;;
    --ort-int8-onnx) require_value "$@"; ORT_INT8_ONNX="$2"; shift 2 ;;
    --engine-output) require_value "$@"; ENGINE_OUTPUT="$2"; shift 2 ;;
    --device) require_value "$@"; DEVICE="$2"; shift 2 ;;
    --batch-size) require_value "$@"; BATCH_SIZE="$2"; shift 2 ;;
    --token-length) require_value "$@"; TOKEN_LENGTH="$2"; shift 2 ;;
    --input-dtype) require_value "$@"; INPUT_DTYPE="$2"; shift 2 ;;
    --include-node-regex) require_value "$@"; INCLUDE_NODE_REGEX="$2"; shift 2 ;;
    --fake-quant-module-regex) require_value "$@"; FAKE_QUANT_MODULE_REGEX="$2"; shift 2 ;;
    --stop-before-node-regex) require_value "$@"; STOP_BEFORE_NODE_REGEX="$2"; shift 2 ;;
    --check) require_value "$@"; CHECK="$2"; shift 2 ;;
    --insert-qdq) require_value "$@"; INSERT_QDQ="$2"; shift 2 ;;
    --make-ort-onnx) require_value "$@"; MAKE_ORT_ONNX="$2"; shift 2 ;;
    --build-engine) require_value "$@"; BUILD_ENGINE="$2"; shift 2 ;;
    --compare) require_value "$@"; COMPARE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

case "$CHECK" in true|false) ;; *) die "--check must be true or false" ;; esac
case "$INSERT_QDQ" in true|false) ;; *) die "--insert-qdq must be true or false" ;; esac
case "$MAKE_ORT_ONNX" in true|false) ;; *) die "--make-ort-onnx must be true or false" ;; esac
case "$BUILD_ENGINE" in true|false) ;; *) die "--build-engine must be true or false" ;; esac
case "$COMPARE" in true|false) ;; *) die "--compare must be true or false" ;; esac
case "$INPUT_DTYPE" in fp32|bf16) ;; *) die "--input-dtype must be fp32 or bf16" ;; esac

mkdir -p "$OUTPUT_ROOT"
INT8_ONNX="${INT8_ONNX:-$OUTPUT_ROOT/smolvla_debug_core_vlm_mlp_w8a8_qdq.onnx}"
ORT_INT8_ONNX="${ORT_INT8_ONNX:-$OUTPUT_ROOT/smolvla_debug_core_vlm_mlp_w8a8_qdq_ort_fp32.onnx}"
ENGINE_OUTPUT="${ENGINE_OUTPUT:-$OUTPUT_ROOT/smolvla_debug_core_vlm_mlp_w8a8_precision_obey.plan}"

export LEROBOT_SRC="${LEROBOT_SRC:-$REPO_ROOT/lerobot/src}"
export PYTHONPATH="$LEROBOT_SRC${PYTHONPATH:+:$PYTHONPATH}"

[[ -f "$BF16_ONNX" ]] || die "$BF16_ONNX not found; run deploy/3-4diagnose.sh first"
[[ -f "$ORT_BF16_ONNX" ]] || die "$ORT_BF16_ONNX not found; run deploy/3-4diagnose.sh first"

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

if [[ "$MAKE_ORT_ONNX" == "true" ]]; then
  ort_args=(
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
    ort_args+=(--check)
  fi
  "$PYTHON" "$SCRIPT_DIR/diagnose_3_4_numeric_baseline.py" "${ort_args[@]}"
else
  [[ -f "$ORT_INT8_ONNX" ]] || die "$ORT_INT8_ONNX not found; enable --make-ort-onnx true or pass --ort-int8-onnx"
fi

if [[ "$BUILD_ENGINE" == "true" ]]; then
  command -v trtexec >/dev/null 2>&1 || die "trtexec not found; install TensorRT or rerun with --build-engine false"
  trtexec \
    --onnx="$INT8_ONNX" \
    --saveEngine="$ENGINE_OUTPUT" \
    --precisionConstraints=obey \
    --profilingVerbosity=detailed \
    --timingCacheFile="$OUTPUT_ROOT/smolvla_debug_vlm_mlp_w8a8_precision_obey.cache"
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
    --quantize-module-regex "$FAKE_QUANT_MODULE_REGEX" \
    --output-dir "$OUTPUT_ROOT/numeric_baseline"
fi

printf '%s\n' "step 3-6A MLP-only INT8 precision-obey outputs:" \
  "  BF16 source ONNX: $BF16_ONNX" \
  "  MLP-only INT8 Q/DQ ONNX: $INT8_ONNX" \
  "  ORT-compatible INT8 shadow ONNX: $ORT_INT8_ONNX" \
  "  TensorRT engine: $ENGINE_OUTPUT" \
  "  PyTorch fake-quant module regex: $FAKE_QUANT_MODULE_REGEX" \
  "  Q/DQ report: ${INT8_ONNX%.onnx}.qdq_report.json" \
  "  numeric baseline: $OUTPUT_ROOT/numeric_baseline"
