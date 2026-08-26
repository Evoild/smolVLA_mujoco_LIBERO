#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$REPO_ROOT/scripts/common.sh"

POLICY_PATH="$REPO_ROOT/smolvla_libero"
OUTPUT_ROOT="$REPO_ROOT/runs/deploy/3-6/B-mlp-only-static-fake-quant"
SCALES_JSON="$REPO_ROOT/runs/deploy/3-3B4/activation_scales.json"
BF16_ONNX="$REPO_ROOT/runs/deploy/3-4/smolvla_debug_core_bf16.onnx"
ORT_BF16_ONNX="$REPO_ROOT/runs/deploy/3-4/smolvla_debug_core_ort_fp32.onnx"
DEVICE=cuda
BATCH_SIZE=1
TOKEN_LENGTH=48
INPUT_DTYPE=fp32
FAKE_QUANT_MODULE_REGEX="^model\\.vlm_with_expert\\.vlm\\.model\\.text_model\\.layers\\.[0-9]+\\.mlp\\.(gate_proj|up_proj|down_proj)$"
PYTHON="${PYTHON_BIN:-python3}"

usage() {
  printf '%s\n' "Usage: $0 [options]" \
    "  --policy-path PATH          default: $POLICY_PATH" \
    "  --output-root DIR          default: $OUTPUT_ROOT" \
    "  --scales-json PATH         default: $SCALES_JSON" \
    "  --bf16-onnx PATH           default: $BF16_ONNX" \
    "  --ort-bf16-onnx PATH       default: $ORT_BF16_ONNX" \
    "  --device DEVICE            default: $DEVICE" \
    "  --batch-size N             default: $BATCH_SIZE" \
    "  --token-length N           default: $TOKEN_LENGTH" \
    "  --input-dtype fp32|bf16    default: $INPUT_DTYPE" \
    "  --fake-quant-module-regex RE default: $FAKE_QUANT_MODULE_REGEX" \
    "  PYTHON_BIN=/path/python    optional Python with torch/lerobot/onnxruntime dependencies"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --policy-path) require_value "$@"; POLICY_PATH="$2"; shift 2 ;;
    --output-root) require_value "$@"; OUTPUT_ROOT="$2"; shift 2 ;;
    --scales-json) require_value "$@"; SCALES_JSON="$2"; shift 2 ;;
    --bf16-onnx) require_value "$@"; BF16_ONNX="$2"; shift 2 ;;
    --ort-bf16-onnx) require_value "$@"; ORT_BF16_ONNX="$2"; shift 2 ;;
    --device) require_value "$@"; DEVICE="$2"; shift 2 ;;
    --batch-size) require_value "$@"; BATCH_SIZE="$2"; shift 2 ;;
    --token-length) require_value "$@"; TOKEN_LENGTH="$2"; shift 2 ;;
    --input-dtype) require_value "$@"; INPUT_DTYPE="$2"; shift 2 ;;
    --fake-quant-module-regex) require_value "$@"; FAKE_QUANT_MODULE_REGEX="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

case "$INPUT_DTYPE" in fp32|bf16) ;; *) die "--input-dtype must be fp32 or bf16" ;; esac

mkdir -p "$OUTPUT_ROOT"

export LEROBOT_SRC="${LEROBOT_SRC:-$REPO_ROOT/lerobot/src}"
export PYTHONPATH="$LEROBOT_SRC${PYTHONPATH:+:$PYTHONPATH}"

[[ -f "$SCALES_JSON" ]] || die "$SCALES_JSON not found; run deploy/3-3B4export.sh first or pass --scales-json"
[[ -f "$BF16_ONNX" ]] || die "$BF16_ONNX not found; run deploy/3-4diagnose.sh first"
[[ -f "$ORT_BF16_ONNX" ]] || die "$ORT_BF16_ONNX not found; run deploy/3-4diagnose.sh first"

"$PYTHON" "$SCRIPT_DIR/diagnose_3_4_numeric_baseline.py" compare \
  --policy-path "$POLICY_PATH" \
  --device "$DEVICE" \
  --batch-size "$BATCH_SIZE" \
  --token-length "$TOKEN_LENGTH" \
  --input-dtype "$INPUT_DTYPE" \
  --bf16-onnx "$BF16_ONNX" \
  --ort-bf16-onnx "$ORT_BF16_ONNX" \
  --quantize-module-regex "$FAKE_QUANT_MODULE_REGEX" \
  --fake-quant-activation-scale-mode calibrated \
  --fake-quant-activation-scales-json "$SCALES_JSON" \
  --output-dir "$OUTPUT_ROOT/numeric_baseline"

printf '%s\n' "step 3-6B MLP-only calibrated-static PyTorch fake quant outputs:" \
  "  activation scales: $SCALES_JSON" \
  "  PyTorch fake-quant module regex: $FAKE_QUANT_MODULE_REGEX" \
  "  numeric baseline: $OUTPUT_ROOT/numeric_baseline"
