#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$REPO_ROOT/scripts/common.sh"

POLICY_PATH="$REPO_ROOT/smolvla_libero"
OUTPUT="$REPO_ROOT/runs/deploy/tensorrt/smolvla_sample_actions.onnx"
ENGINE_OUTPUT=""
DEVICE=cuda
DTYPE=fp32
INPUT_MODE=image-embeds
BATCH_SIZE=1
TOKEN_LENGTH=48
OPSET=18
CHECK=true
BUILD_ENGINE=false
PYTHON="${PYTHON_BIN:-python3}"

usage() {
  printf '%s\n' "Usage: $0 [options]" \
    "  --policy-path PATH          default: $POLICY_PATH" \
    "  --output PATH               default: $OUTPUT" \
    "  --engine-output PATH        default: OUTPUT with .plan suffix" \
    "  --device DEVICE            default: $DEVICE" \
    "  --dtype fp32|bf16          default: $DTYPE" \
    "  --input-mode image-embeds|images default: $INPUT_MODE" \
    "  --batch-size N             default: $BATCH_SIZE" \
    "  --token-length N           default: $TOKEN_LENGTH" \
    "  --opset N                  default: $OPSET" \
    "  --check true|false         default: $CHECK" \
    "  --build-engine true|false  default: $BUILD_ENGINE" \
    "  PYTHON_BIN=/path/python    optional Python with torch/lerobot/onnx dependencies"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --policy-path) require_value "$@"; POLICY_PATH="$2"; shift 2 ;;
    --output) require_value "$@"; OUTPUT="$2"; shift 2 ;;
    --engine-output) require_value "$@"; ENGINE_OUTPUT="$2"; shift 2 ;;
    --device) require_value "$@"; DEVICE="$2"; shift 2 ;;
    --dtype) require_value "$@"; DTYPE="$2"; shift 2 ;;
    --input-mode) require_value "$@"; INPUT_MODE="$2"; shift 2 ;;
    --batch-size) require_value "$@"; BATCH_SIZE="$2"; shift 2 ;;
    --token-length) require_value "$@"; TOKEN_LENGTH="$2"; shift 2 ;;
    --opset) require_value "$@"; OPSET="$2"; shift 2 ;;
    --check) require_value "$@"; CHECK="$2"; shift 2 ;;
    --build-engine) require_value "$@"; BUILD_ENGINE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

case "$DTYPE" in
  fp32|bf16) ;;
  *) die "--dtype must be fp32 or bf16" ;;
esac
case "$INPUT_MODE" in
  image-embeds|images) ;;
  *) die "--input-mode must be image-embeds or images" ;;
esac
case "$CHECK" in
  true|false) ;;
  *) die "--check must be true or false" ;;
esac
case "$BUILD_ENGINE" in
  true|false) ;;
  *) die "--build-engine must be true or false" ;;
esac

export LEROBOT_SRC="${LEROBOT_SRC:-$REPO_ROOT/lerobot/src}"
export PYTHONPATH="$LEROBOT_SRC${PYTHONPATH:+:$PYTHONPATH}"

args=(
  --policy-path "$POLICY_PATH"
  --output "$OUTPUT"
  --device "$DEVICE"
  --dtype "$DTYPE"
  --input-mode "$INPUT_MODE"
  --batch-size "$BATCH_SIZE"
  --token-length "$TOKEN_LENGTH"
  --opset "$OPSET"
)
if [[ "$CHECK" == "true" ]]; then
  args+=(--check)
fi
if [[ "$BUILD_ENGINE" == "true" ]]; then
  args+=(--build-engine)
fi
if [[ -n "$ENGINE_OUTPUT" ]]; then
  args+=(--engine-output "$ENGINE_OUTPUT")
fi

"$PYTHON" "$SCRIPT_DIR/export_smolvla_sample_actions_onnx.py" "${args[@]}"
