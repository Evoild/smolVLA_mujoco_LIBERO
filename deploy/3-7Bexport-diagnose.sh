#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$REPO_ROOT/scripts/common.sh"

POLICY_PATH="$REPO_ROOT/smolvla_libero"
OUTPUT_ROOT="$REPO_ROOT/runs/deploy/3-7/B-formal-quant-deploy"
TASK_SUITE=libero_spatial
DEVICE=cuda
SEED=1000
CALIBRATION_SAMPLES=32
BATCH_SIZE=1
TOKEN_LENGTH=48
INPUT_DTYPE=fp32
PERCENTILE=99.99
RUN_CALIBRATION=true
BUILD_ENGINE=true
COMPARE=true
PYTHON="${PYTHON_BIN:-python3}"

GATE_UP_MODULE_REGEX="^model\\.vlm_with_expert\\.vlm\\.model\\.text_model\\.layers\\.[0-9]+\\.mlp\\.(gate_proj|up_proj)$"
DOWN_MODULE_REGEX="^model\\.vlm_with_expert\\.vlm\\.model\\.text_model\\.layers\\.[0-9]+\\.mlp\\.down_proj$"
GATE_UP_NODE_REGEX="^/mlp/(gate_proj|up_proj)(_[0-9]+)?/MatMul$"
DOWN_NODE_REGEX="^/mlp/down_proj(_[0-9]+)?/MatMul$"
STOP_BEFORE_ACTION_REGEX="^/action_in_proj/"

usage() {
  printf '%s\n' "Usage: $0 [options]" \
    "  --policy-path PATH             default: $POLICY_PATH" \
    "  --output-root DIR             default: $OUTPUT_ROOT" \
    "  --tasks NAME                  default: $TASK_SUITE" \
    "  --device DEVICE               default: $DEVICE" \
    "  --seed N                      default: $SEED" \
    "  --calibration-samples N       default: $CALIBRATION_SAMPLES" \
    "  --batch-size N                default: $BATCH_SIZE" \
    "  --token-length N              default: $TOKEN_LENGTH" \
    "  --input-dtype fp32|bf16       default: $INPUT_DTYPE" \
    "  --percentile P                default: $PERCENTILE" \
    "  --run-calibration true|false  default: $RUN_CALIBRATION" \
    "  --build-engine true|false     default: $BUILD_ENGINE" \
    "  --compare true|false          default: $COMPARE" \
    "  PYTHON_BIN=/path/python       optional Python with torch/lerobot/onnxruntime/tensorrt dependencies"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --policy-path) require_value "$@"; POLICY_PATH="$2"; shift 2 ;;
    --output-root) require_value "$@"; OUTPUT_ROOT="$2"; shift 2 ;;
    --tasks) require_value "$@"; TASK_SUITE="$2"; shift 2 ;;
    --device) require_value "$@"; DEVICE="$2"; shift 2 ;;
    --seed) require_value "$@"; SEED="$2"; shift 2 ;;
    --calibration-samples) require_value "$@"; CALIBRATION_SAMPLES="$2"; shift 2 ;;
    --batch-size) require_value "$@"; BATCH_SIZE="$2"; shift 2 ;;
    --token-length) require_value "$@"; TOKEN_LENGTH="$2"; shift 2 ;;
    --input-dtype) require_value "$@"; INPUT_DTYPE="$2"; shift 2 ;;
    --percentile) require_value "$@"; PERCENTILE="$2"; shift 2 ;;
    --run-calibration) require_value "$@"; RUN_CALIBRATION="$2"; shift 2 ;;
    --build-engine) require_value "$@"; BUILD_ENGINE="$2"; shift 2 ;;
    --compare) require_value "$@"; COMPARE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

case "$INPUT_DTYPE" in fp32|bf16) ;; *) die "--input-dtype must be fp32 or bf16" ;; esac
case "$RUN_CALIBRATION" in true|false) ;; *) die "--run-calibration must be true or false" ;; esac
case "$BUILD_ENGINE" in true|false) ;; *) die "--build-engine must be true or false" ;; esac
case "$COMPARE" in true|false) ;; *) die "--compare must be true or false" ;; esac

mkdir -p "$OUTPUT_ROOT"
SCALES_JSON="$OUTPUT_ROOT/gate_up_activation_channel_scales.json"
BF16_ONNX="$OUTPUT_ROOT/smolvla_debug_core_native_mixed.onnx"
ORT_BF16_ONNX="$OUTPUT_ROOT/smolvla_debug_core_native_mixed_ort_fp32.onnx"
QDQ_ONNX="$OUTPUT_ROOT/smolvla_debug_core_gate_up_w8a8_down_w8a16_qdq.onnx"
ORT_QDQ_ONNX="$OUTPUT_ROOT/smolvla_debug_core_gate_up_w8a8_down_w8a16_qdq_ort_fp32.onnx"
ENGINE_OUTPUT="$OUTPUT_ROOT/smolvla_debug_core_gate_up_w8a8_down_w8a16_precision_obey.plan"

export LEROBOT_SRC="${LEROBOT_SRC:-$REPO_ROOT/lerobot/src}"
export PYTHONPATH="$SCRIPT_DIR:$LEROBOT_SRC${PYTHONPATH:+:$PYTHONPATH}"

if [[ "$RUN_CALIBRATION" == "true" ]]; then
  "$PYTHON" "$SCRIPT_DIR/calibrate_smolvla_activation_channel_scales.py" \
    --policy-path "$POLICY_PATH" \
    --output "$SCALES_JSON" \
    --device "$DEVICE" \
    --tasks "$TASK_SUITE" \
    --seed "$SEED" \
    --samples "$CALIBRATION_SAMPLES" \
    --percentile "$PERCENTILE" \
    --batch-size "$BATCH_SIZE" \
    --token-length "$TOKEN_LENGTH" \
    --include-module-regex "$GATE_UP_MODULE_REGEX"
else
  [[ -f "$SCALES_JSON" ]] || die "$SCALES_JSON not found; enable --run-calibration true"
fi

"$PYTHON" "$SCRIPT_DIR/diagnose_3_4_numeric_baseline.py" export \
  --policy-path "$POLICY_PATH" \
  --device "$DEVICE" \
  --seed "$SEED" \
  --task "$TASK_SUITE step 3-7B numeric diagnosis" \
  --batch-size "$BATCH_SIZE" \
  --token-length "$TOKEN_LENGTH" \
  --input-dtype "$INPUT_DTYPE" \
  --output "$BF16_ONNX"

"$PYTHON" "$SCRIPT_DIR/diagnose_3_4_numeric_baseline.py" make-ort-compatible \
  --policy-path "$POLICY_PATH" \
  --device "$DEVICE" \
  --input "$BF16_ONNX" \
  --output "$ORT_BF16_ONNX"

"$PYTHON" "$SCRIPT_DIR/insert_linear_w8a8_qdq.py" \
  --input "$BF16_ONNX" \
  --output "$QDQ_ONNX" \
  --activation-scale-mode calibrated \
  --activation-scales-json "$SCALES_JSON" \
  --include-module-regex "$GATE_UP_MODULE_REGEX" \
  --include-node-regex "$GATE_UP_NODE_REGEX" \
  --stop-before-node-regex "$STOP_BEFORE_ACTION_REGEX" \
  --weight-only-include-node-regex "$DOWN_NODE_REGEX"

"$PYTHON" - "$QDQ_ONNX" <<'PY'
import json
import sys
from pathlib import Path

report_path = Path(sys.argv[1]).with_suffix(".qdq_report.json")
with open(report_path) as f:
    report = json.load(f)

expected = {
    "rewritten_linear_nodes": 64,
    "rewritten_weight_only_linear_nodes": 32,
}
bad = {key: (report.get(key), value) for key, value in expected.items() if report.get(key) != value}
if bad:
    details = ", ".join(f"{key}: got {got}, expected {want}" for key, (got, want) in bad.items())
    raise SystemExit(f"unexpected 3-7B Q/DQ coverage in {report_path}: {details}")
PY

"$PYTHON" "$SCRIPT_DIR/diagnose_3_4_numeric_baseline.py" make-ort-compatible \
  --policy-path "$POLICY_PATH" \
  --device "$DEVICE" \
  --input "$QDQ_ONNX" \
  --output "$ORT_QDQ_ONNX"

if [[ "$BUILD_ENGINE" == "true" ]]; then
  command -v trtexec >/dev/null 2>&1 || die "trtexec not found; rerun with --build-engine false after building manually"
  trtexec \
    --onnx="$QDQ_ONNX" \
    --saveEngine="$ENGINE_OUTPUT" \
    --precisionConstraints=obey \
    --profilingVerbosity=detailed \
    --timingCacheFile="$OUTPUT_ROOT/smolvla_debug_gate_up_w8a8_down_w8a16_precision_obey.cache"
  [[ -s "$ENGINE_OUTPUT" ]] || die "TensorRT produced empty engine: $ENGINE_OUTPUT"
else
  [[ -f "$ENGINE_OUTPUT" ]] || die "$ENGINE_OUTPUT not found; enable --build-engine true"
fi

if [[ "$COMPARE" == "true" ]]; then
  "$PYTHON" "$SCRIPT_DIR/diagnose_3_4_numeric_baseline.py" compare \
    --policy-path "$POLICY_PATH" \
    --device "$DEVICE" \
    --seed "$SEED" \
    --task "$TASK_SUITE step 3-7B numeric diagnosis" \
    --batch-size "$BATCH_SIZE" \
    --token-length "$TOKEN_LENGTH" \
    --input-dtype "$INPUT_DTYPE" \
    --bf16-onnx "$BF16_ONNX" \
    --ort-bf16-onnx "$ORT_BF16_ONNX" \
    --int8-onnx "$QDQ_ONNX" \
    --ort-int8-onnx "$ORT_QDQ_ONNX" \
    --int8-engine "$ENGINE_OUTPUT" \
    --quantize-module-regex "$GATE_UP_MODULE_REGEX" \
    --fake-quant-kind w8a8 \
    --fake-quant-activation-scale-mode calibrated \
    --fake-quant-activation-scales-json "$SCALES_JSON" \
    --extra-w8a16-module-regex "$DOWN_MODULE_REGEX" \
    --output-dir "$OUTPUT_ROOT/numeric_baseline"
fi

printf '%s\n' "step 3-7B stage 1 outputs:" \
  "  native mixed ONNX: $BF16_ONNX" \
  "  ORT native mixed ONNX: $ORT_BF16_ONNX" \
  "  mixed Q/DQ ONNX: $QDQ_ONNX" \
  "  ORT mixed Q/DQ ONNX: $ORT_QDQ_ONNX" \
  "  TensorRT engine: $ENGINE_OUTPUT" \
  "  numeric baseline: $OUTPUT_ROOT/numeric_baseline" \
  "  Q/DQ report: ${QDQ_ONNX%.onnx}.qdq_report.json"
