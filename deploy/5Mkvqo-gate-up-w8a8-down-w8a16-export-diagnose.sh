#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$REPO_ROOT/scripts/common.sh"

POLICY_PATH="$REPO_ROOT/smolvla_libero"
OUTPUT_ROOT="$REPO_ROOT/runs/deploy/5/M-kvqo-gate-up-w8a8-down-w8a16-deploy"
TASK_SUITE=libero_spatial
DEVICE=cuda
SEED=1000
BATCH_SIZE=1
TOKEN_LENGTH=48
CALIBRATION_SAMPLES=512
CALIBRATION_PERCENTILE=99.999
INPUT_SOURCE=rollout
COMPARE_SAMPLES=50
SAMPLE_STRIDE=5
RUN_CALIBRATION=false
BUILD_ENGINE=true
COMPARE=true
TRT_PRECISION_CONSTRAINTS=prefer
PYTHON="${PYTHON_BIN:-python3}"

# W8A8:
#   q/k/v/o: all Text VLM layers
#   FFN gate/up: all Text VLM layers
# W8A16:
#   FFN down: all Text VLM layers
W8A8_MODULE_REGEX="^model\\.vlm_with_expert\\.vlm\\.model\\.text_model\\.layers\\.[0-9]+\\.(self_attn\\.(q_proj|k_proj|v_proj|o_proj)|mlp\\.(gate_proj|up_proj))$"
DOWN_W8A16_MODULE_REGEX="^model\\.vlm_with_expert\\.vlm\\.model\\.text_model\\.layers\\.[0-9]+\\.mlp\\.down_proj$"
W8A8_NODE_REGEX="^/(debug_core/)?(q_proj|k_proj|v_proj|o_proj|mlp/(gate_proj|up_proj))(_[0-9]+)?/MatMul$"
DOWN_W8A16_NODE_REGEX="^/(debug_core/)?mlp/down_proj(_[0-9]+)?/MatMul$"
STOP_BEFORE_ACTION_REGEX="^/action_in_proj/"
EXPECTED_W8A8_NODES=192
EXPECTED_W8A16_NODES=32

usage() {
  printf '%s\n' "Usage: $0 [options]" \
    "  --policy-path PATH             default: $POLICY_PATH" \
    "  --output-root DIR             default: $OUTPUT_ROOT" \
    "  --tasks NAME                  default: $TASK_SUITE" \
    "  --device DEVICE               default: $DEVICE" \
    "  --seed N                      default: $SEED" \
    "  --batch-size N                default: $BATCH_SIZE" \
    "  --token-length N              default: $TOKEN_LENGTH" \
    "  --calibration-samples N       default: $CALIBRATION_SAMPLES" \
    "  --calibration-percentile P    default: $CALIBRATION_PERCENTILE" \
    "  --input-source synthetic|rollout default: $INPUT_SOURCE" \
    "  --compare-samples N           default: $COMPARE_SAMPLES" \
    "  --sample-stride N             default: $SAMPLE_STRIDE" \
    "  --run-calibration true|false  default: $RUN_CALIBRATION" \
    "  --build-engine true|false     default: $BUILD_ENGINE" \
    "  --compare true|false          default: $COMPARE" \
    "  --trt-precision-constraints obey|prefer|none default: $TRT_PRECISION_CONSTRAINTS" \
    "  PYTHON_BIN=/path/python       optional Python with torch/lerobot/onnxruntime/tensorrt dependencies"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --policy-path) require_value "$@"; POLICY_PATH="$2"; shift 2 ;;
    --output-root) require_value "$@"; OUTPUT_ROOT="$2"; shift 2 ;;
    --tasks) require_value "$@"; TASK_SUITE="$2"; shift 2 ;;
    --device) require_value "$@"; DEVICE="$2"; shift 2 ;;
    --seed) require_value "$@"; SEED="$2"; shift 2 ;;
    --batch-size) require_value "$@"; BATCH_SIZE="$2"; shift 2 ;;
    --token-length) require_value "$@"; TOKEN_LENGTH="$2"; shift 2 ;;
    --calibration-samples) require_value "$@"; CALIBRATION_SAMPLES="$2"; shift 2 ;;
    --calibration-percentile) require_value "$@"; CALIBRATION_PERCENTILE="$2"; shift 2 ;;
    --input-source) require_value "$@"; INPUT_SOURCE="$2"; shift 2 ;;
    --compare-samples) require_value "$@"; COMPARE_SAMPLES="$2"; shift 2 ;;
    --sample-stride) require_value "$@"; SAMPLE_STRIDE="$2"; shift 2 ;;
    --run-calibration) require_value "$@"; RUN_CALIBRATION="$2"; shift 2 ;;
    --build-engine) require_value "$@"; BUILD_ENGINE="$2"; shift 2 ;;
    --compare) require_value "$@"; COMPARE="$2"; shift 2 ;;
    --trt-precision-constraints) require_value "$@"; TRT_PRECISION_CONSTRAINTS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

case "$INPUT_SOURCE" in synthetic|rollout) ;; *) die "--input-source must be synthetic or rollout" ;; esac
case "$RUN_CALIBRATION" in true|false) ;; *) die "--run-calibration must be true or false" ;; esac
case "$BUILD_ENGINE" in true|false) ;; *) die "--build-engine must be true or false" ;; esac
case "$COMPARE" in true|false) ;; *) die "--compare must be true or false" ;; esac
case "$TRT_PRECISION_CONSTRAINTS" in obey|prefer|none) ;; *) die "--trt-precision-constraints must be obey, prefer, or none" ;; esac

mkdir -p "$OUTPUT_ROOT"
export NUMBA_DISABLE_JIT="${NUMBA_DISABLE_JIT:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-smolvla}"
export LEROBOT_SRC="${LEROBOT_SRC:-$REPO_ROOT/lerobot/src}"
export PYTHONPATH="$SCRIPT_DIR:$LEROBOT_SRC${PYTHONPATH:+:$PYTHONPATH}"

SCALES_JSON="$OUTPUT_ROOT/activation_scales_${CALIBRATION_SAMPLES}_p${CALIBRATION_PERCENTILE}_kvqo_gate_up.json"
SOURCE_SCALES="$REPO_ROOT/runs/deploy/5/H-bf16-int8-w8a8/kvqo_gate_up_recalibrated_no_down/activation_scales_${CALIBRATION_SAMPLES}_p${CALIBRATION_PERCENTILE}.json"
BF16_ONNX="$OUTPUT_ROOT/smolvla_debug_core_native_mixed.onnx"
ORT_BF16_ONNX="$OUTPUT_ROOT/smolvla_debug_core_native_mixed_ort_fp32.onnx"
QDQ_ONNX="$OUTPUT_ROOT/smolvla_debug_core_kvqo_gate_up_w8a8_down_w8a16_qdq.onnx"
ORT_QDQ_ONNX="$OUTPUT_ROOT/smolvla_debug_core_kvqo_gate_up_w8a8_down_w8a16_qdq_ort_fp32.onnx"
ENGINE_OUTPUT="$OUTPUT_ROOT/smolvla_debug_core_kvqo_gate_up_w8a8_down_w8a16_precision_${TRT_PRECISION_CONSTRAINTS}.plan"
TRT_FLAGS=(--profilingVerbosity=detailed --timingCacheFile="$OUTPUT_ROOT/smolvla_debug_core_kvqo_gate_up_w8a8_down_w8a16_precision_${TRT_PRECISION_CONSTRAINTS}.cache")
if [[ "$TRT_PRECISION_CONSTRAINTS" != "none" ]]; then
  TRT_FLAGS=(--precisionConstraints="$TRT_PRECISION_CONSTRAINTS" "${TRT_FLAGS[@]}")
fi

if [[ "$RUN_CALIBRATION" == "true" ]]; then
  "$PYTHON" "$SCRIPT_DIR/calibrate_smolvla_activation_scales.py" \
    --policy-path "$POLICY_PATH" \
    --output "$SCALES_JSON" \
    --device "$DEVICE" \
    --tasks "$TASK_SUITE" \
    --seed "$SEED" \
    --samples "$CALIBRATION_SAMPLES" \
    --percentile "$CALIBRATION_PERCENTILE" \
    --batch-size "$BATCH_SIZE" \
    --max-parallel-tasks 1 \
    --token-length "$TOKEN_LENGTH" \
    --include-module-regex "$W8A8_MODULE_REGEX"
else
  [[ -f "$SOURCE_SCALES" ]] || die "missing source scales: $SOURCE_SCALES; rerun with --run-calibration true"
  cp "$SOURCE_SCALES" "$SCALES_JSON"
fi

"$PYTHON" "$SCRIPT_DIR/diagnose_3_4_numeric_baseline.py" export \
  --policy-path "$POLICY_PATH" \
  --device "$DEVICE" \
  --seed "$SEED" \
  --task "$TASK_SUITE" \
  --batch-size "$BATCH_SIZE" \
  --token-length "$TOKEN_LENGTH" \
  --model-dtype bf16 \
  --input-dtype bf16 \
  --output-mode debug \
  --output "$BF16_ONNX" \
  --check

"$PYTHON" "$SCRIPT_DIR/diagnose_3_4_numeric_baseline.py" make-ort-compatible \
  --policy-path "$POLICY_PATH" \
  --device "$DEVICE" \
  --input "$BF16_ONNX" \
  --output "$ORT_BF16_ONNX" \
  --check

"$PYTHON" "$SCRIPT_DIR/insert_linear_w8a8_qdq.py" \
  --input "$BF16_ONNX" \
  --output "$QDQ_ONNX" \
  --activation-scale-mode calibrated \
  --activation-scales-json "$SCALES_JSON" \
  --include-module-regex "$W8A8_MODULE_REGEX" \
  --include-node-regex "$W8A8_NODE_REGEX" \
  --stop-before-node-regex "$STOP_BEFORE_ACTION_REGEX" \
  --weight-only-include-node-regex "$DOWN_W8A16_NODE_REGEX" \
  --cast-output-to bf16 \
  --check

"$PYTHON" - "$QDQ_ONNX" "$EXPECTED_W8A8_NODES" "$EXPECTED_W8A16_NODES" <<'PY'
import json
import sys
from pathlib import Path

report_path = Path(sys.argv[1]).with_suffix(".qdq_report.json")
expected_w8a8 = int(sys.argv[2])
expected_w8a16 = int(sys.argv[3])
report = json.loads(report_path.read_text())
actual_w8a8 = int(report.get("rewritten_linear_nodes", -1))
actual_w8a16 = int(report.get("rewritten_weight_only_linear_nodes", -1))
casts = int(report.get("output_cast_nodes", -1))
expected_casts = expected_w8a8 + expected_w8a16
if actual_w8a8 != expected_w8a8 or actual_w8a16 != expected_w8a16 or casts != expected_casts:
    raise SystemExit(
        f"unexpected 5M Q/DQ coverage in {report_path}: "
        f"rewritten_linear_nodes={actual_w8a8}, expected={expected_w8a8}; "
        f"rewritten_weight_only_linear_nodes={actual_w8a16}, expected={expected_w8a16}; "
        f"output_cast_nodes={casts}, expected={expected_casts}"
    )
PY

"$PYTHON" "$SCRIPT_DIR/diagnose_3_4_numeric_baseline.py" make-ort-compatible \
  --policy-path "$POLICY_PATH" \
  --device "$DEVICE" \
  --input "$QDQ_ONNX" \
  --output "$ORT_QDQ_ONNX" \
  --check

if [[ "$BUILD_ENGINE" == "true" ]]; then
  command -v trtexec >/dev/null 2>&1 || die "trtexec not found; rerun with --build-engine false after building manually"
  trtexec \
    --onnx="$QDQ_ONNX" \
    --saveEngine="$ENGINE_OUTPUT" \
    "${TRT_FLAGS[@]}"
  [[ -s "$ENGINE_OUTPUT" ]] || die "TensorRT produced empty engine: $ENGINE_OUTPUT"
else
  [[ -f "$ENGINE_OUTPUT" ]] || die "$ENGINE_OUTPUT not found; enable --build-engine true"
fi

if [[ "$COMPARE" == "true" ]]; then
  "$PYTHON" "$SCRIPT_DIR/diagnose_3_4_numeric_baseline.py" compare \
    --policy-path "$POLICY_PATH" \
    --device "$DEVICE" \
    --seed "$SEED" \
    --task "$TASK_SUITE" \
    --batch-size "$BATCH_SIZE" \
    --token-length "$TOKEN_LENGTH" \
    --model-dtype bf16 \
    --input-dtype bf16 \
    --input-source "$INPUT_SOURCE" \
    --compare-samples "$COMPARE_SAMPLES" \
    --sample-stride "$SAMPLE_STRIDE" \
    --max-parallel-tasks 1 \
    --bf16-onnx "$BF16_ONNX" \
    --ort-bf16-onnx "$ORT_BF16_ONNX" \
    --int8-onnx "$QDQ_ONNX" \
    --ort-int8-onnx "$ORT_QDQ_ONNX" \
    --int8-engine "$ENGINE_OUTPUT" \
    --quantize-module-regex "$W8A8_MODULE_REGEX" \
    --fake-quant-kind w8a8 \
    --fake-quant-activation-scale-mode calibrated \
    --fake-quant-activation-scales-json "$SCALES_JSON" \
    --extra-w8a16-module-regex "$DOWN_W8A16_MODULE_REGEX" \
    --output-dir "$OUTPUT_ROOT/numeric_baseline"
fi

printf '%s\n' "step 5M KVQO+gate/up W8A8 + down W8A16 deploy diagnosis outputs:" \
  "  output root: $OUTPUT_ROOT" \
  "  activation scales: $SCALES_JSON" \
  "  native mixed ONNX: $BF16_ONNX" \
  "  ORT native mixed ONNX: $ORT_BF16_ONNX" \
  "  mixed Q/DQ ONNX: $QDQ_ONNX" \
  "  ORT mixed Q/DQ ONNX: $ORT_QDQ_ONNX" \
  "  TensorRT engine: $ENGINE_OUTPUT" \
  "  Q/DQ report: ${QDQ_ONNX%.onnx}.qdq_report.json" \
  "  numeric summary: $OUTPUT_ROOT/numeric_baseline/numeric_baseline_summary.json"
