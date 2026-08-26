#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$REPO_ROOT/scripts/common.sh"

POLICY_PATH="$REPO_ROOT/smolvla_libero"
OUTPUT_ROOT="$REPO_ROOT/runs/deploy/5/H-bf16-int8-w8a8"
TASK_SUITE=libero_spatial
DEVICE=cuda
SEED=1000
BATCH_SIZE=1
TOKEN_LENGTH=48
CALIBRATION_SAMPLES=512
CALIBRATION_PERCENTILE=99.995
COMPARE_SAMPLES=50
SAMPLE_STRIDE=5
STAGE=all
RUN_CALIBRATION=true
TRT_PRECISION_CONSTRAINTS=prefer
PYTHON="${PYTHON_BIN:-python3}"

QUANT_MODULE_REGEX="^model\\.vlm_with_expert\\.vlm\\.model\\.text_model\\.layers\\.[0-9]+\\.(mlp\\.(gate_proj|up_proj|down_proj)|self_attn\\.(q_proj|k_proj|v_proj|o_proj))$"
QUANT_NODE_REGEX="^/(debug_core/)?(q_proj|k_proj|v_proj|o_proj|mlp/(gate_proj|up_proj|down_proj))(_([1-9]|[12][0-9]|3[01]))?/MatMul$"
EXPECTED_LINEAR_NODES=224

usage() {
  printf '%s\n' "Usage: $0 [options]" \
    "  --stage fake|onnx|engine|all       default: $STAGE" \
    "  --policy-path PATH                 default: $POLICY_PATH" \
    "  --output-root DIR                 default: $OUTPUT_ROOT" \
    "  --tasks NAME                      default: $TASK_SUITE" \
    "  --device DEVICE                   default: $DEVICE" \
    "  --seed N                          default: $SEED" \
    "  --batch-size N                    default: $BATCH_SIZE" \
    "  --token-length N                  default: $TOKEN_LENGTH" \
    "  --calibration-samples N           default: $CALIBRATION_SAMPLES" \
    "  --calibration-percentile P        default: $CALIBRATION_PERCENTILE" \
    "  --compare-samples N               default: $COMPARE_SAMPLES" \
    "  --sample-stride N                 default: $SAMPLE_STRIDE" \
    "  --run-calibration true|false      default: $RUN_CALIBRATION" \
    "  --trt-precision-constraints obey|prefer|none default: $TRT_PRECISION_CONSTRAINTS" \
    "  PYTHON_BIN=/path/python           optional Python with torch/lerobot/tensorrt dependencies"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage) require_value "$@"; STAGE="$2"; shift 2 ;;
    --policy-path) require_value "$@"; POLICY_PATH="$2"; shift 2 ;;
    --output-root) require_value "$@"; OUTPUT_ROOT="$2"; shift 2 ;;
    --tasks) require_value "$@"; TASK_SUITE="$2"; shift 2 ;;
    --device) require_value "$@"; DEVICE="$2"; shift 2 ;;
    --seed) require_value "$@"; SEED="$2"; shift 2 ;;
    --batch-size) require_value "$@"; BATCH_SIZE="$2"; shift 2 ;;
    --token-length) require_value "$@"; TOKEN_LENGTH="$2"; shift 2 ;;
    --calibration-samples) require_value "$@"; CALIBRATION_SAMPLES="$2"; shift 2 ;;
    --calibration-percentile) require_value "$@"; CALIBRATION_PERCENTILE="$2"; shift 2 ;;
    --compare-samples) require_value "$@"; COMPARE_SAMPLES="$2"; shift 2 ;;
    --sample-stride) require_value "$@"; SAMPLE_STRIDE="$2"; shift 2 ;;
    --run-calibration) require_value "$@"; RUN_CALIBRATION="$2"; shift 2 ;;
    --trt-precision-constraints) require_value "$@"; TRT_PRECISION_CONSTRAINTS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

case "$STAGE" in fake|onnx|engine|all) ;; *) die "--stage must be fake, onnx, engine, or all" ;; esac
case "$RUN_CALIBRATION" in true|false) ;; *) die "--run-calibration must be true or false" ;; esac
case "$TRT_PRECISION_CONSTRAINTS" in obey|prefer|none) ;; *) die "--trt-precision-constraints must be obey, prefer, or none" ;; esac

mkdir -p "$OUTPUT_ROOT"
export LEROBOT_SRC="${LEROBOT_SRC:-$REPO_ROOT/lerobot/src}"
export PYTHONPATH="$SCRIPT_DIR:$LEROBOT_SRC${PYTHONPATH:+:$PYTHONPATH}"

SCALES_JSON="$OUTPUT_ROOT/activation_scales_${CALIBRATION_SAMPLES}_p${CALIBRATION_PERCENTILE}_scalar_bf16.json"
BF16_ONNX="$OUTPUT_ROOT/smolvla_debug_core_bf16.onnx"
ORT_BF16_ONNX="$OUTPUT_ROOT/smolvla_debug_core_bf16_ort.onnx"
QDQ_ONNX="$OUTPUT_ROOT/smolvla_debug_core_text_vlm_full_w8a8_qdq.onnx"
ORT_QDQ_ONNX="$OUTPUT_ROOT/smolvla_debug_core_text_vlm_full_w8a8_qdq_ort.onnx"
INT8_ENGINE="$OUTPUT_ROOT/smolvla_debug_core_text_vlm_full_w8a8_cast_bf16_precision_${TRT_PRECISION_CONSTRAINTS}.plan"
TRT_FLAGS=(--profilingVerbosity=detailed --timingCacheFile="$OUTPUT_ROOT/smolvla_debug_core_text_vlm_full_w8a8_precision_${TRT_PRECISION_CONSTRAINTS}.cache")
if [[ "$TRT_PRECISION_CONSTRAINTS" != "none" ]]; then
  TRT_FLAGS=(--precisionConstraints="$TRT_PRECISION_CONSTRAINTS" "${TRT_FLAGS[@]}")
fi

maybe_calibrate() {
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
      --include-module-regex "$QUANT_MODULE_REGEX"
  else
    [[ -f "$SCALES_JSON" ]] || die "$SCALES_JSON not found; use --run-calibration true first"
  fi
}

run_fake() {
  maybe_calibrate
  "$PYTHON" "$SCRIPT_DIR/diagnose_3_4_numeric_baseline.py" compare \
    --policy-path "$POLICY_PATH" \
    --device "$DEVICE" \
    --seed "$SEED" \
    --task "$TASK_SUITE" \
    --batch-size "$BATCH_SIZE" \
    --token-length "$TOKEN_LENGTH" \
    --model-dtype bf16 \
    --input-dtype bf16 \
    --input-source rollout \
    --compare-samples "$COMPARE_SAMPLES" \
    --sample-stride "$SAMPLE_STRIDE" \
    --max-parallel-tasks 1 \
    --quantize-module-regex "$QUANT_MODULE_REGEX" \
    --fake-quant-kind w8a8 \
    --fake-quant-activation-scale-mode calibrated \
    --fake-quant-activation-scales-json "$SCALES_JSON" \
    --output-dir "$OUTPUT_ROOT/fake_dequant_numeric"
}

run_onnx() {
  [[ -f "$SCALES_JSON" ]] || maybe_calibrate
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
    --input "$BF16_ONNX" \
    --output "$ORT_BF16_ONNX" \
    --check

  "$PYTHON" "$SCRIPT_DIR/insert_linear_w8a8_qdq.py" \
    --input "$BF16_ONNX" \
    --output "$QDQ_ONNX" \
    --activation-scale-mode calibrated \
    --activation-scales-json "$SCALES_JSON" \
    --include-module-regex "$QUANT_MODULE_REGEX" \
    --include-node-regex "$QUANT_NODE_REGEX" \
    --cast-output-to bf16 \
    --check

  "$PYTHON" - "$QDQ_ONNX" "$EXPECTED_LINEAR_NODES" <<'PY'
import json
import sys
from pathlib import Path

onnx_path = Path(sys.argv[1])
expected = int(sys.argv[2])
report_path = onnx_path.with_suffix(".qdq_report.json")
report = json.loads(report_path.read_text())
actual = int(report.get("rewritten_linear_nodes", -1))
casts = int(report.get("output_cast_nodes", -1))
weight_only = int(report.get("weight_only_rewritten_linear_nodes", 0))
if actual != expected or casts != expected or weight_only != 0:
    raise SystemExit(
        f"unexpected 5H Q/DQ coverage in {report_path}: "
        f"rewritten_linear_nodes={actual}, output_cast_nodes={casts}, "
        f"weight_only_rewritten_linear_nodes={weight_only}, expected={expected}"
    )
PY

  "$PYTHON" "$SCRIPT_DIR/diagnose_3_4_numeric_baseline.py" make-ort-compatible \
    --input "$QDQ_ONNX" \
    --output "$ORT_QDQ_ONNX" \
    --check

  "$PYTHON" "$SCRIPT_DIR/diagnose_3_4_numeric_baseline.py" compare \
    --policy-path "$POLICY_PATH" \
    --device "$DEVICE" \
    --seed "$SEED" \
    --task "$TASK_SUITE" \
    --batch-size "$BATCH_SIZE" \
    --token-length "$TOKEN_LENGTH" \
    --model-dtype bf16 \
    --input-dtype bf16 \
    --input-source rollout \
    --compare-samples "$COMPARE_SAMPLES" \
    --sample-stride "$SAMPLE_STRIDE" \
    --max-parallel-tasks 1 \
    --bf16-onnx "$BF16_ONNX" \
    --ort-bf16-onnx "$ORT_BF16_ONNX" \
    --int8-onnx "$QDQ_ONNX" \
    --ort-int8-onnx "$ORT_QDQ_ONNX" \
    --quantize-module-regex "$QUANT_MODULE_REGEX" \
    --fake-quant-kind w8a8 \
    --fake-quant-activation-scale-mode calibrated \
    --fake-quant-activation-scales-json "$SCALES_JSON" \
    --output-dir "$OUTPUT_ROOT/onnx_numeric"
}

run_engine() {
  [[ -f "$QDQ_ONNX" ]] || die "$QDQ_ONNX not found; run --stage onnx first"
  trtexec \
    --onnx="$QDQ_ONNX" \
    --saveEngine="$INT8_ENGINE" \
    "${TRT_FLAGS[@]}" \
    > "$OUTPUT_ROOT/trtexec_int8_build.log" 2>&1

  "$PYTHON" "$SCRIPT_DIR/diagnose_3_4_numeric_baseline.py" compare \
    --policy-path "$POLICY_PATH" \
    --device "$DEVICE" \
    --seed "$SEED" \
    --task "$TASK_SUITE" \
    --batch-size "$BATCH_SIZE" \
    --token-length "$TOKEN_LENGTH" \
    --model-dtype bf16 \
    --input-dtype bf16 \
    --input-source rollout \
    --compare-samples "$COMPARE_SAMPLES" \
    --sample-stride "$SAMPLE_STRIDE" \
    --max-parallel-tasks 1 \
    --bf16-onnx "$BF16_ONNX" \
    --ort-bf16-onnx "$ORT_BF16_ONNX" \
    --int8-onnx "$QDQ_ONNX" \
    --ort-int8-onnx "$ORT_QDQ_ONNX" \
    --int8-engine "$INT8_ENGINE" \
    --quantize-module-regex "$QUANT_MODULE_REGEX" \
    --fake-quant-kind w8a8 \
    --fake-quant-activation-scale-mode calibrated \
    --fake-quant-activation-scales-json "$SCALES_JSON" \
    --output-dir "$OUTPUT_ROOT/engine_numeric"

  "$SCRIPT_DIR/5Dinspect-int8-kernels.sh" \
    --engine "$INT8_ENGINE" \
    --output-root "$OUTPUT_ROOT/inspect"
}

case "$STAGE" in
  fake) run_fake ;;
  onnx) run_onnx ;;
  engine) run_engine ;;
  all)
    run_fake
    run_onnx
    run_engine
    ;;
esac

printf '%s\n' "step 5H outputs:" \
  "  scales: $SCALES_JSON" \
  "  fake-dequant numeric: $OUTPUT_ROOT/fake_dequant_numeric/numeric_baseline_summary.json" \
  "  BF16 debug ONNX: $BF16_ONNX" \
  "  Q/DQ debug ONNX: $QDQ_ONNX" \
  "  ONNX numeric: $OUTPUT_ROOT/onnx_numeric/numeric_baseline_summary.json" \
  "  TensorRT engine: $INT8_ENGINE" \
  "  engine numeric: $OUTPUT_ROOT/engine_numeric/numeric_baseline_summary.json" \
  "  precision inspect: $OUTPUT_ROOT/inspect/layer_precision_summary.json"
