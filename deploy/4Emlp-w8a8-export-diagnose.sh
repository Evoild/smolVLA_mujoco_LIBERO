#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$REPO_ROOT/scripts/common.sh"

POLICY_PATH="$REPO_ROOT/smolvla_libero"
OUTPUT_ROOT="$REPO_ROOT/runs/deploy/4/E-mlp-w8a8-formal-deploy"
REUSE_BF16_ONNX=""
REUSE_BF16_ENGINE=""
SCALES_JSON="$REPO_ROOT/runs/deploy/4/D-shared-mlp-calibration-heldout/activation_scales_best.json"
QUANT_LABEL=vlm_mlp_w8a8
TASK_SUITE=libero_spatial
DEVICE=cuda
SEED=1000
BATCH_SIZE=1
TOKEN_LENGTH=48
INPUT_DTYPE=fp32
EXPORT_OUTPUT_MODE=debug
CAST_OUTPUT_TO=bf16
TRT_PRECISION_CONSTRAINTS=obey
BUILD_ENGINE=true
BUILD_BF16_ENGINE=true
COMPARE=true
COMPARE_INPUT_SOURCE=rollout
COMPARE_SAMPLES=50
SAMPLE_STRIDE=5
MAX_PARALLEL_TASKS=1
PYTHON="${PYTHON_BIN:-python3}"

MLP_MODULE_REGEX="^model\\.vlm_with_expert\\.vlm\\.model\\.text_model\\.layers\\.[0-9]+\\.mlp\\.(gate_proj|up_proj|down_proj)$"
MLP_NODE_REGEX="^/mlp/(gate_proj|up_proj|down_proj)(_[0-9]+)?/MatMul$"
EXPECTED_LINEAR_NODES=96
STOP_BEFORE_ACTION_REGEX="^/(debug_core/)?action_in_proj/"

usage() {
  printf '%s\n' "Usage: $0 [options]" \
    "  --policy-path PATH           default: $POLICY_PATH" \
    "  --output-root DIR           default: $OUTPUT_ROOT" \
    "  --reuse-bf16-onnx PATH       reuse an existing native mixed ONNX instead of exporting a new one" \
    "  --reuse-bf16-engine PATH     reuse an existing native mixed TensorRT engine instead of building a new one" \
    "  --scales-json PATH          default: $SCALES_JSON" \
    "  --quant-label NAME          default: $QUANT_LABEL" \
    "  --quant-module-regex REGEX  default: VLM text_model MLP gate/up/down" \
    "  --quant-node-regex REGEX    default: ONNX MLP gate/up/down MatMul" \
    "  --expected-linear-nodes N   default: $EXPECTED_LINEAR_NODES" \
    "  --tasks NAME                default: $TASK_SUITE" \
    "  --device DEVICE             default: $DEVICE" \
    "  --seed N                    default: $SEED" \
    "  --batch-size N              default: $BATCH_SIZE" \
    "  --token-length N            default: $TOKEN_LENGTH" \
    "  --input-dtype fp32|bf16     default: $INPUT_DTYPE" \
    "  --export-output-mode debug|action-only default: $EXPORT_OUTPUT_MODE" \
    "  --cast-output-to none|bf16|fp16|fp32 default: $CAST_OUTPUT_TO" \
    "  --trt-precision-constraints obey|prefer|none default: $TRT_PRECISION_CONSTRAINTS" \
    "  --build-engine true|false   default: $BUILD_ENGINE" \
    "  --build-bf16-engine true|false default: $BUILD_BF16_ENGINE" \
    "  --compare true|false        default: $COMPARE" \
    "  --compare-input-source synthetic|rollout default: $COMPARE_INPUT_SOURCE" \
    "  --compare-samples N         default: $COMPARE_SAMPLES" \
    "  --sample-stride N           default: $SAMPLE_STRIDE" \
    "  --max-parallel-tasks N      default: $MAX_PARALLEL_TASKS" \
    "  PYTHON_BIN=/path/python     optional Python with torch/onnxruntime/tensorrt dependencies"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --policy-path) require_value "$@"; POLICY_PATH="$2"; shift 2 ;;
    --output-root) require_value "$@"; OUTPUT_ROOT="$2"; shift 2 ;;
    --reuse-bf16-onnx) require_value "$@"; REUSE_BF16_ONNX="$2"; shift 2 ;;
    --reuse-bf16-engine) require_value "$@"; REUSE_BF16_ENGINE="$2"; shift 2 ;;
    --scales-json) require_value "$@"; SCALES_JSON="$2"; shift 2 ;;
    --quant-label) require_value "$@"; QUANT_LABEL="$2"; shift 2 ;;
    --quant-module-regex) require_value "$@"; MLP_MODULE_REGEX="$2"; shift 2 ;;
    --quant-node-regex) require_value "$@"; MLP_NODE_REGEX="$2"; shift 2 ;;
    --expected-linear-nodes) require_value "$@"; EXPECTED_LINEAR_NODES="$2"; shift 2 ;;
    --tasks) require_value "$@"; TASK_SUITE="$2"; shift 2 ;;
    --device) require_value "$@"; DEVICE="$2"; shift 2 ;;
    --seed) require_value "$@"; SEED="$2"; shift 2 ;;
    --batch-size) require_value "$@"; BATCH_SIZE="$2"; shift 2 ;;
    --token-length) require_value "$@"; TOKEN_LENGTH="$2"; shift 2 ;;
    --input-dtype) require_value "$@"; INPUT_DTYPE="$2"; shift 2 ;;
    --export-output-mode) require_value "$@"; EXPORT_OUTPUT_MODE="$2"; shift 2 ;;
    --cast-output-to) require_value "$@"; CAST_OUTPUT_TO="$2"; shift 2 ;;
    --trt-precision-constraints) require_value "$@"; TRT_PRECISION_CONSTRAINTS="$2"; shift 2 ;;
    --build-engine) require_value "$@"; BUILD_ENGINE="$2"; shift 2 ;;
    --build-bf16-engine) require_value "$@"; BUILD_BF16_ENGINE="$2"; shift 2 ;;
    --compare) require_value "$@"; COMPARE="$2"; shift 2 ;;
    --compare-input-source) require_value "$@"; COMPARE_INPUT_SOURCE="$2"; shift 2 ;;
    --compare-samples) require_value "$@"; COMPARE_SAMPLES="$2"; shift 2 ;;
    --sample-stride) require_value "$@"; SAMPLE_STRIDE="$2"; shift 2 ;;
    --max-parallel-tasks) require_value "$@"; MAX_PARALLEL_TASKS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

case "$INPUT_DTYPE" in fp32|bf16) ;; *) die "--input-dtype must be fp32 or bf16" ;; esac
case "$EXPORT_OUTPUT_MODE" in debug|action-only) ;; *) die "--export-output-mode must be debug or action-only" ;; esac
case "$CAST_OUTPUT_TO" in none|bf16|fp16|fp32) ;; *) die "--cast-output-to must be none, bf16, fp16, or fp32" ;; esac
case "$TRT_PRECISION_CONSTRAINTS" in obey|prefer|none) ;; *) die "--trt-precision-constraints must be obey, prefer, or none" ;; esac
case "$BUILD_ENGINE" in true|false) ;; *) die "--build-engine must be true or false" ;; esac
case "$BUILD_BF16_ENGINE" in true|false) ;; *) die "--build-bf16-engine must be true or false" ;; esac
case "$COMPARE" in true|false) ;; *) die "--compare must be true or false" ;; esac
case "$COMPARE_INPUT_SOURCE" in synthetic|rollout) ;; *) die "--compare-input-source must be synthetic or rollout" ;; esac

mkdir -p "$OUTPUT_ROOT"
BF16_ONNX="$OUTPUT_ROOT/smolvla_debug_core_native_mixed.onnx"
ORT_BF16_ONNX="$OUTPUT_ROOT/smolvla_debug_core_native_mixed_ort_fp32.onnx"
QDQ_ONNX="$OUTPUT_ROOT/smolvla_debug_core_vlm_mlp_w8a8_qdq.onnx"
ORT_QDQ_ONNX="$OUTPUT_ROOT/smolvla_debug_core_vlm_mlp_w8a8_qdq_ort_fp32.onnx"
BF16_ENGINE="$OUTPUT_ROOT/smolvla_debug_core_native_mixed_precision_${TRT_PRECISION_CONSTRAINTS}.plan"
MLP_W8A8_ENGINE="$OUTPUT_ROOT/smolvla_debug_core_vlm_mlp_w8a8_precision_obey.plan"
QDQ_ONNX="$OUTPUT_ROOT/smolvla_debug_core_${QUANT_LABEL}_qdq.onnx"
ORT_QDQ_ONNX="$OUTPUT_ROOT/smolvla_debug_core_${QUANT_LABEL}_qdq_ort_fp32.onnx"
MLP_W8A8_ENGINE="$OUTPUT_ROOT/smolvla_debug_core_${QUANT_LABEL}_precision_${TRT_PRECISION_CONSTRAINTS}.plan"

export LEROBOT_SRC="${LEROBOT_SRC:-$REPO_ROOT/lerobot/src}"
export PYTHONPATH="$SCRIPT_DIR:$LEROBOT_SRC${PYTHONPATH:+:$PYTHONPATH}"

[[ -f "$SCALES_JSON" ]] || die "$SCALES_JSON not found; run deploy/4Dshared-mlp-calibration-heldout.sh first"
if [[ -n "$REUSE_BF16_ONNX" ]]; then
  [[ -f "$REUSE_BF16_ONNX" ]] || die "$REUSE_BF16_ONNX not found"
  BF16_ONNX="$REUSE_BF16_ONNX"
fi
if [[ -n "$REUSE_BF16_ENGINE" ]]; then
  [[ -f "$REUSE_BF16_ENGINE" ]] || die "$REUSE_BF16_ENGINE not found"
  BF16_ENGINE="$REUSE_BF16_ENGINE"
  BUILD_BF16_ENGINE=false
fi

if [[ -z "$REUSE_BF16_ONNX" ]]; then
  "$PYTHON" "$SCRIPT_DIR/diagnose_3_4_numeric_baseline.py" export \
    --policy-path "$POLICY_PATH" \
    --device "$DEVICE" \
    --seed "$SEED" \
    --task "$TASK_SUITE step 4E formal MLP W8A8 deployment" \
    --batch-size "$BATCH_SIZE" \
    --token-length "$TOKEN_LENGTH" \
    --input-dtype "$INPUT_DTYPE" \
    --output-mode "$EXPORT_OUTPUT_MODE" \
    --output "$BF16_ONNX"
fi

if [[ "$COMPARE" == "true" ]]; then
  "$PYTHON" "$SCRIPT_DIR/diagnose_3_4_numeric_baseline.py" make-ort-compatible \
    --policy-path "$POLICY_PATH" \
    --device "$DEVICE" \
    --input "$BF16_ONNX" \
    --output "$ORT_BF16_ONNX" \
    --check
fi

"$PYTHON" "$SCRIPT_DIR/insert_linear_w8a8_qdq.py" \
  --input "$BF16_ONNX" \
  --output "$QDQ_ONNX" \
  --activation-scale-mode calibrated \
  --activation-scales-json "$SCALES_JSON" \
  --include-module-regex "$MLP_MODULE_REGEX" \
  --include-node-regex "$MLP_NODE_REGEX" \
  --stop-before-node-regex "$STOP_BEFORE_ACTION_REGEX" \
  --cast-output-to "$CAST_OUTPUT_TO" \
  --check

"$PYTHON" - "$QDQ_ONNX" "$EXPECTED_LINEAR_NODES" "$CAST_OUTPUT_TO" <<'PY'
import json
import sys
from pathlib import Path

report_path = Path(sys.argv[1]).with_suffix(".qdq_report.json")
expected_linear_nodes = int(sys.argv[2])
cast_output_to = sys.argv[3]
with open(report_path) as f:
    report = json.load(f)

expected = {
    "rewritten_linear_nodes": expected_linear_nodes,
    "rewritten_weight_only_linear_nodes": 0,
    "cast_output_to": cast_output_to,
    "output_cast_nodes": 0 if cast_output_to == "none" else expected_linear_nodes,
}
bad = {key: (report.get(key), value) for key, value in expected.items() if report.get(key) != value}
if bad:
    details = ", ".join(f"{key}: got {got}, expected {want}" for key, (got, want) in bad.items())
    raise SystemExit(f"unexpected 4E Q/DQ coverage in {report_path}: {details}")
PY

if [[ "$COMPARE" == "true" ]]; then
  "$PYTHON" "$SCRIPT_DIR/diagnose_3_4_numeric_baseline.py" make-ort-compatible \
    --policy-path "$POLICY_PATH" \
    --device "$DEVICE" \
    --input "$QDQ_ONNX" \
    --output "$ORT_QDQ_ONNX" \
    --check
fi

if [[ "$BUILD_ENGINE" == "true" ]]; then
  command -v trtexec >/dev/null 2>&1 || die "trtexec not found; rerun with --build-engine false after building manually"
  TRT_PRECISION_ARGS=()
  if [[ "$TRT_PRECISION_CONSTRAINTS" != "none" ]]; then
    TRT_PRECISION_ARGS=(--precisionConstraints="$TRT_PRECISION_CONSTRAINTS")
  fi
  if [[ "$BUILD_BF16_ENGINE" == "true" ]]; then
    trtexec \
      --onnx="$BF16_ONNX" \
      --saveEngine="$BF16_ENGINE" \
      "${TRT_PRECISION_ARGS[@]}" \
      --profilingVerbosity=detailed \
      --timingCacheFile="$OUTPUT_ROOT/smolvla_debug_core_native_mixed_precision_${TRT_PRECISION_CONSTRAINTS}.cache"
  fi
  [[ -s "$BF16_ENGINE" ]] || die "TensorRT produced empty engine: $BF16_ENGINE"

  trtexec \
    --onnx="$QDQ_ONNX" \
    --saveEngine="$MLP_W8A8_ENGINE" \
    "${TRT_PRECISION_ARGS[@]}" \
    --profilingVerbosity=detailed \
    --timingCacheFile="$OUTPUT_ROOT/smolvla_debug_core_${QUANT_LABEL}_precision_${TRT_PRECISION_CONSTRAINTS}.cache"
  [[ -s "$MLP_W8A8_ENGINE" ]] || die "TensorRT produced empty engine: $MLP_W8A8_ENGINE"
else
  [[ -f "$BF16_ENGINE" ]] || die "$BF16_ENGINE not found; enable --build-engine true"
  [[ -f "$MLP_W8A8_ENGINE" ]] || die "$MLP_W8A8_ENGINE not found; enable --build-engine true"
fi

if [[ "$COMPARE" == "true" ]]; then
  "$PYTHON" "$SCRIPT_DIR/diagnose_3_4_numeric_baseline.py" compare \
    --policy-path "$POLICY_PATH" \
    --device "$DEVICE" \
    --seed "$SEED" \
    --task "$TASK_SUITE" \
    --batch-size "$BATCH_SIZE" \
    --token-length "$TOKEN_LENGTH" \
    --input-dtype "$INPUT_DTYPE" \
    --input-source "$COMPARE_INPUT_SOURCE" \
    --compare-samples "$COMPARE_SAMPLES" \
    --sample-stride "$SAMPLE_STRIDE" \
    --max-parallel-tasks "$MAX_PARALLEL_TASKS" \
    --bf16-onnx "$BF16_ONNX" \
    --ort-bf16-onnx "$ORT_BF16_ONNX" \
    --int8-onnx "$QDQ_ONNX" \
    --ort-int8-onnx "$ORT_QDQ_ONNX" \
    --bf16-engine "$BF16_ENGINE" \
    --int8-engine "$MLP_W8A8_ENGINE" \
    --quantize-module-regex "$MLP_MODULE_REGEX" \
    --fake-quant-kind w8a8 \
    --fake-quant-activation-scale-mode calibrated \
    --fake-quant-activation-scales-json "$SCALES_JSON" \
    --output-dir "$OUTPUT_ROOT/numeric_baseline"
fi

printf '%s\n' "step 4E stage 1 outputs:" \
  "  BF16 ONNX: $BF16_ONNX" \
  "  BF16 ORT ONNX: $ORT_BF16_ONNX" \
  "  $QUANT_LABEL Q/DQ ONNX: $QDQ_ONNX" \
  "  $QUANT_LABEL ORT ONNX: $ORT_QDQ_ONNX" \
  "  BF16 TensorRT engine: $BF16_ENGINE" \
  "  $QUANT_LABEL TensorRT engine: $MLP_W8A8_ENGINE" \
  "  numeric baseline: $OUTPUT_ROOT/numeric_baseline" \
  "  numeric aggregate: $OUTPUT_ROOT/numeric_baseline/numeric_baseline_aggregate_rows.csv" \
  "  Q/DQ report: ${QDQ_ONNX%.onnx}.qdq_report.json"
