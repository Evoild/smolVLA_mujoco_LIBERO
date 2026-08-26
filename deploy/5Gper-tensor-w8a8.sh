#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$REPO_ROOT/scripts/common.sh"

POLICY_PATH="$REPO_ROOT/smolvla_libero"
OUTPUT_ROOT="$REPO_ROOT/runs/deploy/5/G-per-tensor-w8a8"
BF16_ENGINE="$REPO_ROOT/runs/deploy/4/E4-fps-diagnose-mlp-full-attn-w8a8-action-only/smolvla_action_only_native_mixed_precision_obey.plan"
SCALES_JSON=""
TASK_SUITE=libero_spatial
DEVICE=cuda
SEED=1000
CALIBRATION_SAMPLES=512
CALIBRATION_PERCENTILE=99.995
BATCH_SIZE=1
TOKEN_LENGTH=48
PROFILE_WARMUP=10
PROFILE_ITERS=100
TRT_ITERATIONS=200
EPISODES=10
RUN_CALIBRATION=true
RUN_EXPORT=true
RUN_TRTEXEC=true
RUN_POLICY_PROFILE=true
RUN_EVAL=false
CAST_OUTPUT_TO=bf16
TRT_PRECISION_CONSTRAINTS=prefer
QUANT_LABEL=vlm_mlp_full_attn_w8a8_scalar_act
PYTHON="${PYTHON_BIN:-python3}"

QUANT_MODULE_REGEX="^model\\.vlm_with_expert\\.vlm\\.model\\.text_model\\.layers\\.[0-9]+\\.(mlp\\.(gate_proj|up_proj|down_proj)|self_attn\\.(q_proj|k_proj|v_proj|o_proj))$"
QUANT_NODE_REGEX="^/(debug_core/)?(q_proj|k_proj|v_proj|o_proj|mlp/(gate_proj|up_proj|down_proj))(_[0-9]+)?/MatMul$"
EXPECTED_LINEAR_NODES=219

usage() {
  printf '%s\n' "Usage: $0 [options]" \
    "  --policy-path PATH                  default: $POLICY_PATH" \
    "  --output-root DIR                  default: $OUTPUT_ROOT" \
    "  --bf16-engine PATH                 existing BF16 engine placeholder; default: $BF16_ENGINE" \
    "  --scales-json PATH                 default: OUTPUT_ROOT/activation_scales_512_p99.995_scalar.json" \
    "  --tasks NAME                       default: $TASK_SUITE" \
    "  --device DEVICE                    default: $DEVICE" \
    "  --seed N                           default: $SEED" \
    "  --calibration-samples N            default: $CALIBRATION_SAMPLES" \
    "  --calibration-percentile P         default: $CALIBRATION_PERCENTILE" \
    "  --batch-size N                     default: $BATCH_SIZE" \
    "  --token-length N                   default: $TOKEN_LENGTH" \
    "  --profile-warmup N                 default: $PROFILE_WARMUP" \
    "  --profile-iters N                  default: $PROFILE_ITERS" \
    "  --trtexec-iterations N             default: $TRT_ITERATIONS" \
    "  --episodes N                       default: $EPISODES" \
    "  --run-calibration true|false       default: $RUN_CALIBRATION" \
    "  --run-export true|false            default: $RUN_EXPORT" \
    "  --run-trtexec true|false           default: $RUN_TRTEXEC" \
    "  --run-policy-profile true|false    default: $RUN_POLICY_PROFILE" \
    "  --run-eval true|false              default: $RUN_EVAL" \
    "  --cast-output-to none|bf16|fp16|fp32 default: $CAST_OUTPUT_TO" \
    "  --trt-precision-constraints obey|prefer|none default: $TRT_PRECISION_CONSTRAINTS" \
    "  PYTHON_BIN=/path/python            optional Python with torch/lerobot/tensorrt dependencies"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --policy-path) require_value "$@"; POLICY_PATH="$2"; shift 2 ;;
    --output-root) require_value "$@"; OUTPUT_ROOT="$2"; shift 2 ;;
    --bf16-engine) require_value "$@"; BF16_ENGINE="$2"; shift 2 ;;
    --scales-json) require_value "$@"; SCALES_JSON="$2"; shift 2 ;;
    --tasks) require_value "$@"; TASK_SUITE="$2"; shift 2 ;;
    --device) require_value "$@"; DEVICE="$2"; shift 2 ;;
    --seed) require_value "$@"; SEED="$2"; shift 2 ;;
    --calibration-samples) require_value "$@"; CALIBRATION_SAMPLES="$2"; shift 2 ;;
    --calibration-percentile) require_value "$@"; CALIBRATION_PERCENTILE="$2"; shift 2 ;;
    --batch-size) require_value "$@"; BATCH_SIZE="$2"; shift 2 ;;
    --token-length) require_value "$@"; TOKEN_LENGTH="$2"; shift 2 ;;
    --profile-warmup) require_value "$@"; PROFILE_WARMUP="$2"; shift 2 ;;
    --profile-iters) require_value "$@"; PROFILE_ITERS="$2"; shift 2 ;;
    --trtexec-iterations) require_value "$@"; TRT_ITERATIONS="$2"; shift 2 ;;
    --episodes) require_value "$@"; EPISODES="$2"; shift 2 ;;
    --run-calibration) require_value "$@"; RUN_CALIBRATION="$2"; shift 2 ;;
    --run-export) require_value "$@"; RUN_EXPORT="$2"; shift 2 ;;
    --run-trtexec) require_value "$@"; RUN_TRTEXEC="$2"; shift 2 ;;
    --run-policy-profile) require_value "$@"; RUN_POLICY_PROFILE="$2"; shift 2 ;;
    --run-eval) require_value "$@"; RUN_EVAL="$2"; shift 2 ;;
    --cast-output-to) require_value "$@"; CAST_OUTPUT_TO="$2"; shift 2 ;;
    --trt-precision-constraints) require_value "$@"; TRT_PRECISION_CONSTRAINTS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

for value in "$RUN_CALIBRATION" "$RUN_EXPORT" "$RUN_TRTEXEC" "$RUN_POLICY_PROFILE" "$RUN_EVAL"; do
  case "$value" in true|false) ;; *) die "boolean arguments must be true or false" ;; esac
done
case "$CAST_OUTPUT_TO" in none|bf16|fp16|fp32) ;; *) die "--cast-output-to must be none, bf16, fp16, or fp32" ;; esac
case "$TRT_PRECISION_CONSTRAINTS" in obey|prefer|none) ;; *) die "--trt-precision-constraints must be obey, prefer, or none" ;; esac

mkdir -p "$OUTPUT_ROOT"
SCALES_JSON="${SCALES_JSON:-$OUTPUT_ROOT/activation_scales_${CALIBRATION_SAMPLES}_p${CALIBRATION_PERCENTILE}_scalar.json}"
BF16_ONNX="$OUTPUT_ROOT/smolvla_action_only_native_mixed.onnx"
QDQ_ONNX="$OUTPUT_ROOT/smolvla_action_only_${QUANT_LABEL}_qdq.onnx"
INT8_ENGINE="$OUTPUT_ROOT/smolvla_action_only_${QUANT_LABEL}_cast_${CAST_OUTPUT_TO}_precision_${TRT_PRECISION_CONSTRAINTS}.plan"

export LEROBOT_SRC="${LEROBOT_SRC:-$REPO_ROOT/lerobot/src}"
export PYTHONPATH="$SCRIPT_DIR:$LEROBOT_SRC${PYTHONPATH:+:$PYTHONPATH}"

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
  [[ -f "$SCALES_JSON" ]] || die "$SCALES_JSON not found; enable --run-calibration true or pass --scales-json"
fi

if [[ "$RUN_EXPORT" == "true" || "$RUN_TRTEXEC" == "true" || "$RUN_POLICY_PROFILE" == "true" ]]; then
  [[ -f "$BF16_ENGINE" ]] || die "$BF16_ENGINE not found; pass --bf16-engine with an existing baseline engine"
fi

if [[ "$RUN_EXPORT" == "true" ]]; then
  "$SCRIPT_DIR/4Efps-diagnose.sh" \
    --policy-path "$POLICY_PATH" \
    --output-root "$OUTPUT_ROOT" \
    --reuse-bf16-engine "$BF16_ENGINE" \
    --scales-json "$SCALES_JSON" \
    --quant-label "$QUANT_LABEL" \
    --quant-module-regex "$QUANT_MODULE_REGEX" \
    --quant-node-regex "$QUANT_NODE_REGEX" \
    --expected-linear-nodes "$EXPECTED_LINEAR_NODES" \
    --cast-output-to "$CAST_OUTPUT_TO" \
    --trt-precision-constraints "$TRT_PRECISION_CONSTRAINTS" \
    --tasks "$TASK_SUITE" \
    --device "$DEVICE" \
    --seed "$SEED" \
    --profile-warmup "$PROFILE_WARMUP" \
    --profile-iters "$PROFILE_ITERS" \
    --trtexec-iterations "$TRT_ITERATIONS" \
    --run-export true \
    --run-trtexec "$RUN_TRTEXEC" \
    --run-policy-profile "$RUN_POLICY_PROFILE" \
    --run-bf16-profile false
else
  [[ -f "$BF16_ONNX" ]] || die "$BF16_ONNX not found; enable --run-export true"
  [[ -f "$QDQ_ONNX" ]] || die "$QDQ_ONNX not found; enable --run-export true"
  [[ -f "$INT8_ENGINE" ]] || die "$INT8_ENGINE not found; enable --run-export true"
fi

if [[ -f "$INT8_ENGINE" ]]; then
  "$SCRIPT_DIR/5Dinspect-int8-kernels.sh" \
    --engine "$INT8_ENGINE" \
    --output-root "$OUTPUT_ROOT/inspect"
fi

if [[ "$RUN_EVAL" == "true" ]]; then
  "$SCRIPT_DIR/4Emlp-w8a8-deploy-eval.sh" \
    --policy-path "$POLICY_PATH" \
    --output-root "$OUTPUT_ROOT" \
    --bf16-engine "$BF16_ENGINE" \
    --bf16-onnx "$BF16_ONNX" \
    --quant-engine "$INT8_ENGINE" \
    --quant-onnx "$QDQ_ONNX" \
    --quant-output-name per_tensor_w8a8 \
    --quant-config-name "per-tensor activation W8A8" \
    --tasks "$TASK_SUITE" \
    --seed "$SEED" \
    --episodes "$EPISODES" \
    --batch-size "$BATCH_SIZE" \
    --max-parallel-tasks 1 \
    --profile-warmup "$PROFILE_WARMUP" \
    --profile-iters "$PROFILE_ITERS" \
    --run-bf16 false \
    --run-profile false \
    --run-eval true
fi

printf '%s\n' "step 5G per-tensor activation W8A8 outputs:" \
  "  scales: $SCALES_JSON" \
  "  action-only native ONNX: $BF16_ONNX" \
  "  Q/DQ ONNX: $QDQ_ONNX" \
  "  TensorRT engine: $INT8_ENGINE" \
  "  Q/DQ report: ${QDQ_ONNX%.onnx}.qdq_report.json" \
  "  trtexec profile: $OUTPUT_ROOT/trtexec_int8_profile.json" \
  "  policy profile: $OUTPUT_ROOT/deploy_eval/int8_action_only/profile/profile_summary.json" \
  "  precision inspect: $OUTPUT_ROOT/inspect/layer_precision_summary.json"
