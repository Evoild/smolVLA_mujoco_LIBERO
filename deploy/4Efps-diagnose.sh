#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$REPO_ROOT/scripts/common.sh"

POLICY_PATH="$REPO_ROOT/smolvla_libero"
OUTPUT_ROOT="$REPO_ROOT/runs/deploy/4/E4-fps-diagnose-mlp-full-attn-w8a8-action-only"
REUSE_BF16_ONNX=""
REUSE_BF16_ENGINE=""
SCALES_JSON="$REPO_ROOT/runs/deploy/4/E3-C-mlp-full-attn-w8a8/activation_scales_512_p99.995.json"
QUANT_LABEL=vlm_mlp_full_attn_w8a8
QUANT_MODULE_REGEX="^model\\.vlm_with_expert\\.vlm\\.model\\.text_model\\.layers\\.[0-9]+\\.(mlp\\.(gate_proj|up_proj|down_proj)|self_attn\\.(q_proj|k_proj|v_proj|o_proj))$"
QUANT_NODE_REGEX="^/(debug_core/)?(q_proj|k_proj|v_proj|o_proj|mlp/(gate_proj|up_proj|down_proj))(_[0-9]+)?/MatMul$"
EXPECTED_LINEAR_NODES=219
CAST_OUTPUT_TO=bf16
TRT_PRECISION_CONSTRAINTS=obey
TASK_SUITE=libero_spatial
DEVICE=cuda
SEED=1000
PROFILE_WARMUP=10
PROFILE_ITERS=100
TRT_ITERATIONS=200
RUN_EXPORT=true
RUN_TRTEXEC=true
RUN_POLICY_PROFILE=true
RUN_BF16_PROFILE=true
PYTHON="${PYTHON_BIN:-python3}"

usage() {
  printf '%s\n' "Usage: $0 [options]" \
    "  --policy-path PATH             default: $POLICY_PATH" \
    "  --output-root DIR             default: $OUTPUT_ROOT" \
    "  --reuse-bf16-onnx PATH         reuse an existing action-only native mixed ONNX" \
    "  --reuse-bf16-engine PATH       reuse an existing action-only native mixed TensorRT engine" \
    "  --scales-json PATH            default: $SCALES_JSON" \
    "  --quant-label NAME            default: $QUANT_LABEL" \
    "  --quant-module-regex REGEX    default: E3-C full text attention" \
    "  --quant-node-regex REGEX      default: E3-C full text attention ONNX MatMul" \
    "  --expected-linear-nodes N     default: $EXPECTED_LINEAR_NODES" \
    "  --cast-output-to none|bf16|fp16|fp32 default: $CAST_OUTPUT_TO" \
    "  --trt-precision-constraints obey|prefer|none default: $TRT_PRECISION_CONSTRAINTS" \
    "  --tasks NAME                  default: $TASK_SUITE" \
    "  --device DEVICE               default: $DEVICE" \
    "  --seed N                      default: $SEED" \
    "  --profile-warmup N            default: $PROFILE_WARMUP" \
    "  --profile-iters N             default: $PROFILE_ITERS" \
    "  --trtexec-iterations N        default: $TRT_ITERATIONS" \
    "  --run-export true|false       default: $RUN_EXPORT" \
    "  --run-trtexec true|false      default: $RUN_TRTEXEC" \
    "  --run-policy-profile true|false default: $RUN_POLICY_PROFILE" \
    "  --run-bf16-profile true|false default: $RUN_BF16_PROFILE" \
    "  PYTHON_BIN=/path/python       optional Python with torch/onnxruntime/tensorrt dependencies"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --policy-path) require_value "$@"; POLICY_PATH="$2"; shift 2 ;;
    --output-root) require_value "$@"; OUTPUT_ROOT="$2"; shift 2 ;;
    --reuse-bf16-onnx) require_value "$@"; REUSE_BF16_ONNX="$2"; shift 2 ;;
    --reuse-bf16-engine) require_value "$@"; REUSE_BF16_ENGINE="$2"; shift 2 ;;
    --scales-json) require_value "$@"; SCALES_JSON="$2"; shift 2 ;;
    --quant-label) require_value "$@"; QUANT_LABEL="$2"; shift 2 ;;
    --quant-module-regex) require_value "$@"; QUANT_MODULE_REGEX="$2"; shift 2 ;;
    --quant-node-regex) require_value "$@"; QUANT_NODE_REGEX="$2"; shift 2 ;;
    --expected-linear-nodes) require_value "$@"; EXPECTED_LINEAR_NODES="$2"; shift 2 ;;
    --cast-output-to) require_value "$@"; CAST_OUTPUT_TO="$2"; shift 2 ;;
    --trt-precision-constraints) require_value "$@"; TRT_PRECISION_CONSTRAINTS="$2"; shift 2 ;;
    --tasks) require_value "$@"; TASK_SUITE="$2"; shift 2 ;;
    --device) require_value "$@"; DEVICE="$2"; shift 2 ;;
    --seed) require_value "$@"; SEED="$2"; shift 2 ;;
    --profile-warmup) require_value "$@"; PROFILE_WARMUP="$2"; shift 2 ;;
    --profile-iters) require_value "$@"; PROFILE_ITERS="$2"; shift 2 ;;
    --trtexec-iterations) require_value "$@"; TRT_ITERATIONS="$2"; shift 2 ;;
    --run-export) require_value "$@"; RUN_EXPORT="$2"; shift 2 ;;
    --run-trtexec) require_value "$@"; RUN_TRTEXEC="$2"; shift 2 ;;
    --run-policy-profile) require_value "$@"; RUN_POLICY_PROFILE="$2"; shift 2 ;;
    --run-bf16-profile) require_value "$@"; RUN_BF16_PROFILE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

for value in "$RUN_EXPORT" "$RUN_TRTEXEC" "$RUN_POLICY_PROFILE" "$RUN_BF16_PROFILE"; do
  case "$value" in true|false) ;; *) die "boolean arguments must be true or false" ;; esac
done
case "$CAST_OUTPUT_TO" in none|bf16|fp16|fp32) ;; *) die "--cast-output-to must be none, bf16, fp16, or fp32" ;; esac
case "$TRT_PRECISION_CONSTRAINTS" in obey|prefer|none) ;; *) die "--trt-precision-constraints must be obey, prefer, or none" ;; esac

mkdir -p "$OUTPUT_ROOT"
BF16_ONNX="$OUTPUT_ROOT/smolvla_action_only_native_mixed.onnx"
QDQ_ONNX="$OUTPUT_ROOT/smolvla_action_only_${QUANT_LABEL}_qdq.onnx"
BF16_ENGINE="$OUTPUT_ROOT/smolvla_action_only_native_mixed_precision_${TRT_PRECISION_CONSTRAINTS}.plan"
INT8_ENGINE="$OUTPUT_ROOT/smolvla_action_only_${QUANT_LABEL}_cast_${CAST_OUTPUT_TO}_precision_${TRT_PRECISION_CONSTRAINTS}.plan"
if [[ -n "$REUSE_BF16_ONNX" ]]; then
  [[ -f "$REUSE_BF16_ONNX" ]] || die "$REUSE_BF16_ONNX not found"
  BF16_ONNX="$REUSE_BF16_ONNX"
fi
if [[ -n "$REUSE_BF16_ENGINE" ]]; then
  [[ -f "$REUSE_BF16_ENGINE" ]] || die "$REUSE_BF16_ENGINE not found"
  BF16_ENGINE="$REUSE_BF16_ENGINE"
fi

if [[ "$RUN_EXPORT" == "true" ]]; then
  REUSE_ARGS=()
  if [[ -n "$REUSE_BF16_ONNX" ]]; then
    REUSE_ARGS+=(--reuse-bf16-onnx "$REUSE_BF16_ONNX")
  fi
  if [[ -n "$REUSE_BF16_ENGINE" ]]; then
    REUSE_ARGS+=(--reuse-bf16-engine "$REUSE_BF16_ENGINE")
  fi
  "$SCRIPT_DIR/4Emlp-w8a8-export-diagnose.sh" \
    --policy-path "$POLICY_PATH" \
    --output-root "$OUTPUT_ROOT" \
    "${REUSE_ARGS[@]}" \
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
    --export-output-mode action-only \
    --compare false \
    --build-engine true

  if [[ -z "$REUSE_BF16_ONNX" ]]; then
    mv "$OUTPUT_ROOT/smolvla_debug_core_native_mixed.onnx" "$BF16_ONNX"
  fi
  DEBUG_QDQ_ONNX="$OUTPUT_ROOT/smolvla_debug_core_${QUANT_LABEL}_qdq.onnx"
  DEBUG_QDQ_REPORT="${DEBUG_QDQ_ONNX%.onnx}.qdq_report.json"
  ACTION_QDQ_REPORT="${QDQ_ONNX%.onnx}.qdq_report.json"
  mv "$DEBUG_QDQ_ONNX" "$QDQ_ONNX"
  if [[ -f "$DEBUG_QDQ_REPORT" ]]; then
    mv "$DEBUG_QDQ_REPORT" "$ACTION_QDQ_REPORT"
  fi
  if [[ -z "$REUSE_BF16_ENGINE" ]]; then
    mv "$OUTPUT_ROOT/smolvla_debug_core_native_mixed_precision_obey.plan" "$BF16_ENGINE" 2>/dev/null || \
      mv "$OUTPUT_ROOT/smolvla_debug_core_native_mixed_precision_${TRT_PRECISION_CONSTRAINTS}.plan" "$BF16_ENGINE" 2>/dev/null || true
  fi
  mv "$OUTPUT_ROOT/smolvla_debug_core_${QUANT_LABEL}_precision_obey.plan" "$INT8_ENGINE" 2>/dev/null || \
    mv "$OUTPUT_ROOT/smolvla_debug_core_${QUANT_LABEL}_precision_${TRT_PRECISION_CONSTRAINTS}.plan" "$INT8_ENGINE" 2>/dev/null || true
fi

[[ -f "$BF16_ENGINE" ]] || die "$BF16_ENGINE not found"
[[ -f "$INT8_ENGINE" ]] || die "$INT8_ENGINE not found"
[[ -f "$BF16_ONNX" ]] || die "$BF16_ONNX not found"
[[ -f "$QDQ_ONNX" ]] || die "$QDQ_ONNX not found"

if [[ "$RUN_TRTEXEC" == "true" ]]; then
  command -v trtexec >/dev/null 2>&1 || die "trtexec not found"
  if [[ "$RUN_BF16_PROFILE" == "true" ]]; then
    trtexec \
      --loadEngine="$BF16_ENGINE" \
      --warmUp=1000 \
      --iterations="$TRT_ITERATIONS" \
      --dumpProfile \
      --separateProfileRun \
      --profilingVerbosity=detailed \
      --exportProfile="$OUTPUT_ROOT/trtexec_bf16_profile.json" \
      > "$OUTPUT_ROOT/trtexec_bf16.log" 2>&1
  fi

  trtexec \
    --loadEngine="$INT8_ENGINE" \
    --warmUp=1000 \
    --iterations="$TRT_ITERATIONS" \
    --dumpProfile \
    --separateProfileRun \
    --profilingVerbosity=detailed \
    --exportProfile="$OUTPUT_ROOT/trtexec_int8_profile.json" \
    > "$OUTPUT_ROOT/trtexec_int8.log" 2>&1
fi

if [[ "$RUN_POLICY_PROFILE" == "true" ]]; then
  "$SCRIPT_DIR/4Emlp-w8a8-deploy-eval.sh" \
    --policy-path "$POLICY_PATH" \
    --output-root "$OUTPUT_ROOT" \
    --bf16-engine "$BF16_ENGINE" \
    --bf16-onnx "$BF16_ONNX" \
    --quant-engine "$INT8_ENGINE" \
    --quant-onnx "$QDQ_ONNX" \
    --quant-output-name int8_action_only \
    --quant-config-name "$QUANT_LABEL action-only" \
    --tasks "$TASK_SUITE" \
    --seed "$SEED" \
    --profile-warmup "$PROFILE_WARMUP" \
    --profile-iters "$PROFILE_ITERS" \
    --run-bf16 "$RUN_BF16_PROFILE" \
    --run-eval false \
    --run-profile true
fi

printf '%s\n' "step 4E FPS diagnosis outputs:" \
  "  output root: $OUTPUT_ROOT" \
  "  BF16 action-only ONNX: $BF16_ONNX" \
  "  INT8 action-only ONNX: $QDQ_ONNX" \
  "  BF16 action-only engine: $BF16_ENGINE" \
  "  INT8 action-only engine: $INT8_ENGINE" \
  "  BF16 trtexec log: $OUTPUT_ROOT/trtexec_bf16.log" \
  "  INT8 trtexec log: $OUTPUT_ROOT/trtexec_int8.log" \
  "  BF16 trtexec profile: $OUTPUT_ROOT/trtexec_bf16_profile.json" \
  "  INT8 trtexec profile: $OUTPUT_ROOT/trtexec_int8_profile.json" \
  "  policy profile root: $OUTPUT_ROOT/deploy_eval"
