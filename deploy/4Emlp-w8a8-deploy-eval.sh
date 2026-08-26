#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$REPO_ROOT/scripts/common.sh"

POLICY_PATH="$REPO_ROOT/smolvla_libero"
OUTPUT_ROOT="$REPO_ROOT/runs/deploy/4/E-mlp-w8a8-formal-deploy"
BF16_ENGINE="$OUTPUT_ROOT/smolvla_debug_core_native_mixed_precision_obey.plan"
MLP_W8A8_ENGINE="$OUTPUT_ROOT/smolvla_debug_core_vlm_mlp_w8a8_precision_obey.plan"
BF16_ONNX="$OUTPUT_ROOT/smolvla_debug_core_native_mixed.onnx"
MLP_W8A8_ONNX="$OUTPUT_ROOT/smolvla_debug_core_vlm_mlp_w8a8_qdq.onnx"
TASK_SUITE=libero_spatial
SEED=1000
EPISODES=10
BATCH_SIZE=1
MAX_PARALLEL_TASKS=1
DEVICE=cuda
PROFILE_WARMUP=5
PROFILE_ITERS=30
RUN_BF16=true
RUN_MLP_W8A8=true
QUANT_OUTPUT_NAME=mlp_w8a8
QUANT_CONFIG_NAME="MLP W8A8"
RUN_PROFILE=true
RUN_EVAL=true
PYTHON="${PYTHON_BIN:-python3}"

usage() {
  printf '%s\n' "Usage: $0 [options]" \
    "  --policy-path PATH              default: $POLICY_PATH" \
    "  --output-root DIR              default: $OUTPUT_ROOT" \
    "  --bf16-engine PATH             default: $BF16_ENGINE" \
    "  --mlp-w8a8-engine PATH         default: $MLP_W8A8_ENGINE" \
    "  --quant-engine PATH            alias of --mlp-w8a8-engine" \
    "  --bf16-onnx PATH               default: $BF16_ONNX" \
    "  --mlp-w8a8-onnx PATH           default: $MLP_W8A8_ONNX" \
    "  --quant-onnx PATH              alias of --mlp-w8a8-onnx" \
    "  --quant-output-name NAME       default: $QUANT_OUTPUT_NAME" \
    "  --quant-config-name NAME       default: $QUANT_CONFIG_NAME" \
    "  --tasks NAME                   default: $TASK_SUITE" \
    "  --device DEVICE                default: $DEVICE" \
    "  --seed N                       default: $SEED" \
    "  --episodes N                   default: $EPISODES" \
    "  --batch-size N                 default: $BATCH_SIZE" \
    "  --max-parallel-tasks N         default: $MAX_PARALLEL_TASKS" \
    "  --profile-warmup N             default: $PROFILE_WARMUP" \
    "  --profile-iters N              default: $PROFILE_ITERS" \
    "  --run-bf16 true|false          default: $RUN_BF16" \
    "  --run-mlp-w8a8 true|false      default: $RUN_MLP_W8A8" \
    "  --run-profile true|false       default: $RUN_PROFILE" \
    "  --run-eval true|false          default: $RUN_EVAL" \
    "  PYTHON_BIN=/path/python        optional Python with torch/lerobot/tensorrt dependencies"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --policy-path) require_value "$@"; POLICY_PATH="$2"; shift 2 ;;
    --output-root) require_value "$@"; OUTPUT_ROOT="$2"; shift 2 ;;
    --bf16-engine) require_value "$@"; BF16_ENGINE="$2"; shift 2 ;;
    --mlp-w8a8-engine) require_value "$@"; MLP_W8A8_ENGINE="$2"; shift 2 ;;
    --quant-engine) require_value "$@"; MLP_W8A8_ENGINE="$2"; shift 2 ;;
    --bf16-onnx) require_value "$@"; BF16_ONNX="$2"; shift 2 ;;
    --mlp-w8a8-onnx) require_value "$@"; MLP_W8A8_ONNX="$2"; shift 2 ;;
    --quant-onnx) require_value "$@"; MLP_W8A8_ONNX="$2"; shift 2 ;;
    --quant-output-name) require_value "$@"; QUANT_OUTPUT_NAME="$2"; shift 2 ;;
    --quant-config-name) require_value "$@"; QUANT_CONFIG_NAME="$2"; shift 2 ;;
    --tasks) require_value "$@"; TASK_SUITE="$2"; shift 2 ;;
    --device) require_value "$@"; DEVICE="$2"; shift 2 ;;
    --seed) require_value "$@"; SEED="$2"; shift 2 ;;
    --episodes) require_value "$@"; EPISODES="$2"; shift 2 ;;
    --batch-size) require_value "$@"; BATCH_SIZE="$2"; shift 2 ;;
    --max-parallel-tasks) require_value "$@"; MAX_PARALLEL_TASKS="$2"; shift 2 ;;
    --profile-warmup) require_value "$@"; PROFILE_WARMUP="$2"; shift 2 ;;
    --profile-iters) require_value "$@"; PROFILE_ITERS="$2"; shift 2 ;;
    --run-bf16) require_value "$@"; RUN_BF16="$2"; shift 2 ;;
    --run-mlp-w8a8) require_value "$@"; RUN_MLP_W8A8="$2"; shift 2 ;;
    --run-profile) require_value "$@"; RUN_PROFILE="$2"; shift 2 ;;
    --run-eval) require_value "$@"; RUN_EVAL="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

for value in "$RUN_BF16" "$RUN_MLP_W8A8" "$RUN_PROFILE" "$RUN_EVAL"; do
  case "$value" in true|false) ;; *) die "boolean arguments must be true or false" ;; esac
done

[[ -f "$BF16_ENGINE" ]] || die "$BF16_ENGINE not found; run deploy/4Emlp-w8a8-export-diagnose.sh first"
[[ -f "$MLP_W8A8_ENGINE" ]] || die "$MLP_W8A8_ENGINE not found; run deploy/4Emlp-w8a8-export-diagnose.sh first"
[[ -f "$BF16_ONNX" ]] || die "$BF16_ONNX not found; run deploy/4Emlp-w8a8-export-diagnose.sh first"
[[ -f "$MLP_W8A8_ONNX" ]] || die "$MLP_W8A8_ONNX not found; run deploy/4Emlp-w8a8-export-diagnose.sh first"

export LEROBOT_SRC="${LEROBOT_SRC:-$REPO_ROOT/lerobot/src}"
export PYTHONPATH="$SCRIPT_DIR:$LEROBOT_SRC${PYTHONPATH:+:$PYTHONPATH}"
export NUMBA_DISABLE_JIT="${NUMBA_DISABLE_JIT:-1}"
mkdir -p "$OUTPUT_ROOT/deploy_eval"

"$PYTHON" -c "import tensorrt" >/dev/null 2>&1 || die "TensorRT Python binding is missing in $PYTHON"

run_one() {
  local label="$1"
  local engine="$2"
  local backend="$3"
  local out_dir="$OUTPUT_ROOT/deploy_eval/$label"
  mkdir -p "$out_dir"
  if [[ "$RUN_PROFILE" == "true" ]]; then
    "$PYTHON" "$SCRIPT_DIR/trt_sample_actions_core_deploy.py" profile \
      --backend "$backend" \
      --engine-path "$engine" \
      --trt-output-name action_chunk \
      --policy-path "$POLICY_PATH" \
      --device "$DEVICE" \
      --output-dir "$out_dir/profile" \
      --warmup "$PROFILE_WARMUP" \
      --iters "$PROFILE_ITERS" \
      --task "$TASK_SUITE 4E $label profile"
  fi
  if [[ "$RUN_EVAL" == "true" ]]; then
    "$PYTHON" "$SCRIPT_DIR/trt_sample_actions_core_deploy.py" eval \
      --backend "$backend" \
      --engine-path "$engine" \
      --trt-output-name action_chunk \
      --policy-path "$POLICY_PATH" \
      --device "$DEVICE" \
      --output-dir "$out_dir/eval" \
      --tasks "$TASK_SUITE" \
      --seed "$SEED" \
      --episodes "$EPISODES" \
      --batch-size "$BATCH_SIZE" \
      --max-parallel-tasks "$MAX_PARALLEL_TASKS"

    "$PYTHON" "$REPO_ROOT/scripts/analyze_eval.py" \
      "$out_dir/eval/eval_info.json" \
      --plot-suite "$TASK_SUITE" \
      --output-dir "$out_dir/report"
  fi
}

if [[ "$RUN_BF16" == "true" ]]; then
  run_one "bf16" "$BF16_ENGINE" "trt"
fi
if [[ "$RUN_MLP_W8A8" == "true" ]]; then
  run_one "$QUANT_OUTPUT_NAME" "$MLP_W8A8_ENGINE" "trt-int8"
fi

if [[ "$RUN_PROFILE" == "true" && "$RUN_EVAL" == "true" ]]; then
  "$PYTHON" "$SCRIPT_DIR/summarize_4e_deployment_table.py" \
    --output-root "$OUTPUT_ROOT/deploy_eval" \
    --bf16-engine "$BF16_ENGINE" \
    --bf16-onnx "$BF16_ONNX" \
    --bf16-profile "$OUTPUT_ROOT/deploy_eval/bf16/profile/profile_summary.json" \
    --bf16-eval "$OUTPUT_ROOT/deploy_eval/bf16/eval/eval_info.json" \
    --mlp-engine "$MLP_W8A8_ENGINE" \
    --mlp-onnx "$MLP_W8A8_ONNX" \
    --mlp-profile "$OUTPUT_ROOT/deploy_eval/$QUANT_OUTPUT_NAME/profile/profile_summary.json" \
    --mlp-eval "$OUTPUT_ROOT/deploy_eval/$QUANT_OUTPUT_NAME/eval/eval_info.json" \
    --quant-config "$QUANT_CONFIG_NAME"
fi

printf '%s\n' "step 4E stage 2 outputs:" \
  "  BF16 profile: $OUTPUT_ROOT/deploy_eval/bf16/profile/profile_summary.json" \
  "  BF16 eval: $OUTPUT_ROOT/deploy_eval/bf16/eval/eval_info.json" \
  "  $QUANT_CONFIG_NAME profile: $OUTPUT_ROOT/deploy_eval/$QUANT_OUTPUT_NAME/profile/profile_summary.json" \
  "  $QUANT_CONFIG_NAME eval: $OUTPUT_ROOT/deploy_eval/$QUANT_OUTPUT_NAME/eval/eval_info.json" \
  "  deployment table: $OUTPUT_ROOT/deploy_eval/deployment_table.md (only when --run-profile true and --run-eval true)"
