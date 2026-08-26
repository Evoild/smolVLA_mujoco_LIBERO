#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$REPO_ROOT/scripts/common.sh"

POLICY_PATH="$REPO_ROOT/smolvla_libero"
ENGINE_PATH="$REPO_ROOT/runs/deploy/3-7/B-formal-quant-deploy/smolvla_debug_core_gate_up_w8a8_down_w8a16_precision_obey.plan"
OUTPUT_ROOT="$REPO_ROOT/runs/deploy/3-7/B-formal-quant-deploy/deploy_eval"
TASK_SUITE=libero_spatial
SEED=1000
EPISODES=10
BATCH_SIZE=1
MAX_PARALLEL_TASKS=1
DEVICE=cuda
PROFILE_WARMUP=5
PROFILE_ITERS=30
RUN_PROFILE=true
RUN_EVAL=true
PYTHON="${PYTHON_BIN:-python3}"

usage() {
  printf '%s\n' "Usage: $0 [options]" \
    "  --policy-path PATH          default: $POLICY_PATH" \
    "  --engine-path PATH          default: $ENGINE_PATH" \
    "  --output-root DIR          default: $OUTPUT_ROOT" \
    "  --tasks NAME               default: $TASK_SUITE" \
    "  --device DEVICE            default: $DEVICE" \
    "  --seed N                   default: $SEED" \
    "  --episodes N               default: $EPISODES" \
    "  --batch-size N             default: $BATCH_SIZE" \
    "  --max-parallel-tasks N     default: $MAX_PARALLEL_TASKS" \
    "  --profile-warmup N         default: $PROFILE_WARMUP" \
    "  --profile-iters N          default: $PROFILE_ITERS" \
    "  --run-profile true|false   default: $RUN_PROFILE" \
    "  --run-eval true|false      default: $RUN_EVAL" \
    "  PYTHON_BIN=/path/python    optional Python with torch/lerobot/tensorrt dependencies"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --policy-path) require_value "$@"; POLICY_PATH="$2"; shift 2 ;;
    --engine-path) require_value "$@"; ENGINE_PATH="$2"; shift 2 ;;
    --output-root) require_value "$@"; OUTPUT_ROOT="$2"; shift 2 ;;
    --tasks) require_value "$@"; TASK_SUITE="$2"; shift 2 ;;
    --device) require_value "$@"; DEVICE="$2"; shift 2 ;;
    --seed) require_value "$@"; SEED="$2"; shift 2 ;;
    --episodes) require_value "$@"; EPISODES="$2"; shift 2 ;;
    --batch-size) require_value "$@"; BATCH_SIZE="$2"; shift 2 ;;
    --max-parallel-tasks) require_value "$@"; MAX_PARALLEL_TASKS="$2"; shift 2 ;;
    --profile-warmup) require_value "$@"; PROFILE_WARMUP="$2"; shift 2 ;;
    --profile-iters) require_value "$@"; PROFILE_ITERS="$2"; shift 2 ;;
    --run-profile) require_value "$@"; RUN_PROFILE="$2"; shift 2 ;;
    --run-eval) require_value "$@"; RUN_EVAL="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

case "$RUN_PROFILE" in true|false) ;; *) die "--run-profile must be true or false" ;; esac
case "$RUN_EVAL" in true|false) ;; *) die "--run-eval must be true or false" ;; esac

[[ -f "$ENGINE_PATH" ]] || die "$ENGINE_PATH not found; run deploy/3-7Bexport-diagnose.sh first"

export LEROBOT_SRC="${LEROBOT_SRC:-$REPO_ROOT/lerobot/src}"
export PYTHONPATH="$LEROBOT_SRC${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUTPUT_ROOT"

"$PYTHON" -c "import tensorrt" >/dev/null 2>&1 || die "TensorRT Python binding is missing in $PYTHON"

if [[ "$RUN_PROFILE" == "true" ]]; then
  "$PYTHON" "$SCRIPT_DIR/trt_sample_actions_core_deploy.py" profile \
    --backend trt-int8 \
    --engine-path "$ENGINE_PATH" \
    --trt-output-name action_chunk \
    --policy-path "$POLICY_PATH" \
    --device "$DEVICE" \
    --output-dir "$OUTPUT_ROOT/profile" \
    --warmup "$PROFILE_WARMUP" \
    --iters "$PROFILE_ITERS" \
    --task "$TASK_SUITE 3-7B TensorRT mixed quant profile"
fi

if [[ "$RUN_EVAL" == "true" ]]; then
  "$PYTHON" "$SCRIPT_DIR/trt_sample_actions_core_deploy.py" eval \
    --backend trt-int8 \
    --engine-path "$ENGINE_PATH" \
    --trt-output-name action_chunk \
    --policy-path "$POLICY_PATH" \
    --device "$DEVICE" \
    --output-dir "$OUTPUT_ROOT/eval" \
    --tasks "$TASK_SUITE" \
    --seed "$SEED" \
    --episodes "$EPISODES" \
    --batch-size "$BATCH_SIZE" \
    --max-parallel-tasks "$MAX_PARALLEL_TASKS"

  "$PYTHON" "$REPO_ROOT/scripts/analyze_eval.py" \
    "$OUTPUT_ROOT/eval/eval_info.json" \
    --plot-suite "$TASK_SUITE" \
    --output-dir "$OUTPUT_ROOT/report"
fi

printf '%s\n' "step 3-7B stage 2 outputs:" \
  "  engine: $ENGINE_PATH" \
  "  profile summary: $OUTPUT_ROOT/profile/profile_summary.json" \
  "  eval info: $OUTPUT_ROOT/eval/eval_info.json" \
  "  eval report: $OUTPUT_ROOT/report" \
  "  metrics: latency ms, FPS, peak_gpu_memory_mb, success rate"
