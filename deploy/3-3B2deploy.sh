#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$REPO_ROOT/scripts/common.sh"

POLICY_PATH="$REPO_ROOT/smolvla_libero"
ENGINE_PATH="$REPO_ROOT/runs/deploy/3-3B2/smolvla_sample_actions_core_w8a8_dynamic.plan"
OUTPUT_ROOT="$REPO_ROOT/runs/deploy/3-3B2/deploy"
TASK_SUITE=libero_goal
SEED=1000
EPISODES=10
BATCH_SIZE=1
MAX_PARALLEL_TASKS=1
DEVICE=cuda
PROFILE_WARMUP=5
PROFILE_ITERS=30
BACKEND=both
RUN_EVAL=true
RUN_PROFILE=true
PYTHON="${PYTHON_BIN:-python3}"

usage() {
  printf '%s\n' "Usage: $0 [options]" \
    "  --policy-path PATH          default: $POLICY_PATH" \
    "  --engine-path PATH          default: $ENGINE_PATH" \
    "  --output-root DIR          default: $OUTPUT_ROOT" \
    "  --backend both|pytorch|trt-int8 default: $BACKEND" \
    "  --device DEVICE            default: $DEVICE" \
    "  --seed N                   default: $SEED" \
    "  --episodes N               default: $EPISODES" \
    "  --batch-size N             default: $BATCH_SIZE" \
    "  --max-parallel-tasks N     default: $MAX_PARALLEL_TASKS" \
    "  --profile-warmup N         default: $PROFILE_WARMUP" \
    "  --profile-iters N          default: $PROFILE_ITERS" \
    "  --run-eval true|false      default: $RUN_EVAL" \
    "  --run-profile true|false   default: $RUN_PROFILE" \
    "  PYTHON_BIN=/path/python    optional Python with torch/lerobot/tensorrt dependencies"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --policy-path) require_value "$@"; POLICY_PATH="$2"; shift 2 ;;
    --engine-path) require_value "$@"; ENGINE_PATH="$2"; shift 2 ;;
    --output-root) require_value "$@"; OUTPUT_ROOT="$2"; shift 2 ;;
    --backend) require_value "$@"; BACKEND="$2"; shift 2 ;;
    --device) require_value "$@"; DEVICE="$2"; shift 2 ;;
    --seed) require_value "$@"; SEED="$2"; shift 2 ;;
    --episodes) require_value "$@"; EPISODES="$2"; shift 2 ;;
    --batch-size) require_value "$@"; BATCH_SIZE="$2"; shift 2 ;;
    --max-parallel-tasks) require_value "$@"; MAX_PARALLEL_TASKS="$2"; shift 2 ;;
    --profile-warmup) require_value "$@"; PROFILE_WARMUP="$2"; shift 2 ;;
    --profile-iters) require_value "$@"; PROFILE_ITERS="$2"; shift 2 ;;
    --run-eval) require_value "$@"; RUN_EVAL="$2"; shift 2 ;;
    --run-profile) require_value "$@"; RUN_PROFILE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

case "$BACKEND" in
  both|pytorch|trt-int8) ;;
  *) die "--backend must be both, pytorch, or trt-int8" ;;
esac
case "$RUN_EVAL" in
  true|false) ;;
  *) die "--run-eval must be true or false" ;;
esac
case "$RUN_PROFILE" in
  true|false) ;;
  *) die "--run-profile must be true or false" ;;
esac

if [[ "$BACKEND" != "pytorch" && ! -f "$ENGINE_PATH" ]]; then
  die "$ENGINE_PATH not found; run deploy/3-3B2export.sh first"
fi

export LEROBOT_SRC="${LEROBOT_SRC:-$REPO_ROOT/lerobot/src}"
export PYTHONPATH="$LEROBOT_SRC${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUTPUT_ROOT"

if [[ "$BACKEND" == "both" || "$BACKEND" == "trt-int8" ]]; then
  "$PYTHON" -c "import tensorrt" >/dev/null 2>&1 || die "TensorRT Python binding is missing in $PYTHON; install a TensorRT Python package matching your trtexec/TensorRT version, then rerun"
fi

run_backend() {
  local backend="$1"
  local backend_dir="$OUTPUT_ROOT/$backend"
  local engine_args=()
  if [[ "$backend" == "trt-int8" ]]; then
    engine_args=(--engine-path "$ENGINE_PATH")
  fi

  if [[ "$RUN_PROFILE" == "true" ]]; then
    "$PYTHON" "$SCRIPT_DIR/trt_sample_actions_core_deploy.py" profile \
      --backend "$backend" \
      "${engine_args[@]}" \
      --policy-path "$POLICY_PATH" \
      --device "$DEVICE" \
      --output-dir "$backend_dir/profile" \
      --warmup "$PROFILE_WARMUP" \
      --iters "$PROFILE_ITERS" \
      --task "libero_goal $backend dynamic-scale sample_actions inference profile"
  fi

  if [[ "$RUN_EVAL" == "true" ]]; then
    "$PYTHON" "$SCRIPT_DIR/trt_sample_actions_core_deploy.py" eval \
      --backend "$backend" \
      "${engine_args[@]}" \
      --policy-path "$POLICY_PATH" \
      --device "$DEVICE" \
      --output-dir "$backend_dir/eval" \
      --tasks "$TASK_SUITE" \
      --seed "$SEED" \
      --episodes "$EPISODES" \
      --batch-size "$BATCH_SIZE" \
      --max-parallel-tasks "$MAX_PARALLEL_TASKS"
  fi
}

if [[ "$BACKEND" == "both" || "$BACKEND" == "pytorch" ]]; then
  run_backend pytorch
fi
if [[ "$BACKEND" == "both" || "$BACKEND" == "trt-int8" ]]; then
  run_backend trt-int8
fi

printf '%s\n' "step 3-3B2 deployment outputs:" \
  "  output root: $OUTPUT_ROOT" \
  "  pytorch profile/eval: $OUTPUT_ROOT/pytorch" \
  "  trt-int8 profile/eval: $OUTPUT_ROOT/trt-int8"
