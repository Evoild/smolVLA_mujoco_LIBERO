#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$REPO_ROOT/scripts/common.sh"

POLICY_PATH="$REPO_ROOT/smolvla_libero"
TASK_SUITE=libero_goal
SEED=1000
EPISODES=10
EVAL_BATCH_SIZE=1
MAX_PARALLEL_TASKS=1
DEVICE=cuda
OUTPUT_ROOT="$REPO_ROOT/runs/deploy/3-1baseline"
EVAL_DIR=""
REPORT_DIR=""
PROFILE_DIR=""
PROFILE_WARMUP=5
PROFILE_ITERS=30
RUN_EVAL=true
RUN_PROFILE=true
PYTHON="${PYTHON_BIN:-python3}"

usage() {
  printf '%s\n' "Usage: $0 [options]" \
    "  --policy-path PATH          default: $POLICY_PATH" \
    "  --device DEVICE            default: $DEVICE" \
    "  --seed N                   default: $SEED" \
    "  --episodes N               default: $EPISODES" \
    "  --eval-batch-size N        default: $EVAL_BATCH_SIZE" \
    "  --max-parallel-tasks N     default: $MAX_PARALLEL_TASKS" \
    "  --output-root DIR          default: $OUTPUT_ROOT" \
    "  --eval-dir DIR             default: OUTPUT_ROOT/eval" \
    "  --report-dir DIR           default: OUTPUT_ROOT/report" \
    "  --profile-dir DIR          default: OUTPUT_ROOT/profile" \
    "  --profile-warmup N         default: $PROFILE_WARMUP" \
    "  --profile-iters N          default: $PROFILE_ITERS" \
    "  PYTHON_BIN=/path/python    optional Python with torch/lerobot dependencies" \
    "  --run-eval true|false      default: $RUN_EVAL" \
    "  --run-profile true|false   default: $RUN_PROFILE"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --policy-path) require_value "$@"; POLICY_PATH="$2"; shift 2 ;;
    --device) require_value "$@"; DEVICE="$2"; shift 2 ;;
    --seed) require_value "$@"; SEED="$2"; shift 2 ;;
    --episodes) require_value "$@"; EPISODES="$2"; shift 2 ;;
    --eval-batch-size) require_value "$@"; EVAL_BATCH_SIZE="$2"; shift 2 ;;
    --max-parallel-tasks) require_value "$@"; MAX_PARALLEL_TASKS="$2"; shift 2 ;;
    --output-root) require_value "$@"; OUTPUT_ROOT="$2"; shift 2 ;;
    --eval-dir) require_value "$@"; EVAL_DIR="$2"; shift 2 ;;
    --report-dir) require_value "$@"; REPORT_DIR="$2"; shift 2 ;;
    --profile-dir) require_value "$@"; PROFILE_DIR="$2"; shift 2 ;;
    --profile-warmup) require_value "$@"; PROFILE_WARMUP="$2"; shift 2 ;;
    --profile-iters) require_value "$@"; PROFILE_ITERS="$2"; shift 2 ;;
    --run-eval) require_value "$@"; RUN_EVAL="$2"; shift 2 ;;
    --run-profile) require_value "$@"; RUN_PROFILE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

case "$RUN_EVAL" in
  true|false) ;;
  *) die "--run-eval must be true or false" ;;
esac
case "$RUN_PROFILE" in
  true|false) ;;
  *) die "--run-profile must be true or false" ;;
esac

EVAL_DIR="${EVAL_DIR:-$OUTPUT_ROOT/eval}"
REPORT_DIR="${REPORT_DIR:-$OUTPUT_ROOT/report}"
PROFILE_DIR="${PROFILE_DIR:-$OUTPUT_ROOT/profile}"

if [[ "$RUN_EVAL" == "true" && -e "$EVAL_DIR" ]]; then
  die "$EVAL_DIR already exists; pass a new --eval-dir or --output-root"
fi
if [[ "$RUN_PROFILE" == "true" && -e "$PROFILE_DIR" ]]; then
  die "$PROFILE_DIR already exists; pass a new --profile-dir or --output-root"
fi

if [[ "$RUN_EVAL" == "true" ]]; then
  bash "$REPO_ROOT/scripts/eval_libero.sh" \
    --policy-path "$POLICY_PATH" \
    --tasks "$TASK_SUITE" \
    --seeds "$SEED" \
    --episodes "$EPISODES" \
    --batch-size "$EVAL_BATCH_SIZE" \
    --max-parallel-tasks "$MAX_PARALLEL_TASKS" \
    --device "$DEVICE" \
    --output-dir "$EVAL_DIR"

  "$PYTHON" "$REPO_ROOT/scripts/analyze_eval.py" \
    "$EVAL_DIR/eval_info.json" \
    --plot-suite "$TASK_SUITE" \
    --output-dir "$REPORT_DIR"
fi

if [[ "$RUN_PROFILE" == "true" ]]; then
  LEROBOT_SRC="${LEROBOT_SRC:-$REPO_ROOT/lerobot/src}" \
  PYTHONPATH="${LEROBOT_SRC:-$REPO_ROOT/lerobot/src}${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON" "$SCRIPT_DIR/profile_3_1_baseline.py" \
    --policy-path "$POLICY_PATH" \
    --device "$DEVICE" \
    --output-dir "$PROFILE_DIR" \
    --warmup "$PROFILE_WARMUP" \
    --iters "$PROFILE_ITERS" \
    --task "libero_goal baseline forward profile"
fi

printf '%s\n' "step 3-1 baseline outputs:" \
  "  success eval: $EVAL_DIR" \
  "  success report: $REPORT_DIR" \
  "  forward profile: $PROFILE_DIR"
