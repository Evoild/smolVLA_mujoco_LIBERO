#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

POLICY_PATH=""
TASKS="libero_spatial,libero_object,libero_goal,libero_10"
SEEDS="0"
EPISODES=10
BATCH_SIZE=1
MAX_PARALLEL_TASKS=1
DEVICE=cuda
OUTPUT_ROOT=runs/eval
OUTPUT_DIR=""

usage() {
  printf '%s\n' "Usage: $0 --policy-path PATH [options]" \
    "  --tasks CSV                default: $TASKS" \
    "  --seeds CSV                default: $SEEDS" \
    "  --episodes N               episodes per task (default: $EPISODES)" \
    "  --batch-size N             environments per task (default: $BATCH_SIZE)" \
    "  --max-parallel-tasks N     task workers (default: $MAX_PARALLEL_TASKS)" \
    "  --device DEVICE            default: $DEVICE" \
    "  --output-root DIR          organize results as DIR/model/seed" \
    "  --output-dir DIR           exact directory; single seed only"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --policy-path) require_value "$@"; POLICY_PATH="$2"; shift 2 ;;
    --tasks|--task-suite) require_value "$@"; TASKS="$2"; shift 2 ;;
    --seeds) require_value "$@"; SEEDS="$2"; shift 2 ;;
    --episodes) require_value "$@"; EPISODES="$2"; shift 2 ;;
    --batch-size) require_value "$@"; BATCH_SIZE="$2"; shift 2 ;;
    --max-parallel-tasks) require_value "$@"; MAX_PARALLEL_TASKS="$2"; shift 2 ;;
    --device) require_value "$@"; DEVICE="$2"; shift 2 ;;
    --output-root) require_value "$@"; OUTPUT_ROOT="$2"; shift 2 ;;
    --output-dir) require_value "$@"; OUTPUT_DIR="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ -n "$POLICY_PATH" ]] || { usage >&2; die "--policy-path is required"; }
[[ "$EPISODES" =~ ^[1-9][0-9]*$ ]] || die "--episodes must be a positive integer"
[[ "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || die "--batch-size must be a positive integer"

policy_name="$(basename "${POLICY_PATH%/}")"
IFS=',' read -r -a seed_list <<< "$SEEDS"
if [[ -n "$OUTPUT_DIR" && ${#seed_list[@]} -ne 1 ]]; then
  die "--output-dir requires exactly one seed; use --output-root for multiple seeds"
fi
for seed in "${seed_list[@]}"; do
  [[ "$seed" =~ ^[0-9]+$ ]] || die "invalid seed: $seed"
  set_reproducible_env "$seed"
  output_dir="${OUTPUT_DIR:-$OUTPUT_ROOT/$policy_name/seed_$seed}"
  [[ ! -e "$output_dir" ]] || die "$output_dir already exists; choose another output root"
  run_lerobot lerobot-eval \
    "--policy.path=$POLICY_PATH" \
    "--policy.device=$DEVICE" \
    --env.type=libero \
    "--env.task=$TASKS" \
    "--eval.batch_size=$BATCH_SIZE" \
    "--eval.n_episodes=$EPISODES" \
    "--env.max_parallel_tasks=$MAX_PARALLEL_TASKS" \
    "--seed=$seed" \
    "--output_dir=$output_dir"
done

if [[ -n "$OUTPUT_DIR" ]]; then
  python3 "$SCRIPT_DIR/analyze_eval.py" "$OUTPUT_DIR" --output-dir "$OUTPUT_DIR/report"
else
  python3 "$SCRIPT_DIR/analyze_eval.py" "$OUTPUT_ROOT/$policy_name" --output-dir "$OUTPUT_ROOT/$policy_name/report"
fi
