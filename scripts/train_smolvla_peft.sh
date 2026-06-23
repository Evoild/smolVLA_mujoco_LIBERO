#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

DATASET=""
DATASET_ROOT=""
DATASET_SUITE=libero_spatial
POLICY_PATH=lerobot/smolvla_base
OUTPUT_DIR=""
SEED=0
STEPS=20000
BATCH_SIZE=8
RANK=16
ALPHA=""
LR=5e-5
SAVE_FREQ=10000
NUM_WORKERS=4
WANDB=false
FREEZE_STRATEGY=lora_only

usage() {
  printf '%s\n' "Usage: $0 --dataset-repo-id ID --output-dir DIR [options]" \
    "  --policy-path PATH|HF_ID    default: $POLICY_PATH" \
    "  --dataset-root DIR          optional local dataset snapshot" \
    "  --dataset-suite NAME        libero_spatial|libero_object|libero_goal|libero_10|all (default: $DATASET_SUITE)" \
    "  --seed N                    default: $SEED" \
    "  --steps N                   default: $STEPS" \
    "  --batch-size N              default: $BATCH_SIZE" \
    "  --lora-rank N               default: $RANK" \
    "  --lora-alpha N              default: 2 * rank" \
    "  --lr FLOAT                  default: $LR" \
    "  --save-freq N               default: $SAVE_FREQ" \
    "  --num-workers N             default: $NUM_WORKERS" \
    "  --freeze-strategy NAME      lora_only|lora_action_head (default: $FREEZE_STRATEGY)" \
    "  --wandb true|false          default: $WANDB"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-repo-id) require_value "$@"; DATASET="$2"; shift 2 ;;
    --dataset-root) require_value "$@"; DATASET_ROOT="$2"; shift 2 ;;
    --dataset-suite) require_value "$@"; DATASET_SUITE="$2"; shift 2 ;;
    --policy-path) require_value "$@"; POLICY_PATH="$2"; shift 2 ;;
    --output-dir) require_value "$@"; OUTPUT_DIR="$2"; shift 2 ;;
    --seed) require_value "$@"; SEED="$2"; shift 2 ;;
    --steps) require_value "$@"; STEPS="$2"; shift 2 ;;
    --batch-size) require_value "$@"; BATCH_SIZE="$2"; shift 2 ;;
    --lora-rank) require_value "$@"; RANK="$2"; shift 2 ;;
    --lora-alpha) require_value "$@"; ALPHA="$2"; shift 2 ;;
    --lr) require_value "$@"; LR="$2"; shift 2 ;;
    --save-freq) require_value "$@"; SAVE_FREQ="$2"; shift 2 ;;
    --num-workers) require_value "$@"; NUM_WORKERS="$2"; shift 2 ;;
    --freeze-strategy) require_value "$@"; FREEZE_STRATEGY="$2"; shift 2 ;;
    --wandb) require_value "$@"; WANDB="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ -n "$DATASET" ]] || { usage >&2; die "--dataset-repo-id is required"; }
if [[ -n "$DATASET_ROOT" ]]; then
  [[ -f "$DATASET_ROOT/meta/info.json" ]] || die "--dataset-root does not contain meta/info.json: $DATASET_ROOT"
fi
case "$DATASET_SUITE" in
  libero_spatial|libero_object|libero_goal|libero_10|all) ;;
  *) die "unsupported --dataset-suite: $DATASET_SUITE" ;;
esac
if [[ "$DATASET_SUITE" != "all" && -z "$DATASET_ROOT" ]]; then
  die "--dataset-root is required when --dataset-suite is not all"
fi
[[ -n "$OUTPUT_DIR" ]] || { usage >&2; die "--output-dir is required"; }
[[ "$RANK" =~ ^[1-9][0-9]*$ ]] || die "--lora-rank must be a positive integer"
[[ "$FREEZE_STRATEGY" == "lora_only" || "$FREEZE_STRATEGY" == "lora_action_head" ]] || \
  die "--freeze-strategy must be lora_only or lora_action_head"
[[ ! -e "$OUTPUT_DIR" ]] || die "$OUTPUT_DIR already exists; use a new directory or resume manually"
ALPHA="${ALPHA:-$((2 * RANK))}"
set_reproducible_env "$SEED"

dataset_args=("--dataset.repo_id=$DATASET")
if [[ -n "$DATASET_ROOT" ]]; then
  dataset_args+=("--dataset.root=$DATASET_ROOT")
fi
if [[ "$DATASET_SUITE" != "all" ]]; then
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    python_bin="$PYTHON_BIN"
  elif command -v lerobot-train >/dev/null 2>&1 && \
      [[ -x "$(dirname "$(command -v lerobot-train)")/python" ]]; then
    # Use the same environment as lerobot-train. This avoids accidentally
    # selecting /usr/bin/python3, which may not contain the pyarrow dependency.
    python_bin="$(dirname "$(command -v lerobot-train)")/python"
  else
    python_bin=python3
  fi
  selected_episodes="$("$python_bin" "$SCRIPT_DIR/select_libero_episodes.py" \
    --dataset-root "$DATASET_ROOT" \
    --suite "$DATASET_SUITE")"
  dataset_args+=("--dataset.episodes=$selected_episodes")
fi

peft_args=(
  --peft.method_type=LORA
  "--peft.r=$RANK"
  "--peft.lora_alpha=$ALPHA"
)
if [[ "$FREEZE_STRATEGY" == "lora_action_head" ]]; then
  peft_args+=(
    '--peft.target_modules=(model\.vlm_with_expert\.lm_expert\..*\.(q|v)_proj|model\.state_proj)'
    '--peft.full_training_modules=["action_in_proj","action_out_proj","action_time_mlp_in","action_time_mlp_out"]'
  )
fi

run_lerobot lerobot-train \
  "--policy.path=$POLICY_PATH" \
  --policy.push_to_hub=false \
  "--policy.optimizer_lr=$LR" \
  "${peft_args[@]}" \
  "${dataset_args[@]}" \
  "--output_dir=$OUTPUT_DIR" \
  "--seed=$SEED" \
  --cudnn_deterministic=true \
  "--steps=$STEPS" \
  "--batch_size=$BATCH_SIZE" \
  "--num_workers=$NUM_WORKERS" \
  "--save_freq=$SAVE_FREQ" \
  "--wandb.enable=$WANDB"
