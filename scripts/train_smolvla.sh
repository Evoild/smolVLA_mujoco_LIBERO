#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

DATASET=""
POLICY_PATH=lerobot/smolvla_base
OUTPUT_DIR=""
SEED=0
STEPS=20000
BATCH_SIZE=8
LR=1e-4
SAVE_FREQ=10000
NUM_WORKERS=4
WANDB=false

usage() {
  printf '%s\n' "Usage: $0 --dataset-repo-id ID --output-dir DIR [options]" \
    "  --policy-path PATH|HF_ID    default: $POLICY_PATH" \
    "  --seed N                    default: $SEED" \
    "  --steps N                   default: $STEPS" \
    "  --batch-size N              default: $BATCH_SIZE" \
    "  --lr FLOAT                  default: $LR" \
    "  --save-freq N               default: $SAVE_FREQ" \
    "  --num-workers N             default: $NUM_WORKERS" \
    "  --wandb true|false          default: $WANDB"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-repo-id) require_value "$@"; DATASET="$2"; shift 2 ;;
    --policy-path) require_value "$@"; POLICY_PATH="$2"; shift 2 ;;
    --output-dir) require_value "$@"; OUTPUT_DIR="$2"; shift 2 ;;
    --seed) require_value "$@"; SEED="$2"; shift 2 ;;
    --steps) require_value "$@"; STEPS="$2"; shift 2 ;;
    --batch-size) require_value "$@"; BATCH_SIZE="$2"; shift 2 ;;
    --lr) require_value "$@"; LR="$2"; shift 2 ;;
    --save-freq) require_value "$@"; SAVE_FREQ="$2"; shift 2 ;;
    --num-workers) require_value "$@"; NUM_WORKERS="$2"; shift 2 ;;
    --wandb) require_value "$@"; WANDB="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ -n "$DATASET" ]] || { usage >&2; die "--dataset-repo-id is required"; }
[[ -n "$OUTPUT_DIR" ]] || { usage >&2; die "--output-dir is required"; }
[[ ! -e "$OUTPUT_DIR" ]] || die "$OUTPUT_DIR already exists; use a new directory or resume manually"
set_reproducible_env "$SEED"

run_lerobot lerobot-train \
  "--policy.path=$POLICY_PATH" \
  --policy.push_to_hub=false \
  --policy.freeze_vision_encoder=true \
  --policy.train_expert_only=true \
  --policy.train_state_proj=true \
  "--policy.optimizer_lr=$LR" \
  "--dataset.repo_id=$DATASET" \
  "--output_dir=$OUTPUT_DIR" \
  "--seed=$SEED" \
  --cudnn_deterministic=true \
  "--steps=$STEPS" \
  "--batch_size=$BATCH_SIZE" \
  "--num_workers=$NUM_WORKERS" \
  "--save_freq=$SAVE_FREQ" \
  "--wandb.enable=$WANDB"
