#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

[[ $# -ge 1 ]] || { printf 'Usage: %s DATASET_REPO_ID [POLICY_PATH] [STEPS]\n' "$0" >&2; exit 2; }
dataset="$1"
policy="${2:-lerobot/smolvla_base}"
steps="${3:-20000}"
seeds="${SEEDS:-0}"
lr="${LR:-5e-5}"
lora_alpha="${LORA_ALPHA:-32}"
ranks_csv="${RANKS:-8,32}"
dataset_root="${DATASET_ROOT:-}"
dataset_suite="${DATASET_SUITE:-libero_spatial}"
batch_size="${BATCH_SIZE:-8}"
save_freq="${SAVE_FREQ:-10000}"
num_workers="${NUM_WORKERS:-4}"
freeze_strategy="${FREEZE_STRATEGY:-lora_only}"
wandb="${WANDB:-false}"

mkdir -p runs/logs

IFS=',' read -r -a ranks <<< "$ranks_csv"
IFS=',' read -r -a seed_list <<< "$seeds"
for rank in "${ranks[@]}"; do
  for seed in "${seed_list[@]}"; do
    run_name="${dataset_suite}_lora_r${rank}_lr${lr}_seed${seed}"
    dataset_args=(--dataset-repo-id "$dataset")
    if [[ -n "$dataset_root" ]]; then
      dataset_args+=(--dataset-root "$dataset_root")
    fi
    dataset_args+=(--dataset-suite "$dataset_suite")
    /usr/bin/time -v "$SCRIPT_DIR/train_smolvla_peft.sh" \
      "${dataset_args[@]}" \
      --policy-path "$policy" \
      --output-dir "runs/peft/rank/$run_name" \
      --seed "$seed" \
      --lora-rank "$rank" \
      --lora-alpha "$lora_alpha" \
      --lr "$lr" \
      --steps "$steps" \
      --batch-size "$batch_size" \
      --save-freq "$save_freq" \
      --num-workers "$num_workers" \
      --freeze-strategy "$freeze_strategy" \
      --wandb "$wandb" \
      2>&1 | tee "runs/logs/${run_name}.log"
  done
done
