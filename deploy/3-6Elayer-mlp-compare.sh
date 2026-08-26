#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$REPO_ROOT/scripts/common.sh"

POLICY_PATH="$REPO_ROOT/smolvla_libero"
SCALES_JSON="$REPO_ROOT/runs/deploy/3-3B4/activation_scales.json"
OUTPUT_ROOT="$REPO_ROOT/runs/deploy/3-6/E-layer-mlp-compare"
DEVICE=cuda
SEED=1000
BATCH_SIZE=1
TOKEN_LENGTH=48
INPUT_DTYPE=fp32
RUN_CLIPPING=true
PYTHON="${PYTHON_BIN:-python3}"

usage() {
  printf '%s\n' "Usage: $0 [options]" \
    "  --policy-path PATH       default: $POLICY_PATH" \
    "  --scales-json PATH      default: $SCALES_JSON" \
    "  --output-root DIR       default: $OUTPUT_ROOT" \
    "  --device DEVICE         default: $DEVICE" \
    "  --seed N                default: $SEED" \
    "  --batch-size N          default: $BATCH_SIZE" \
    "  --token-length N        default: $TOKEN_LENGTH" \
    "  --input-dtype fp32|bf16 default: $INPUT_DTYPE" \
    "  --run-clipping true|false default: $RUN_CLIPPING" \
    "  PYTHON_BIN=/path/python optional Python with torch/lerobot dependencies"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --policy-path) require_value "$@"; POLICY_PATH="$2"; shift 2 ;;
    --scales-json) require_value "$@"; SCALES_JSON="$2"; shift 2 ;;
    --output-root) require_value "$@"; OUTPUT_ROOT="$2"; shift 2 ;;
    --device) require_value "$@"; DEVICE="$2"; shift 2 ;;
    --seed) require_value "$@"; SEED="$2"; shift 2 ;;
    --batch-size) require_value "$@"; BATCH_SIZE="$2"; shift 2 ;;
    --token-length) require_value "$@"; TOKEN_LENGTH="$2"; shift 2 ;;
    --input-dtype) require_value "$@"; INPUT_DTYPE="$2"; shift 2 ;;
    --run-clipping) require_value "$@"; RUN_CLIPPING="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

case "$INPUT_DTYPE" in fp32|bf16) ;; *) die "--input-dtype must be fp32 or bf16" ;; esac
case "$RUN_CLIPPING" in true|false) ;; *) die "--run-clipping must be true or false" ;; esac

mkdir -p "$OUTPUT_ROOT"

export LEROBOT_SRC="${LEROBOT_SRC:-$REPO_ROOT/lerobot/src}"
export PYTHONPATH="$SCRIPT_DIR:$LEROBOT_SRC${PYTHONPATH:+:$PYTHONPATH}"

[[ -f "$SCALES_JSON" ]] || die "$SCALES_JSON not found; run deploy/3-3B4export.sh first or pass --scales-json"

VARIANTS=(gate_only up_only down_only gate_up all_mlp)
declare -A REGEXES
REGEXES[gate_only]="^model\\.vlm_with_expert\\.vlm\\.model\\.text_model\\.layers\\.[0-9]+\\.mlp\\.gate_proj$"
REGEXES[up_only]="^model\\.vlm_with_expert\\.vlm\\.model\\.text_model\\.layers\\.[0-9]+\\.mlp\\.up_proj$"
REGEXES[down_only]="^model\\.vlm_with_expert\\.vlm\\.model\\.text_model\\.layers\\.[0-9]+\\.mlp\\.down_proj$"
REGEXES[gate_up]="^model\\.vlm_with_expert\\.vlm\\.model\\.text_model\\.layers\\.[0-9]+\\.mlp\\.(gate_proj|up_proj)$"
REGEXES[all_mlp]="^model\\.vlm_with_expert\\.vlm\\.model\\.text_model\\.layers\\.[0-9]+\\.mlp\\.(gate_proj|up_proj|down_proj)$"

for variant in "${VARIANTS[@]}"; do
  run_dir="$OUTPUT_ROOT/$variant"
  module_regex="${REGEXES[$variant]}"
  mkdir -p "$run_dir"

  "$PYTHON" "$SCRIPT_DIR/diagnose_3_4_numeric_baseline.py" compare \
    --policy-path "$POLICY_PATH" \
    --device "$DEVICE" \
    --seed "$SEED" \
    --batch-size "$BATCH_SIZE" \
    --token-length "$TOKEN_LENGTH" \
    --input-dtype "$INPUT_DTYPE" \
    --quantize-module-regex "$module_regex" \
    --fake-quant-activation-scale-mode calibrated \
    --fake-quant-activation-scales-json "$SCALES_JSON" \
    --output-dir "$run_dir/numeric_baseline"

  if [[ "$RUN_CLIPPING" == "true" ]]; then
    "$PYTHON" "$SCRIPT_DIR/check_static_scale_clipping.py" \
      --policy-path "$POLICY_PATH" \
      --activation-scales-json "$SCALES_JSON" \
      --output-dir "$run_dir/clipping" \
      --device "$DEVICE" \
      --seed "$SEED" \
      --num-seeds 1 \
      --batch-size "$BATCH_SIZE" \
      --token-length "$TOKEN_LENGTH" \
      --input-dtype "$INPUT_DTYPE" \
      --module-regex "$module_regex"
  fi
done

"$PYTHON" "$SCRIPT_DIR/summarize_layer_mlp_compare.py" \
  --output-root "$OUTPUT_ROOT" \
  --variants "${VARIANTS[@]}"

printf '%s\n' "step 3-6E MLP sub-layer compare outputs:" \
  "  output root: $OUTPUT_ROOT" \
  "  summary json: $OUTPUT_ROOT/layer_mlp_compare_summary.json" \
  "  summary csv: $OUTPUT_ROOT/layer_mlp_compare_summary.csv" \
  "  activation scales: $SCALES_JSON"
