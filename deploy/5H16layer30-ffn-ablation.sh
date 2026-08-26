#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$REPO_ROOT/scripts/common.sh"

POLICY_PATH="$REPO_ROOT/smolvla_libero"
OUTPUT_ROOT="$REPO_ROOT/runs/deploy/5/H-bf16-int8-w8a8/layer30_ffn_ablation"
TASK_SUITE=libero_spatial
DEVICE=cuda
SEED=1000
BATCH_SIZE=1
TOKEN_LENGTH=48
CALIBRATION_SAMPLES=512
CALIBRATION_PERCENTILE=99.999
COMPARE_SAMPLES=50
SAMPLE_STRIDE=5
INPUT_SOURCE=rollout
RUN_CALIBRATION=true
CONFIGS="skip_l30_ffn,skip_l30_down,skip_l30_gate_up,skip_l30_gate,skip_l30_up"
PYTHON="${PYTHON_BIN:-python3}"

HOOK_MODULE_REGEX="^model\\.vlm_with_expert\\.vlm\\.model\\.text_model\\.layers\\.([0-9]+)\\.(?:self_attn\\.(q_proj|k_proj|v_proj|o_proj)|mlp\\.(gate_proj|up_proj|down_proj))$"
FULL_TARGET="self_attn\\.(q_proj|k_proj|v_proj|o_proj)|mlp\\.(gate_proj|up_proj|down_proj)"

usage() {
  printf '%s\n' "Usage: $0 [options]" \
    "  --policy-path PATH              default: $POLICY_PATH" \
    "  --output-root DIR              default: $OUTPUT_ROOT" \
    "  --tasks NAME                   default: $TASK_SUITE" \
    "  --device DEVICE                default: $DEVICE" \
    "  --seed N                       default: $SEED" \
    "  --batch-size N                 default: $BATCH_SIZE" \
    "  --token-length N               default: $TOKEN_LENGTH" \
    "  --calibration-samples N        default: $CALIBRATION_SAMPLES" \
    "  --calibration-percentile P     default: $CALIBRATION_PERCENTILE" \
    "  --input-source synthetic|rollout default: $INPUT_SOURCE" \
    "  --compare-samples N            default: $COMPARE_SAMPLES" \
    "  --sample-stride N              default: $SAMPLE_STRIDE" \
    "  --run-calibration true|false   default: $RUN_CALIBRATION" \
    "  --configs CSV                  default: $CONFIGS" \
    "                                 known: skip_l30_ffn,skip_l30_down,skip_l30_gate_up,skip_l30_gate,skip_l30_up" \
    "  PYTHON_BIN=/path/python        optional Python with torch/lerobot dependencies"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --policy-path) require_value "$@"; POLICY_PATH="$2"; shift 2 ;;
    --output-root) require_value "$@"; OUTPUT_ROOT="$2"; shift 2 ;;
    --tasks) require_value "$@"; TASK_SUITE="$2"; shift 2 ;;
    --device) require_value "$@"; DEVICE="$2"; shift 2 ;;
    --seed) require_value "$@"; SEED="$2"; shift 2 ;;
    --batch-size) require_value "$@"; BATCH_SIZE="$2"; shift 2 ;;
    --token-length) require_value "$@"; TOKEN_LENGTH="$2"; shift 2 ;;
    --calibration-samples) require_value "$@"; CALIBRATION_SAMPLES="$2"; shift 2 ;;
    --calibration-percentile) require_value "$@"; CALIBRATION_PERCENTILE="$2"; shift 2 ;;
    --input-source) require_value "$@"; INPUT_SOURCE="$2"; shift 2 ;;
    --compare-samples) require_value "$@"; COMPARE_SAMPLES="$2"; shift 2 ;;
    --sample-stride) require_value "$@"; SAMPLE_STRIDE="$2"; shift 2 ;;
    --run-calibration) require_value "$@"; RUN_CALIBRATION="$2"; shift 2 ;;
    --configs) require_value "$@"; CONFIGS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

case "$INPUT_SOURCE" in synthetic|rollout) ;; *) die "--input-source must be synthetic or rollout" ;; esac
case "$RUN_CALIBRATION" in true|false) ;; *) die "--run-calibration must be true or false" ;; esac

mkdir -p "$OUTPUT_ROOT"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-smolvla}"
export LEROBOT_SRC="${LEROBOT_SRC:-$REPO_ROOT/lerobot/src}"
export PYTHONPATH="$SCRIPT_DIR:$LEROBOT_SRC${PYTHONPATH:+:$PYTHONPATH}"

quant_regex_for_config() {
  local config="$1"
  case "$config" in
    skip_l30_ffn)
      printf '^model\\.vlm_with_expert\\.vlm\\.model\\.text_model\\.layers\\.(?!(30\\.mlp\\.(gate_proj|up_proj|down_proj)$))[0-9]+\\.(%s)$' "$FULL_TARGET"
      ;;
    skip_l30_down)
      printf '^model\\.vlm_with_expert\\.vlm\\.model\\.text_model\\.layers\\.(?!(30\\.mlp\\.down_proj$))[0-9]+\\.(%s)$' "$FULL_TARGET"
      ;;
    skip_l30_gate_up)
      printf '^model\\.vlm_with_expert\\.vlm\\.model\\.text_model\\.layers\\.(?!(30\\.mlp\\.(gate_proj|up_proj)$))[0-9]+\\.(%s)$' "$FULL_TARGET"
      ;;
    skip_l30_gate)
      printf '^model\\.vlm_with_expert\\.vlm\\.model\\.text_model\\.layers\\.(?!(30\\.mlp\\.gate_proj$))[0-9]+\\.(%s)$' "$FULL_TARGET"
      ;;
    skip_l30_up)
      printf '^model\\.vlm_with_expert\\.vlm\\.model\\.text_model\\.layers\\.(?!(30\\.mlp\\.up_proj$))[0-9]+\\.(%s)$' "$FULL_TARGET"
      ;;
    *)
      die "unknown config: $config"
      ;;
  esac
}

run_one_config() {
  local config="$1"
  local run_dir="$OUTPUT_ROOT/$config"
  local scales_json="$run_dir/activation_scales_${CALIBRATION_SAMPLES}_p${CALIBRATION_PERCENTILE}.json"
  local quant_module_regex
  quant_module_regex="$(quant_regex_for_config "$config")"

  mkdir -p "$run_dir"
  printf '%s\n' "==== 5H.16 $config ====" "quantize regex: $quant_module_regex"

  if [[ "$RUN_CALIBRATION" == "true" ]]; then
    "$PYTHON" "$SCRIPT_DIR/calibrate_smolvla_activation_scales.py" \
      --policy-path "$POLICY_PATH" \
      --output "$scales_json" \
      --device "$DEVICE" \
      --tasks "$TASK_SUITE" \
      --seed "$SEED" \
      --samples "$CALIBRATION_SAMPLES" \
      --percentile "$CALIBRATION_PERCENTILE" \
      --batch-size "$BATCH_SIZE" \
      --max-parallel-tasks 1 \
      --token-length "$TOKEN_LENGTH" \
      --include-module-regex "$quant_module_regex"
  else
    [[ -f "$scales_json" ]] || die "$scales_json not found; enable --run-calibration true"
  fi

  "$PYTHON" "$SCRIPT_DIR/diagnose_3_4_numeric_baseline.py" compare \
    --policy-path "$POLICY_PATH" \
    --device "$DEVICE" \
    --seed "$SEED" \
    --task "$TASK_SUITE" \
    --batch-size "$BATCH_SIZE" \
    --token-length "$TOKEN_LENGTH" \
    --model-dtype bf16 \
    --input-dtype bf16 \
    --input-source "$INPUT_SOURCE" \
    --compare-samples "$COMPARE_SAMPLES" \
    --sample-stride "$SAMPLE_STRIDE" \
    --max-parallel-tasks 1 \
    --quantize-module-regex "$quant_module_regex" \
    --fake-quant-kind w8a8 \
    --fake-quant-activation-scale-mode calibrated \
    --fake-quant-activation-scales-json "$scales_json" \
    --output-dir "$run_dir/fake_dequant_numeric"

  "$PYTHON" "$SCRIPT_DIR/compare_5h_text_vlm_layer_outputs.py" \
    --policy-path "$POLICY_PATH" \
    --activation-scales-json "$scales_json" \
    --output-dir "$run_dir/layer_outputs" \
    --device "$DEVICE" \
    --seed "$SEED" \
    --task "$TASK_SUITE" \
    --batch-size "$BATCH_SIZE" \
    --token-length "$TOKEN_LENGTH" \
    --model-dtype bf16 \
    --input-dtype bf16 \
    --input-source "$INPUT_SOURCE" \
    --compare-samples "$COMPARE_SAMPLES" \
    --sample-stride "$SAMPLE_STRIDE" \
    --module-regex "$HOOK_MODULE_REGEX" \
    --quantize-module-regex "$quant_module_regex"

  "$PYTHON" "$SCRIPT_DIR/compare_5h_transformer_block_outputs.py" \
    --policy-path "$POLICY_PATH" \
    --activation-scales-json "$scales_json" \
    --output-dir "$run_dir/block_outputs" \
    --device "$DEVICE" \
    --seed "$SEED" \
    --task "$TASK_SUITE" \
    --batch-size "$BATCH_SIZE" \
    --token-length "$TOKEN_LENGTH" \
    --model-dtype bf16 \
    --input-dtype bf16 \
    --input-source "$INPUT_SOURCE" \
    --compare-samples "$COMPARE_SAMPLES" \
    --sample-stride "$SAMPLE_STRIDE" \
    --max-parallel-tasks 1 \
    --quantize-module-regex "$quant_module_regex"
}

IFS=',' read -r -a config_list <<< "$CONFIGS"
for config in "${config_list[@]}"; do
  config="${config//[[:space:]]/}"
  [[ -n "$config" ]] || continue
  run_one_config "$config"
done

"$PYTHON" "$SCRIPT_DIR/summarize_5h16_layer30_ablation.py" \
  --root "$OUTPUT_ROOT"

printf '%s\n' "step 5H.16 layer30 FFN ablation outputs:" \
  "  output root: $OUTPUT_ROOT" \
  "  summary: $OUTPUT_ROOT/summary.json" \
  "  summary csv: $OUTPUT_ROOT/summary.csv" \
  "  action/block plot: $OUTPUT_ROOT/summary_action_block_l2.png" \
  "  layer30 FFN sublayer plot: $OUTPUT_ROOT/summary_layer30_ffn_sublayer_l2.png"
