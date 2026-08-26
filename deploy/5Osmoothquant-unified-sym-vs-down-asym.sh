#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$REPO_ROOT/scripts/common.sh"

POLICY_PATH="$REPO_ROOT/smolvla_libero"
OUTPUT_ROOT="$REPO_ROOT/runs/deploy/5/O-smoothquant-unified-sym-vs-down-asym"
SOURCE_CHANNEL_SCALES="$REPO_ROOT/runs/deploy/5/O-smoothquant-fake-dequant/activation_channel_scales_512_p99.995.json"
TASK_SUITE=libero_spatial
DEVICE=cuda
SEED=1000
BATCH_SIZE=1
TOKEN_LENGTH=48
CALIBRATION_SAMPLES=512
CALIBRATION_PERCENTILE=99.995
ALPHAS=(0.85 0.9)
COMPARE_SAMPLES=50
LAYER_COMPARE_SAMPLES=1
SAMPLE_STRIDE=5
RUN_SOURCE_CALIBRATION=true
RUN_LAYER_CURVES=true
PYTHON="${PYTHON_BIN:-python3}"

QUANT_MODULE_REGEX="^model\\.vlm_with_expert\\.vlm\\.model\\.text_model\\.layers\\.[0-9]+\\.(mlp\\.(gate_proj|up_proj|down_proj)|self_attn\\.(q_proj|k_proj|v_proj|o_proj))$"
DOWN_MODULE_REGEX="^model\\.vlm_with_expert\\.vlm\\.model\\.text_model\\.layers\\.[0-9]+\\.mlp\\.down_proj$"

usage() {
  printf '%s\n' "Usage: $0 [options]" \
    "  --policy-path PATH              default: $POLICY_PATH" \
    "  --output-root DIR              default: $OUTPUT_ROOT" \
    "  --source-channel-scales PATH   default: $SOURCE_CHANNEL_SCALES" \
    "  --tasks NAME                   default: $TASK_SUITE" \
    "  --device DEVICE                default: $DEVICE" \
    "  --seed N                       default: $SEED" \
    "  --batch-size N                 default: $BATCH_SIZE" \
    "  --token-length N               default: $TOKEN_LENGTH" \
    "  --calibration-samples N        default: $CALIBRATION_SAMPLES" \
    "  --calibration-percentile P     default: $CALIBRATION_PERCENTILE" \
    "  --alphas LIST                  default: ${ALPHAS[*]}" \
    "  --compare-samples N            default: $COMPARE_SAMPLES" \
    "  --layer-compare-samples N      default: $LAYER_COMPARE_SAMPLES" \
    "  --sample-stride N              default: $SAMPLE_STRIDE" \
    "  --run-source-calibration true|false default: $RUN_SOURCE_CALIBRATION" \
    "  --run-layer-curves true|false  default: $RUN_LAYER_CURVES" \
    "  PYTHON_BIN=/path/python        optional Python with torch/lerobot dependencies"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --policy-path) require_value "$@"; POLICY_PATH="$2"; shift 2 ;;
    --output-root) require_value "$@"; OUTPUT_ROOT="$2"; shift 2 ;;
    --source-channel-scales) require_value "$@"; SOURCE_CHANNEL_SCALES="$2"; shift 2 ;;
    --tasks) require_value "$@"; TASK_SUITE="$2"; shift 2 ;;
    --device) require_value "$@"; DEVICE="$2"; shift 2 ;;
    --seed) require_value "$@"; SEED="$2"; shift 2 ;;
    --batch-size) require_value "$@"; BATCH_SIZE="$2"; shift 2 ;;
    --token-length) require_value "$@"; TOKEN_LENGTH="$2"; shift 2 ;;
    --calibration-samples) require_value "$@"; CALIBRATION_SAMPLES="$2"; shift 2 ;;
    --calibration-percentile) require_value "$@"; CALIBRATION_PERCENTILE="$2"; shift 2 ;;
    --alphas) require_value "$@"; read -r -a ALPHAS <<< "$2"; shift 2 ;;
    --compare-samples) require_value "$@"; COMPARE_SAMPLES="$2"; shift 2 ;;
    --layer-compare-samples) require_value "$@"; LAYER_COMPARE_SAMPLES="$2"; shift 2 ;;
    --sample-stride) require_value "$@"; SAMPLE_STRIDE="$2"; shift 2 ;;
    --run-source-calibration) require_value "$@"; RUN_SOURCE_CALIBRATION="$2"; shift 2 ;;
    --run-layer-curves) require_value "$@"; RUN_LAYER_CURVES="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

case "$RUN_SOURCE_CALIBRATION" in true|false) ;; *) die "--run-source-calibration must be true or false" ;; esac
case "$RUN_LAYER_CURVES" in true|false) ;; *) die "--run-layer-curves must be true or false" ;; esac

mkdir -p "$OUTPUT_ROOT"
export NUMBA_DISABLE_JIT="${NUMBA_DISABLE_JIT:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-smolvla}"
export LEROBOT_SRC="${LEROBOT_SRC:-$REPO_ROOT/lerobot/src}"
export PYTHONPATH="$SCRIPT_DIR:$LEROBOT_SRC${PYTHONPATH:+:$PYTHONPATH}"

if [[ "$RUN_SOURCE_CALIBRATION" == "true" ]]; then
  SOURCE_CHANNEL_SCALES="$OUTPUT_ROOT/activation_channel_scales_${CALIBRATION_SAMPLES}_p${CALIBRATION_PERCENTILE}.json"
  "$PYTHON" "$SCRIPT_DIR/calibrate_smolvla_activation_channel_scales.py" \
    --policy-path "$POLICY_PATH" \
    --output "$SOURCE_CHANNEL_SCALES" \
    --device "$DEVICE" \
    --tasks "$TASK_SUITE" \
    --seed "$SEED" \
    --samples "$CALIBRATION_SAMPLES" \
    --percentile "$CALIBRATION_PERCENTILE" \
    --batch-size "$BATCH_SIZE" \
    --max-parallel-tasks 1 \
    --token-length "$TOKEN_LENGTH" \
    --include-module-regex "$QUANT_MODULE_REGEX"
else
  [[ -f "$SOURCE_CHANNEL_SCALES" ]] || die "$SOURCE_CHANNEL_SCALES not found"
fi

for alpha in "${ALPHAS[@]}"; do
  alpha_dir="alpha_${alpha//./_}"
  run_dir="$OUTPUT_ROOT/$alpha_dir"
  smooth_scales_json="$run_dir/symmetric/smoothquant_alpha_${alpha}_activation_scales.json"
  down_asym_scales_json="$run_dir/down_asym/smoothquant_alpha_${alpha}_down_asym_activation_scales.json"
  mkdir -p "$run_dir/symmetric" "$run_dir/down_asym"

  "$PYTHON" "$SCRIPT_DIR/build_smoothquant_scales.py" \
    --policy-path "$POLICY_PATH" \
    --activation-channel-scales-json "$SOURCE_CHANNEL_SCALES" \
    --output "$smooth_scales_json" \
    --device "$DEVICE" \
    --alpha "$alpha" \
    --include-module-regex "$QUANT_MODULE_REGEX"

  "$PYTHON" "$SCRIPT_DIR/calibrate_smoothquant_down_asym_scales.py" \
    --policy-path "$POLICY_PATH" \
    --smoothquant-scales-json "$smooth_scales_json" \
    --output "$down_asym_scales_json" \
    --device "$DEVICE" \
    --tasks "$TASK_SUITE" \
    --seed "$SEED" \
    --samples "$CALIBRATION_SAMPLES" \
    --percentile "$CALIBRATION_PERCENTILE" \
    --batch-size "$BATCH_SIZE" \
    --max-parallel-tasks 1 \
    --token-length "$TOKEN_LENGTH" \
    --down-module-regex "$DOWN_MODULE_REGEX"

  for variant in symmetric down_asym; do
    if [[ "$variant" == "symmetric" ]]; then
      scales="$smooth_scales_json"
    else
      scales="$down_asym_scales_json"
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
      --input-source rollout \
      --compare-samples "$COMPARE_SAMPLES" \
      --sample-stride "$SAMPLE_STRIDE" \
      --max-parallel-tasks 1 \
      --quantize-module-regex "$QUANT_MODULE_REGEX" \
      --fake-quant-kind w8a8 \
      --fake-quant-activation-scale-mode calibrated \
      --fake-quant-activation-scales-json "$scales" \
      --output-dir "$run_dir/$variant/fake_dequant_numeric"

    if [[ "$RUN_LAYER_CURVES" == "true" ]]; then
      "$PYTHON" "$SCRIPT_DIR/compare_5h_text_vlm_layer_outputs.py" \
        --policy-path "$POLICY_PATH" \
        --activation-scales-json "$scales" \
        --output-dir "$run_dir/$variant/layer_outputs" \
        --device "$DEVICE" \
        --seed "$SEED" \
        --task "$TASK_SUITE" \
        --batch-size "$BATCH_SIZE" \
        --token-length "$TOKEN_LENGTH" \
        --model-dtype bf16 \
        --input-dtype bf16 \
        --input-source rollout \
        --compare-samples "$LAYER_COMPARE_SAMPLES" \
        --sample-stride "$SAMPLE_STRIDE" \
        --max-parallel-tasks 1 \
        --quantize-module-regex "$QUANT_MODULE_REGEX"
    fi
  done
done

"$PYTHON" - "$OUTPUT_ROOT" "${ALPHAS[@]}" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
alphas = sys.argv[2:]
rows = []

def metric(summary, output, key):
    for row in summary["critical_outputs"]:
        if row.get("pair") == "A_vs_E" and row.get("output") == output:
            return row.get(key)
    return None

for alpha in alphas:
    alpha_dir = f"alpha_{alpha.replace('.', '_')}"
    for variant in ("symmetric", "down_asym"):
        path = root / alpha_dir / variant / "fake_dequant_numeric" / "numeric_baseline_summary.json"
        if not path.exists():
            rows.append({"alpha": float(alpha), "variant": variant, "status": "missing"})
            continue
        summary = json.loads(path.read_text())
        rows.append(
            {
                "alpha": float(alpha),
                "variant": variant,
                "status": "ok",
                "sample_count": summary.get("sample_count"),
                "action_l2_mean": metric(summary, "action_chunk", "relative_l2_error_mean"),
                "action_l2_p95": metric(summary, "action_chunk", "relative_l2_error_p95"),
                "action_cosine": metric(summary, "action_chunk", "cosine_similarity_mean"),
                "prefix_l2_mean": metric(summary, "prefix_out", "relative_l2_error_mean"),
                "v_t_step09_l2_mean": metric(summary, "v_t_step_09", "relative_l2_error_mean"),
                "x_t_step09_l2_mean": metric(summary, "x_t_step_09", "relative_l2_error_mean"),
            }
        )

(root / "unified_sym_vs_down_asym_summary.json").write_text(json.dumps({"rows": rows}, indent=2))
with (root / "unified_sym_vs_down_asym_summary.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=sorted({key for row in rows for key in row}))
    writer.writeheader()
    writer.writerows(rows)
print(json.dumps({"rows": rows}, indent=2))
PY

printf '%s\n' "step 5O unified SmoothQuant symmetric vs down-asym outputs:" \
  "  output root: $OUTPUT_ROOT" \
  "  source per-channel scales: $SOURCE_CHANNEL_SCALES" \
  "  summary json: $OUTPUT_ROOT/unified_sym_vs_down_asym_summary.json" \
  "  summary csv: $OUTPUT_ROOT/unified_sym_vs_down_asym_summary.csv" \
  "  alphas: ${ALPHAS[*]}"
