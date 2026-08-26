#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
POLICY_PATH="${POLICY_PATH:-${REPO_ROOT}/smolvla_libero}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/runs/deploy/6/B-attention-breakdown}"
TASK_SUITE="${TASK_SUITE:-libero_spatial}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-1000}"
BATCH_SIZE="${BATCH_SIZE:-1}"
TOKEN_LENGTH="${TOKEN_LENGTH:-48}"
INPUT_SOURCE="${INPUT_SOURCE:-rollout}"
SAMPLES="${SAMPLES:-10}"
SAMPLE_STRIDE="${SAMPLE_STRIDE:-5}"
MAX_PARALLEL_TASKS="${MAX_PARALLEL_TASKS:-1}"
WARMUP="${WARMUP:-2}"
ITERS="${ITERS:-10}"
MODEL_DTYPE="${MODEL_DTYPE:-bf16}"
INPUT_DTYPE="${INPUT_DTYPE:-bf16}"
RUN_NATIVE="${RUN_NATIVE:-true}"
RUN_SMOOTHQUANT="${RUN_SMOOTHQUANT:-true}"

SCALES_JSON="${SCALES_JSON:-${REPO_ROOT}/runs/deploy/5/O-smoothquant-alpha085-full-w8a8-deploy-cudagraph/smoothquant_alpha_0.85_text_and_lm_expert_activation_scales.json}"
QUANT_MODULE_REGEX="${QUANT_MODULE_REGEX:-^(model\\.vlm_with_expert\\.vlm\\.model\\.text_model\\.layers|model\\.vlm_with_expert\\.lm_expert\\.layers)\\.[0-9]+\\.(self_attn\\.(q_proj|k_proj|v_proj|o_proj)|mlp\\.(gate_proj|up_proj|down_proj))$}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --policy-path) POLICY_PATH="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --task) TASK_SUITE="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --token-length) TOKEN_LENGTH="$2"; shift 2 ;;
    --input-source) INPUT_SOURCE="$2"; shift 2 ;;
    --samples) SAMPLES="$2"; shift 2 ;;
    --sample-stride) SAMPLE_STRIDE="$2"; shift 2 ;;
    --max-parallel-tasks) MAX_PARALLEL_TASKS="$2"; shift 2 ;;
    --warmup) WARMUP="$2"; shift 2 ;;
    --iters) ITERS="$2"; shift 2 ;;
    --model-dtype) MODEL_DTYPE="$2"; shift 2 ;;
    --input-dtype) INPUT_DTYPE="$2"; shift 2 ;;
    --run-native) RUN_NATIVE="$2"; shift 2 ;;
    --run-smoothquant) RUN_SMOOTHQUANT="$2"; shift 2 ;;
    --scales-json) SCALES_JSON="$2"; shift 2 ;;
    --quant-module-regex) QUANT_MODULE_REGEX="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "${OUTPUT_ROOT}"

COMMON_ARGS=(
  --policy-path "${POLICY_PATH}"
  --device "${DEVICE}"
  --task "${TASK_SUITE}"
  --seed "${SEED}"
  --batch-size "${BATCH_SIZE}"
  --token-length "${TOKEN_LENGTH}"
  --input-source "${INPUT_SOURCE}"
  --samples "${SAMPLES}"
  --sample-stride "${SAMPLE_STRIDE}"
  --max-parallel-tasks "${MAX_PARALLEL_TASKS}"
  --warmup "${WARMUP}"
  --iters "${ITERS}"
  --model-dtype "${MODEL_DTYPE}"
  --input-dtype "${INPUT_DTYPE}"
)

if [[ "${RUN_NATIVE}" == "true" ]]; then
  "${PYTHON_BIN}" deploy/profile_6b_attention_breakdown.py \
    "${COMMON_ARGS[@]}" \
    --output-dir "${OUTPUT_ROOT}/native"
fi

if [[ "${RUN_SMOOTHQUANT}" == "true" ]]; then
  "${PYTHON_BIN}" deploy/profile_6b_attention_breakdown.py \
    "${COMMON_ARGS[@]}" \
    --output-dir "${OUTPUT_ROOT}/smoothquant_alpha085_full_a8w8_fake_quant" \
    --quantize-module-regex "${QUANT_MODULE_REGEX}" \
    --activation-scales-json "${SCALES_JSON}"
fi

"${PYTHON_BIN}" - "${OUTPUT_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
cases = [
    ("native", root / "native" / "attention_breakdown_summary.json"),
    ("smoothquant_alpha085_full_a8w8_fake_quant", root / "smoothquant_alpha085_full_a8w8_fake_quant" / "attention_breakdown_summary.json"),
]

candidate_segments = {
    "rope",
    "kv_expand_reshape",
    "qk_cast_fp32",
    "qk_transpose",
    "qk_matmul",
    "score_scale",
    "mask_where",
    "softmax",
    "probs_cast",
    "pv_matmul",
    "output_permute_reshape",
}
projection_segments = {"projection_q_proj", "projection_k_proj", "projection_v_proj", "projection_o_proj"}
transpose_segments = {"kv_expand_reshape", "qk_transpose", "output_permute_reshape"}

rows = []
for name, path in cases:
    if not path.exists():
        continue
    data = json.loads(path.read_text())
    seg = {k: float(v) for k, v in data.get("total_by_segment_ms", {}).items()}
    e2e = seg.get("action_core_e2e", 0.0)
    attention_chain = sum(seg.get(k, 0.0) for k in candidate_segments)
    projections = sum(seg.get(k, 0.0) for k in projection_segments)
    rows.append(
        {
            "case": name,
            "action_core_e2e_ms": e2e,
            "fused_candidate_chain_ms": attention_chain,
            "fused_candidate_share_of_e2e": attention_chain / e2e * 100.0 if e2e else 0.0,
            "projections_ms": projections,
            "projections_share_of_e2e": projections / e2e * 100.0 if e2e else 0.0,
            "rope_ms": seg.get("rope", 0.0),
            "qk_matmul_ms": seg.get("qk_matmul", 0.0),
            "softmax_ms": seg.get("softmax", 0.0),
            "pv_matmul_ms": seg.get("pv_matmul", 0.0),
            "transpose_reshape_ms": sum(seg.get(k, 0.0) for k in transpose_segments),
            "mask_scale_cast_ms": seg.get("qk_cast_fp32", 0.0) + seg.get("score_scale", 0.0) + seg.get("mask_where", 0.0) + seg.get("probs_cast", 0.0),
        }
    )

report = {
    "output_root": str(root),
    "candidate_definition": sorted(candidate_segments),
    "projection_definition": sorted(projection_segments),
    "rows": rows,
}
(root / "attention_breakdown_comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

lines = [
    "# Step 6B Attention Breakdown Comparison",
    "",
    "| case | action core E2E ms | fused candidate chain ms | chain share | projections ms | projection share | RoPE ms | QK^T ms | Softmax ms | PV ms | transpose/reshape ms | mask/scale/cast ms |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
]
for row in rows:
    lines.append(
        f"| `{row['case']}` | `{row['action_core_e2e_ms']:.3f}` | "
        f"`{row['fused_candidate_chain_ms']:.3f}` | `{row['fused_candidate_share_of_e2e']:.2f}%` | "
        f"`{row['projections_ms']:.3f}` | `{row['projections_share_of_e2e']:.2f}%` | "
        f"`{row['rope_ms']:.3f}` | `{row['qk_matmul_ms']:.3f}` | `{row['softmax_ms']:.3f}` | "
        f"`{row['pv_matmul_ms']:.3f}` | `{row['transpose_reshape_ms']:.3f}` | `{row['mask_scale_cast_ms']:.3f}` |"
    )

lines += [
    "",
    "Fusion candidate chain is `RoPE -> QK^T -> scale/mask -> softmax -> PV -> transpose/reshape`.",
    "Projection timing is shown only as the counterfactual: step 6 should not spend effort optimizing the already-small INT8 projection GEMM first.",
]
(root / "attention_breakdown_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print((root / "attention_breakdown_comparison.md").read_text(), end="")
PY
