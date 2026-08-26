#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON="${PYTHON_BIN:-python}"
OUTPUT_ROOT="runs/deploy/6/G-smoothquant-linear-epilogue"
POLICY_PATH="smolvla_libero"
TASK="libero_spatial"
SEED="1000"
INPUT_SOURCE="rollout"
TOKEN_LENGTH="48"
BATCH_SIZE="1"
SAMPLE_STRIDE="1"
MAX_PARALLEL_TASKS="10"
MODEL_DTYPE="bf16"
INPUT_DTYPE="bf16"
ACTIVATION_SCALES_JSON="runs/deploy/5/O-smoothquant-alpha085-full-w8a8-deploy-cudagraph/smoothquant_alpha_0.85_text_and_lm_expert_activation_scales.json"
TARGET_MODULE="model.vlm_with_expert.vlm.model.text_model.layers.0.self_attn.q_proj"
RUN_EXPORT="true"
RUN_BENCH="true"
RUN_CUBLASLT="true"
WARMUP="20"
ITERS="200"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --policy-path) POLICY_PATH="$2"; shift 2 ;;
    --task) TASK="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --input-source) INPUT_SOURCE="$2"; shift 2 ;;
    --token-length) TOKEN_LENGTH="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --sample-stride) SAMPLE_STRIDE="$2"; shift 2 ;;
    --max-parallel-tasks) MAX_PARALLEL_TASKS="$2"; shift 2 ;;
    --model-dtype) MODEL_DTYPE="$2"; shift 2 ;;
    --input-dtype) INPUT_DTYPE="$2"; shift 2 ;;
    --activation-scales-json) ACTIVATION_SCALES_JSON="$2"; shift 2 ;;
    --target-module) TARGET_MODULE="$2"; shift 2 ;;
    --run-export) RUN_EXPORT="$2"; shift 2 ;;
    --run-bench) RUN_BENCH="$2"; shift 2 ;;
    --run-cublaslt) RUN_CUBLASLT="$2"; shift 2 ;;
    --warmup) WARMUP="$2"; shift 2 ;;
    --iters) ITERS="$2"; shift 2 ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

mkdir -p "${OUTPUT_ROOT}"
TENSOR_DIR="${OUTPUT_ROOT}/tensors"
BUILD_DIR="${OUTPUT_ROOT}/build"

if [[ "${RUN_EXPORT}" == "true" ]]; then
  "${PYTHON}" deploy/export_6g_smoothquant_linear_tensors.py \
    --policy-path "${POLICY_PATH}" \
    --output-dir "${TENSOR_DIR}" \
    --device cuda \
    --seed "${SEED}" \
    --task "${TASK}" \
    --input-source "${INPUT_SOURCE}" \
    --token-length "${TOKEN_LENGTH}" \
    --batch-size "${BATCH_SIZE}" \
    --sample-stride "${SAMPLE_STRIDE}" \
    --max-parallel-tasks "${MAX_PARALLEL_TASKS}" \
    --model-dtype "${MODEL_DTYPE}" \
    --input-dtype "${INPUT_DTYPE}" \
    --activation-scales-json "${ACTIVATION_SCALES_JSON}" \
    --target-module "${TARGET_MODULE}" \
    --warmup "${WARMUP}" \
    --iters "${ITERS}" \
    2>&1 | tee "${OUTPUT_ROOT}/export_smoothquant_linear_tensors.log"
fi

cmake -S deploy/cuda -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD_DIR}" --target smolvla_smoothquant_linear_epilogue -j "${JOBS:-8}"
cmake --build "${BUILD_DIR}" --target smolvla_smoothquant_linear_cublaslt -j "${JOBS:-8}"

ACTIVATION_SCALE="$("${PYTHON}" - "${TENSOR_DIR}" <<'PY'
import json
import sys
from pathlib import Path
meta = json.loads((Path(sys.argv[1]) / "smoothquant_linear_tensors_meta.json").read_text())
print(meta["activation_scale"])
PY
)"

if [[ "${RUN_BENCH}" == "true" ]]; then
  "${BUILD_DIR}/smolvla_smoothquant_linear_epilogue" "${TENSOR_DIR}" "${OUTPUT_ROOT}" "${ACTIVATION_SCALE}" "${WARMUP}" "${ITERS}" \
    2>&1 | tee "${OUTPUT_ROOT}/smoothquant_linear_epilogue.log"
fi

if [[ "${RUN_CUBLASLT}" == "true" ]]; then
  "${BUILD_DIR}/smolvla_smoothquant_linear_cublaslt" "${TENSOR_DIR}" "${OUTPUT_ROOT}" "${ACTIVATION_SCALE}" "${WARMUP}" "${ITERS}" \
    2>&1 | tee "${OUTPUT_ROOT}/smoothquant_linear_cublaslt.log"
fi

"${PYTHON}" - "${OUTPUT_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
tensor_meta = root / "tensors" / "smoothquant_linear_tensors_meta.json"
kernel_report = root / "smoothquant_linear_epilogue_report.json"
cublaslt_report = root / "smoothquant_linear_cublaslt_report.json"
summary = {}
if tensor_meta.exists():
    summary["tensor_export"] = json.loads(tensor_meta.read_text())
if kernel_report.exists():
    summary["kernel_report"] = json.loads(kernel_report.read_text())
if cublaslt_report.exists():
    summary["cublaslt_report"] = json.loads(cublaslt_report.read_text())
if "kernel_report" in summary and "cublaslt_report" in summary:
    naive_total = summary["kernel_report"]["latency_ms"].get("fused_total")
    lt_latency = summary["cublaslt_report"]["latency_ms"]
    lt_total = lt_latency.get("total_with_activation_quantize")
    lt_vec4_total = lt_latency.get("total_with_activation_quantize_vec4")
    best_lt_total = lt_vec4_total or lt_total
    if naive_total and best_lt_total:
        summary["naive_vs_cublaslt"] = {
            "naive_fused_total_ms": naive_total,
            "cublaslt_total_with_activation_quantize_ms": lt_total,
            "cublaslt_total_with_activation_quantize_vec4_ms": lt_vec4_total,
            "cublaslt_best_total_ms": best_lt_total,
            "cublaslt_speedup_vs_naive": naive_total / best_lt_total,
        }
(root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
PY
