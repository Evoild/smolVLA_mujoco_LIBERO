#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON="${PYTHON_BIN:-python}"
OUTPUT_ROOT="runs/deploy/6/F-layer0-real-attention"
POLICY_PATH="smolvla_libero"
TASK="libero_spatial"
SEED="1000"
INPUT_SOURCE="rollout"
TOKEN_LENGTH="48"
MODEL_DTYPE="bf16"
INPUT_DTYPE="bf16"
ACTIVATION_SCALES_JSON="runs/deploy/5/O-smoothquant-alpha085-full-w8a8-deploy-cudagraph/smoothquant_alpha_0.85_text_and_lm_expert_activation_scales.json"
DISABLE_FAKE_QUANT="false"
RUN_EXPORT="true"
RUN_SMOKE="true"
RUN_UNFUSED_BASELINE="true"
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
    --model-dtype) MODEL_DTYPE="$2"; shift 2 ;;
    --input-dtype) INPUT_DTYPE="$2"; shift 2 ;;
    --activation-scales-json) ACTIVATION_SCALES_JSON="$2"; shift 2 ;;
    --disable-fake-quant) DISABLE_FAKE_QUANT="$2"; shift 2 ;;
    --run-export) RUN_EXPORT="$2"; shift 2 ;;
    --run-smoke) RUN_SMOKE="$2"; shift 2 ;;
    --run-unfused-baseline) RUN_UNFUSED_BASELINE="$2"; shift 2 ;;
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
  export_args=(
    --policy-path "${POLICY_PATH}"
    --output-dir "${TENSOR_DIR}"
    --device cuda
    --seed "${SEED}"
    --task "${TASK}"
    --input-source "${INPUT_SOURCE}"
    --token-length "${TOKEN_LENGTH}"
    --model-dtype "${MODEL_DTYPE}"
    --input-dtype "${INPUT_DTYPE}"
    --activation-scales-json "${ACTIVATION_SCALES_JSON}"
    --warmup "${WARMUP}"
    --iters "${ITERS}"
  )
  if [[ "${DISABLE_FAKE_QUANT}" == "true" ]]; then
    export_args+=(--disable-fake-quant)
  fi
  "${PYTHON}" deploy/export_6f_layer0_attention_tensors.py "${export_args[@]}" \
    2>&1 | tee "${OUTPUT_ROOT}/export_layer0_tensors.log"
fi

cmake -S deploy/cuda -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD_DIR}" --target smolvla_fused_attention_v3_real -j "${JOBS:-8}"
cmake --build "${BUILD_DIR}" --target smolvla_unfused_attention_trt_real -j "${JOBS:-8}"

if [[ "${RUN_SMOKE}" == "true" ]]; then
  "${BUILD_DIR}/smolvla_fused_attention_v3_real" "${TENSOR_DIR}" "${OUTPUT_ROOT}" "${WARMUP}" "${ITERS}" \
    2>&1 | tee "${OUTPUT_ROOT}/network_api_real.log"
fi

if [[ "${RUN_UNFUSED_BASELINE}" == "true" ]]; then
  "${BUILD_DIR}/smolvla_unfused_attention_trt_real" "${TENSOR_DIR}" "${OUTPUT_ROOT}" "${WARMUP}" "${ITERS}" \
    2>&1 | tee "${OUTPUT_ROOT}/network_api_unfused_trt_real.log"
fi

"${PYTHON}" - "${OUTPUT_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
meta_path = root / "tensors" / "layer0_attention_tensors_meta.json"
report_path = root / "fused_attention_v3_real_report.json"
unfused_report_path = root / "unfused_attention_trt_real_report.json"
summary = {}
if meta_path.exists():
    summary["tensor_export"] = json.loads(meta_path.read_text())
if report_path.exists():
    summary["plugin_report"] = json.loads(report_path.read_text())
if unfused_report_path.exists():
    summary["unfused_trt_report"] = json.loads(unfused_report_path.read_text())
if "plugin_report" in summary and "unfused_trt_report" in summary:
    fused_ms = summary["plugin_report"].get("plugin_latency_ms")
    unfused_ms = summary["unfused_trt_report"].get("unfused_trt_latency_ms")
    if fused_ms and unfused_ms:
        summary["trt_unfused_vs_fused"] = {
            "unfused_trt_latency_ms": unfused_ms,
            "fused_plugin_latency_ms": fused_ms,
            "speedup_unfused_over_fused": unfused_ms / fused_ms,
            "fused_slowdown_vs_unfused": fused_ms / unfused_ms,
            "fused_latency_delta_ms": fused_ms - unfused_ms,
        }
(root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
PY
