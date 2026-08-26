#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON="${PYTHON_BIN:-python}"
OUTPUT_ROOT="runs/deploy/6/K-qkv-plugin-deployment-compare"
TENSOR_ROOT="runs/deploy/6/H-qkv-smoothquant-cublaslt/tensors"
RUN_6H="false"
RUN_6I="false"
RUN_6J="false"
WARMUP="20"
ITERS="200"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --tensor-root) TENSOR_ROOT="$2"; shift 2 ;;
    --run-6h) RUN_6H="$2"; shift 2 ;;
    --run-6i) RUN_6I="$2"; shift 2 ;;
    --run-6j) RUN_6J="$2"; shift 2 ;;
    --warmup) WARMUP="$2"; shift 2 ;;
    --iters) ITERS="$2"; shift 2 ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

mkdir -p "${OUTPUT_ROOT}"

if [[ "${RUN_6H}" == "true" ]]; then
  HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" NUMBA_DISABLE_JIT="${NUMBA_DISABLE_JIT:-1}" \
  PYTHON_BIN="${PYTHON}" \
  bash deploy/6Hqkv-smoothquant-cublaslt.sh \
    --output-root "$(dirname "${TENSOR_ROOT}")" \
    --run-export false \
    --warmup "${WARMUP}" \
    --iters "${ITERS}"
fi

if [[ "${RUN_6I}" == "true" ]]; then
  HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" NUMBA_DISABLE_JIT="${NUMBA_DISABLE_JIT:-1}" \
  PYTHON_BIN="${PYTHON}" \
  bash deploy/6Iqkv-plugin-network-api.sh \
    --output-root runs/deploy/6/I-qkv-plugin-network-api \
    --tensor-root "${TENSOR_ROOT}" \
    --run-export false \
    --warmup "${WARMUP}" \
    --iters "${ITERS}"
fi

if [[ "${RUN_6J}" == "true" ]]; then
  HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" NUMBA_DISABLE_JIT="${NUMBA_DISABLE_JIT:-1}" \
  PYTHON_BIN="${PYTHON}" \
  bash deploy/6Jqkv-plugin-serialized-network-api.sh \
    --output-root runs/deploy/6/J-qkv-plugin-serialized-network-api \
    --tensor-root "${TENSOR_ROOT}" \
    --run-export false \
    --warmup "${WARMUP}" \
    --iters "${ITERS}"
fi

"${PYTHON}" - "${OUTPUT_ROOT}" "$(dirname "${TENSOR_ROOT}")" \
  "runs/deploy/6/I-qkv-plugin-network-api" \
  "runs/deploy/6/J-qkv-plugin-serialized-network-api" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
root_6h = Path(sys.argv[2])
root_6i = Path(sys.argv[3])
root_6j = Path(sys.argv[4])

def load(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text())

summary_6h = load(root_6h / "summary.json")
summary_6i = load(root_6i / "summary.json")
summary_6j = load(root_6j / "summary.json")

def get_6h_latency(summary):
    if not summary:
        return None
    report = summary.get("qkv_cublaslt_report") or summary.get("cublaslt_report") or summary.get("qkv_report")
    if not report:
        report = summary.get("benchmark_report") or summary
    lat = report.get("latency_ms", {}) if isinstance(report, dict) else {}
    return {
        "baseline_three_quant_three_gemm_three_dequant_ms": lat.get("baseline_three_quant_three_gemm_three_dequant"),
        "fused_dequant_three_quant_three_gemm_one_dequant_ms": lat.get("fused_dequant_three_quant_three_gemm_one_dequant"),
    }

def get_plugin(summary):
    if not summary:
        return None
    report = summary.get("plugin_report")
    if not report:
        return None
    return {
        "plan_size_bytes": report.get("plan_size_bytes"),
        "plugin_latency_ms": report.get("plugin_latency_ms"),
        "metrics": report.get("metrics"),
    }

lat_6h = get_6h_latency(summary_6h)
plugin_6i = get_plugin(summary_6i)
plugin_6j = get_plugin(summary_6j)

comparison = {}
raw_fused = lat_6h and lat_6h.get("fused_dequant_three_quant_three_gemm_one_dequant_ms")
raw_unfused = lat_6h and lat_6h.get("baseline_three_quant_three_gemm_three_dequant_ms")
lat_6i = plugin_6i and plugin_6i.get("plugin_latency_ms")
lat_6j = plugin_6j and plugin_6j.get("plugin_latency_ms")

if raw_fused and lat_6j:
    comparison["serialized_plugin_vs_raw_fused"] = {
        "raw_fused_ms": raw_fused,
        "serialized_plugin_ms": lat_6j,
        "slowdown": lat_6j / raw_fused,
        "delta_ms": lat_6j - raw_fused,
    }
if raw_unfused and lat_6j:
    comparison["serialized_plugin_vs_raw_unfused"] = {
        "raw_unfused_ms": raw_unfused,
        "serialized_plugin_ms": lat_6j,
        "speedup": raw_unfused / lat_6j,
        "delta_ms": raw_unfused - lat_6j,
    }
if lat_6i and lat_6j:
    comparison["serialized_plugin_vs_runtime_weight_plugin"] = {
        "runtime_weight_plugin_ms": lat_6i,
        "serialized_weight_plugin_ms": lat_6j,
        "speedup": lat_6i / lat_6j,
        "latency_reduction_pct": (lat_6i - lat_6j) / lat_6i * 100.0,
    }

result = {
    "step": "6K qkv plugin deployment compare",
    "inputs": {
        "tensor_root": str(root_6h / "tensors"),
        "summary_6h": str(root_6h / "summary.json"),
        "summary_6i": str(root_6i / "summary.json"),
        "summary_6j": str(root_6j / "summary.json"),
    },
    "raw_6h": lat_6h,
    "plugin_6i_runtime_weights": plugin_6i,
    "plugin_6j_serialized_weights": plugin_6j,
    "comparison": comparison,
    "conclusion": (
        "6J serialized-weight plugin is the deployable qkv path if q/k/v relative_l2 stays below 1e-6 "
        "and latency remains close to the 6H raw fused-dequant benchmark. "
        "The remaining gap is TensorRT/plugin enqueue overhead plus plugin boundary cost."
    ),
}

out.mkdir(parents=True, exist_ok=True)
(out / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
PY
