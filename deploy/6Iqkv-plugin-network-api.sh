#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON="${PYTHON_BIN:-python}"
OUTPUT_ROOT="runs/deploy/6/I-qkv-plugin-network-api"
TENSOR_ROOT="runs/deploy/6/H-qkv-smoothquant-cublaslt/tensors"
RUN_EXPORT="false"
WARMUP="20"
ITERS="200"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --tensor-root) TENSOR_ROOT="$2"; shift 2 ;;
    --run-export) RUN_EXPORT="$2"; shift 2 ;;
    --warmup) WARMUP="$2"; shift 2 ;;
    --iters) ITERS="$2"; shift 2 ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

mkdir -p "${OUTPUT_ROOT}"
BUILD_DIR="${OUTPUT_ROOT}/build"

if [[ "${RUN_EXPORT}" == "true" ]]; then
  HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" NUMBA_DISABLE_JIT="${NUMBA_DISABLE_JIT:-1}" \
  PYTHON_BIN="${PYTHON}" \
  bash deploy/6Hqkv-smoothquant-cublaslt.sh \
    --output-root "$(dirname "${TENSOR_ROOT}")" \
    --run-bench false
fi

cmake -S deploy/cuda -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD_DIR}" --target smolvla_smoothquant_qkv_plugin_v3_smoke -j "${JOBS:-8}"

read -r Q_SCALE K_SCALE V_SCALE < <("${PYTHON}" - "${TENSOR_ROOT}" <<'PY'
import json
import sys
from pathlib import Path
meta = json.loads((Path(sys.argv[1]) / "qkv_smoothquant_tensors_meta.json").read_text())
print(meta["modules"]["q"]["activation_scale"], meta["modules"]["k"]["activation_scale"], meta["modules"]["v"]["activation_scale"])
PY
)

"${BUILD_DIR}/smolvla_smoothquant_qkv_plugin_v3_smoke" "${TENSOR_ROOT}" "${OUTPUT_ROOT}" "${Q_SCALE}" "${K_SCALE}" "${V_SCALE}" "${WARMUP}" "${ITERS}" \
  2>&1 | tee "${OUTPUT_ROOT}/smoothquant_qkv_plugin_v3_smoke.log"

"${PYTHON}" - "${OUTPUT_ROOT}" "${TENSOR_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
tensor_root = Path(sys.argv[2])
summary = {}
meta_path = tensor_root / "qkv_smoothquant_tensors_meta.json"
report_path = root / "smoothquant_qkv_plugin_v3_smoke_report.json"
if meta_path.exists():
    summary["tensor_export"] = json.loads(meta_path.read_text())
if report_path.exists():
    summary["plugin_report"] = json.loads(report_path.read_text())
(root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
PY
