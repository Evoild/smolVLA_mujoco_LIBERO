#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT="runs/deploy/6/E-network-api-layer0-attention"
BUILD_DIR=""
RUN_SMOKE="true"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --build-dir)
      BUILD_DIR="$2"
      shift 2
      ;;
    --run-smoke)
      RUN_SMOKE="$2"
      shift 2
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "${BUILD_DIR}" ]]; then
  BUILD_DIR="${OUTPUT_ROOT}/build"
fi

mkdir -p "${OUTPUT_ROOT}" "${BUILD_DIR}"

cmake -S deploy/cuda -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD_DIR}" --target smolvla_fused_attention_v3_smoke -j "${JOBS:-8}"

PLAN_PATH="${OUTPUT_ROOT}/fused_attention_v3_smoke.plan"
LOG_PATH="${OUTPUT_ROOT}/network_api_smoke.log"

if [[ "${RUN_SMOKE}" == "true" ]]; then
  "${BUILD_DIR}/smolvla_fused_attention_v3_smoke" "${PLAN_PATH}" 2>&1 | tee "${LOG_PATH}"
else
  echo "build-only finished; smoke run skipped"
  echo "executable: ${BUILD_DIR}/smolvla_fused_attention_v3_smoke"
  echo "plan path for smoke: ${PLAN_PATH}"
fi
