#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/home/evoild/miniconda3/envs/LIBERO-smolvla/bin/python}"
POLICY_PATH="${POLICY_PATH:-${REPO_ROOT}/smolvla_libero}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/deploy/6/D-fused-rope-layout}"
BASELINE_ROOT="${BASELINE_ROOT:-runs/deploy/5/O-smoothquant-alpha085-full-w8a8-deploy-cudagraph}"
INPUT_ONNX="${INPUT_ONNX:-${BASELINE_ROOT}/smolvla_action_only_smoothquant_alpha085_full_w8a8_qdq.onnx}"
FUSED_ONNX="${FUSED_ONNX:-${OUTPUT_ROOT}/smolvla_action_only_smoothquant_alpha085_full_w8a8_fused_rope_layout.onnx}"
ENGINE_OUTPUT="${ENGINE_OUTPUT:-${OUTPUT_ROOT}/smolvla_action_only_smoothquant_alpha085_full_w8a8_fused_rope_layout_precision_prefer.plan}"
PLUGIN_SO="${PLUGIN_SO:-${OUTPUT_ROOT}/build/libsmolvla_fused_rope_layout.so}"
CACHE_FILE="${CACHE_FILE:-${OUTPUT_ROOT}/fused_rope_layout_precision_prefer.cache}"
TRT_PRECISION_CONSTRAINTS="${TRT_PRECISION_CONSTRAINTS:-prefer}"
TRT_BUILDER_OPT_LEVEL="${TRT_BUILDER_OPT_LEVEL:-}"
USE_CUDA_GRAPH="${USE_CUDA_GRAPH:-true}"
TASK_SUITE="${TASK_SUITE:-libero_spatial}"
DEVICE="${DEVICE:-cuda}"
PROFILE_WARMUP="${PROFILE_WARMUP:-5}"
PROFILE_ITERS="${PROFILE_ITERS:-30}"
TRT_ITERATIONS="${TRT_ITERATIONS:-200}"
RUN_BUILD_PLUGIN="${RUN_BUILD_PLUGIN:-true}"
RUN_REWRITE_ONNX="${RUN_REWRITE_ONNX:-true}"
RUN_BUILD_ENGINE="${RUN_BUILD_ENGINE:-true}"
RUN_TRTEXEC_PROFILE="${RUN_TRTEXEC_PROFILE:-true}"
RUN_POLICY_PROFILE="${RUN_POLICY_PROFILE:-true}"
RUN_ATTENTION_BREAKDOWN="${RUN_ATTENTION_BREAKDOWN:-true}"
RUN_MATH_CHECK="${RUN_MATH_CHECK:-true}"
FUSION_TARGET="${FUSION_TARGET:-qk}"
REWRITE_MAX_LAYER_INDEX="${REWRITE_MAX_LAYER_INDEX:-31}"
REWRITE_MIN_REWRITES="${REWRITE_MIN_REWRITES:-1}"
PLUGIN_INPUT_DTYPE="${PLUGIN_INPUT_DTYPE:-fp32}"
FUSED_ONNX_EXPLICIT=false
ENGINE_OUTPUT_EXPLICIT=false
PLUGIN_SO_EXPLICIT=false
CACHE_FILE_EXPLICIT=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --input-onnx) INPUT_ONNX="$2"; shift 2 ;;
    --fused-onnx) FUSED_ONNX="$2"; FUSED_ONNX_EXPLICIT=true; shift 2 ;;
    --engine-output) ENGINE_OUTPUT="$2"; ENGINE_OUTPUT_EXPLICIT=true; shift 2 ;;
    --plugin-so) PLUGIN_SO="$2"; PLUGIN_SO_EXPLICIT=true; shift 2 ;;
    --cache-file) CACHE_FILE="$2"; CACHE_FILE_EXPLICIT=true; shift 2 ;;
    --policy-path) POLICY_PATH="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --task) TASK_SUITE="$2"; shift 2 ;;
    --profile-warmup) PROFILE_WARMUP="$2"; shift 2 ;;
    --profile-iters) PROFILE_ITERS="$2"; shift 2 ;;
    --trt-iterations) TRT_ITERATIONS="$2"; shift 2 ;;
    --run-build-plugin) RUN_BUILD_PLUGIN="$2"; shift 2 ;;
    --run-rewrite-onnx) RUN_REWRITE_ONNX="$2"; shift 2 ;;
    --run-build-engine) RUN_BUILD_ENGINE="$2"; shift 2 ;;
    --run-trtexec-profile) RUN_TRTEXEC_PROFILE="$2"; shift 2 ;;
    --run-policy-profile) RUN_POLICY_PROFILE="$2"; shift 2 ;;
    --run-attention-breakdown) RUN_ATTENTION_BREAKDOWN="$2"; shift 2 ;;
    --run-math-check) RUN_MATH_CHECK="$2"; shift 2 ;;
    --fusion-target) FUSION_TARGET="$2"; shift 2 ;;
    --max-layer-index) REWRITE_MAX_LAYER_INDEX="$2"; shift 2 ;;
    --min-rewrites) REWRITE_MIN_REWRITES="$2"; shift 2 ;;
    --plugin-input-dtype) PLUGIN_INPUT_DTYPE="$2"; shift 2 ;;
    --trt-precision-constraints) TRT_PRECISION_CONSTRAINTS="$2"; shift 2 ;;
    --builder-optimization-level) TRT_BUILDER_OPT_LEVEL="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ "${FUSED_ONNX_EXPLICIT}" != "true" ]]; then
  FUSED_ONNX="${OUTPUT_ROOT}/smolvla_action_only_smoothquant_alpha085_full_w8a8_fused_rope_layout.onnx"
fi
if [[ "${ENGINE_OUTPUT_EXPLICIT}" != "true" ]]; then
  ENGINE_OUTPUT="${OUTPUT_ROOT}/smolvla_action_only_smoothquant_alpha085_full_w8a8_fused_rope_layout_precision_prefer.plan"
fi
if [[ "${PLUGIN_SO_EXPLICIT}" != "true" ]]; then
  PLUGIN_SO="${OUTPUT_ROOT}/build/libsmolvla_fused_rope_layout.so"
fi
if [[ "${CACHE_FILE_EXPLICIT}" != "true" ]]; then
  CACHE_FILE="${OUTPUT_ROOT}/fused_rope_layout_precision_prefer.cache"
fi

mkdir -p "${OUTPUT_ROOT}"

case "${FUSION_TARGET}" in
  layout|qk|qk-softmax|attention) ;;
  *) echo "--fusion-target must be layout, qk, qk-softmax, or attention" >&2; exit 2 ;;
esac

if [[ "${RUN_BUILD_PLUGIN}" == "true" ]]; then
  cmake -S deploy/cuda -B "${OUTPUT_ROOT}/build" -DCMAKE_BUILD_TYPE=Release
  cmake --build "${OUTPUT_ROOT}/build" -j"$(nproc)"
fi

if [[ "${RUN_MATH_CHECK}" == "true" ]]; then
  if [[ "${FUSION_TARGET}" == "qk" || "${FUSION_TARGET}" == "qk-softmax" || "${FUSION_TARGET}" == "attention" ]]; then
    "${PYTHON_BIN}" deploy/validate_fused_rope_qk_math.py \
      --output-json "${OUTPUT_ROOT}/fused_rope_qk_math_check.json"
  else
    "${PYTHON_BIN}" deploy/validate_fused_rope_layout_math.py \
      --output-json "${OUTPUT_ROOT}/fused_rope_layout_math_check.json"
  fi
fi

if [[ "${RUN_REWRITE_ONNX}" == "true" ]]; then
  if [[ "${FUSION_TARGET}" == "qk" || "${FUSION_TARGET}" == "qk-softmax" || "${FUSION_TARGET}" == "attention" ]]; then
    QK_MODE="qk"
    if [[ "${FUSION_TARGET}" == "qk-softmax" ]]; then
      QK_MODE="qk_softmax"
    elif [[ "${FUSION_TARGET}" == "attention" ]]; then
      QK_MODE="attention"
    fi
    "${PYTHON_BIN}" deploy/insert_fused_rope_qk_plugin.py \
      --input-onnx "${INPUT_ONNX}" \
      --output-onnx "${FUSED_ONNX}" \
      --report-json "${OUTPUT_ROOT}/fused_rope_qk_rewrite_report.json" \
      --max-layer-index "${REWRITE_MAX_LAYER_INDEX}" \
      --min-rewrites "${REWRITE_MIN_REWRITES}" \
      --mode "${QK_MODE}"
  else
    "${PYTHON_BIN}" deploy/insert_fused_rope_layout_plugin.py \
      --input-onnx "${INPUT_ONNX}" \
      --output-onnx "${FUSED_ONNX}" \
      --report-json "${OUTPUT_ROOT}/fused_rope_layout_rewrite_report.json" \
      --max-layer-index "${REWRITE_MAX_LAYER_INDEX}" \
      --min-rewrites "${REWRITE_MIN_REWRITES}" \
      --plugin-input-dtype "${PLUGIN_INPUT_DTYPE}"
  fi
fi

TRT_FLAGS=(--staticPlugins="${PLUGIN_SO}" --profilingVerbosity=detailed --timingCacheFile="${CACHE_FILE}")
if [[ "${TRT_PRECISION_CONSTRAINTS}" != "none" ]]; then
  TRT_FLAGS=(--precisionConstraints="${TRT_PRECISION_CONSTRAINTS}" "${TRT_FLAGS[@]}")
fi
if [[ -n "${TRT_BUILDER_OPT_LEVEL}" ]]; then
  TRT_FLAGS=(--builderOptimizationLevel="${TRT_BUILDER_OPT_LEVEL}" "${TRT_FLAGS[@]}")
fi

if [[ "${RUN_BUILD_ENGINE}" == "true" ]]; then
  command -v trtexec >/dev/null 2>&1 || { echo "trtexec not found" >&2; exit 1; }
  trtexec \
    --onnx="${FUSED_ONNX}" \
    --saveEngine="${ENGINE_OUTPUT}" \
    --skipInference \
    --exportLayerInfo="${OUTPUT_ROOT}/layer_info.json" \
    "${TRT_FLAGS[@]}" \
    > "${OUTPUT_ROOT}/build_trtexec.log" 2>&1
fi

if [[ "${RUN_TRTEXEC_PROFILE}" == "true" ]]; then
  trtexec \
    --loadEngine="${ENGINE_OUTPUT}" \
    --useCudaGraph \
    --iterations="${TRT_ITERATIONS}" \
    --exportProfile="${OUTPUT_ROOT}/trtexec_profile.json" \
    --exportLayerInfo="${OUTPUT_ROOT}/layer_info_profile.json" \
    --staticPlugins="${PLUGIN_SO}" \
    > "${OUTPUT_ROOT}/trtexec_profile.log" 2>&1

  "${PYTHON_BIN}" - "${OUTPUT_ROOT}/trtexec_profile.log" "${OUTPUT_ROOT}/trtexec_latency.json" <<'PY'
import json
import re
import sys
from pathlib import Path

log = Path(sys.argv[1]).read_text(errors="replace")
patterns = [
    r"Latency:\s+min\s*=\s*[-+0-9.eE]+\s*ms,\s*max\s*=\s*[-+0-9.eE]+\s*ms,\s*mean\s*=\s*([-+0-9.eE]+)\s*ms",
    r"GPU Compute Time:\s+min\s*=\s*[-+0-9.eE]+\s*ms,\s*max\s*=\s*[-+0-9.eE]+\s*ms,\s*mean\s*=\s*([-+0-9.eE]+)\s*ms",
]
for pattern in patterns:
    match = re.search(pattern, log)
    if match:
        value = float(match.group(1))
        Path(sys.argv[2]).write_text(json.dumps({"engine_latency_ms": value}, indent=2), encoding="utf-8")
        print(value)
        break
else:
    raise SystemExit("could not parse trtexec mean latency from log")
PY

  FUSED_ENGINE_LATENCY_MS="$("${PYTHON_BIN}" - "${OUTPUT_ROOT}/trtexec_latency.json" <<'PY'
import json
import sys
print(json.loads(open(sys.argv[1]).read())["engine_latency_ms"])
PY
)"
  "${PYTHON_BIN}" deploy/summarize_5m_trt_profile.py \
    --profile "${OUTPUT_ROOT}/trtexec_profile.json" \
    --layer-info "${OUTPUT_ROOT}/layer_info_profile.json" \
    --engine-latency-ms "${FUSED_ENGINE_LATENCY_MS}" \
    --output-json "${OUTPUT_ROOT}/per_layer_category_summary.json" \
    --output-md "${OUTPUT_ROOT}/per_layer_category_summary.md"
fi

CUDA_GRAPH_ARGS=()
if [[ "${USE_CUDA_GRAPH}" == "true" ]]; then
  CUDA_GRAPH_ARGS=(--use-cuda-graph)
fi

if [[ "${RUN_POLICY_PROFILE}" == "true" ]]; then
  "${PYTHON_BIN}" deploy/trt_sample_actions_core_deploy.py profile \
    --backend trt-int8 \
    --engine-path "${ENGINE_OUTPUT}" \
    --trt-output-name action_chunk \
    --trt-plugin-library "${PLUGIN_SO}" \
    --policy-path "${POLICY_PATH}" \
    --device "${DEVICE}" \
    --output-dir "${OUTPUT_ROOT}/deploy_eval/int8_action_only/profile" \
    --warmup "${PROFILE_WARMUP}" \
    --iters "${PROFILE_ITERS}" \
    --task "${TASK_SUITE} SmoothQuant alpha=0.85 full A8W8 + fused RoPE/Layout" \
    "${CUDA_GRAPH_ARGS[@]}"
fi

if [[ "${RUN_ATTENTION_BREAKDOWN}" == "true" ]]; then
  "${PYTHON_BIN}" deploy/profile_6c_trt_attention_breakdown.py \
    --profile "${OUTPUT_ROOT}/trtexec_profile.json" \
    --layer-info "${OUTPUT_ROOT}/layer_info_profile.json" \
    --category-summary "${OUTPUT_ROOT}/per_layer_category_summary.json" \
    --output-dir "${OUTPUT_ROOT}/attention_breakdown"
fi

printf '%s\n' \
  "Step 6D outputs:" \
  "  plugin: ${PLUGIN_SO}" \
  "  fused ONNX: ${FUSED_ONNX}" \
  "  engine: ${ENGINE_OUTPUT}" \
  "  trtexec profile: ${OUTPUT_ROOT}/trtexec_profile.json" \
  "  policy profile: ${OUTPUT_ROOT}/deploy_eval/int8_action_only/profile/profile_summary.json" \
  "  attention breakdown: ${OUTPUT_ROOT}/attention_breakdown/trt_attention_breakdown_summary.md"
