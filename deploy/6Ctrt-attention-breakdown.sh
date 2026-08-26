#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
PROFILE="${PROFILE:-runs/deploy/5/N-cuda-graph-on-off-5m/cuda_graph_on/profile.json}"
LAYER_INFO="${LAYER_INFO:-runs/deploy/6/A-trt-kernel-tactic-inspect-5m/layer_info.json}"
CATEGORY_SUMMARY="${CATEGORY_SUMMARY:-runs/deploy/6/B-5m-final-profile/per_layer_category_summary.json}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/deploy/6/C-trt-attention-breakdown}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile) PROFILE="$2"; shift 2 ;;
    --layer-info) LAYER_INFO="$2"; shift 2 ;;
    --category-summary) CATEGORY_SUMMARY="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

"${PYTHON_BIN}" deploy/profile_6c_trt_attention_breakdown.py \
  --profile "${PROFILE}" \
  --layer-info "${LAYER_INFO}" \
  --category-summary "${CATEGORY_SUMMARY}" \
  --output-dir "${OUTPUT_DIR}"
