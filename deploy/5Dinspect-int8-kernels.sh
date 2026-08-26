#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$REPO_ROOT/scripts/common.sh"

ENGINE="$REPO_ROOT/runs/deploy/5/B-cast-bf16-prefer/smolvla_action_only_vlm_mlp_full_attn_w8a8_cast_bf16_precision_prefer.plan"
OUTPUT_ROOT="$REPO_ROOT/runs/deploy/5/D-inspect-int8-kernels-prefer"
PYTHON="${PYTHON_BIN:-python3}"

usage() {
  printf '%s\n' "Usage: $0 [options]" \
    "  --engine PATH          default: $ENGINE" \
    "  --output-root DIR      default: $OUTPUT_ROOT" \
    "  PYTHON_BIN=/path/python optional Python environment"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --engine) require_value "$@"; ENGINE="$2"; shift 2 ;;
    --output-root) require_value "$@"; OUTPUT_ROOT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ -f "$ENGINE" ]] || die "$ENGINE not found"
command -v trtexec >/dev/null 2>&1 || die "trtexec not found"
mkdir -p "$OUTPUT_ROOT"

trtexec \
  --loadEngine="$ENGINE" \
  --profilingVerbosity=detailed \
  --dumpLayerInfo \
  --exportLayerInfo="$OUTPUT_ROOT/layer_info.json" \
  > "$OUTPUT_ROOT/layer_info.log" 2>&1

"$PYTHON" "$SCRIPT_DIR/inspect_trt_layer_precision.py" \
  --layer-info "$OUTPUT_ROOT/layer_info.json" \
  --output "$OUTPUT_ROOT/layer_precision_summary.json"

printf '%s\n' "TensorRT layer precision inspection outputs:" \
  "  engine: $ENGINE" \
  "  layer info: $OUTPUT_ROOT/layer_info.json" \
  "  trtexec log: $OUTPUT_ROOT/layer_info.log" \
  "  summary: $OUTPUT_ROOT/layer_precision_summary.json"
