#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$REPO_ROOT/scripts/common.sh"

INPUT_ROOT="$REPO_ROOT/runs/deploy/5/O-smoothquant-alpha085-full-w8a8-deploy-cudagraph"
ONNX_PATH="$INPUT_ROOT/smolvla_action_only_smoothquant_alpha085_full_w8a8_qdq.onnx"
QDQ_REPORT="$INPUT_ROOT/smolvla_action_only_smoothquant_alpha085_full_w8a8_qdq.qdq_report.json"
OUTPUT_ROOT="$REPO_ROOT/runs/deploy/6/A-attention-fusion-inspect"
PYTHON="${PYTHON_BIN:-python3}"

usage() {
  printf '%s\n' "Usage: $0 [options]" \
    "  --onnx PATH              default: $ONNX_PATH" \
    "  --qdq-report PATH        default: $QDQ_REPORT" \
    "  --output-root DIR        default: $OUTPUT_ROOT" \
    "  PYTHON_BIN=/path/python  optional Python with onnx installed"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --onnx) require_value "$@"; ONNX_PATH="$2"; shift 2 ;;
    --qdq-report) require_value "$@"; QDQ_REPORT="$2"; shift 2 ;;
    --output-root) require_value "$@"; OUTPUT_ROOT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ -f "$ONNX_PATH" ]] || die "missing ONNX: $ONNX_PATH"
[[ -f "$QDQ_REPORT" ]] || die "missing Q/DQ report: $QDQ_REPORT"

mkdir -p "$OUTPUT_ROOT"
export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON" "$SCRIPT_DIR/inspect_attention_fusion_candidates.py" \
  --onnx "$ONNX_PATH" \
  --qdq-report "$QDQ_REPORT" \
  --output-dir "$OUTPUT_ROOT"

printf '%s\n' "step 6A attention fusion inspection outputs:" \
  "  summary md: $OUTPUT_ROOT/attention_fusion_inspection.md" \
  "  summary json: $OUTPUT_ROOT/attention_fusion_inspection.json"
