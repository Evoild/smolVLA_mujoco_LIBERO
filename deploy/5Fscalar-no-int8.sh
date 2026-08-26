#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/evoild/miniconda3/envs/LIBERO-smolvla/bin/python}"
NATIVE_ONNX="$REPO_ROOT/runs/deploy/4/E4-fps-diagnose-mlp-full-attn-w8a8-action-only/smolvla_action_only_native_mixed.onnx"
OUT_ROOT="$REPO_ROOT/runs/deploy/5/F-qkernel-ablation"
SCALAR_ROOT="$OUT_ROOT/scalar-act"
SCALAR_NO_INT8_ROOT="$OUT_ROOT/scalar-act-no-int8"
SCALAR_QDQ_ONNX="$SCALAR_ROOT/smolvla_action_only_vlm_mlp_full_attn_w8a8_scalar_act_qdq.onnx"
SCALAR_NO_INT8_ENGINE="$SCALAR_NO_INT8_ROOT/smolvla_action_only_vlm_mlp_full_attn_w8a8_scalar_act_no_int8.plan"
TARGET_NODE_REGEX='^/(debug_core/)?(q_proj|k_proj|v_proj|o_proj|mlp/(gate_proj|up_proj|down_proj))(_[0-9]+)?/MatMul$'

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

inspect_engine() {
  local engine_path="$1"
  local output_root="$2"
  mkdir -p "$output_root"
  trtexec \
    --loadEngine="$engine_path" \
    --profilingVerbosity=detailed \
    --dumpLayerInfo \
    --exportLayerInfo="$output_root/layer_info.json" \
    > "$output_root/layer_info.log" 2>&1
  "$PYTHON_BIN" "$SCRIPT_DIR/inspect_trt_layer_precision.py" \
    --layer-info "$output_root/layer_info.json" \
    --output "$output_root/layer_precision_summary.json"
}

command -v trtexec >/dev/null 2>&1 || die "trtexec not found"
[[ -x "$PYTHON_BIN" ]] || die "PYTHON_BIN not executable: $PYTHON_BIN"

mkdir -p "$SCALAR_ROOT" "$SCALAR_NO_INT8_ROOT"

if [[ ! -f "$SCALAR_QDQ_ONNX" ]]; then
  [[ -f "$NATIVE_ONNX" ]] || die "missing native ONNX: $NATIVE_ONNX"
  "$PYTHON_BIN" "$SCRIPT_DIR/insert_linear_w8a8_qdq.py" \
    --input "$NATIVE_ONNX" \
    --output "$SCALAR_QDQ_ONNX" \
    --activation-scale-mode static \
    --activation-scale 0.01 \
    --include-node-regex "$TARGET_NODE_REGEX" \
    --cast-output-to bf16 \
    --check \
    > "$SCALAR_ROOT/insert_qdq.log" 2>&1
fi

trtexec \
  --onnx="$SCALAR_QDQ_ONNX" \
  --saveEngine="$SCALAR_NO_INT8_ENGINE" \
  --precisionConstraints=prefer \
  --profilingVerbosity=detailed \
  --timingCacheFile="$SCALAR_NO_INT8_ROOT/scalar_act_no_int8.cache" \
  > "$SCALAR_NO_INT8_ROOT/build.log" 2>&1

inspect_engine "$SCALAR_NO_INT8_ENGINE" "$SCALAR_NO_INT8_ROOT/inspect"

printf '%s\n' \
  "5F scalar no --int8 finished:" \
  "  Q/DQ ONNX: $SCALAR_QDQ_ONNX" \
  "  Engine: $SCALAR_NO_INT8_ENGINE" \
  "  Build log: $SCALAR_NO_INT8_ROOT/build.log" \
  "  Summary: $SCALAR_NO_INT8_ROOT/inspect/layer_precision_summary.json"
