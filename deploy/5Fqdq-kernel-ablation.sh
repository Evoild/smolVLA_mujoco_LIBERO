#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/evoild/miniconda3/envs/LIBERO-smolvla/bin/python}"

BASE_QDQ_ONNX="$REPO_ROOT/runs/deploy/5/B-cast-bf16-prefer/smolvla_action_only_vlm_mlp_full_attn_w8a8_qdq.onnx"
BASE_ENGINE="$REPO_ROOT/runs/deploy/5/B-cast-bf16-prefer/smolvla_action_only_vlm_mlp_full_attn_w8a8_cast_bf16_precision_prefer.plan"
NATIVE_ONNX="$REPO_ROOT/runs/deploy/4/E4-fps-diagnose-mlp-full-attn-w8a8-action-only/smolvla_action_only_native_mixed.onnx"

OUT_ROOT="$REPO_ROOT/runs/deploy/5/F-qkernel-ablation"
SCALAR_ROOT="$OUT_ROOT/scalar-act"
SCALAR_NO_INT8_ROOT="$OUT_ROOT/scalar-act-no-int8"
INT8_FLAG_ROOT="$OUT_ROOT/calibrated-act-plus-int8-flag"
BASE_INSPECT_ROOT="$OUT_ROOT/baseline-prefer-no-int8"

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
[[ -f "$BASE_QDQ_ONNX" ]] || die "missing base Q/DQ ONNX: $BASE_QDQ_ONNX"
[[ -f "$BASE_ENGINE" ]] || die "missing base engine: $BASE_ENGINE"
[[ -f "$NATIVE_ONNX" ]] || die "missing native ONNX: $NATIVE_ONNX"

mkdir -p "$OUT_ROOT"

printf '\n==== 0. Inspect existing calibrated per-channel activation Q/DQ engine, no --int8 build flag ====\n'
inspect_engine "$BASE_ENGINE" "$BASE_INSPECT_ROOT"

printf '\n==== 1. Rebuild calibrated per-channel activation Q/DQ with --int8 ====\n'
mkdir -p "$INT8_FLAG_ROOT"
trtexec \
  --onnx="$BASE_QDQ_ONNX" \
  --saveEngine="$INT8_FLAG_ROOT/smolvla_action_only_vlm_mlp_full_attn_w8a8_calibrated_act_int8_flag.plan" \
  --int8 \
  --precisionConstraints=prefer \
  --profilingVerbosity=detailed \
  --timingCacheFile="$INT8_FLAG_ROOT/calibrated_act_int8_flag.cache" \
  > "$INT8_FLAG_ROOT/build.log" 2>&1
inspect_engine \
  "$INT8_FLAG_ROOT/smolvla_action_only_vlm_mlp_full_attn_w8a8_calibrated_act_int8_flag.plan" \
  "$INT8_FLAG_ROOT/inspect"

printf '\n==== 2. Generate scalar per-tensor activation Q/DQ ONNX ====\n'
mkdir -p "$SCALAR_ROOT"
"$PYTHON_BIN" "$SCRIPT_DIR/insert_linear_w8a8_qdq.py" \
  --input "$NATIVE_ONNX" \
  --output "$SCALAR_ROOT/smolvla_action_only_vlm_mlp_full_attn_w8a8_scalar_act_qdq.onnx" \
  --activation-scale-mode static \
  --activation-scale 0.01 \
  --include-node-regex "$TARGET_NODE_REGEX" \
  --cast-output-to bf16 \
  --check \
  > "$SCALAR_ROOT/insert_qdq.log" 2>&1

printf '\n==== 3. Build scalar per-tensor activation Q/DQ with --int8 ====\n'
trtexec \
  --onnx="$SCALAR_ROOT/smolvla_action_only_vlm_mlp_full_attn_w8a8_scalar_act_qdq.onnx" \
  --saveEngine="$SCALAR_ROOT/smolvla_action_only_vlm_mlp_full_attn_w8a8_scalar_act_int8_flag.plan" \
  --int8 \
  --precisionConstraints=prefer \
  --profilingVerbosity=detailed \
  --timingCacheFile="$SCALAR_ROOT/scalar_act_int8_flag.cache" \
  > "$SCALAR_ROOT/build.log" 2>&1
inspect_engine \
  "$SCALAR_ROOT/smolvla_action_only_vlm_mlp_full_attn_w8a8_scalar_act_int8_flag.plan" \
  "$SCALAR_ROOT/inspect"

printf '\n==== 4. Build scalar per-tensor activation Q/DQ without --int8 ====\n'
mkdir -p "$SCALAR_NO_INT8_ROOT"
trtexec \
  --onnx="$SCALAR_ROOT/smolvla_action_only_vlm_mlp_full_attn_w8a8_scalar_act_qdq.onnx" \
  --saveEngine="$SCALAR_NO_INT8_ROOT/smolvla_action_only_vlm_mlp_full_attn_w8a8_scalar_act_no_int8.plan" \
  --precisionConstraints=prefer \
  --profilingVerbosity=detailed \
  --timingCacheFile="$SCALAR_NO_INT8_ROOT/scalar_act_no_int8.cache" \
  > "$SCALAR_NO_INT8_ROOT/build.log" 2>&1
inspect_engine \
  "$SCALAR_NO_INT8_ROOT/smolvla_action_only_vlm_mlp_full_attn_w8a8_scalar_act_no_int8.plan" \
  "$SCALAR_NO_INT8_ROOT/inspect"

printf '\n==== summaries ====\n'
printf '%s\n' \
  "$BASE_INSPECT_ROOT/layer_precision_summary.json" \
  "$INT8_FLAG_ROOT/inspect/layer_precision_summary.json" \
  "$SCALAR_ROOT/inspect/layer_precision_summary.json" \
  "$SCALAR_NO_INT8_ROOT/inspect/layer_precision_summary.json"
