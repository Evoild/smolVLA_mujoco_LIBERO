#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$REPO_ROOT/scripts/common.sh"

POLICY_PATH="$REPO_ROOT/smolvla_libero"
TASK_SUITE=libero_spatial
DEVICE=cuda
SEED=1000
PROFILE_WARMUP=10
PROFILE_ITERS=100
TRT_ITERATIONS=200
RUN_POLICY_PROFILE=true

BASELINE_ROOT="$REPO_ROOT/runs/deploy/4/E4-fps-diagnose-mlp-full-attn-w8a8-action-only"
REUSE_BF16_ONNX="$BASELINE_ROOT/smolvla_action_only_native_mixed.onnx"
REUSE_BF16_ENGINE="$BASELINE_ROOT/smolvla_action_only_native_mixed_precision_obey.plan"

FULL_ATTN_SCALES="$REPO_ROOT/runs/deploy/4/E3-C-mlp-full-attn-w8a8/activation_scales_512_p99.995.json"
MLP_SCALES="$REPO_ROOT/runs/deploy/4/D-shared-mlp-calibration-heldout/activation_scales_best.json"

usage() {
  printf '%s\n' "Usage: $0 [options]" \
    "  --policy-path PATH             default: $POLICY_PATH" \
    "  --tasks NAME                  default: $TASK_SUITE" \
    "  --device DEVICE               default: $DEVICE" \
    "  --seed N                      default: $SEED" \
    "  --profile-warmup N            default: $PROFILE_WARMUP" \
    "  --profile-iters N             default: $PROFILE_ITERS" \
    "  --trtexec-iterations N        default: $TRT_ITERATIONS" \
    "  --run-policy-profile true|false default: $RUN_POLICY_PROFILE" \
    "  --reuse-bf16-onnx PATH        default: $REUSE_BF16_ONNX" \
    "  --reuse-bf16-engine PATH      default: $REUSE_BF16_ENGINE" \
    "  PYTHON_BIN=/path/python       optional Python with torch/onnxruntime/tensorrt dependencies"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --policy-path) require_value "$@"; POLICY_PATH="$2"; shift 2 ;;
    --tasks) require_value "$@"; TASK_SUITE="$2"; shift 2 ;;
    --device) require_value "$@"; DEVICE="$2"; shift 2 ;;
    --seed) require_value "$@"; SEED="$2"; shift 2 ;;
    --profile-warmup) require_value "$@"; PROFILE_WARMUP="$2"; shift 2 ;;
    --profile-iters) require_value "$@"; PROFILE_ITERS="$2"; shift 2 ;;
    --trtexec-iterations) require_value "$@"; TRT_ITERATIONS="$2"; shift 2 ;;
    --run-policy-profile) require_value "$@"; RUN_POLICY_PROFILE="$2"; shift 2 ;;
    --reuse-bf16-onnx) require_value "$@"; REUSE_BF16_ONNX="$2"; shift 2 ;;
    --reuse-bf16-engine) require_value "$@"; REUSE_BF16_ENGINE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

case "$RUN_POLICY_PROFILE" in true|false) ;; *) die "--run-policy-profile must be true or false" ;; esac
[[ -f "$REUSE_BF16_ONNX" ]] || die "$REUSE_BF16_ONNX not found; run E4 action-only baseline first"
[[ -f "$REUSE_BF16_ENGINE" ]] || die "$REUSE_BF16_ENGINE not found; run E4 action-only baseline first"
[[ -f "$FULL_ATTN_SCALES" ]] || die "$FULL_ATTN_SCALES not found"
[[ -f "$MLP_SCALES" ]] || die "$MLP_SCALES not found"

run_case() {
  local output_root="$1"
  shift
  printf '\n==== Running %s ====\n' "$output_root"
  "$SCRIPT_DIR/4Efps-diagnose.sh" \
    --policy-path "$POLICY_PATH" \
    --tasks "$TASK_SUITE" \
    --device "$DEVICE" \
    --seed "$SEED" \
    --profile-warmup "$PROFILE_WARMUP" \
    --profile-iters "$PROFILE_ITERS" \
    --trtexec-iterations "$TRT_ITERATIONS" \
    --reuse-bf16-onnx "$REUSE_BF16_ONNX" \
    --reuse-bf16-engine "$REUSE_BF16_ENGINE" \
    --run-bf16-profile false \
    --run-policy-profile "$RUN_POLICY_PROFILE" \
    --output-root "$output_root" \
    "$@"
}

run_case "$REPO_ROOT/runs/deploy/5/B-cast-bf16-prefer" \
  --scales-json "$FULL_ATTN_SCALES" \
  --quant-label vlm_mlp_full_attn_w8a8 \
  --cast-output-to bf16 \
  --trt-precision-constraints prefer

run_case "$REPO_ROOT/runs/deploy/5/B-cast-bf16-no-constraints" \
  --scales-json "$FULL_ATTN_SCALES" \
  --quant-label vlm_mlp_full_attn_w8a8 \
  --cast-output-to bf16 \
  --trt-precision-constraints none

run_case "$REPO_ROOT/runs/deploy/5/C-mlp-only-cast-bf16-obey" \
  --scales-json "$MLP_SCALES" \
  --quant-label vlm_mlp_w8a8 \
  --quant-module-regex '^model\.vlm_with_expert\.vlm\.model\.text_model\.layers\.[0-9]+\.mlp\.(gate_proj|up_proj|down_proj)$' \
  --quant-node-regex '^/(debug_core/)?mlp/(gate_proj|up_proj|down_proj)(_[0-9]+)?/MatMul$' \
  --expected-linear-nodes 93 \
  --cast-output-to bf16 \
  --trt-precision-constraints obey

run_case "$REPO_ROOT/runs/deploy/5/C-attn-only-cast-bf16-obey" \
  --scales-json "$FULL_ATTN_SCALES" \
  --quant-label vlm_attn_w8a8 \
  --quant-module-regex '^model\.vlm_with_expert\.vlm\.model\.text_model\.layers\.[0-9]+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)$' \
  --quant-node-regex '^/(debug_core/)?(q_proj|k_proj|v_proj|o_proj)(_[0-9]+)?/MatMul$' \
  --expected-linear-nodes 126 \
  --cast-output-to bf16 \
  --trt-precision-constraints obey

printf '\n%s\n' "Step 5B/5C FPS sweep finished. Output roots:" \
  "  $REPO_ROOT/runs/deploy/5/B-cast-bf16-prefer" \
  "  $REPO_ROOT/runs/deploy/5/B-cast-bf16-no-constraints" \
  "  $REPO_ROOT/runs/deploy/5/C-mlp-only-cast-bf16-obey" \
  "  $REPO_ROOT/runs/deploy/5/C-attn-only-cast-bf16-obey"
