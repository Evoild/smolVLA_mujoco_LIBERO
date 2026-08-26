#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$REPO_ROOT/scripts/common.sh"

POLICY_PATH="$REPO_ROOT/smolvla_libero"
OUTPUT_ROOT="$REPO_ROOT/runs/deploy/5/M-kvqo-gate-up-w8a8-down-w8a16-deploy"
BF16_ONNX="$REPO_ROOT/runs/deploy/4/E4-fps-diagnose-mlp-full-attn-w8a8-action-only/smolvla_action_only_native_mixed.onnx"
BF16_ENGINE="$REPO_ROOT/runs/deploy/4/E4-fps-diagnose-mlp-full-attn-w8a8-action-only/smolvla_action_only_native_mixed_precision_obey.plan"
BF16_PROFILE="$REPO_ROOT/runs/deploy/4/E4-fps-diagnose-mlp-full-attn-w8a8-action-only/deploy_eval/bf16/profile/profile_summary.json"
BF16_EVAL="$REPO_ROOT/runs/deploy/4/E3-C-mlp-full-attn-w8a8/deploy_eval/bf16/eval/eval_info.json"
TASK_SUITE=libero_spatial
DEVICE=cuda
SEED=1000
EPISODES=10
BATCH_SIZE=1
MAX_PARALLEL_TASKS=1
PROFILE_WARMUP=5
PROFILE_ITERS=30
TRT_PRECISION_CONSTRAINTS=prefer
BUILD_ENGINE=true
RUN_PROFILE=true
RUN_EVAL=true
USE_CUDA_GRAPH=false
PYTHON="${PYTHON_BIN:-python3}"

W8A8_MODULE_REGEX="^model\\.vlm_with_expert\\.vlm\\.model\\.text_model\\.layers\\.[0-9]+\\.(self_attn\\.(q_proj|k_proj|v_proj|o_proj)|mlp\\.(gate_proj|up_proj))$"
W8A8_NODE_REGEX="^/(debug_core/)?(q_proj|k_proj|v_proj|o_proj|mlp/(gate_proj|up_proj))(_([1-9]|[12][0-9]|3[01]))?/MatMul$"
DOWN_W8A16_NODE_REGEX="^/(debug_core/)?mlp/down_proj(_([1-9]|[12][0-9]|3[01]))?/MatMul$"
EXPECTED_W8A8_NODES=192
EXPECTED_W8A16_NODES=32

usage() {
  printf '%s\n' "Usage: $0 [options]" \
    "  --policy-path PATH             default: $POLICY_PATH" \
    "  --output-root DIR             default: $OUTPUT_ROOT" \
    "  --bf16-onnx PATH              default: $BF16_ONNX" \
    "  --bf16-engine PATH            default: $BF16_ENGINE" \
    "  --bf16-profile PATH           default: $BF16_PROFILE" \
    "  --bf16-eval PATH              default: $BF16_EVAL" \
    "  --tasks NAME                  default: $TASK_SUITE" \
    "  --device DEVICE               default: $DEVICE" \
    "  --seed N                      default: $SEED" \
    "  --episodes N                  default: $EPISODES" \
    "  --batch-size N                default: $BATCH_SIZE" \
    "  --max-parallel-tasks N        default: $MAX_PARALLEL_TASKS" \
    "  --profile-warmup N            default: $PROFILE_WARMUP" \
    "  --profile-iters N             default: $PROFILE_ITERS" \
    "  --build-engine true|false     default: $BUILD_ENGINE" \
    "  --run-profile true|false      default: $RUN_PROFILE" \
    "  --run-eval true|false         default: $RUN_EVAL" \
    "  --use-cuda-graph true|false   default: $USE_CUDA_GRAPH" \
    "  --trt-precision-constraints obey|prefer|none default: $TRT_PRECISION_CONSTRAINTS" \
    "  PYTHON_BIN=/path/python       optional Python with torch/lerobot/tensorrt dependencies"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --policy-path) require_value "$@"; POLICY_PATH="$2"; shift 2 ;;
    --output-root) require_value "$@"; OUTPUT_ROOT="$2"; shift 2 ;;
    --bf16-onnx) require_value "$@"; BF16_ONNX="$2"; shift 2 ;;
    --bf16-engine) require_value "$@"; BF16_ENGINE="$2"; shift 2 ;;
    --bf16-profile) require_value "$@"; BF16_PROFILE="$2"; shift 2 ;;
    --bf16-eval) require_value "$@"; BF16_EVAL="$2"; shift 2 ;;
    --tasks) require_value "$@"; TASK_SUITE="$2"; shift 2 ;;
    --device) require_value "$@"; DEVICE="$2"; shift 2 ;;
    --seed) require_value "$@"; SEED="$2"; shift 2 ;;
    --episodes) require_value "$@"; EPISODES="$2"; shift 2 ;;
    --batch-size) require_value "$@"; BATCH_SIZE="$2"; shift 2 ;;
    --max-parallel-tasks) require_value "$@"; MAX_PARALLEL_TASKS="$2"; shift 2 ;;
    --profile-warmup) require_value "$@"; PROFILE_WARMUP="$2"; shift 2 ;;
    --profile-iters) require_value "$@"; PROFILE_ITERS="$2"; shift 2 ;;
    --build-engine) require_value "$@"; BUILD_ENGINE="$2"; shift 2 ;;
    --run-profile) require_value "$@"; RUN_PROFILE="$2"; shift 2 ;;
    --run-eval) require_value "$@"; RUN_EVAL="$2"; shift 2 ;;
    --use-cuda-graph) require_value "$@"; USE_CUDA_GRAPH="$2"; shift 2 ;;
    --trt-precision-constraints) require_value "$@"; TRT_PRECISION_CONSTRAINTS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

for value in "$BUILD_ENGINE" "$RUN_PROFILE" "$RUN_EVAL" "$USE_CUDA_GRAPH"; do
  case "$value" in true|false) ;; *) die "boolean arguments must be true or false" ;; esac
done
case "$TRT_PRECISION_CONSTRAINTS" in obey|prefer|none) ;; *) die "--trt-precision-constraints must be obey, prefer, or none" ;; esac

mkdir -p "$OUTPUT_ROOT/deploy_eval/int8_action_only"
export NUMBA_DISABLE_JIT="${NUMBA_DISABLE_JIT:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-smolvla}"
export LEROBOT_SRC="${LEROBOT_SRC:-$REPO_ROOT/lerobot/src}"
export PYTHONPATH="$SCRIPT_DIR:$LEROBOT_SRC${PYTHONPATH:+:$PYTHONPATH}"

SCALES_JSON="$OUTPUT_ROOT/activation_scales_512_p99.999_kvqo_gate_up.json"
QDQ_ONNX="$OUTPUT_ROOT/smolvla_action_only_kvqo_gate_up_w8a8_down_w8a16_qdq.onnx"
ENGINE_OUTPUT="$OUTPUT_ROOT/smolvla_action_only_kvqo_gate_up_w8a8_down_w8a16_precision_${TRT_PRECISION_CONSTRAINTS}.plan"
CACHE_FILE="$OUTPUT_ROOT/smolvla_action_only_kvqo_gate_up_w8a8_down_w8a16_precision_${TRT_PRECISION_CONSTRAINTS}.cache"
SUMMARY_JSON="$OUTPUT_ROOT/deploy_eval/deployment_summary.json"
SUMMARY_MD="$OUTPUT_ROOT/deploy_eval/deployment_summary.md"

[[ -f "$BF16_ONNX" ]] || die "missing BF16 action-only ONNX: $BF16_ONNX"
[[ -f "$SCALES_JSON" ]] || die "missing 5M activation scales: $SCALES_JSON; run deploy/5Mkvqo-gate-up-w8a8-down-w8a16-export-diagnose.sh first"

"$PYTHON" "$SCRIPT_DIR/insert_linear_w8a8_qdq.py" \
  --input "$BF16_ONNX" \
  --output "$QDQ_ONNX" \
  --activation-scale-mode calibrated \
  --activation-scales-json "$SCALES_JSON" \
  --include-module-regex "$W8A8_MODULE_REGEX" \
  --include-node-regex "$W8A8_NODE_REGEX" \
  --weight-only-include-node-regex "$DOWN_W8A16_NODE_REGEX" \
  --cast-output-to bf16 \
  --check

"$PYTHON" - "$QDQ_ONNX" "$EXPECTED_W8A8_NODES" "$EXPECTED_W8A16_NODES" <<'PY'
import json
import sys
from pathlib import Path

report_path = Path(sys.argv[1]).with_suffix(".qdq_report.json")
expected_w8a8 = int(sys.argv[2])
expected_w8a16 = int(sys.argv[3])
report = json.loads(report_path.read_text())
actual_w8a8 = int(report.get("rewritten_linear_nodes", -1))
actual_w8a16 = int(report.get("rewritten_weight_only_linear_nodes", -1))
casts = int(report.get("output_cast_nodes", -1))
expected_casts = expected_w8a8 + expected_w8a16
if actual_w8a8 != expected_w8a8 or actual_w8a16 != expected_w8a16 or casts != expected_casts:
    raise SystemExit(
        f"unexpected 5M action-only Q/DQ coverage in {report_path}: "
        f"rewritten_linear_nodes={actual_w8a8}, expected={expected_w8a8}; "
        f"rewritten_weight_only_linear_nodes={actual_w8a16}, expected={expected_w8a16}; "
        f"output_cast_nodes={casts}, expected={expected_casts}"
    )
PY

TRT_FLAGS=(--profilingVerbosity=detailed --timingCacheFile="$CACHE_FILE")
if [[ "$TRT_PRECISION_CONSTRAINTS" != "none" ]]; then
  TRT_FLAGS=(--precisionConstraints="$TRT_PRECISION_CONSTRAINTS" "${TRT_FLAGS[@]}")
fi

if [[ "$BUILD_ENGINE" == "true" ]]; then
  command -v trtexec >/dev/null 2>&1 || die "trtexec not found; rerun with --build-engine false after building manually"
  trtexec \
    --onnx="$QDQ_ONNX" \
    --saveEngine="$ENGINE_OUTPUT" \
    "${TRT_FLAGS[@]}"
  [[ -s "$ENGINE_OUTPUT" ]] || die "TensorRT produced empty engine: $ENGINE_OUTPUT"
elif [[ "$RUN_PROFILE" == "true" || "$RUN_EVAL" == "true" ]]; then
  [[ -f "$ENGINE_OUTPUT" ]] || die "$ENGINE_OUTPUT not found; enable --build-engine true"
fi

if [[ "$RUN_PROFILE" == "true" ]]; then
  CUDA_GRAPH_ARGS=()
  if [[ "$USE_CUDA_GRAPH" == "true" ]]; then
    CUDA_GRAPH_ARGS=(--use-cuda-graph)
  fi
  "$PYTHON" -c "import tensorrt" >/dev/null 2>&1 || die "TensorRT Python binding is missing in $PYTHON"
  "$PYTHON" "$SCRIPT_DIR/trt_sample_actions_core_deploy.py" profile \
    --backend trt-int8 \
    --engine-path "$ENGINE_OUTPUT" \
    --trt-output-name action_chunk \
    --policy-path "$POLICY_PATH" \
    --device "$DEVICE" \
    --output-dir "$OUTPUT_ROOT/deploy_eval/int8_action_only/profile" \
    --warmup "$PROFILE_WARMUP" \
    --iters "$PROFILE_ITERS" \
    --task "$TASK_SUITE 5M action-only profile" \
    "${CUDA_GRAPH_ARGS[@]}"
fi

if [[ "$RUN_EVAL" == "true" ]]; then
  CUDA_GRAPH_ARGS=()
  if [[ "$USE_CUDA_GRAPH" == "true" ]]; then
    CUDA_GRAPH_ARGS=(--use-cuda-graph)
  fi
  "$PYTHON" -c "import tensorrt" >/dev/null 2>&1 || die "TensorRT Python binding is missing in $PYTHON"
  "$PYTHON" "$SCRIPT_DIR/trt_sample_actions_core_deploy.py" eval \
    --backend trt-int8 \
    --engine-path "$ENGINE_OUTPUT" \
    --trt-output-name action_chunk \
    --policy-path "$POLICY_PATH" \
    --device "$DEVICE" \
    --output-dir "$OUTPUT_ROOT/deploy_eval/int8_action_only/eval" \
    --tasks "$TASK_SUITE" \
    --seed "$SEED" \
    --episodes "$EPISODES" \
    --batch-size "$BATCH_SIZE" \
    --max-parallel-tasks "$MAX_PARALLEL_TASKS" \
    "${CUDA_GRAPH_ARGS[@]}"

  "$PYTHON" "$REPO_ROOT/scripts/analyze_eval.py" \
    "$OUTPUT_ROOT/deploy_eval/int8_action_only/eval/eval_info.json" \
    --plot-suite "$TASK_SUITE" \
    --output-dir "$OUTPUT_ROOT/deploy_eval/int8_action_only/report"
fi

"$PYTHON" - "$BF16_ENGINE" "$BF16_PROFILE" "$BF16_EVAL" "$ENGINE_OUTPUT" "$QDQ_ONNX" "$OUTPUT_ROOT/deploy_eval/int8_action_only/profile/profile_summary.json" "$OUTPUT_ROOT/deploy_eval/int8_action_only/eval/eval_info.json" "$SUMMARY_JSON" "$SUMMARY_MD" <<'PY'
import json
import os
import sys
from pathlib import Path

(
    bf16_engine,
    bf16_profile,
    bf16_eval,
    int8_engine,
    int8_onnx,
    int8_profile,
    int8_eval,
    summary_json,
    summary_md,
) = map(Path, sys.argv[1:])

def load_json(path: Path):
    if not path.is_file():
        return None
    return json.loads(path.read_text())

def file_mb(path: Path):
    return path.stat().st_size / (1024**2) if path.is_file() else None

def success_pct(info):
    if not info:
        return None
    overall = info.get("overall")
    if isinstance(overall, dict):
        for key in ("success_rate_pct", "pc_success", "success_rate"):
            if key in overall:
                value = float(overall[key])
                return value * 100.0 if key == "success_rate" and value <= 1.0 else value
    successes = []
    for task in info.get("per_task", []):
        successes.extend(bool(x) for x in task.get("metrics", {}).get("successes", []))
    if not successes:
        return None
    return 100.0 * sum(successes) / len(successes)

def episodes(info):
    if not info:
        return None
    total = 0
    for task in info.get("per_task", []):
        total += len(task.get("metrics", {}).get("successes", []))
    return total or None

bf16_profile_data = load_json(bf16_profile)
bf16_eval_data = load_json(bf16_eval)
int8_profile_data = load_json(int8_profile)
int8_eval_data = load_json(int8_eval)

rows = [
    {
        "config": "BF16 action-only TensorRT baseline",
        "engine_size_mb": file_mb(bf16_engine),
        "latency_ms": None if not bf16_profile_data else bf16_profile_data.get("policy_inference_e2e_mean_ms"),
        "fps": None if not bf16_profile_data else bf16_profile_data.get("policy_inference_fps"),
        "peak_gpu_memory_mb": None if not bf16_profile_data else bf16_profile_data.get("peak_gpu_memory_mb"),
        "success_rate_pct": success_pct(bf16_eval_data),
        "episodes": episodes(bf16_eval_data),
        "profile_source": str(bf16_profile) if bf16_profile.is_file() else None,
        "eval_source": str(bf16_eval) if bf16_eval.is_file() else None,
    },
    {
        "config": "5M q/k/v/o + gate/up W8A8, down W8A16",
        "engine_size_mb": file_mb(int8_engine),
        "onnx_size_mb": file_mb(int8_onnx),
        "latency_ms": None if not int8_profile_data else int8_profile_data.get("policy_inference_e2e_mean_ms"),
        "fps": None if not int8_profile_data else int8_profile_data.get("policy_inference_fps"),
        "peak_gpu_memory_mb": None if not int8_profile_data else int8_profile_data.get("peak_gpu_memory_mb"),
        "success_rate_pct": success_pct(int8_eval_data),
        "episodes": episodes(int8_eval_data),
        "profile_source": str(int8_profile) if int8_profile.is_file() else None,
        "eval_source": str(int8_eval) if int8_eval.is_file() else None,
    },
]

summary = {"rows": rows}
summary_json.parent.mkdir(parents=True, exist_ok=True)
summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

def fmt(value, suffix=""):
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.3f}{suffix}"
    return f"{value}{suffix}"

lines = [
    "| 配置 | Engine Size | Peak GPU Memory | Latency | FPS | Success Rate | Episodes |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
]
for row in rows:
    lines.append(
        "| {config} | {engine} | {mem} | {lat} | {fps} | {succ} | {eps} |".format(
            config=row["config"],
            engine=fmt(row.get("engine_size_mb"), " MB"),
            mem=fmt(row.get("peak_gpu_memory_mb"), " MB"),
            lat=fmt(row.get("latency_ms"), " ms"),
            fps=fmt(row.get("fps")),
            succ=fmt(row.get("success_rate_pct"), "%"),
            eps=fmt(row.get("episodes")),
        )
    )
summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(summary_md.read_text(), end="")
PY

printf '%s\n' "step 5M stage 2 deployment outputs:" \
  "  action-only Q/DQ ONNX: $QDQ_ONNX" \
  "  action-only TensorRT engine: $ENGINE_OUTPUT" \
  "  Q/DQ report: ${QDQ_ONNX%.onnx}.qdq_report.json" \
  "  profile summary: $OUTPUT_ROOT/deploy_eval/int8_action_only/profile/profile_summary.json" \
  "  eval info: $OUTPUT_ROOT/deploy_eval/int8_action_only/eval/eval_info.json" \
  "  eval report: $OUTPUT_ROOT/deploy_eval/int8_action_only/report" \
  "  deployment summary: $SUMMARY_MD"
