#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$REPO_ROOT/scripts/common.sh"

ENGINE="$REPO_ROOT/runs/deploy/5/M-kvqo-gate-up-w8a8-down-w8a16-deploy/smolvla_action_only_kvqo_gate_up_w8a8_down_w8a16_precision_prefer.plan"
OUTPUT_ROOT="$REPO_ROOT/runs/deploy/5/N-cuda-graph-on-off-5m"
WARMUP_MS=1000
ITERATIONS=200
AVG_RUNS=10
PYTHON="${PYTHON_BIN:-python3}"

usage() {
  printf '%s\n' "Usage: $0 [options]" \
    "  --engine PATH          default: $ENGINE" \
    "  --output-root DIR      default: $OUTPUT_ROOT" \
    "  --warmup-ms N          default: $WARMUP_MS" \
    "  --iterations N         default: $ITERATIONS" \
    "  --avg-runs N           default: $AVG_RUNS" \
    "  PYTHON_BIN=/path/python optional Python environment"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --engine) require_value "$@"; ENGINE="$2"; shift 2 ;;
    --output-root) require_value "$@"; OUTPUT_ROOT="$2"; shift 2 ;;
    --warmup-ms) require_value "$@"; WARMUP_MS="$2"; shift 2 ;;
    --iterations) require_value "$@"; ITERATIONS="$2"; shift 2 ;;
    --avg-runs) require_value "$@"; AVG_RUNS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ -f "$ENGINE" ]] || die "engine not found: $ENGINE"
command -v trtexec >/dev/null 2>&1 || die "trtexec not found"
mkdir -p "$OUTPUT_ROOT/cuda_graph_off" "$OUTPUT_ROOT/cuda_graph_on"

run_trtexec() {
  local label="$1"
  local extra_flag="${2:-}"
  local out_dir="$OUTPUT_ROOT/$label"
  local cmd=(
    trtexec
    --loadEngine="$ENGINE"
    --warmUp="$WARMUP_MS"
    --iterations="$ITERATIONS"
    --avgRuns="$AVG_RUNS"
    --dumpProfile
    --separateProfileRun
    --profilingVerbosity=detailed
    --exportTimes="$out_dir/times.json"
    --exportProfile="$out_dir/profile.json"
  )
  if [[ -n "$extra_flag" ]]; then
    cmd+=("$extra_flag")
  fi
  "${cmd[@]}" > "$out_dir/trtexec.log" 2>&1
}

run_trtexec cuda_graph_off ""
run_trtexec cuda_graph_on "--useCudaGraph"

"$PYTHON" - "$OUTPUT_ROOT" "$ENGINE" <<'PY'
import json
import re
import sys
from pathlib import Path

output_root = Path(sys.argv[1])
engine = Path(sys.argv[2])

def read_metric(log: str, label: str) -> float | None:
    pattern = rf"{re.escape(label)}:.*?mean = ([0-9.]+) ms"
    match = re.search(pattern, log)
    return float(match.group(1)) if match else None

def read_throughput(log: str) -> float | None:
    match = re.search(r"Throughput: ([0-9.]+) qps", log)
    return float(match.group(1)) if match else None

def read_cuda_graph(log: str) -> str:
    match = re.search(r"CUDA Graph: ([A-Za-z]+)", log)
    return match.group(1) if match else "unknown"

rows = []
for label in ["cuda_graph_off", "cuda_graph_on"]:
    log_path = output_root / label / "trtexec.log"
    log = log_path.read_text(errors="replace")
    rows.append(
        {
            "config": label,
            "cuda_graph": read_cuda_graph(log),
            "throughput_qps": read_throughput(log),
            "latency_mean_ms": read_metric(log, "Latency"),
            "gpu_compute_mean_ms": read_metric(log, "GPU Compute Time"),
            "enqueue_mean_ms": read_metric(log, "Enqueue Time"),
            "h2d_mean_ms": read_metric(log, "H2D Latency"),
            "d2h_mean_ms": read_metric(log, "D2H Latency"),
            "log": str(log_path),
        }
    )

summary = {"engine": str(engine), "rows": rows}
(output_root / "cuda_graph_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

def fmt(value: float | None, suffix: str = "") -> str:
    if value is None:
        return "N/A"
    return f"{value:.4f}{suffix}"

lines = [
    "| 配置 | CUDA Graph | Throughput | Latency mean | GPU Compute mean | Enqueue mean | H2D mean | D2H mean |",
    "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
]
for row in rows:
    lines.append(
        "| {config} | {cuda_graph} | {throughput} | {latency} | {gpu} | {enqueue} | {h2d} | {d2h} |".format(
            config=row["config"],
            cuda_graph=row["cuda_graph"],
            throughput=fmt(row["throughput_qps"], " qps"),
            latency=fmt(row["latency_mean_ms"], " ms"),
            gpu=fmt(row["gpu_compute_mean_ms"], " ms"),
            enqueue=fmt(row["enqueue_mean_ms"], " ms"),
            h2d=fmt(row["h2d_mean_ms"], " ms"),
            d2h=fmt(row["d2h_mean_ms"], " ms"),
        )
    )

off, on = rows
if off["enqueue_mean_ms"] and on["enqueue_mean_ms"]:
    delta = 100.0 * (off["enqueue_mean_ms"] - on["enqueue_mean_ms"]) / off["enqueue_mean_ms"]
    lines.append("")
    lines.append(f"CUDA Graph ON 相比 OFF 的 Enqueue mean 变化：{delta:.2f}%。")
if off["latency_mean_ms"] and on["latency_mean_ms"]:
    delta = 100.0 * (off["latency_mean_ms"] - on["latency_mean_ms"]) / off["latency_mean_ms"]
    lines.append(f"CUDA Graph ON 相比 OFF 的 Latency mean 变化：{delta:.2f}%。")
if off["throughput_qps"] and on["throughput_qps"]:
    delta = 100.0 * (on["throughput_qps"] - off["throughput_qps"]) / off["throughput_qps"]
    lines.append(f"CUDA Graph ON 相比 OFF 的 Throughput 变化：{delta:.2f}%。")

(output_root / "cuda_graph_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print((output_root / "cuda_graph_summary.md").read_text(), end="")
PY

printf '%s\n' "step 5N CUDA Graph ON/OFF outputs:" \
  "  engine: $ENGINE" \
  "  off log: $OUTPUT_ROOT/cuda_graph_off/trtexec.log" \
  "  on log: $OUTPUT_ROOT/cuda_graph_on/trtexec.log" \
  "  summary json: $OUTPUT_ROOT/cuda_graph_summary.json" \
  "  summary md: $OUTPUT_ROOT/cuda_graph_summary.md"
