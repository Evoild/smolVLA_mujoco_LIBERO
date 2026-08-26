#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$REPO_ROOT/scripts/common.sh"

CHANNEL_STATS_CSV="$REPO_ROOT/runs/deploy/5/J-down-channel-attribution/down_channel_activation_error.csv"
LAYER_STATS_CSV="$REPO_ROOT/runs/deploy/5/J-down-channel-attribution/down_layer_error_amplification.csv"
SCALES_JSON="$REPO_ROOT/runs/deploy/5/H-bf16-int8-w8a8/percentile_sweep/p99_999/activation_scales_p99.999.json"
OUTPUT_ROOT="$REPO_ROOT/runs/deploy/5/H-bf16-int8-w8a8/down_activation_distribution"
LAYERS="3,13,30,31"
HIST_BINS=60
PYTHON="${PYTHON_BIN:-python3}"

usage() {
  printf '%s\n' "Usage: $0 [options]" \
    "  --channel-stats-csv PATH    default: $CHANNEL_STATS_CSV" \
    "  --layer-stats-csv PATH      default: $LAYER_STATS_CSV" \
    "  --scales-json PATH          default: $SCALES_JSON" \
    "  --output-root DIR           default: $OUTPUT_ROOT" \
    "  --layers LIST               default: $LAYERS" \
    "  --hist-bins N               default: $HIST_BINS" \
    "  PYTHON_BIN=/path/python     optional Python; stdlib only is enough"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --channel-stats-csv) require_value "$@"; CHANNEL_STATS_CSV="$2"; shift 2 ;;
    --layer-stats-csv) require_value "$@"; LAYER_STATS_CSV="$2"; shift 2 ;;
    --scales-json) require_value "$@"; SCALES_JSON="$2"; shift 2 ;;
    --output-root) require_value "$@"; OUTPUT_ROOT="$2"; shift 2 ;;
    --layers) require_value "$@"; LAYERS="$2"; shift 2 ;;
    --hist-bins) require_value "$@"; HIST_BINS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ -f "$CHANNEL_STATS_CSV" ]] || die "missing channel stats csv: $CHANNEL_STATS_CSV; run deploy/5Jdown-channel-attribution.sh first"
[[ -f "$LAYER_STATS_CSV" ]] || die "missing layer stats csv: $LAYER_STATS_CSV; run deploy/5Jdown-channel-attribution.sh first"
[[ -f "$SCALES_JSON" ]] || die "missing activation scales: $SCALES_JSON"

mkdir -p "$OUTPUT_ROOT"

"$PYTHON" "$SCRIPT_DIR/plot_down_activation_distribution.py" \
  --channel-stats-csv "$CHANNEL_STATS_CSV" \
  --layer-stats-csv "$LAYER_STATS_CSV" \
  --activation-scales-json "$SCALES_JSON" \
  --output-dir "$OUTPUT_ROOT" \
  --layers "$LAYERS" \
  --hist-bins "$HIST_BINS"

printf '%s\n' "step 5H.17 no-SmoothQuant down activation distribution outputs:" \
  "  output root: $OUTPUT_ROOT" \
  "  summary: $OUTPUT_ROOT/down_activation_distribution_summary.csv" \
  "  range by layer: $OUTPUT_ROOT/down_activation_range_by_layer.svg" \
  "  selected max_abs hist: $OUTPUT_ROOT/selected_down_channel_max_abs_hist.svg" \
  "  selected p99.99 hist: $OUTPUT_ROOT/selected_down_channel_p99_99_hist.svg" \
  "  selected range vs quant error: $OUTPUT_ROOT/selected_down_range_vs_quant_error.svg"
