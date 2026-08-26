#!/usr/bin/env python3

"""Plot no-SmoothQuant down_proj activation distribution diagnostics.

This script is intentionally stdlib-only so it can summarize existing 5J CSVs
without requiring the training/runtime Python environment. The plots show
per-channel activation statistics captured from real rollout samples.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any


DOWN_RE = re.compile(r"\.layers\.([0-9]+)\.mlp\.down_proj$")
COLORS = {
    3: "#dc2626",
    13: "#2563eb",
    30: "#16a34a",
    31: "#7c3aed",
}
DEFAULT_COLOR = "#525252"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--channel-stats-csv",
        default="runs/deploy/5/J-down-channel-attribution/down_channel_activation_error.csv",
    )
    parser.add_argument(
        "--layer-stats-csv",
        default="runs/deploy/5/J-down-channel-attribution/down_layer_error_amplification.csv",
    )
    parser.add_argument(
        "--activation-scales-json",
        default="runs/deploy/5/H-bf16-int8-w8a8/percentile_sweep/p99_999/activation_scales_p99.999.json",
    )
    parser.add_argument(
        "--output-dir",
        default="runs/deploy/5/H-bf16-int8-w8a8/down_activation_distribution",
    )
    parser.add_argument("--layers", default="3,13,30,31")
    parser.add_argument("--hist-bins", type=int, default=60)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, Any]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        if not rows:
            return
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def as_float(row: dict[str, Any], key: str) -> float:
    value = row.get(key, "")
    if value == "":
        return float("nan")
    return float(value)


def quantile(values: list[float], q: float) -> float:
    clean = sorted(v for v in values if math.isfinite(v))
    if not clean:
        return float("nan")
    if len(clean) == 1:
        return clean[0]
    pos = q * (len(clean) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return clean[lo]
    frac = pos - lo
    return clean[lo] * (1.0 - frac) + clean[hi] * frac


def load_scales(path: Path) -> dict[int, dict[str, Any]]:
    with open(path) as f:
        data = json.load(f)
    rows = data.get("linear_call_scales", [])
    result = {}
    for row in rows:
        module = str(row.get("module", ""))
        match = DOWN_RE.search(module)
        if not match:
            continue
        layer = int(match.group(1))
        result[layer] = row
    return result


def summarize_channels(
    channel_rows: list[dict[str, Any]],
    layer_rows: list[dict[str, Any]],
    scales: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows_by_layer: dict[int, list[dict[str, Any]]] = {}
    for row in channel_rows:
        rows_by_layer.setdefault(int(row["layer"]), []).append(row)
    layer_by_id = {int(row["layer"]): row for row in layer_rows}

    summary = []
    for layer in sorted(rows_by_layer):
        rows = rows_by_layer[layer]
        max_abs = [as_float(row, "max_abs") for row in rows]
        p99 = [as_float(row, "p99_99") for row in rows]
        std = [as_float(row, "std") for row in rows]
        qerr = [as_float(row, "quant_rel_l2") for row in rows]
        scale_row = scales.get(layer, {})
        layer_row = layer_by_id.get(layer, {})
        step = float(scale_row.get("scale", layer_row.get("scale", "nan")))
        covered = 127.0 * step if math.isfinite(step) else float("nan")
        summary.append(
            {
                "layer": layer,
                "channels": len(rows),
                "scale": step,
                "symmetric_int8_positive_limit": covered,
                "calibration_amax": float(scale_row.get("amax", "nan")),
                "calibration_percentile_amax": float(scale_row.get("percentile_amax", "nan")),
                "observed_max_abs": max(max_abs),
                "observed_p99_99_max": max(p99),
                "channel_max_abs_median": quantile(max_abs, 0.5),
                "channel_max_abs_p95": quantile(max_abs, 0.95),
                "channel_max_abs_p99": quantile(max_abs, 0.99),
                "channel_p99_99_median": quantile(p99, 0.5),
                "channel_p99_99_p95": quantile(p99, 0.95),
                "channel_p99_99_p99": quantile(p99, 0.99),
                "channel_std_median": quantile(std, 0.5),
                "channel_std_p99": quantile(std, 0.99),
                "channel_quant_rel_l2_median": quantile(qerr, 0.5),
                "channel_quant_rel_l2_p95": quantile(qerr, 0.95),
                "layer_input_rel_l2": as_float(layer_row, "input_rel_l2") if layer_row else float("nan"),
                "layer_output_rel_l2": as_float(layer_row, "output_rel_l2") if layer_row else float("nan"),
            }
        )
    return summary


def svg_header(width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#171717} .axis{stroke:#737373;stroke-width:1} .grid{stroke:#e5e5e5;stroke-width:1} .label{font-size:12px} .title{font-size:16px;font-weight:700}</style>',
    ]


def svg_footer() -> str:
    return "</svg>\n"


def log_bins(values: list[float], bins: int) -> tuple[list[int], float, float]:
    clean = [v for v in values if v > 0 and math.isfinite(v)]
    lo = math.log10(min(clean)) if clean else -6.0
    hi = math.log10(max(clean)) if clean else 1.0
    if hi <= lo:
        hi = lo + 1.0
    counts = [0] * bins
    for value in clean:
        idx = int((math.log10(value) - lo) / (hi - lo) * bins)
        counts[min(max(idx, 0), bins - 1)] += 1
    return counts, lo, hi


def plot_histogram_grid(
    path: Path,
    rows_by_layer: dict[int, list[dict[str, Any]]],
    layers: list[int],
    key: str,
    title: str,
    bins: int,
) -> None:
    width = 1120
    panel_h = 170
    margin_l = 70
    margin_r = 25
    margin_t = 55
    margin_b = 35
    height = margin_t + margin_b + panel_h * len(layers)
    out = svg_header(width, height)
    out.append(f'<text class="title" x="24" y="28">{title}</text>')
    plot_w = width - margin_l - margin_r
    bar_gap = 1

    for panel_idx, layer in enumerate(layers):
        top = margin_t + panel_idx * panel_h
        rows = rows_by_layer.get(layer, [])
        values = [as_float(row, key) for row in rows]
        counts, lo, hi = log_bins(values, bins)
        max_count = max(counts) if counts else 1
        color = COLORS.get(layer, DEFAULT_COLOR)
        out.append(f'<text class="label" x="24" y="{top + 18}">layer {layer}</text>')
        out.append(f'<line class="axis" x1="{margin_l}" y1="{top + panel_h - margin_b}" x2="{width - margin_r}" y2="{top + panel_h - margin_b}"/>')
        out.append(f'<line class="axis" x1="{margin_l}" y1="{top + 10}" x2="{margin_l}" y2="{top + panel_h - margin_b}"/>')
        for tick in range(math.floor(lo), math.ceil(hi) + 1):
            x = margin_l + (tick - lo) / (hi - lo) * plot_w
            out.append(f'<line class="grid" x1="{x:.1f}" y1="{top + 10}" x2="{x:.1f}" y2="{top + panel_h - margin_b}"/>')
            out.append(f'<text class="label" x="{x - 12:.1f}" y="{top + panel_h - 12}">1e{tick}</text>')
        bar_w = max(1.0, plot_w / bins - bar_gap)
        for idx, count in enumerate(counts):
            h = 0.0 if max_count == 0 else count / max_count * (panel_h - margin_b - 18)
            x = margin_l + idx * plot_w / bins
            y = top + panel_h - margin_b - h
            out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{color}" opacity="0.82"/>')
        out.append(f'<text class="label" x="{width - 210}" y="{top + 22}">channels={len(rows)}, max bin={max_count}</text>')
    out.append(svg_footer())
    path.write_text("\n".join(out))


def plot_layer_summary(path: Path, summary: list[dict[str, Any]], selected_layers: list[int]) -> None:
    width, height = 1120, 560
    ml, mr, mt, mb = 80, 35, 55, 65
    plot_w, plot_h = width - ml - mr, height - mt - mb
    layers = [int(row["layer"]) for row in summary]
    max_y = max(float(row["observed_max_abs"]) for row in summary)
    max_y = max(max_y, max(float(row["symmetric_int8_positive_limit"]) for row in summary))
    log_hi = math.ceil(math.log10(max_y))
    log_lo = -2

    def sx(layer: int) -> float:
        return ml + layer / max(layers) * plot_w

    def sy(value: float) -> float:
        value = max(value, 10**log_lo)
        return mt + (log_hi - math.log10(value)) / (log_hi - log_lo) * plot_h

    out = svg_header(width, height)
    out.append('<text class="title" x="24" y="28">No-SmoothQuant down_proj activation range by layer</text>')
    for tick in range(log_lo, log_hi + 1):
        y = sy(10.0**tick)
        out.append(f'<line class="grid" x1="{ml}" y1="{y:.1f}" x2="{width - mr}" y2="{y:.1f}"/>')
        out.append(f'<text class="label" x="28" y="{y + 4:.1f}">1e{tick}</text>')
    for layer in range(0, max(layers) + 1, 4):
        x = sx(layer)
        out.append(f'<line class="grid" x1="{x:.1f}" y1="{mt}" x2="{x:.1f}" y2="{height - mb}"/>')
        out.append(f'<text class="label" x="{x - 6:.1f}" y="{height - 35}">{layer}</text>')
    out.append(f'<line class="axis" x1="{ml}" y1="{height - mb}" x2="{width - mr}" y2="{height - mb}"/>')
    out.append(f'<line class="axis" x1="{ml}" y1="{mt}" x2="{ml}" y2="{height - mb}"/>')

    series = [
        ("observed_max_abs", "#dc2626", "observed max_abs"),
        ("channel_max_abs_p99", "#f59e0b", "channel max_abs p99"),
        ("channel_p99_99_median", "#2563eb", "channel p99.99 median"),
        ("symmetric_int8_positive_limit", "#525252", "127 * scale"),
    ]
    for key, color, label in series:
        points = [(sx(int(row["layer"])), sy(float(row[key]))) for row in summary]
        d = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        out.append(f'<polyline points="{d}" fill="none" stroke="{color}" stroke-width="2"/>')
        out.append(f'<circle cx="{points[-1][0]:.1f}" cy="{points[-1][1]:.1f}" r="3" fill="{color}"/>')
        out.append(f'<text class="label" x="{width - 230}" y="{mt + 20 + series.index((key, color, label)) * 20}" fill="{color}">{label}</text>')
    for layer in selected_layers:
        x = sx(layer)
        out.append(f'<line x1="{x:.1f}" y1="{mt}" x2="{x:.1f}" y2="{height - mb}" stroke="{COLORS.get(layer, DEFAULT_COLOR)}" stroke-width="1.5" stroke-dasharray="5 4"/>')
        out.append(f'<text class="label" x="{x + 4:.1f}" y="{mt + 15}">L{layer}</text>')
    out.append('<text class="label" x="500" y="535">layer</text>')
    out.append('<text class="label" transform="translate(16 330) rotate(-90)">absolute activation value, log scale</text>')
    out.append(svg_footer())
    path.write_text("\n".join(out))


def plot_error_scatter(path: Path, channel_rows: list[dict[str, Any]], selected_layers: list[int]) -> None:
    width, height = 900, 620
    ml, mr, mt, mb = 80, 140, 45, 65
    plot_w, plot_h = width - ml - mr, height - mt - mb
    rows = [row for row in channel_rows if int(row["layer"]) in selected_layers]
    x_vals = [max(as_float(row, "max_abs"), 1.0e-8) for row in rows]
    y_vals = [max(as_float(row, "quant_rel_l2"), 1.0e-8) for row in rows]
    xlo, xhi = math.floor(math.log10(min(x_vals))), math.ceil(math.log10(max(x_vals)))
    ylo, yhi = -3, max(1, math.ceil(math.log10(max(y_vals))))

    def sx(value: float) -> float:
        return ml + (math.log10(max(value, 1.0e-8)) - xlo) / (xhi - xlo) * plot_w

    def sy(value: float) -> float:
        return mt + (yhi - math.log10(max(value, 1.0e-8))) / (yhi - ylo) * plot_h

    out = svg_header(width, height)
    out.append('<text class="title" x="24" y="28">Selected down_proj channels: range vs A8 input error</text>')
    for tick in range(xlo, xhi + 1):
        x = sx(10.0**tick)
        out.append(f'<line class="grid" x1="{x:.1f}" y1="{mt}" x2="{x:.1f}" y2="{height - mb}"/>')
        out.append(f'<text class="label" x="{x - 12:.1f}" y="{height - 38}">1e{tick}</text>')
    for tick in range(ylo, yhi + 1):
        y = sy(10.0**tick)
        out.append(f'<line class="grid" x1="{ml}" y1="{y:.1f}" x2="{width - mr}" y2="{y:.1f}"/>')
        out.append(f'<text class="label" x="28" y="{y + 4:.1f}">1e{tick}</text>')
    out.append(f'<line class="axis" x1="{ml}" y1="{height - mb}" x2="{width - mr}" y2="{height - mb}"/>')
    out.append(f'<line class="axis" x1="{ml}" y1="{mt}" x2="{ml}" y2="{height - mb}"/>')
    for row in rows:
        layer = int(row["layer"])
        out.append(
            f'<circle cx="{sx(as_float(row, "max_abs")):.1f}" cy="{sy(as_float(row, "quant_rel_l2")):.1f}" '
            f'r="2.2" fill="{COLORS.get(layer, DEFAULT_COLOR)}" opacity="0.38"/>'
        )
    for idx, layer in enumerate(selected_layers):
        out.append(f'<circle cx="{width - 110}" cy="{mt + 22 + idx * 22}" r="5" fill="{COLORS.get(layer, DEFAULT_COLOR)}"/>')
        out.append(f'<text class="label" x="{width - 96}" y="{mt + 26 + idx * 22}">layer {layer}</text>')
    out.append('<text class="label" x="350" y="585">channel max_abs, log scale</text>')
    out.append('<text class="label" transform="translate(16 390) rotate(-90)">channel quant_rel_l2, log scale</text>')
    out.append(svg_footer())
    path.write_text("\n".join(out))


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_layers = [int(item) for item in args.layers.split(",") if item.strip()]

    channel_rows = read_csv(Path(args.channel_stats_csv))
    layer_rows = read_csv(Path(args.layer_stats_csv))
    scales = load_scales(Path(args.activation_scales_json))
    summary = summarize_channels(channel_rows, layer_rows, scales)
    write_csv(output_dir / "down_activation_distribution_summary.csv", summary)

    rows_by_layer: dict[int, list[dict[str, Any]]] = {}
    for row in channel_rows:
        rows_by_layer.setdefault(int(row["layer"]), []).append(row)

    plot_histogram_grid(
        output_dir / "selected_down_channel_max_abs_hist.svg",
        rows_by_layer,
        selected_layers,
        "max_abs",
        "Selected no-SmoothQuant down_proj channel max_abs distribution",
        args.hist_bins,
    )
    plot_histogram_grid(
        output_dir / "selected_down_channel_p99_99_hist.svg",
        rows_by_layer,
        selected_layers,
        "p99_99",
        "Selected no-SmoothQuant down_proj channel p99.99 distribution",
        args.hist_bins,
    )
    plot_layer_summary(output_dir / "down_activation_range_by_layer.svg", summary, selected_layers)
    plot_error_scatter(output_dir / "selected_down_range_vs_quant_error.svg", channel_rows, selected_layers)

    report = {
        "channel_stats_csv": args.channel_stats_csv,
        "layer_stats_csv": args.layer_stats_csv,
        "activation_scales_json": args.activation_scales_json,
        "selected_layers": selected_layers,
        "note": "Plots use per-channel activation statistics from no-SmoothQuant rollout captures, not raw per-element activations.",
        "outputs": {
            "summary_csv": str(output_dir / "down_activation_distribution_summary.csv"),
            "channel_max_abs_hist_svg": str(output_dir / "selected_down_channel_max_abs_hist.svg"),
            "channel_p99_99_hist_svg": str(output_dir / "selected_down_channel_p99_99_hist.svg"),
            "range_by_layer_svg": str(output_dir / "down_activation_range_by_layer.svg"),
            "range_vs_quant_error_svg": str(output_dir / "selected_down_range_vs_quant_error.svg"),
        },
    }
    (output_dir / "down_activation_distribution_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
