#!/usr/bin/env python3

"""Summarize SmoothQuant alpha sweep fake-dequant results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--alphas", nargs="+", required=True)
    return parser.parse_args()


def alpha_dir_name(alpha: str) -> str:
    return f"alpha_{alpha.replace('.', '_')}"


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def find_metric(rows: list[dict[str, Any]], output: str) -> dict[str, Any] | None:
    for row in rows:
        if row.get("pair") == "A_vs_E" and row.get("output") == output:
            return row
    return None


def add_output_metrics(row: dict[str, Any], prefix: str, metric: dict[str, Any] | None) -> None:
    if metric is None:
        row[f"{prefix}_relative_l2_mean"] = None
        row[f"{prefix}_relative_l2_p95"] = None
        row[f"{prefix}_relative_l2_max"] = None
        row[f"{prefix}_cosine_mean"] = None
        row[f"{prefix}_l2_norm_ratio_mean"] = None
        return
    row[f"{prefix}_relative_l2_mean"] = metric.get("relative_l2_error_mean")
    row[f"{prefix}_relative_l2_p95"] = metric.get("relative_l2_error_p95")
    row[f"{prefix}_relative_l2_max"] = metric.get("relative_l2_error_max")
    row[f"{prefix}_cosine_mean"] = metric.get("cosine_similarity_mean")
    row[f"{prefix}_l2_norm_ratio_mean"] = metric.get("l2_norm_ratio_mean")


def add_smoothquant_stats(row: dict[str, Any], calibration: dict[str, Any] | None) -> None:
    if calibration is None:
        return
    scales = calibration.get("linear_call_scales", [])
    row["linear_call_scales"] = len(scales)
    row["smoothquant_alpha"] = calibration.get("smoothquant_alpha")
    for key in (
        "source_activation_amax_max",
        "smoothed_activation_amax_max",
        "smooth_scale_min",
        "smooth_scale_max",
    ):
        values = [float(entry[key]) for entry in scales if key in entry]
        if values:
            row[f"{key}_min"] = min(values)
            row[f"{key}_mean"] = sum(values) / len(values)
            row[f"{key}_max"] = max(values)


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    rows = []
    for alpha in args.alphas:
        run_dir = output_root / alpha_dir_name(alpha)
        numeric = load_json(run_dir / "fake_dequant_numeric" / "numeric_baseline_summary.json")
        calibration = load_json(run_dir / f"smoothquant_alpha_{alpha}_activation_scales.json")
        row: dict[str, Any] = {
            "alpha": float(alpha),
            "status": "ok" if numeric is not None else "missing_numeric_summary",
        }
        add_smoothquant_stats(row, calibration)
        if numeric is not None:
            row["sample_count"] = numeric.get("sample_count")
            critical = numeric.get("critical_outputs", [])
            add_output_metrics(row, "action_chunk", find_metric(critical, "action_chunk"))
            add_output_metrics(row, "prefix_out", find_metric(critical, "prefix_out"))
            add_output_metrics(row, "v_t_step_09", find_metric(critical, "v_t_step_09"))
            add_output_metrics(row, "x_t_step_09", find_metric(critical, "x_t_step_09"))
        rows.append(row)

    rows = sorted(rows, key=lambda item: float(item["alpha"]))
    summary = {
        "output_root": str(output_root),
        "alphas": [float(alpha) for alpha in args.alphas],
        "rows": rows,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    with open(output_root / "smoothquant_alpha_sweep_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(output_root / "smoothquant_alpha_sweep_summary.csv", "w", newline="") as f:
        fieldnames = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
