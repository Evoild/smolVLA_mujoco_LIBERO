#!/usr/bin/env python3

"""Summarize Step 5H per-tensor activation percentile sweep."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--percentiles", nargs="+", required=True)
    return parser.parse_args()


def percentile_dir_name(percentile: str) -> str:
    return f"p{percentile.replace('.', '_')}"


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


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    rows = []
    for percentile in args.percentiles:
        run_dir = output_root / percentile_dir_name(percentile)
        numeric = load_json(run_dir / "fake_dequant_numeric" / "numeric_baseline_summary.json")
        calibration = load_json(run_dir / f"activation_scales_p{percentile}.json")
        row: dict[str, Any] = {
            "percentile": percentile,
            "status": "ok" if numeric is not None else "missing_numeric_summary",
        }
        if calibration:
            scales = calibration.get("linear_call_scales", [])
            row["calibration_samples_collected"] = calibration.get("samples_collected")
            row["linear_call_scales"] = len(scales)
            numeric_scales = [entry.get("scale") for entry in scales if isinstance(entry.get("scale"), (int, float))]
            if numeric_scales:
                row["min_scale"] = min(numeric_scales)
                row["max_scale"] = max(numeric_scales)
        if numeric is not None:
            critical = numeric.get("critical_outputs", [])
            add_output_metrics(row, "action_chunk", find_metric(critical, "action_chunk"))
            add_output_metrics(row, "prefix_out", find_metric(critical, "prefix_out"))
            add_output_metrics(row, "v_t_step_09", find_metric(critical, "v_t_step_09"))
            add_output_metrics(row, "x_t_step_09", find_metric(critical, "x_t_step_09"))
        rows.append(row)

    summary = {
        "output_root": str(output_root),
        "percentiles": args.percentiles,
        "rows": rows,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    with open(output_root / "percentile_sweep_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(output_root / "percentile_sweep_summary.csv", "w", newline="") as f:
        fieldnames = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
