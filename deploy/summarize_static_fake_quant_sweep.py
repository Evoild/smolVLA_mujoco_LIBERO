#!/usr/bin/env python3

"""Summarize static fake-quant percentile sweep results."""

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


def find_metric(rows: list[dict[str, Any]], pair: str, output: str) -> dict[str, Any] | None:
    for row in rows:
        if row.get("pair") == pair and row.get("output") == output:
            return row
    return None


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    rows = []
    for percentile in args.percentiles:
        run_dir = output_root / percentile_dir_name(percentile)
        numeric = load_json(run_dir / "numeric_baseline" / "numeric_baseline_summary.json")
        clipping = load_json(run_dir / "clipping" / "clipping_summary.json")
        if numeric is None:
            rows.append({"percentile": percentile, "status": "missing_numeric_summary"})
            continue

        action = find_metric(numeric.get("critical_outputs", []), "A_vs_E", "action_chunk")
        vt = find_metric(numeric.get("critical_outputs", []), "A_vs_E", "v_t_step_09")
        xt = find_metric(numeric.get("critical_outputs", []), "A_vs_E", "x_t_step_09")
        if action is None:
            rows.append({"percentile": percentile, "status": "missing_A_vs_E_action_chunk"})
            continue

        row = {
            "percentile": percentile,
            "status": "ok",
            "action_cosine": action["cosine_similarity"],
            "action_relative_l2_error": action["relative_l2_error"],
            "action_l2_norm_ratio": action["l2_norm_ratio"],
            "action_std_ratio": action["std_ratio"],
            "action_mean_diff": action["mean_diff"],
            "action_max_abs_error": action["max_abs_error"],
            "v_t_step_09_relative_l2_error": vt["relative_l2_error"] if vt else None,
            "x_t_step_09_relative_l2_error": xt["relative_l2_error"] if xt else None,
            "max_clipping_ratio": clipping.get("max_clipping_ratio") if clipping else None,
            "mean_clipping_ratio": clipping.get("mean_clipping_ratio") if clipping else None,
        }
        rows.append(row)

    summary = {
        "output_root": str(output_root),
        "percentiles": args.percentiles,
        "rows": rows,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    with open(output_root / "static_fake_quant_sweep_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(output_root / "static_fake_quant_sweep_summary.csv", "w", newline="") as f:
        fieldnames = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
