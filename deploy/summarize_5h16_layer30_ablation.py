#!/usr/bin/env python3

"""Summarize and plot Step 5H.16 layer30 FFN ablation results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def collect_rows(root: Path) -> list[dict[str, Any]]:
    rows = []
    for run_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        config = run_dir.name
        numeric_path = run_dir / "fake_dequant_numeric" / "numeric_baseline_summary.json"
        layer_path = run_dir / "layer_outputs" / "text_vlm_layer_output_l2_summary.json"
        block_path = run_dir / "block_outputs" / "text_vlm_block_output_l2_summary.json"
        if not numeric_path.exists() or not layer_path.exists() or not block_path.exists():
            continue
        numeric = read_json(numeric_path)
        layer = read_json(layer_path)
        block = read_json(block_path)
        action = next(item for item in numeric["critical_outputs"] if item["output"] == "action_chunk")
        layer30 = {row["stage"]: row for row in block["rows"] if int(row["layer"]) == 30}
        layer31 = {row["stage"]: row for row in block["rows"] if int(row["layer"]) == 31}
        row = {
            "config": config,
            "num_linear_replaced": layer["num_linear_replaced"],
            "action_relative_l2": action["relative_l2_error_mean"],
            "action_cosine": action["cosine_similarity_mean"],
            "action_l2_p95": action["relative_l2_error_p95"],
            "layer30_attention_residual_l2": layer30["attention_residual_output"]["relative_l2_mean"],
            "layer30_block_end_l2": layer30["ffn_residual_output"]["relative_l2_mean"],
            "layer31_block_end_l2": layer31["ffn_residual_output"]["relative_l2_mean"],
        }
        for sublayer in ("gate_proj", "up_proj", "down_proj"):
            row[f"layer30_{sublayer}_output_l2"] = next(
                item["relative_l2_error_mean"]
                for item in layer["rows"]
                if int(item["layer"]) == 30 and item["sublayer"] == sublayer
            )
        rows.append(row)
    order = {
        "skip_l30_ffn": 0,
        "skip_l30_down": 1,
        "skip_l30_gate_up": 2,
        "skip_l30_gate": 3,
        "skip_l30_up": 4,
    }
    return sorted(rows, key=lambda row: order.get(row["config"], 99))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with open(path, "w", newline="") as f:
        if not rows:
            return
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_plots(root: Path, rows: list[dict[str, Any]]) -> dict[str, str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return {"plot_status": f"skipped: {exc!r}"}

    plots: dict[str, str] = {}
    labels = [row["config"] for row in rows]
    x = list(range(len(rows)))

    fig, ax = plt.subplots(figsize=(11.5, 5.2), dpi=160)
    series = (
        ("action_relative_l2", "action rel L2", "#2563eb"),
        ("layer30_block_end_l2", "layer30 block end", "#dc2626"),
        ("layer31_block_end_l2", "layer31 block end", "#7c3aed"),
    )
    for key, label, color in series:
        ax.plot(x, [row[key] for row in rows], marker="o", linewidth=1.8, markersize=4, label=label, color=color)
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.set_ylabel("Relative L2 vs BF16")
    ax.set_title("5H.16 layer30 FFN fallback ablation summary")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)
    ax.legend()
    fig.tight_layout()
    path = root / "summary_action_block_l2.png"
    fig.savefig(path)
    plt.close(fig)
    plots["summary_action_block_l2_png"] = str(path)

    fig, ax = plt.subplots(figsize=(11.5, 5.2), dpi=160)
    series = (
        ("layer30_gate_proj_output_l2", "layer30 gate", "#16a34a"),
        ("layer30_up_proj_output_l2", "layer30 up", "#ca8a04"),
        ("layer30_down_proj_output_l2", "layer30 down", "#dc2626"),
    )
    for key, label, color in series:
        ax.plot(x, [row[key] for row in rows], marker="o", linewidth=1.8, markersize=4, label=label, color=color)
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.set_ylabel("Linear output relative L2 vs BF16")
    ax.set_title("5H.16 layer30 FFN sublayer output L2")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)
    ax.legend()
    fig.tight_layout()
    path = root / "summary_layer30_ffn_sublayer_l2.png"
    fig.savefig(path)
    plt.close(fig)
    plots["summary_layer30_ffn_sublayer_l2_png"] = str(path)
    return plots


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    rows = collect_rows(root)
    plots = write_plots(root, rows)
    summary = {"configs": rows, "plots": plots}
    (root / "summary.json").write_text(json.dumps(summary, indent=2))
    write_csv(root / "summary.csv", rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
