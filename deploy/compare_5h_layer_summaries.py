#!/usr/bin/env python3

"""Plot side-by-side layer output L2 curves from two Step 5H summaries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


SUBLAYERS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
ATTN_SUBLAYERS = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP_SUBLAYERS = ("gate_proj", "up_proj", "down_proj")
COLORS = {
    "q_proj": "#2563eb",
    "k_proj": "#0891b2",
    "v_proj": "#7c3aed",
    "o_proj": "#db2777",
    "gate_proj": "#16a34a",
    "up_proj": "#ca8a04",
    "down_proj": "#dc2626",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-summary", required=True)
    parser.add_argument("--left-label", required=True)
    parser.add_argument("--right-summary", required=True)
    parser.add_argument("--right-label", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def load_rows(path: str, label: str) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text())
    rows = []
    for row in data["rows"]:
        rows.append(
            {
                "config": label,
                "layer": int(row["layer"]),
                "sublayer": row["sublayer"],
                "family": row["family"],
                "relative_l2": float(row["relative_l2_error_mean"]),
                "cosine": float(row["cosine_similarity_mean"]),
                "norm_ratio": float(row["l2_norm_ratio_mean"]),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for config in sorted({row["config"] for row in rows}):
        for sublayer in SUBLAYERS:
            vals = [row["relative_l2"] for row in rows if row["config"] == config and row["sublayer"] == sublayer]
            sub_rows = [row for row in rows if row["config"] == config and row["sublayer"] == sublayer]
            worst = max(sub_rows, key=lambda row: row["relative_l2"])
            result.append(
                {
                    "config": config,
                    "sublayer": sublayer,
                    "mean_relative_l2": sum(vals) / len(vals),
                    "max_relative_l2": worst["relative_l2"],
                    "worst_layer": worst["layer"],
                }
            )
    return result


def write_group_plot(rows: list[dict[str, Any]], labels: tuple[str, str], sublayers: tuple[str, ...], path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10.8, 5.6), dpi=160)
    linestyles = {labels[0]: "-", labels[1]: "--"}
    markers = {labels[0]: "o", labels[1]: "s"}
    for config in labels:
        for sublayer in sublayers:
            sub_rows = sorted(
                (row for row in rows if row["config"] == config and row["sublayer"] == sublayer),
                key=lambda row: row["layer"],
            )
            ax.plot(
                [row["layer"] for row in sub_rows],
                [row["relative_l2"] for row in sub_rows],
                color=COLORS[sublayer],
                linestyle=linestyles[config],
                marker=markers[config],
                linewidth=1.5,
                markersize=3.0,
                label=f"{config} {sublayer}",
                alpha=0.9,
            )
    ax.set_xlabel("VLM text_model layer")
    ax.set_ylabel("Output relative L2 error vs BF16")
    ax.set_title(title)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_facets(rows: list[dict[str, Any]], labels: tuple[str, str], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 4, figsize=(15.0, 7.2), dpi=160, sharex=True)
    axes_flat = axes.flatten()
    linestyles = {labels[0]: "-", labels[1]: "--"}
    markers = {labels[0]: "o", labels[1]: "s"}
    for idx, sublayer in enumerate(SUBLAYERS):
        ax = axes_flat[idx]
        for config in labels:
            sub_rows = sorted(
                (row for row in rows if row["config"] == config and row["sublayer"] == sublayer),
                key=lambda row: row["layer"],
            )
            ax.plot(
                [row["layer"] for row in sub_rows],
                [row["relative_l2"] for row in sub_rows],
                color=COLORS[sublayer],
                linestyle=linestyles[config],
                marker=markers[config],
                linewidth=1.6,
                markersize=3.0,
                label=config,
            )
        ax.set_title(sublayer)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)
    axes_flat[-1].axis("off")
    handles, legend_labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower right")
    fig.supxlabel("VLM text_model layer")
    fig.supylabel("Output relative L2 error vs BF16")
    fig.suptitle("Layer output relative L2 by sublayer")
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = (args.left_label, args.right_label)
    rows = load_rows(args.left_summary, args.left_label) + load_rows(args.right_summary, args.right_label)
    summary = summarize(rows)
    write_csv(output_dir / "layer_output_comparison_rows.csv", rows)
    write_csv(output_dir / "layer_output_comparison_summary.csv", summary)

    write_group_plot(rows, labels, ATTN_SUBLAYERS, output_dir / "attention_layer_output_relative_l2_compare.png", "Attention output relative L2")
    write_group_plot(rows, labels, MLP_SUBLAYERS, output_dir / "mlp_layer_output_relative_l2_compare.png", "MLP output relative L2")
    write_facets(rows, labels, output_dir / "text_vlm_layer_output_relative_l2_facets.png")

    report = {
        "left_summary": args.left_summary,
        "left_label": args.left_label,
        "right_summary": args.right_summary,
        "right_label": args.right_label,
        "plots": {
            "attention_compare": str(output_dir / "attention_layer_output_relative_l2_compare.png"),
            "mlp_compare": str(output_dir / "mlp_layer_output_relative_l2_compare.png"),
            "facets": str(output_dir / "text_vlm_layer_output_relative_l2_facets.png"),
        },
        "summary": summary,
    }
    (output_dir / "layer_output_comparison_summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
