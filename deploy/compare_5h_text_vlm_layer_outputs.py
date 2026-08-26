#!/usr/bin/env python3

"""Compare Step 5H fake-quant text VLM layer outputs against BF16 PyTorch."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch import nn

from diagnose_3_4_numeric_baseline import (
    SmolVLADebugCoreWrapper,
    collect_rollout_core_inputs,
    compare_tensors,
    core_inputs_from_policy,
    device_inputs,
    load_policy,
)
from linear_only_quant import load_activation_scales_by_module, replace_linear_modules


DEFAULT_REGEX = (
    r"^model\.vlm_with_expert\.vlm\.model\.text_model\.layers\.([0-9]+)"
    r"\.(?:self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", default="smolvla_libero")
    parser.add_argument("--activation-scales-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--task", default="libero_spatial")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--token-length", type=int, default=48)
    parser.add_argument("--input-dtype", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--model-dtype", choices=["native", "bf16"], default="bf16")
    parser.add_argument("--input-source", choices=["synthetic", "rollout"], default="synthetic")
    parser.add_argument("--compare-samples", type=int, default=1)
    parser.add_argument("--sample-stride", type=int, default=5)
    parser.add_argument("--max-parallel-tasks", type=int, default=1)
    parser.add_argument("--module-regex", default=DEFAULT_REGEX)
    parser.add_argument(
        "--quantize-module-regex",
        default=None,
        help="Regex for Linear modules to fake-quantize. Defaults to --module-regex.",
    )
    return parser.parse_args()


def register_output_hooks(
    wrapper: SmolVLADebugCoreWrapper,
    pattern: re.Pattern[str],
) -> tuple[dict[str, list[torch.Tensor]], list[Any], dict[str, dict[str, Any]]]:
    captures: dict[str, list[torch.Tensor]] = {}
    metadata: dict[str, dict[str, Any]] = {}
    handles = []

    def make_hook(name: str):
        def hook(_module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            if isinstance(output, torch.Tensor):
                captures.setdefault(name, []).append(output.detach().to(torch.float32).cpu())

        return hook

    for name, module in wrapper.named_modules():
        match = pattern.search(name)
        if not match or not isinstance(module, nn.Module):
            continue
        layer = int(match.group(1))
        sublayer = match.group(2) or match.group(3)
        if sublayer is None:
            continue
        family = "attention" if sublayer in {"q_proj", "k_proj", "v_proj", "o_proj"} else "mlp"
        captures[name] = []
        metadata[name] = {"layer": layer, "sublayer": sublayer, "family": family, "module": name}
        handles.append(module.register_forward_hook(make_hook(name)))
    return captures, handles, metadata


def concat_captures(values: list[torch.Tensor]) -> torch.Tensor:
    if not values:
        return torch.empty(0, dtype=torch.float32)
    return torch.cat([value.flatten() for value in values], dim=0)


def run_with_hooks(
    policy: nn.Module,
    inputs: tuple[torch.Tensor, ...],
    pattern: re.Pattern[str],
) -> tuple[dict[str, list[torch.Tensor]], dict[str, dict[str, Any]]]:
    wrapper = SmolVLADebugCoreWrapper(policy).to(device=str(policy.config.device)).eval()
    captures, handles, metadata = register_output_hooks(wrapper, pattern)
    try:
        with torch.no_grad():
            wrapper(*inputs)
    finally:
        for handle in handles:
            handle.remove()
    return captures, metadata


def metric_mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    if not values:
        return None
    return float(sum(values) / len(values))


def aggregate_rows(sample_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sample_rows:
        grouped[row["module"]].append(row)
    aggregate = []
    for module_name, rows in grouped.items():
        first = rows[0]
        aggregate.append(
            {
                "module": module_name,
                "layer": first["layer"],
                "sublayer": first["sublayer"],
                "family": first["family"],
                "num_samples": len(rows),
                "native_num_calls_mean": metric_mean(rows, "native_num_calls"),
                "fake_quant_num_calls_mean": metric_mean(rows, "fake_quant_num_calls"),
                "cosine_similarity_mean": metric_mean(rows, "cosine_similarity"),
                "relative_l2_error_mean": metric_mean(rows, "relative_l2_error"),
                "l2_norm_ratio_mean": metric_mean(rows, "l2_norm_ratio"),
                "max_abs_error_mean": metric_mean(rows, "max_abs_error"),
                "mean_abs_error_mean": metric_mean(rows, "mean_abs_error"),
                "std_ratio_mean": metric_mean(rows, "std_ratio"),
            }
        )
    return sorted(aggregate, key=lambda row: (int(row["layer"]), str(row["sublayer"])))


def write_plot(rows: list[dict[str, Any]], output_dir: Path) -> dict[str, str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional local dependency
        return {"plot_status": f"skipped: {exc!r}"}

    colors = {
        "q_proj": "#2563eb",
        "k_proj": "#0891b2",
        "v_proj": "#7c3aed",
        "o_proj": "#db2777",
        "gate_proj": "#16a34a",
        "up_proj": "#ca8a04",
        "down_proj": "#dc2626",
    }
    groups = {
        "text_vlm_layer_output_relative_l2": ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
        "attention_layer_output_relative_l2": ("q_proj", "k_proj", "v_proj", "o_proj"),
        "mlp_layer_output_relative_l2": ("gate_proj", "up_proj", "down_proj"),
    }
    paths: dict[str, str] = {}
    for stem, sublayers in groups.items():
        fig, ax = plt.subplots(figsize=(9.8, 5.2), dpi=160)
        for sublayer in sublayers:
            sub_rows = sorted((row for row in rows if row["sublayer"] == sublayer), key=lambda row: row["layer"])
            ax.plot(
                [row["layer"] for row in sub_rows],
                [row["relative_l2_error_mean"] for row in sub_rows],
                marker="o",
                linewidth=1.6,
                markersize=3.2,
                label=sublayer,
                color=colors[sublayer],
            )
        ax.set_xlabel("VLM text_model layer")
        ax.set_ylabel("Output relative L2 error vs BF16")
        ax.set_title(stem.replace("_", " "))
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)
        ax.legend(ncol=2 if len(sublayers) > 4 else 1)
        fig.tight_layout()
        png_path = output_dir / f"{stem}.png"
        svg_path = output_dir / f"{stem}.svg"
        fig.savefig(png_path)
        fig.savefig(svg_path)
        plt.close(fig)
        paths[f"{stem}_png"] = str(png_path)
        paths[f"{stem}_svg"] = str(svg_path)

    for sublayer, color in colors.items():
        fig, ax = plt.subplots(figsize=(8.0, 4.4), dpi=160)
        sub_rows = sorted((row for row in rows if row["sublayer"] == sublayer), key=lambda row: row["layer"])
        ax.plot(
            [row["layer"] for row in sub_rows],
            [row["relative_l2_error_mean"] for row in sub_rows],
            marker="o",
            linewidth=1.8,
            markersize=3.5,
            color=color,
        )
        ax.set_xlabel("VLM text_model layer")
        ax.set_ylabel("Output relative L2 error vs BF16")
        ax.set_title(f"{sublayer} output relative L2")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)
        fig.tight_layout()
        path = output_dir / f"{sublayer}_output_relative_l2.png"
        fig.savefig(path)
        plt.close(fig)
        paths[f"{sublayer}_plot"] = str(path)
    return paths


def make_input_records(args: argparse.Namespace, original: nn.Module, preprocessor: Any, device: str):
    input_dtype = torch.float32 if args.input_dtype == "fp32" else torch.bfloat16
    if args.input_source == "synthetic":
        inputs = core_inputs_from_policy(original, preprocessor, args.task, args.seed, input_dtype, args.token_length)
        return [
            {
                "sample_index": 0,
                "input_source": "synthetic",
                "core_inputs": tuple(tensor.detach().cpu() for tensor in inputs),
            }
        ]
    records, _rollout_device = collect_rollout_core_inputs(args, args.compare_samples)
    return records


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(args.module_regex)

    original, preprocessor, device = load_policy(args.policy_path, args.device, args.model_dtype)
    input_records = make_input_records(args, original, preprocessor, device)

    fake_quant, _, _ = load_policy(args.policy_path, args.device, args.model_dtype)
    quantize_module_regex = args.quantize_module_regex or args.module_regex
    activation_scales = load_activation_scales_by_module(args.activation_scales_json, (quantize_module_regex,))
    replaced = replace_linear_modules(
        fake_quant,
        include_prefixes=None,
        include_regexes=(quantize_module_regex,),
        activation_scales_by_module=activation_scales,
        quantization_kind="w8a8",
    )
    fake_quant.to(device=device).eval()

    rows: list[dict[str, Any]] = []
    metadata_reference: dict[str, dict[str, Any]] | None = None
    for record in input_records:
        inputs = device_inputs(record["core_inputs"], device)
        baseline_captures, metadata = run_with_hooks(original, inputs, pattern)
        quant_captures, quant_metadata = run_with_hooks(fake_quant, inputs, pattern)
        if set(metadata) != set(quant_metadata):
            missing_lhs = sorted(set(quant_metadata) - set(metadata))
            missing_rhs = sorted(set(metadata) - set(quant_metadata))
            raise RuntimeError(f"hook target mismatch: only_quant={missing_lhs}, only_native={missing_rhs}")
        metadata_reference = metadata
        for module_name in sorted(metadata, key=lambda name: (metadata[name]["layer"], metadata[name]["sublayer"])):
            lhs = concat_captures(baseline_captures[module_name])
            rhs = concat_captures(quant_captures[module_name])
            row = compare_tensors("A_vs_E_layer_output", module_name, lhs, rhs)
            row.update(metadata[module_name])
            row["sample_index"] = record["sample_index"]
            row["input_source"] = record.get("input_source", args.input_source)
            row["native_num_calls"] = len(baseline_captures[module_name])
            row["fake_quant_num_calls"] = len(quant_captures[module_name])
            rows.append(row)

    aggregate = aggregate_rows(rows)
    plot_report = write_plot(aggregate, output_dir)
    summary_by_sublayer = {}
    for sublayer in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
        sub_rows = [row for row in aggregate if row["sublayer"] == sublayer]
        values = torch.tensor([row["relative_l2_error_mean"] for row in sub_rows], dtype=torch.float32)
        worst = max(sub_rows, key=lambda row: row["relative_l2_error_mean"]) if sub_rows else None
        summary_by_sublayer[sublayer] = {
            "num_layers": len(sub_rows),
            "mean_relative_l2_error": float(values.mean()) if len(values) else None,
            "max_relative_l2_error": float(values.max()) if len(values) else None,
            "worst_layer": worst["layer"] if worst else None,
            "worst_module": worst["module"] if worst else None,
        }

    summary = {
        "policy_path": args.policy_path,
        "activation_scales_json": args.activation_scales_json,
        "seed": args.seed,
        "task": args.task,
        "device": device,
        "model_dtype": args.model_dtype,
        "input_source": args.input_source,
        "sample_count": len(input_records),
        "module_regex": args.module_regex,
        "quantize_module_regex": quantize_module_regex,
        "num_linear_replaced": replaced,
        "num_modules_compared": len(metadata_reference or {}),
        "summary_by_sublayer": summary_by_sublayer,
        "plots": plot_report,
    }

    with open(output_dir / "text_vlm_layer_output_l2_rows.json", "w") as f:
        json.dump(rows, f, indent=2)
    with open(output_dir / "text_vlm_layer_output_l2_aggregate_rows.json", "w") as f:
        json.dump(aggregate, f, indent=2)
    with open(output_dir / "text_vlm_layer_output_l2_summary.json", "w") as f:
        json.dump({**summary, "rows": aggregate}, f, indent=2)
    with open(output_dir / "text_vlm_layer_output_l2_rows.csv", "w", newline="") as f:
        fieldnames = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with open(output_dir / "text_vlm_layer_output_l2_aggregate_rows.csv", "w", newline="") as f:
        fieldnames = sorted({key for row in aggregate for key in row})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(aggregate)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
