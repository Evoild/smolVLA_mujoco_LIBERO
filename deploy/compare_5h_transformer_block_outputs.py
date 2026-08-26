#!/usr/bin/env python3

"""Compare Text VLM transformer block-end outputs for a fake-quant policy."""

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
    collect_rollout_core_inputs,
    core_inputs_from_policy,
    device_inputs,
    load_policy,
)
from diagnose_5i_attention_error_propagation import run_prefix_trace, tensor_metrics
from linear_only_quant import load_activation_scales_by_module, replace_linear_modules


DEFAULT_QUANT_REGEX = (
    r"^model\.vlm_with_expert\.vlm\.model\.text_model\.layers\.[0-9]+"
    r"\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))$"
)
STAGES = ("attention_residual_output", "ffn_residual_output")


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
    parser.add_argument("--input-source", choices=["synthetic", "rollout"], default="rollout")
    parser.add_argument("--compare-samples", type=int, default=50)
    parser.add_argument("--sample-stride", type=int, default=5)
    parser.add_argument("--max-parallel-tasks", type=int, default=1)
    parser.add_argument("--quantize-module-regex", default=DEFAULT_QUANT_REGEX)
    return parser.parse_args()


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


def load_fake_quant_policy(args: argparse.Namespace, activation_scales: dict[str, Any]) -> nn.Module:
    policy, _, device = load_policy(args.policy_path, args.device, args.model_dtype)
    regex = re.compile(args.quantize_module_regex)
    scales = {module: scale for module, scale in activation_scales.items() if regex.search(module)}
    replaced = replace_linear_modules(
        policy,
        include_prefixes=None,
        include_regexes=(args.quantize_module_regex,),
        activation_scales_by_module=scales,
        quantization_kind="w8a8",
    )
    policy.to(device=device).eval()
    policy._num_fake_quant_linear_replaced = replaced
    return policy


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        fieldnames = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["layer"]), row["stage"])].append(row)
    out = []
    for (layer, stage), values in grouped.items():
        item: dict[str, Any] = {"layer": layer, "stage": stage, "num_samples": len(values)}
        for key in (
            "relative_l2",
            "absolute_l2",
            "cosine",
            "ref_norm",
            "quant_norm",
            "norm_ratio",
            "max_abs_error",
        ):
            item[f"{key}_mean"] = sum(float(row[key]) for row in values) / len(values)
        out.append(item)
    return sorted(out, key=lambda row: (row["layer"], STAGES.index(row["stage"])))


def write_plot(rows: list[dict[str, Any]], output_dir: Path) -> dict[str, str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return {"plot_status": f"skipped: {exc!r}"}

    paths: dict[str, str] = {}
    colors = {
        "attention_residual_output": "#2563eb",
        "ffn_residual_output": "#dc2626",
    }
    labels = {
        "attention_residual_output": "after attention residual",
        "ffn_residual_output": "block end after MLP residual",
    }
    metrics = (
        ("relative_l2_mean", "Relative L2 vs BF16", "text_vlm_block_output_relative_l2"),
        ("absolute_l2_mean", "Absolute L2 vs BF16", "text_vlm_block_output_absolute_l2"),
        ("ref_norm_mean", "BF16 output L2 norm", "text_vlm_block_output_ref_norm"),
    )
    for key, ylabel, stem in metrics:
        fig, ax = plt.subplots(figsize=(9.8, 5.2), dpi=160)
        for stage in STAGES:
            sub_rows = [row for row in rows if row["stage"] == stage]
            ax.plot(
                [row["layer"] for row in sub_rows],
                [row[key] for row in sub_rows],
                marker="o",
                linewidth=1.8,
                markersize=3.5,
                color=colors[stage],
                label=labels[stage],
            )
        ax.set_xlabel("VLM text_model layer")
        ax.set_ylabel(ylabel)
        ax.set_title(stem.replace("_", " "))
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)
        ax.legend()
        fig.tight_layout()
        png_path = output_dir / f"{stem}.png"
        svg_path = output_dir / f"{stem}.svg"
        fig.savefig(png_path)
        fig.savefig(svg_path)
        plt.close(fig)
        paths[f"{stem}_png"] = str(png_path)
        paths[f"{stem}_svg"] = str(svg_path)
    return paths


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reference, preprocessor, device = load_policy(args.policy_path, args.device, args.model_dtype)
    input_records = make_input_records(args, reference, preprocessor, device)
    activation_scales = load_activation_scales_by_module(
        args.activation_scales_json,
        include_regexes=(args.quantize_module_regex,),
    )
    quant_policy = load_fake_quant_policy(args, activation_scales)
    reference.to(device=device).eval()

    target_layers = list(range(32))
    sample_rows: list[dict[str, Any]] = []
    for record in input_records:
        inputs = device_inputs(record["core_inputs"], device)
        sample_index = int(record["sample_index"])
        with torch.no_grad():
            ref_capture = run_prefix_trace(reference, inputs, set(target_layers))
            quant_capture = run_prefix_trace(quant_policy, inputs, set(target_layers))
        for layer in target_layers:
            for stage in STAGES:
                metrics = tensor_metrics(ref_capture[layer][stage], quant_capture[layer][stage])
                sample_rows.append(
                    {
                        "sample_index": sample_index,
                        "input_source": record.get("input_source", args.input_source),
                        "layer": layer,
                        "stage": stage,
                        **metrics,
                    }
                )

    aggregate = aggregate_rows(sample_rows)
    plots = write_plot(aggregate, output_dir)
    block_rows = [row for row in aggregate if row["stage"] == "ffn_residual_output"]
    worst_block = max(block_rows, key=lambda row: row["relative_l2_mean"])
    final_block = next(row for row in block_rows if row["layer"] == 31)
    summary = {
        "policy_path": args.policy_path,
        "activation_scales_json": args.activation_scales_json,
        "seed": args.seed,
        "task": args.task,
        "device": device,
        "model_dtype": args.model_dtype,
        "input_source": args.input_source,
        "sample_count": len(input_records),
        "quantize_module_regex": args.quantize_module_regex,
        "num_linear_replaced": int(getattr(quant_policy, "_num_fake_quant_linear_replaced", -1)),
        "stages": list(STAGES),
        "worst_block_end_relative_l2": worst_block,
        "final_layer_block_end": final_block,
        "plots": plots,
    }

    with open(output_dir / "text_vlm_block_output_l2_rows.json", "w") as f:
        json.dump(sample_rows, f, indent=2)
    with open(output_dir / "text_vlm_block_output_l2_summary.json", "w") as f:
        json.dump({**summary, "rows": aggregate}, f, indent=2)
    write_csv(output_dir / "text_vlm_block_output_l2_rows.csv", sample_rows)
    write_csv(output_dir / "text_vlm_block_output_l2_aggregate_rows.csv", aggregate)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
