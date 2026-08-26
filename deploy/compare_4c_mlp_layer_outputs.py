#!/usr/bin/env python3

"""Compare Step 4C fake-quant VLM MLP layer outputs against native PyTorch."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import torch
from torch import nn

from diagnose_3_4_numeric_baseline import (
    SmolVLADebugCoreWrapper,
    compare_tensors,
    core_inputs_from_policy,
    load_policy,
)
from linear_only_quant import load_activation_scales_by_module, replace_linear_modules


DEFAULT_REGEX = (
    r"^model\.vlm_with_expert\.vlm\.model\.text_model\.layers\.([0-9]+)"
    r"\.mlp\.(gate_proj|up_proj|down_proj)$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", default="smolvla_libero")
    parser.add_argument("--activation-scales-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--task", default="libero_goal step 4C MLP layer output diagnosis")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--token-length", type=int, default=48)
    parser.add_argument("--input-dtype", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--module-regex", default=DEFAULT_REGEX)
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
        sublayer = match.group(2)
        captures[name] = []
        metadata[name] = {"layer": layer, "sublayer": sublayer, "module": name}
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


def write_plot(rows: list[dict[str, Any]], output_dir: Path) -> dict[str, str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional local dependency
        return {"plot_status": f"skipped: {exc!r}"}

    colors = {"gate_proj": "#2563eb", "up_proj": "#16a34a", "down_proj": "#dc2626"}
    fig, ax = plt.subplots(figsize=(9.5, 5.2), dpi=160)
    for sublayer in ("gate_proj", "up_proj", "down_proj"):
        sub_rows = sorted((row for row in rows if row["sublayer"] == sublayer), key=lambda row: row["layer"])
        ax.plot(
            [row["layer"] for row in sub_rows],
            [row["relative_l2_error"] for row in sub_rows],
            marker="o",
            linewidth=1.8,
            markersize=3.5,
            label=sublayer,
            color=colors[sublayer],
        )
    ax.set_xlabel("VLM text_model layer")
    ax.set_ylabel("Output relative L2 error vs native")
    ax.set_title("Step 4C VLM MLP Linear Output Relative L2")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)
    ax.legend()
    fig.tight_layout()

    png_path = output_dir / "mlp_layer_output_relative_l2.png"
    svg_path = output_dir / "mlp_layer_output_relative_l2.svg"
    fig.savefig(png_path)
    fig.savefig(svg_path)
    plt.close(fig)

    per_sublayer_paths = {}
    for sublayer in ("gate_proj", "up_proj", "down_proj"):
        fig, ax = plt.subplots(figsize=(8.0, 4.4), dpi=160)
        sub_rows = sorted((row for row in rows if row["sublayer"] == sublayer), key=lambda row: row["layer"])
        ax.plot(
            [row["layer"] for row in sub_rows],
            [row["relative_l2_error"] for row in sub_rows],
            marker="o",
            linewidth=1.8,
            markersize=3.5,
            color=colors[sublayer],
        )
        ax.set_xlabel("VLM text_model layer")
        ax.set_ylabel("Output relative L2 error vs native")
        ax.set_title(f"Step 4C {sublayer} Output Relative L2")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)
        fig.tight_layout()
        path = output_dir / f"{sublayer}_output_relative_l2.png"
        fig.savefig(path)
        plt.close(fig)
        per_sublayer_paths[f"{sublayer}_plot"] = str(path)

    return {
        "combined_png": str(png_path),
        "combined_svg": str(svg_path),
        **per_sublayer_paths,
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_dtype = torch.float32 if args.input_dtype == "fp32" else torch.bfloat16
    pattern = re.compile(args.module_regex)

    original, preprocessor, device = load_policy(args.policy_path, args.device)
    inputs = core_inputs_from_policy(original, preprocessor, args.task, args.seed, input_dtype, args.token_length)
    baseline_captures, metadata = run_with_hooks(original, inputs, pattern)

    fake_quant, _, _ = load_policy(args.policy_path, args.device)
    activation_scales = load_activation_scales_by_module(args.activation_scales_json, (args.module_regex,))
    replaced = replace_linear_modules(
        fake_quant,
        include_prefixes=None,
        include_regexes=(args.module_regex,),
        activation_scales_by_module=activation_scales,
        quantization_kind="w8a8",
    )
    fake_quant.to(device=device).eval()
    quant_captures, quant_metadata = run_with_hooks(fake_quant, inputs, pattern)

    if set(metadata) != set(quant_metadata):
        missing_lhs = sorted(set(quant_metadata) - set(metadata))
        missing_rhs = sorted(set(metadata) - set(quant_metadata))
        raise RuntimeError(f"hook target mismatch: only_quant={missing_lhs}, only_native={missing_rhs}")

    rows: list[dict[str, Any]] = []
    for module_name in sorted(metadata, key=lambda name: (metadata[name]["layer"], metadata[name]["sublayer"])):
        lhs = concat_captures(baseline_captures[module_name])
        rhs = concat_captures(quant_captures[module_name])
        row = compare_tensors("A_vs_E_layer_output", module_name, lhs, rhs)
        row.update(metadata[module_name])
        row["native_num_calls"] = len(baseline_captures[module_name])
        row["fake_quant_num_calls"] = len(quant_captures[module_name])
        rows.append(row)

    summary_by_sublayer = {}
    for sublayer in ("gate_proj", "up_proj", "down_proj"):
        sub_rows = [row for row in rows if row["sublayer"] == sublayer]
        values = torch.tensor([row["relative_l2_error"] for row in sub_rows], dtype=torch.float32)
        worst = max(sub_rows, key=lambda row: row["relative_l2_error"]) if sub_rows else None
        summary_by_sublayer[sublayer] = {
            "num_layers": len(sub_rows),
            "mean_relative_l2_error": float(values.mean()) if len(values) else None,
            "max_relative_l2_error": float(values.max()) if len(values) else None,
            "worst_layer": worst["layer"] if worst else None,
            "worst_module": worst["module"] if worst else None,
        }

    plot_report = write_plot(rows, output_dir)
    summary = {
        "policy_path": args.policy_path,
        "activation_scales_json": args.activation_scales_json,
        "seed": args.seed,
        "task": args.task,
        "device": device,
        "module_regex": args.module_regex,
        "num_linear_replaced": replaced,
        "num_modules_compared": len(rows),
        "summary_by_sublayer": summary_by_sublayer,
        "plots": plot_report,
        "rows": rows,
    }

    with open(output_dir / "mlp_layer_output_l2_rows.json", "w") as f:
        json.dump(rows, f, indent=2)
    with open(output_dir / "mlp_layer_output_l2_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(output_dir / "mlp_layer_output_l2_rows.csv", "w", newline="") as f:
        fieldnames = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    printable = {key: value for key, value in summary.items() if key != "rows"}
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()
