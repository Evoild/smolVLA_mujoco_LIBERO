#!/usr/bin/env python3

"""Check clipping/resolution risk for calibrated static activation scales."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import torch
from torch import nn

from diagnose_3_4_numeric_baseline import SmolVLADebugCoreWrapper, core_inputs_from_policy, load_policy
from linear_only_quant import load_activation_scales_by_module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", default="smolvla_libero")
    parser.add_argument("--activation-scales-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--task", default="libero_goal step 3-6C clipping ratio")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--token-length", type=int, default=48)
    parser.add_argument("--input-dtype", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument(
        "--module-regex",
        action="append",
        default=None,
        help="Only collect nn.Linear modules whose full module name matches this regex.",
    )
    args = parser.parse_args()
    if args.module_regex is None:
        args.module_regex = [
            r"^model\.vlm_with_expert\.vlm\.model\.text_model\.layers\.[0-9]+\.mlp\.(gate_proj|up_proj|down_proj)$"
        ]
    return args


def percentile(abs_values: torch.Tensor, q: float) -> float:
    if abs_values.numel() == 0:
        return float("nan")
    return float(torch.quantile(abs_values, q).item())


class StaticScaleClippingCollector:
    def __init__(self, scales_by_module: dict[str, float], module_regexes: tuple[str, ...]):
        self.scales_by_module = scales_by_module
        self.patterns = tuple(re.compile(pattern) for pattern in module_regexes)
        self.handles: list[Any] = []
        self.rows_by_module: dict[str, dict[str, Any]] = {}

    def should_collect(self, module_name: str) -> bool:
        return any(pattern.search(module_name) for pattern in self.patterns)

    def hook_modules(self, module: nn.Module) -> None:
        for name, child in module.named_modules():
            if isinstance(child, nn.Linear) and self.should_collect(name):
                if name not in self.scales_by_module:
                    raise KeyError(f"missing activation scale for collected module: {name}")
                self.handles.append(child.register_forward_pre_hook(self._make_hook(name)))

    def _make_hook(self, name: str):
        def hook(_module, inputs):
            x = inputs[0].detach().to(torch.float32).flatten().cpu()
            scale = float(self.scales_by_module[name])
            pos_threshold = 127.0 * scale
            neg_threshold = -128.0 * scale
            abs_threshold = pos_threshold
            if x.numel() == 0:
                return

            abs_x = x.abs()
            clipped_pos = x > pos_threshold
            clipped_neg = x < neg_threshold
            clipped = clipped_pos | clipped_neg
            q = torch.round(x / scale).clamp(-128, 127)
            nonzero = x != 0

            row = self.rows_by_module.setdefault(
                name,
                {
                    "module": name,
                    "scale": scale,
                    "threshold": abs_threshold,
                    "num_calls": 0,
                    "numel": 0,
                    "num_clipped": 0,
                    "num_clipped_pos": 0,
                    "num_clipped_neg": 0,
                    "amax": 0.0,
                    "max_over_threshold": 0.0,
                    "sum_abs": 0.0,
                    "sum_abs_q_error": 0.0,
                    "sum_nonzero": 0,
                    "unique_q_values": set(),
                    "p50_values": [],
                    "p90_values": [],
                    "p99_values": [],
                    "p999_values": [],
                    "p9999_values": [],
                },
            )
            numel = int(x.numel())
            num_clipped = int(clipped.sum().item())
            row["num_calls"] += 1
            row["numel"] += numel
            row["num_clipped"] += num_clipped
            row["num_clipped_pos"] += int(clipped_pos.sum().item())
            row["num_clipped_neg"] += int(clipped_neg.sum().item())
            amax = float(abs_x.max().item())
            row["amax"] = max(float(row["amax"]), amax)
            row["max_over_threshold"] = max(float(row["max_over_threshold"]), amax / max(abs_threshold, 1.0e-12))
            row["sum_abs"] += float(abs_x.sum().item())
            row["sum_abs_q_error"] += float((x - q * scale).abs().sum().item())
            row["sum_nonzero"] += int(nonzero.sum().item())
            row["unique_q_values"].update(int(value) for value in torch.unique(q).tolist())
            row["p50_values"].append(percentile(abs_x, 0.5))
            row["p90_values"].append(percentile(abs_x, 0.9))
            row["p99_values"].append(percentile(abs_x, 0.99))
            row["p999_values"].append(percentile(abs_x, 0.999))
            row["p9999_values"].append(percentile(abs_x, 0.9999))

        return hook

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()

    def rows(self) -> list[dict[str, Any]]:
        rows = []
        for row in self.rows_by_module.values():
            numel = max(int(row["numel"]), 1)
            sum_abs = max(float(row["sum_abs"]), 1.0e-12)
            output = {
                "module": row["module"],
                "scale": row["scale"],
                "threshold": row["threshold"],
                "num_calls": row["num_calls"],
                "numel": row["numel"],
                "num_clipped": row["num_clipped"],
                "clipping_ratio": float(row["num_clipped"]) / numel,
                "clipping_pos_ratio": float(row["num_clipped_pos"]) / numel,
                "clipping_neg_ratio": float(row["num_clipped_neg"]) / numel,
                "amax": row["amax"],
                "max_over_threshold": row["max_over_threshold"],
                "p50_abs": max(row["p50_values"]),
                "p90_abs": max(row["p90_values"]),
                "p99_abs": max(row["p99_values"]),
                "p999_abs": max(row["p999_values"]),
                "p9999_abs": max(row["p9999_values"]),
                "p99_over_threshold": max(row["p99_values"]) / max(float(row["threshold"]), 1.0e-12),
                "mean_abs": sum_abs / numel,
                "mean_abs_quant_error": float(row["sum_abs_q_error"]) / numel,
                "relative_mean_abs_quant_error": float(row["sum_abs_q_error"]) / sum_abs,
                "unique_q_values": len(row["unique_q_values"]),
                "mean_abs_over_scale": (sum_abs / numel) / max(float(row["scale"]), 1.0e-12),
            }
            rows.append(output)
        return sorted(rows, key=lambda item: item["clipping_ratio"], reverse=True)


def write_reports(rows: list[dict[str, Any]], args: argparse.Namespace, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_json = output_dir / "clipping_rows.json"
    rows_csv = output_dir / "clipping_rows.csv"
    summary_json = output_dir / "clipping_summary.json"

    with open(rows_json, "w") as f:
        json.dump(rows, f, indent=2)
    with open(rows_csv, "w", newline="") as f:
        fieldnames = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "policy_path": args.policy_path,
        "activation_scales_json": args.activation_scales_json,
        "seed": args.seed,
        "num_seeds": args.num_seeds,
        "task": args.task,
        "module_regex": args.module_regex,
        "num_modules": len(rows),
        "max_clipping_ratio": max((row["clipping_ratio"] for row in rows), default=0.0),
        "mean_clipping_ratio": sum(row["clipping_ratio"] for row in rows) / max(len(rows), 1),
        "top_clipping_ratio": rows[:20],
        "top_max_over_threshold": sorted(rows, key=lambda item: item["max_over_threshold"], reverse=True)[:20],
        "top_relative_mean_abs_quant_error": sorted(
            rows, key=lambda item: item["relative_mean_abs_quant_error"], reverse=True
        )[:20],
    }
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if not k.startswith("top_")}, indent=2))
    print("top clipping_ratio:")
    for row in summary["top_clipping_ratio"][:10]:
        print(
            f"{row['clipping_ratio']:.8f} "
            f"max/threshold={row['max_over_threshold']:.3f} "
            f"rel_qerr={row['relative_mean_abs_quant_error']:.6f} "
            f"{row['module']}"
        )


def main() -> None:
    args = parse_args()
    input_dtype = torch.float32 if args.input_dtype == "fp32" else torch.bfloat16
    policy, preprocessor, device = load_policy(args.policy_path, args.device)
    wrapper = SmolVLADebugCoreWrapper(policy).to(device=device).eval()
    scales_by_module = load_activation_scales_by_module(args.activation_scales_json, tuple(args.module_regex))
    collector = StaticScaleClippingCollector(scales_by_module, tuple(args.module_regex))
    collector.hook_modules(wrapper)
    try:
        with torch.no_grad():
            for offset in range(args.num_seeds):
                inputs = core_inputs_from_policy(
                    policy,
                    preprocessor,
                    args.task,
                    args.seed + offset,
                    input_dtype,
                    args.token_length,
                )
                wrapper(*inputs)
    finally:
        collector.close()
    write_reports(collector.rows(), args, Path(args.output_dir))


if __name__ == "__main__":
    main()
