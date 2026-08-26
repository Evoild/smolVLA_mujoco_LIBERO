#!/usr/bin/env python3

"""Build SmoothQuant calibration JSON from per-channel activation ranges.

For each Linear, SmoothQuant applies:

    x_smooth = x / s
    W_smooth = W * s

where s[channel] = activation_amax[channel] ** alpha / weight_amax[channel] ** (1 - alpha).
The generated activation scale is a TensorRT-friendly per-tensor scalar for x_smooth.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch
from torch import nn

from lerobot.configs import PreTrainedConfig
from lerobot.policies.factory import get_policy_class


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", required=True)
    parser.add_argument("--activation-channel-scales-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--include-module-regex", default=r".*")
    parser.add_argument("--smooth-scale-min", type=float, default=1.0e-4)
    parser.add_argument("--smooth-scale-max", type=float, default=1.0e4)
    return parser.parse_args()


def load_policy(policy_path: str, device: str) -> nn.Module:
    cfg = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=[f"--device={device}"])
    policy_cls = get_policy_class(cfg.type)
    policy = policy_cls.from_pretrained(policy_path, config=cfg, strict=False)
    policy.eval()
    policy.to(str(policy.config.device))
    return policy


def modules_by_name(module: nn.Module) -> dict[str, nn.Module]:
    return {name: child for name, child in module.named_modules()}


def activation_amax_from_entry(entry: dict[str, Any]) -> torch.Tensor:
    if "percentile_amax" in entry and isinstance(entry["percentile_amax"], list):
        return torch.as_tensor(entry["percentile_amax"], dtype=torch.float32).clamp_min(1.0e-8)
    scale = entry.get("scale")
    if not isinstance(scale, list):
        raise ValueError(f"SmoothQuant requires per-channel activation scale list for {entry.get('module')}")
    return (torch.as_tensor(scale, dtype=torch.float32) * 127.0).clamp_min(1.0e-8)


def build_entry(
    source_entry: dict[str, Any],
    linear: nn.Linear,
    alpha: float,
    smooth_scale_min: float,
    smooth_scale_max: float,
) -> dict[str, Any]:
    act_amax = activation_amax_from_entry(source_entry).to(device="cpu", dtype=torch.float32)
    weight = linear.weight.detach().to(device="cpu", dtype=torch.float32)
    weight_amax = weight.abs().amax(dim=0).clamp_min(1.0e-8)
    if act_amax.numel() != weight_amax.numel():
        raise ValueError(
            f"activation channels {act_amax.numel()} do not match weight input channels {weight_amax.numel()} "
            f"for {source_entry.get('module')}"
        )

    smooth_scale = (act_amax.pow(alpha) / weight_amax.pow(1.0 - alpha)).clamp(
        min=smooth_scale_min,
        max=smooth_scale_max,
    )
    smoothed_act_amax = (act_amax / smooth_scale).clamp_min(1.0e-8)
    scale = float((smoothed_act_amax.max() / 127.0).item())

    return {
        "index": source_entry.get("index"),
        "module": source_entry["module"],
        "shape": source_entry.get("shape"),
        "channels": int(act_amax.numel()),
        "num_samples": source_entry.get("num_samples"),
        "scale": max(scale, 1.0e-8),
        "scale_source": f"smoothquant_alpha_{alpha:g}_from_{source_entry.get('scale_source', 'per_channel')}",
        "quantization": "smoothquant_symmetric_static_per_tensor",
        "smoothquant_alpha": alpha,
        "smooth_scale": smooth_scale.tolist(),
        "source_activation_amax_max": float(act_amax.max().item()),
        "source_activation_amax_min": float(act_amax.min().item()),
        "smoothed_activation_amax_max": float(smoothed_act_amax.max().item()),
        "smoothed_activation_amax_min": float(smoothed_act_amax.min().item()),
        "weight_input_amax_max": float(weight_amax.max().item()),
        "weight_input_amax_min": float(weight_amax.min().item()),
        "smooth_scale_max": float(smooth_scale.max().item()),
        "smooth_scale_min": float(smooth_scale.min().item()),
    }


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha must be in [0, 1]")

    with open(args.activation_channel_scales_json) as f:
        source = json.load(f)
    include_pattern = re.compile(args.include_module_regex)

    policy = load_policy(args.policy_path, args.device)
    named = modules_by_name(policy)

    rows = []
    for entry in source.get("linear_call_scales", []):
        module_name = str(entry.get("module", ""))
        if not include_pattern.search(module_name):
            continue
        module = named.get(module_name)
        if not isinstance(module, nn.Linear):
            raise KeyError(f"module is not an nn.Linear or was not found: {module_name}")
        rows.append(
            build_entry(
                entry,
                module,
                args.alpha,
                args.smooth_scale_min,
                args.smooth_scale_max,
            )
        )

    if not rows:
        raise RuntimeError("no SmoothQuant entries were generated")

    report = {
        "policy_path": args.policy_path,
        "source_activation_channel_scales_json": args.activation_channel_scales_json,
        "include_module_regex": args.include_module_regex,
        "scale_granularity": "smoothquant_per_tensor_activation_with_per_input_channel_smoothing",
        "smoothquant_formula": "s = activation_amax**alpha / weight_input_amax**(1-alpha); x'=x/s; W'=W*s",
        "smoothquant_alpha": args.alpha,
        "smooth_scale_min_clip": args.smooth_scale_min,
        "smooth_scale_max_clip": args.smooth_scale_max,
        "linear_call_scales": rows,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "linear_call_scales"}, indent=2))
    print(f"linear_call_scales: {len(rows)}")


if __name__ == "__main__":
    main()
