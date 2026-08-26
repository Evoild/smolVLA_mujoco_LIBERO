#!/usr/bin/env python3

"""Channel-wise attribution for Text VLM down_proj activation A8 error."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch import nn

from diagnose_3_4_numeric_baseline import (
    SmolVLADebugCoreWrapper,
    collect_rollout_core_inputs,
    core_inputs_from_policy,
    device_inputs,
    load_policy,
)
from linear_only_quant import load_activation_scales_by_module


DOWN_REGEX = r"^model\.vlm_with_expert\.vlm\.model\.text_model\.layers\.([0-9]+)\.mlp\.down_proj$"
EPS = 1.0e-12


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
    parser.add_argument("--compare-samples", type=int, default=20)
    parser.add_argument("--sample-stride", type=int, default=5)
    parser.add_argument("--max-parallel-tasks", type=int, default=1)
    parser.add_argument("--module-regex", default=DOWN_REGEX)
    parser.add_argument("--topk", type=int, default=20)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        if not rows:
            return
        fieldnames = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sqnr_db(signal: torch.Tensor, noise: torch.Tensor) -> float:
    signal_power = signal.detach().to(torch.float32).pow(2).mean()
    noise_power = noise.detach().to(torch.float32).pow(2).mean()
    return float((10.0 * torch.log10(signal_power.clamp_min(EPS) / noise_power.clamp_min(EPS))).item())


def cosine(ref: torch.Tensor, other: torch.Tensor) -> float:
    lhs = ref.detach().to(torch.float32).flatten()
    rhs = other.detach().to(torch.float32).flatten()
    if lhs.numel() == 0:
        return 1.0
    return float(torch.nn.functional.cosine_similarity(lhs[None], rhs[None], dim=1).item())


def rel_l2(ref: torch.Tensor, other: torch.Tensor) -> float:
    lhs = ref.detach().to(torch.float32).flatten()
    rhs = other.detach().to(torch.float32).flatten()
    return float((torch.linalg.vector_norm(rhs - lhs) / torch.linalg.vector_norm(lhs).clamp_min(EPS)).item())


def quant_dequant_symmetric_static(x: torch.Tensor, scale: float) -> torch.Tensor:
    x_fp32 = x.to(torch.float32)
    scale_tensor = torch.tensor(max(float(scale), 1.0e-8), dtype=torch.float32, device=x_fp32.device)
    q = torch.round(x_fp32 / scale_tensor).clamp(-128, 127)
    return q * scale_tensor


def clipping_ratio(x: torch.Tensor, scale: float) -> float:
    x_fp32 = x.detach().to(torch.float32)
    scale = max(float(scale), 1.0e-8)
    clipped = (x_fp32 < -128.0 * scale) | (x_fp32 > 127.0 * scale)
    return float(clipped.to(torch.float32).mean().item())


def tensor_stats(x: torch.Tensor) -> dict[str, float]:
    flat = x.detach().to(torch.float32).flatten()
    return {
        "max_abs": float(flat.abs().max().item()),
        "p99_9": float(torch.quantile(flat.abs(), 0.999).item()),
        "p99_99": float(torch.quantile(flat.abs(), 0.9999).item()),
        "p99_999": float(torch.quantile(flat.abs(), 0.99999).item()),
    }


def make_input_records(args: argparse.Namespace, policy: nn.Module, preprocessor: Any, device: str):
    input_dtype = torch.float32 if args.input_dtype == "fp32" else torch.bfloat16
    if args.input_source == "synthetic":
        inputs = core_inputs_from_policy(policy, preprocessor, args.task, args.seed, input_dtype, args.token_length)
        return [
            {
                "sample_index": 0,
                "input_source": "synthetic",
                "core_inputs": tuple(tensor.detach().cpu() for tensor in inputs),
            }
        ]
    records, _rollout_device = collect_rollout_core_inputs(args, args.compare_samples)
    return records


def register_down_hooks(
    wrapper: SmolVLADebugCoreWrapper,
    pattern: re.Pattern[str],
) -> tuple[dict[int, list[torch.Tensor]], list[Any], dict[int, str]]:
    inputs: dict[int, list[torch.Tensor]] = defaultdict(list)
    layer_modules: dict[int, str] = {}
    handles = []

    def make_pre_hook(layer: int):
        def hook(_module: nn.Module, hook_inputs: tuple[torch.Tensor, ...]) -> None:
            inputs[layer].append(hook_inputs[0].detach().to(torch.float32).cpu())

        return hook

    for name, module in wrapper.named_modules():
        match = pattern.search(name)
        if not match or not isinstance(module, nn.Linear):
            continue
        layer = int(match.group(1))
        layer_modules[layer] = name
        handles.append(module.register_forward_pre_hook(make_pre_hook(layer)))
    return inputs, handles, layer_modules


def collect_down_inputs(
    policy: nn.Module,
    records: list[dict[str, Any]],
    device: str,
    module_regex: str,
) -> tuple[dict[int, list[torch.Tensor]], dict[int, str]]:
    wrapper = SmolVLADebugCoreWrapper(policy).to(device=str(policy.config.device)).eval()
    pattern = re.compile(module_regex)
    inputs_by_layer, handles, layer_modules = register_down_hooks(wrapper, pattern)
    try:
        for record in records:
            inputs = device_inputs(record["core_inputs"], device)
            with torch.no_grad():
                wrapper(*inputs)
    finally:
        for handle in handles:
            handle.remove()
    return inputs_by_layer, layer_modules


def get_down_weights(policy: nn.Module, module_regex: str) -> dict[int, torch.Tensor]:
    pattern = re.compile(module_regex)
    result = {}
    wrapper = SmolVLADebugCoreWrapper(policy)
    for name, module in wrapper.named_modules():
        match = pattern.search(name)
        if match and isinstance(module, nn.Linear):
            result[int(match.group(1))] = module.weight.detach().to(torch.float32).cpu()
    return result


def scale_for_module(scales: dict[str, Any], module_name: str) -> float:
    if module_name not in scales:
        raise KeyError(f"missing activation scale for {module_name}")
    scale = scales[module_name]
    if isinstance(scale, dict):
        raise TypeError(f"expected symmetric scalar scale for {module_name}, got dict")
    if isinstance(scale, list):
        raise TypeError(f"expected per-tensor scalar scale for {module_name}, got per-channel list")
    return float(scale)


def process_layer(
    layer: int,
    module_name: str,
    x_values: list[torch.Tensor],
    weight: torch.Tensor,
    scale: float,
    topk: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    x = torch.cat([value.reshape(-1, value.shape[-1]) for value in x_values], dim=0).to(torch.float32)
    w = weight.to(torch.float32)
    x_q = quant_dequant_symmetric_static(x, scale)
    e = x_q - x
    y_ref = x @ w.t()
    y_a8 = x_q @ w.t()
    delta_y = y_a8 - y_ref

    input_rel = rel_l2(x, x_q)
    output_rel = rel_l2(y_ref, y_a8)
    stats = tensor_stats(x)
    layer_row = {
        "layer": layer,
        "module": module_name,
        "num_tokens": int(x.shape[0]),
        "channels": int(x.shape[1]),
        "scale": scale,
        "input_cosine": cosine(x, x_q),
        "input_absolute_l2": float(torch.linalg.vector_norm(e.flatten()).item()),
        "input_rel_l2": input_rel,
        "input_mse": float(e.pow(2).mean().item()),
        "input_mae": float(e.abs().mean().item()),
        "input_sqnr": sqnr_db(x, e),
        "clip_ratio": clipping_ratio(x, scale),
        "input_max_abs": stats["max_abs"],
        "input_p99_9": stats["p99_9"],
        "input_p99_99": stats["p99_99"],
        "input_p99_999": stats["p99_999"],
        "output_cosine": cosine(y_ref, y_a8),
        "output_absolute_l2": float(torch.linalg.vector_norm(delta_y.flatten()).item()),
        "output_rel_l2": output_rel,
        "output_mae": float(delta_y.abs().mean().item()),
        "output_max_abs_error": float(delta_y.abs().max().item()),
        "output_norm_ratio": float(
            (torch.linalg.vector_norm(y_a8.flatten()) / torch.linalg.vector_norm(y_ref.flatten()).clamp_min(EPS)).item()
        ),
        "amplification": output_rel / max(input_rel, EPS),
    }

    channel_rows = []
    weight_rows = []
    attribution_rows = []
    contribution_l2_values = []
    delta_y_norm = torch.linalg.vector_norm(delta_y.flatten())
    delta_y_norm_sq = delta_y_norm.pow(2)
    delta_y_projected_to_input = delta_y @ w
    for channel in range(x.shape[1]):
        x_j = x[:, channel]
        e_j = e[:, channel]
        w_j = w[:, channel]
        qerr_l2 = torch.linalg.vector_norm(e_j)
        x_norm = torch.linalg.vector_norm(x_j)
        contribution_l2 = qerr_l2 * torch.linalg.vector_norm(w_j)
        contribution_l2_values.append(contribution_l2)
        clip_j = ((x_j < -128.0 * scale) | (x_j > 127.0 * scale)).to(torch.float32).mean()
        channel_rows.append(
            {
                "layer": layer,
                "channel": channel,
                "max_abs": float(x_j.abs().max().item()),
                "p99_99": float(torch.quantile(x_j.abs(), 0.9999).item()),
                "std": float(x_j.std(unbiased=False).item()),
                "quant_mae": float(e_j.abs().mean().item()),
                "quant_mse": float(e_j.pow(2).mean().item()),
                "quant_rel_l2": float((qerr_l2 / x_norm.clamp_min(EPS)).item()),
                "sqnr": sqnr_db(x_j, e_j),
                "clip_ratio": float(clip_j.item()),
            }
        )
        weight_rows.append(
            {
                "layer": layer,
                "channel": channel,
                "weight_l1": float(w_j.abs().sum().item()),
                "weight_l2": float(torch.linalg.vector_norm(w_j).item()),
                "weight_rms": float(w_j.pow(2).mean().sqrt().item()),
                "weight_max_abs": float(w_j.abs().max().item()),
            }
        )

    contribution_sum = torch.stack(contribution_l2_values).sum().clamp_min(EPS)
    for channel in range(x.shape[1]):
        e_j = e[:, channel]
        contribution_l2 = contribution_l2_values[channel]
        contribution_l2_sq = contribution_l2.pow(2)
        dot_delta_delta_j = (e_j * delta_y_projected_to_input[:, channel]).sum()
        without_j_norm = (delta_y_norm_sq + contribution_l2_sq - 2.0 * dot_delta_delta_j).clamp_min(0.0).sqrt()
        attribution_rows.append(
            {
                "layer": layer,
                "channel": channel,
                "activation_quant_rel_l2": channel_rows[channel]["quant_rel_l2"],
                "activation_error_l2": float(torch.linalg.vector_norm(e_j).item()),
                "weight_col_l2": weight_rows[channel]["weight_l2"],
                "contribution_l2": float(contribution_l2.item()),
                "contribution_ratio": float((contribution_l2 / contribution_sum).item()),
                "leave_one_out_importance": float((delta_y_norm - without_j_norm).item()),
            }
        )

    channel_top = []
    for row in sorted(channel_rows, key=lambda item: item["quant_rel_l2"], reverse=True)[:topk]:
        channel_top.append({**row, "rank_by_quant_rel_l2": len(channel_top) + 1})
    return layer_row, channel_rows, weight_rows, attribution_rows + channel_top


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    policy, preprocessor, device = load_policy(args.policy_path, args.device, args.model_dtype)
    records = make_input_records(args, policy, preprocessor, device)
    inputs_by_layer, layer_modules = collect_down_inputs(policy, records, device, args.module_regex)
    weights_by_layer = get_down_weights(policy, args.module_regex)
    scales = load_activation_scales_by_module(args.activation_scales_json, include_regexes=(args.module_regex,))

    layer_rows = []
    channel_rows_all = []
    weight_rows_all = []
    attribution_rows_all = []
    activation_top_rows = []
    attribution_top_rows = []
    for layer in sorted(layer_modules):
        module_name = layer_modules[layer]
        scale = scale_for_module(scales, module_name)
        layer_row, channel_rows, weight_rows, attribution_rows = process_layer(
            layer,
            module_name,
            inputs_by_layer[layer],
            weights_by_layer[layer],
            scale,
            args.topk,
        )
        layer_rows.append(layer_row)
        channel_rows_all.extend(channel_rows)
        weight_rows_all.extend(weight_rows)
        actual_attr_rows = [row for row in attribution_rows if "rank_by_quant_rel_l2" not in row]
        attribution_rows_all.extend(actual_attr_rows)
        activation_top_rows.extend([row for row in attribution_rows if "rank_by_quant_rel_l2" in row])
        for rank, row in enumerate(sorted(actual_attr_rows, key=lambda item: item["contribution_l2"], reverse=True)[: args.topk], 1):
            attribution_top_rows.append({**row, "rank_by_contribution_l2": rank})

    layer_top = sorted(layer_rows, key=lambda item: item["amplification"], reverse=True)[:10]
    write_csv(output_dir / "down_layer_error_amplification.csv", layer_rows)
    write_csv(output_dir / "down_layer_error_amplification_top10.csv", layer_top)
    write_csv(output_dir / "down_channel_activation_error.csv", channel_rows_all)
    write_csv(output_dir / "down_channel_activation_error_top20.csv", activation_top_rows)
    write_csv(output_dir / "down_weight_channel_sensitivity.csv", weight_rows_all)
    write_csv(output_dir / "down_channel_output_attribution.csv", attribution_rows_all)
    write_csv(output_dir / "down_channel_output_attribution_top20.csv", attribution_top_rows)

    summary = {
        "policy_path": args.policy_path,
        "activation_scales_json": args.activation_scales_json,
        "task": args.task,
        "seed": args.seed,
        "input_source": args.input_source,
        "sample_count": len(records),
        "module_regex": args.module_regex,
        "num_layers": len(layer_rows),
        "top10_layers_by_amplification": layer_top,
        "output_files": {
            "layer_amplification": str(output_dir / "down_layer_error_amplification.csv"),
            "layer_amplification_top10": str(output_dir / "down_layer_error_amplification_top10.csv"),
            "channel_activation_error": str(output_dir / "down_channel_activation_error.csv"),
            "channel_activation_error_top20": str(output_dir / "down_channel_activation_error_top20.csv"),
            "weight_channel_sensitivity": str(output_dir / "down_weight_channel_sensitivity.csv"),
            "channel_output_attribution": str(output_dir / "down_channel_output_attribution.csv"),
            "channel_output_attribution_top20": str(output_dir / "down_channel_output_attribution_top20.csv"),
        },
    }
    with open(output_dir / "down_channel_attribution_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
