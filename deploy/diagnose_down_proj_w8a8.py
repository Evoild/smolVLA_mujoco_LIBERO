#!/usr/bin/env python3

"""Diagnose VLM text_model MLP down_proj W8A8 sensitivity.

This is a PyTorch-only PTQ diagnostic. It reuses the existing SmolVLA debug
sample_actions core and fake-dequant Linear wrappers, and writes CSV/JSON files
that explain where down_proj W8A8 error enters and how it propagates through
the denoising steps and final action chunk.
"""

from __future__ import annotations

import argparse
import csv
import gc
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
    compare_tensors,
    core_inputs_from_policy,
    load_policy,
    run_pytorch,
)
from linear_only_quant import (
    W8A8Linear,
    load_activation_scales_by_module,
    quantize_weight_per_out_channel,
    replace_linear_modules,
)


DOWN_REGEX = r"^model\.vlm_with_expert\.vlm\.model\.text_model\.layers\.[0-9]+\.mlp\.down_proj$"
MLP_REGEX = r"^model\.vlm_with_expert\.vlm\.model\.text_model\.layers\.[0-9]+\.mlp$"
LAYER_REGEX = r"^model\.vlm_with_expert\.vlm\.model\.text_model\.layers\.[0-9]+$"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", default="smolvla_libero")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--task", default="libero_spatial step 4A down_proj diagnosis")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--token-length", type=int, default=48)
    parser.add_argument("--input-dtype", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--activation-scales-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--down-module-regex", default=DOWN_REGEX)
    parser.add_argument("--range-percentiles", default="99.9,99.95,99.99,99.995,99.999,max")
    parser.add_argument("--near-zero-threshold", type=float, default=1.0e-6)
    parser.add_argument("--sign-flip-threshold", type=float, default=1.0e-3)
    parser.add_argument("--topk-channels", type=int, default=10)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def tensor_stats(prefix: str, x: torch.Tensor) -> dict[str, Any]:
    x = x.detach().to(torch.float32).cpu()
    flat = x.flatten()
    abs_flat = flat.abs()
    return {
        f"{prefix}_shape": list(x.shape),
        f"{prefix}_numel": int(flat.numel()),
        f"{prefix}_mean": float(flat.mean()),
        f"{prefix}_std": float(flat.std(unbiased=False)),
        f"{prefix}_min": float(flat.min()),
        f"{prefix}_max": float(flat.max()),
        f"{prefix}_max_abs": float(abs_flat.max()),
        f"{prefix}_abs_p99": float(torch.quantile(abs_flat, 0.99)),
        f"{prefix}_abs_p99_9": float(torch.quantile(abs_flat, 0.999)),
        f"{prefix}_abs_p99_99": float(torch.quantile(abs_flat, 0.9999)),
        f"{prefix}_abs_p99_999": float(torch.quantile(abs_flat, 0.99999)),
        f"{prefix}_zero_rate": float((flat == 0).to(torch.float32).mean()),
    }


def vector_metrics(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    row = compare_tensors("tmp", "tmp", a, b)
    diff = a.detach().to(torch.float32).flatten() - b.detach().to(torch.float32).flatten()
    return {
        "cosine_similarity": row["cosine_similarity"],
        "relative_l2_error": row["relative_l2_error"],
        "l2_norm_ratio": row["l2_norm_ratio"],
        "std_ratio": row["std_ratio"],
        "MAE": row["mean_abs_error"],
        "RMSE": float(torch.sqrt(torch.mean(diff * diff))),
        "max_abs_error": row["max_abs_error"],
    }


def qdq_static(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    x_fp32 = x.detach().to(torch.float32)
    scale = scale.to(device=x_fp32.device, dtype=torch.float32).clamp_min(1.0e-8)
    q = torch.round(x_fp32 / scale).clamp(-128, 127).to(torch.int8)
    return q.to(torch.float32) * scale


def scale_for_module(scales_by_module: dict[str, Any], name: str, device: torch.device) -> torch.Tensor:
    if name not in scales_by_module:
        raise KeyError(f"missing activation scale for {name}")
    return torch.as_tensor(scales_by_module[name], dtype=torch.float32, device=device)


class HookCapture:
    def __init__(self, down_regex: str):
        self.down_pattern = re.compile(down_regex)
        self.mlp_pattern = re.compile(MLP_REGEX)
        self.layer_pattern = re.compile(LAYER_REGEX)
        self.down_inputs: dict[str, torch.Tensor] = {}
        self.down_outputs: dict[str, torch.Tensor] = {}
        self.mlp_outputs: dict[str, torch.Tensor] = {}
        self.layer_outputs: dict[str, torch.Tensor] = {}
        self.handles: list[Any] = []

    def install(self, module: nn.Module) -> None:
        for name, child in module.named_modules():
            if self.down_pattern.search(name):
                self.handles.append(child.register_forward_pre_hook(self._down_pre(name)))
                self.handles.append(child.register_forward_hook(self._down_post(name)))
            elif self.mlp_pattern.search(name):
                self.handles.append(child.register_forward_hook(self._output_post(self.mlp_outputs, name)))
            elif self.layer_pattern.search(name):
                self.handles.append(child.register_forward_hook(self._output_post(self.layer_outputs, name)))

    def _down_pre(self, name: str):
        def hook(_module, inputs):
            self.down_inputs[name] = inputs[0].detach().to(torch.float32).cpu()

        return hook

    def _down_post(self, name: str):
        def hook(_module, _inputs, output):
            self.down_outputs[name] = tensor_first(output).detach().to(torch.float32).cpu()

        return hook

    def _output_post(self, store: dict[str, torch.Tensor], name: str):
        def hook(_module, _inputs, output):
            store[name] = tensor_first(output).detach().to(torch.float32).cpu()

        return hook

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def tensor_first(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)):
        for item in output:
            if torch.is_tensor(item):
                return item
    raise TypeError(f"cannot extract tensor from output type {type(output)!r}")


def run_with_hooks(policy: nn.Module, inputs: tuple[torch.Tensor, ...], down_regex: str):
    wrapper = SmolVLADebugCoreWrapper(policy).to(device=str(policy.config.device)).eval()
    capture = HookCapture(down_regex)
    capture.install(wrapper)
    try:
        with torch.no_grad():
            outputs = wrapper(*inputs)
        output_dict = {
            name: tensor.detach().to(torch.float32).cpu()
            for name, tensor in zip(wrapper.output_names, outputs, strict=True)
        }
    finally:
        capture.close()
    return output_dict, capture


def channel_view(x: torch.Tensor) -> torch.Tensor:
    x = x.detach().to(torch.float32)
    return x.reshape(-1, x.shape[-1])


def distribution_rows(
    down_inputs: dict[str, torch.Tensor],
    near_zero_threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    dist_rows = []
    channel_rows = []
    for module, x in sorted(down_inputs.items()):
        flat = x.flatten()
        abs_flat = flat.abs()
        view = channel_view(x)
        channel_abs = view.abs()
        channel_max = channel_abs.amax(dim=0).clamp_min(1.0e-12)
        channel_p9999 = torch.quantile(channel_abs, 0.9999, dim=0).clamp_min(1.0e-12)
        channel_mean_abs = channel_abs.mean(dim=0)
        channel_std = view.std(dim=0, unbiased=False)
        row = tensor_stats("x_down", x)
        row.update(
            {
                "module": module,
                "near_zero_threshold": near_zero_threshold,
                "x_down_near_zero_rate": float((abs_flat < near_zero_threshold).to(torch.float32).mean()),
                "channel_max_abs_max_over_median": float(channel_max.max() / channel_max.median()),
                "channel_p99_99_max_over_median": float(channel_p9999.max() / channel_p9999.median()),
            }
        )
        dist_rows.append(row)
        for channel in range(view.shape[-1]):
            channel_rows.append(
                {
                    "module": module,
                    "channel": channel,
                    "channel_max_abs": float(channel_max[channel]),
                    "channel_p99_99": float(channel_p9999[channel]),
                    "channel_mean_abs": float(channel_mean_abs[channel]),
                    "channel_std": float(channel_std[channel]),
                }
            )
    return dist_rows, channel_rows


def clipping_and_input_error_rows(
    down_inputs: dict[str, torch.Tensor],
    scales_by_module: dict[str, Any],
    topk: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    clipping_rows = []
    input_error_rows = []
    top_channel_rows = []
    for module, x in sorted(down_inputs.items()):
        x_device = x.to(torch.float32)
        scale = scale_for_module(scales_by_module, module, x_device.device)
        qrange_pos = scale * 127.0
        qrange_neg = scale * 128.0
        xq = qdq_static(x_device, scale).cpu()
        metrics = vector_metrics(x, xq)
        view = channel_view(x)
        qview = channel_view(xq)
        abs_view = view.abs()
        scale_cpu = scale.detach().to(torch.float32).cpu()
        if scale_cpu.ndim == 0:
            qrange_pos_view = torch.full((view.shape[-1],), float(scale_cpu * 127.0))
            qrange_neg_view = torch.full((view.shape[-1],), float(scale_cpu * 128.0))
        else:
            qrange_pos_view = scale_cpu * 127.0
            qrange_neg_view = scale_cpu * 128.0
        clipped_pos = view > qrange_pos_view[None, :]
        clipped_neg = view < -qrange_neg_view[None, :]
        clipped = clipped_pos | clipped_neg
        per_channel_clipping = clipped.to(torch.float32).mean(dim=0)
        channel_rel_l2 = torch.linalg.vector_norm(view - qview, dim=0) / torch.linalg.vector_norm(view, dim=0).clamp_min(1e-12)
        channel_mae = (view - qview).abs().mean(dim=0)
        channel_cos = []
        for idx in range(view.shape[-1]):
            denom = torch.linalg.vector_norm(view[:, idx]) * torch.linalg.vector_norm(qview[:, idx])
            if denom.item() == 0.0:
                channel_cos.append(float("nan"))
            else:
                channel_cos.append(float(torch.dot(view[:, idx], qview[:, idx]) / denom))
        clipping_rows.append(
            {
                "module": module,
                "scale_kind": "per_channel" if scale_cpu.ndim > 0 else "per_tensor",
                "mean_scale": float(scale_cpu.mean()),
                "max_scale": float(scale_cpu.max()),
                "mean_quant_range": float(qrange_pos_view.mean()),
                "max_quant_range": float(qrange_pos_view.max()),
                "clipped_elements": int(clipped.sum()),
                "clipping_rate": float(clipped.to(torch.float32).mean()),
                "positive_clipping_rate": float(clipped_pos.to(torch.float32).mean()),
                "negative_clipping_rate": float(clipped_neg.to(torch.float32).mean()),
                "mean_channel_clipping_rate": float(per_channel_clipping.mean()),
                "max_channel_clipping_rate": float(per_channel_clipping.max()),
                "p95_channel_clipping_rate": float(torch.quantile(per_channel_clipping, 0.95)),
            }
        )
        input_error_rows.append({"module": module, **metrics})
        worst = torch.topk(channel_rel_l2, k=min(topk, channel_rel_l2.numel())).indices
        for channel in worst.tolist():
            top_channel_rows.append(
                {
                    "module": module,
                    "channel_id": int(channel),
                    "relative_l2": float(channel_rel_l2[channel]),
                    "MAE": float(channel_mae[channel]),
                    "cosine_similarity": channel_cos[channel],
                    "FP_max_abs": float(abs_view[:, channel].max()),
                    "FP_std": float(view[:, channel].std(unbiased=False)),
                    "quant_range": float(qrange_pos_view[channel]),
                    "clipping_rate": float(per_channel_clipping[channel]),
                }
            )
    return clipping_rows, input_error_rows, top_channel_rows


def output_error_rows(
    fp_capture: HookCapture,
    w8a16_capture: HookCapture,
    w8a8_capture: HookCapture,
    input_error_by_module: dict[str, float],
) -> list[dict[str, Any]]:
    rows = []
    for module, y_fp in sorted(fp_capture.down_outputs.items()):
        for variant, capture in [("W8A16", w8a16_capture), ("W8A8", w8a8_capture)]:
            if module not in capture.down_outputs:
                continue
            metrics = vector_metrics(y_fp, capture.down_outputs[module])
            input_l2 = input_error_by_module.get(module, float("nan"))
            amp = metrics["relative_l2_error"] / input_l2 if input_l2 and input_l2 > 1.0e-12 else float("nan")
            rows.append({"module": module, "variant": variant, **metrics, "error_amplification_ratio": amp})
    return rows


def residual_rows(fp_capture: HookCapture, w8a8_capture: HookCapture) -> list[dict[str, Any]]:
    rows = []
    for module, mlp_fp in sorted(fp_capture.mlp_outputs.items()):
        if module not in w8a8_capture.mlp_outputs:
            continue
        layer_module = module.rsplit(".mlp", 1)[0]
        before = vector_metrics(mlp_fp, w8a8_capture.mlp_outputs[module])
        row = {"module": module, "stage": "mlp_out_before_residual", **before}
        rows.append(row)
        if layer_module in fp_capture.layer_outputs and layer_module in w8a8_capture.layer_outputs:
            after = vector_metrics(fp_capture.layer_outputs[layer_module], w8a8_capture.layer_outputs[layer_module])
            ratio = after["relative_l2_error"] / before["relative_l2_error"] if before["relative_l2_error"] > 1e-12 else float("nan")
            rows.append(
                {
                    "module": module,
                    "stage": "hidden_after_residual",
                    **after,
                    "residual_error_ratio": ratio,
                }
            )
    return rows


def flow_rows(fp_outputs: dict[str, torch.Tensor], quant_outputs: dict[str, torch.Tensor]) -> list[dict[str, Any]]:
    rows = []
    for step in range(10):
        v_name = f"v_t_step_{step:02d}"
        x_name = f"x_t_step_{step:02d}"
        if v_name not in fp_outputs or x_name not in fp_outputs:
            continue
        v = vector_metrics(fp_outputs[v_name], quant_outputs[v_name])
        x = vector_metrics(fp_outputs[x_name], quant_outputs[x_name])
        rows.append(
            {
                "step": step,
                "v_t_cosine": v["cosine_similarity"],
                "v_t_relative_l2_error": v["relative_l2_error"],
                "v_t_l2_norm_ratio": v["l2_norm_ratio"],
                "v_t_std_ratio": v["std_ratio"],
                "x_t_cosine": x["cosine_similarity"],
                "x_t_relative_l2_error": x["relative_l2_error"],
                "x_t_l2_norm_ratio": x["l2_norm_ratio"],
                "x_t_std_ratio": x["std_ratio"],
            }
        )
    return rows


def action_rows(
    fp_action: torch.Tensor,
    quant_action: torch.Tensor,
    sign_flip_threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    fp = fp_action.to(torch.float32)
    qt = quant_action.to(torch.float32)
    timestep_rows = []
    for timestep in range(fp.shape[1]):
        metrics = vector_metrics(fp[:, timestep, :], qt[:, timestep, :])
        timestep_rows.append({"timestep": timestep, **metrics})
    dim_rows = []
    for dim in range(fp.shape[-1]):
        metrics = vector_metrics(fp[:, :, dim], qt[:, :, dim])
        dim_rows.append({"action_dim": dim, **metrics})
    meaningful = fp.abs() > sign_flip_threshold
    sign_flip = (fp.sign() != qt.sign()) & meaningful
    timestep_l2 = torch.tensor([row["relative_l2_error"] for row in timestep_rows], dtype=torch.float32)
    summary = {
        "mean_timestep_relative_l2": float(timestep_l2.mean()),
        "max_timestep_relative_l2": float(timestep_l2.max()),
        "p95_timestep_relative_l2": float(torch.quantile(timestep_l2, 0.95)),
        "worst_timestep_index": int(timestep_l2.argmax()),
        "worst_action_dim": int(max(dim_rows, key=lambda row: row["relative_l2_error"])["action_dim"]),
        "sign_flip_threshold": sign_flip_threshold,
        "meaningful_sign_flip_rate": float(sign_flip.to(torch.float32).sum() / meaningful.to(torch.float32).sum().clamp_min(1.0)),
        "gripper_analysis": "not_reported_no_configured_gripper_dimension",
    }
    return timestep_rows, dim_rows, summary


def percentile_scales_from_inputs(down_inputs: dict[str, torch.Tensor], percentile: str) -> dict[str, Any]:
    scales = {}
    for module, x in down_inputs.items():
        view = channel_view(x)
        abs_view = view.abs()
        if percentile == "max":
            amax = abs_view.amax(dim=0)
        else:
            q = float(percentile) / 100.0
            amax = torch.quantile(abs_view, q, dim=0)
        scales[module] = (amax.clamp_min(1e-8) / 127.0).tolist()
    return scales


def load_down_quant_policy(
    policy_path: str,
    device: str,
    down_regex: str,
    kind: str,
    scales: dict[str, Any] | None = None,
) -> nn.Module:
    policy, _, _ = load_policy(policy_path, device)
    replace_linear_modules(
        policy,
        include_prefixes=None,
        include_regexes=(down_regex,),
        activation_scales_by_module=scales,
        quantization_kind=kind,
    )
    return policy.to(device=str(policy.config.device)).eval()


def aggregate(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if key in row and math.isfinite(float(row[key]))]
    return float(sum(values) / len(values)) if values else float("nan")


def range_sensitivity(
    args: argparse.Namespace,
    fp_outputs: dict[str, torch.Tensor],
    fp_down_inputs: dict[str, torch.Tensor],
    fp_down_outputs: dict[str, torch.Tensor],
    inputs: tuple[torch.Tensor, ...],
) -> list[dict[str, Any]]:
    rows = []
    ranges = [item.strip() for item in args.range_percentiles.split(",") if item.strip()]
    for item in ranges:
        percentile = "max" if item.lower() == "max" else item
        scales = percentile_scales_from_inputs(fp_down_inputs, percentile)
        policy = load_down_quant_policy(args.policy_path, args.device, args.down_module_regex, "w8a8", scales)
        outputs, capture = run_with_hooks(policy, inputs, args.down_module_regex)
        clipping, input_errors, _ = clipping_and_input_error_rows(fp_down_inputs, scales, topk=1)
        out_errors = []
        for module, y_fp in fp_down_outputs.items():
            if module in capture.down_outputs:
                out_errors.append({"module": module, **vector_metrics(y_fp, capture.down_outputs[module])})
        action = vector_metrics(fp_outputs["action_chunk"], outputs["action_chunk"])
        vt9 = vector_metrics(fp_outputs["v_t_step_09"], outputs["v_t_step_09"])
        rows.append(
            {
                "range": percentile,
                "mean_clipping_rate": aggregate(clipping, "clipping_rate"),
                "max_channel_clipping_rate": max(float(row["max_channel_clipping_rate"]) for row in clipping),
                "down_input_relative_l2": aggregate(input_errors, "relative_l2_error"),
                "down_output_relative_l2": aggregate(out_errors, "relative_l2_error"),
                "v_t_step_09_relative_l2": vt9["relative_l2_error"],
                "action_chunk_relative_l2": action["relative_l2_error"],
            }
        )
        del policy, outputs, capture
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return rows


def infer_summary(
    summary: dict[str, Any],
    input_error_rows: list[dict[str, Any]],
    output_rows: list[dict[str, Any]],
    clipping_rows: list[dict[str, Any]],
    residual: list[dict[str, Any]],
    flow: list[dict[str, Any]],
    range_rows: list[dict[str, Any]],
) -> None:
    w8a16_l2 = aggregate([row for row in output_rows if row["variant"] == "W8A16"], "relative_l2_error")
    w8a8_l2 = aggregate([row for row in output_rows if row["variant"] == "W8A8"], "relative_l2_error")
    input_l2 = aggregate(input_error_rows, "relative_l2_error")
    clipping = aggregate(clipping_rows, "clipping_rate")
    max_ch_clip = max(float(row["max_channel_clipping_rate"]) for row in clipping_rows)
    amp = aggregate([row for row in output_rows if row["variant"] == "W8A8"], "error_amplification_ratio")
    residual_after = [row for row in residual if row["stage"] == "hidden_after_residual"]
    residual_ratio = aggregate(residual_after, "residual_error_ratio")
    first_flow = flow[0]["x_t_relative_l2_error"] if flow else float("nan")
    last_flow = flow[-1]["x_t_relative_l2_error"] if flow else float("nan")
    best_range = min(range_rows, key=lambda row: row["action_chunk_relative_l2"]) if range_rows else None

    if w8a16_l2 < w8a8_l2 * 0.25:
        weight_source = "activation_A8_dominant"
    elif w8a16_l2 < w8a8_l2 * 0.75:
        weight_source = "activation_A8_larger_than_weight_W8"
    else:
        weight_source = "weight_W8_and_activation_A8_both_material"

    if clipping > 0.01 or max_ch_clip > 0.05:
        activation_cause = "clipping_or_channel_clipping_material"
    elif input_l2 > 0.05:
        activation_cause = "quantization_resolution_or_channel_range_imbalance_likely"
    else:
        activation_cause = "activation_A8_input_error_small_for_this_sample"

    summary["automatic_diagnosis"] = {
        "weight_int8_role": weight_source,
        "mean_down_input_relative_l2": input_l2,
        "mean_down_output_w8a16_relative_l2": w8a16_l2,
        "mean_down_output_w8a8_relative_l2": w8a8_l2,
        "activation_error_hypothesis": activation_cause,
        "mean_clipping_rate": clipping,
        "max_channel_clipping_rate": max_ch_clip,
        "down_output_error_amplification_ratio": amp,
        "residual_error_ratio_mean": residual_ratio,
        "flow_x_t_step00_relative_l2": first_flow,
        "flow_x_t_step09_relative_l2": last_flow,
        "flow_matching_trend": "amplifies" if last_flow > first_flow else "does_not_amplify",
        "best_range_by_action_relative_l2": best_range,
        "recommendation": (
            "keep_down_proj_w8a16_or_try_more_adaptive_activation_quantization"
            if w8a8_l2 > w8a16_l2 * 2.0
            else "down_proj_w8a8_may_be_worth_further_range_tuning"
        ),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_dtype = torch.float32 if args.input_dtype == "fp32" else torch.bfloat16

    fp_policy, preprocessor, device = load_policy(args.policy_path, args.device)
    inputs = core_inputs_from_policy(fp_policy, preprocessor, args.task, args.seed, input_dtype, args.token_length)
    fp_outputs, fp_capture = run_with_hooks(fp_policy, inputs, args.down_module_regex)

    scales_by_module = load_activation_scales_by_module(
        args.activation_scales_json,
        include_regexes=(args.down_module_regex,),
    )

    w8a16_policy = load_down_quant_policy(args.policy_path, device, args.down_module_regex, "w8a16")
    w8a16_outputs, w8a16_capture = run_with_hooks(w8a16_policy, inputs, args.down_module_regex)

    w8a8_policy = load_down_quant_policy(args.policy_path, device, args.down_module_regex, "w8a8", scales_by_module)
    w8a8_outputs, w8a8_capture = run_with_hooks(w8a8_policy, inputs, args.down_module_regex)

    dist_rows, channel_rows = distribution_rows(fp_capture.down_inputs, args.near_zero_threshold)
    clipping_rows, input_error_rows, top_channel_rows = clipping_and_input_error_rows(
        fp_capture.down_inputs,
        scales_by_module,
        args.topk_channels,
    )
    input_error_by_module = {row["module"]: row["relative_l2_error"] for row in input_error_rows}
    down_output_rows = output_error_rows(fp_capture, w8a16_capture, w8a8_capture, input_error_by_module)
    residual = residual_rows(fp_capture, w8a8_capture)
    flow = flow_rows(fp_outputs, w8a8_outputs)
    timestep_rows, dim_rows, action_summary = action_rows(
        fp_outputs["action_chunk"],
        w8a8_outputs["action_chunk"],
        args.sign_flip_threshold,
    )
    del w8a16_policy, w8a8_policy
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    ranges = range_sensitivity(args, fp_outputs, fp_capture.down_inputs, fp_capture.down_outputs, inputs)

    write_csv(output_dir / "down_activation_distribution.csv", dist_rows)
    write_csv(output_dir / "down_activation_channel_stats.csv", channel_rows)
    write_csv(output_dir / "down_activation_clipping.csv", clipping_rows)
    write_csv(output_dir / "down_quantization_error.csv", input_error_rows + top_channel_rows)
    write_csv(output_dir / "down_output_error.csv", down_output_rows)
    write_csv(output_dir / "residual_error.csv", residual)
    write_csv(output_dir / "flow_matching_step_error.csv", flow)
    write_csv(output_dir / "action_timestep_error.csv", timestep_rows)
    write_csv(output_dir / "action_dimension_error.csv", dim_rows)
    write_csv(output_dir / "range_sensitivity.csv", ranges)

    summary = {
        "policy_path": args.policy_path,
        "device": device,
        "seed": args.seed,
        "task": args.task,
        "input_dtype": args.input_dtype,
        "down_module_regex": args.down_module_regex,
        "activation_scales_json": args.activation_scales_json,
        "near_zero_threshold": args.near_zero_threshold,
        "sign_flip_threshold": args.sign_flip_threshold,
        "num_down_modules": len(fp_capture.down_inputs),
        "baseline_vs_w8a16_action": vector_metrics(fp_outputs["action_chunk"], w8a16_outputs["action_chunk"]),
        "baseline_vs_w8a8_action": vector_metrics(fp_outputs["action_chunk"], w8a8_outputs["action_chunk"]),
        "action_summary_w8a8": action_summary,
        "flow_step00_w8a8": flow[0] if flow else None,
        "flow_step09_w8a8": flow[-1] if flow else None,
        "mean_down_input_w8a8_relative_l2": aggregate(input_error_rows, "relative_l2_error"),
        "mean_down_output_w8a16_relative_l2": aggregate(
            [row for row in down_output_rows if row["variant"] == "W8A16"], "relative_l2_error"
        ),
        "mean_down_output_w8a8_relative_l2": aggregate(
            [row for row in down_output_rows if row["variant"] == "W8A8"], "relative_l2_error"
        ),
        "mean_clipping_rate": aggregate(clipping_rows, "clipping_rate"),
        "max_channel_clipping_rate": max(float(row["max_channel_clipping_rate"]) for row in clipping_rows),
        "mean_channel_max_abs_over_median": aggregate(dist_rows, "channel_max_abs_max_over_median"),
        "mean_channel_p99_99_over_median": aggregate(dist_rows, "channel_p99_99_max_over_median"),
    }
    infer_summary(summary, input_error_rows, down_output_rows, clipping_rows, residual, flow, ranges)
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
