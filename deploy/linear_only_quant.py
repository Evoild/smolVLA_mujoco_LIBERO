#!/usr/bin/env python3

"""Linear-only W8A8 fake-quant runner for SmolVLA deployment experiments.

Only nn.Linear weights and activations are quantized. Weights are statically
quantized per output channel and cached as bf16 dequantized tensors. Linear
inputs are quantized/dequantized to bf16. By default activation scales are
computed dynamically per token; diagnostics can also pass calibrated static
per-module activation scales to match explicit ONNX Q/DQ experiments.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import statistics
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from lerobot.configs import FeatureType, PreTrainedConfig
from lerobot.envs import close_envs, make_env, make_env_pre_post_processors
from lerobot.envs.factory import make_env_config
from lerobot.policies import make_pre_post_processors
from lerobot.policies.factory import get_policy_class
from lerobot.scripts.lerobot_eval import eval_policy_all
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.random_utils import set_seed


class W8A8Linear(nn.Module):
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None, activation_scale: Any | None = None):
        super().__init__()
        weight = weight.detach().to(device="cpu", dtype=torch.float32)
        smooth_scale = None
        if isinstance(activation_scale, dict) and "smooth_scale" in activation_scale:
            smooth_scale = torch.as_tensor(activation_scale["smooth_scale"], dtype=torch.float32).clamp_min(1.0e-8)
            if smooth_scale.ndim != 1 or smooth_scale.numel() != weight.shape[1]:
                raise ValueError(
                    f"smooth_scale shape {tuple(smooth_scale.shape)} does not match Linear in_features {weight.shape[1]}"
                )
            weight = weight * smooth_scale[None, :]

        qweight, weight_scale = quantize_weight_per_out_channel(weight)
        self.register_buffer("weight", dequantize_weight(qweight, weight_scale).to(torch.bfloat16))
        if bias is None:
            self.register_buffer("bias", None)
        else:
            self.register_buffer("bias", bias.detach().to(torch.bfloat16))
        if smooth_scale is None:
            self.register_buffer("smooth_scale", None)
        else:
            self.register_buffer("smooth_scale", smooth_scale)
        if activation_scale is None:
            self.register_buffer("activation_scale", None)
            self.register_buffer("activation_zero_point", None)
            self.activation_quantization = "dynamic_per_token"
        elif isinstance(activation_scale, dict):
            quantization = str(activation_scale.get("quantization", "symmetric_static_per_tensor"))
            scale_tensor = torch.as_tensor(activation_scale.get("scale", 1.0e-8), dtype=torch.float32).clamp_min(1.0e-8)
            self.register_buffer("activation_scale", scale_tensor)
            if quantization == "asymmetric_static_per_tensor":
                zero_point = int(activation_scale.get("zero_point", 0))
                self.register_buffer("activation_zero_point", torch.tensor(zero_point, dtype=torch.int32))
                self.activation_quantization = "calibrated_static_asymmetric_per_tensor"
            elif quantization == "smoothquant_asymmetric_static_per_tensor":
                zero_point = int(activation_scale.get("zero_point", 0))
                self.register_buffer("activation_zero_point", torch.tensor(zero_point, dtype=torch.int32))
                self.activation_quantization = "calibrated_static_asymmetric_per_tensor"
            elif quantization in {
                "symmetric_static_per_tensor",
                "symmetric_static_per_channel",
                "smoothquant_symmetric_static_per_tensor",
            }:
                self.register_buffer("activation_zero_point", None)
                self.activation_quantization = (
                    "calibrated_static_per_channel" if scale_tensor.ndim > 0 else "calibrated_static_per_tensor"
                )
            else:
                raise ValueError(f"unsupported activation quantization: {quantization}")
        else:
            scale_tensor = torch.as_tensor(activation_scale, dtype=torch.float32).clamp_min(1.0e-8)
            self.register_buffer("activation_scale", scale_tensor)
            self.register_buffer("activation_zero_point", None)
            self.activation_quantization = (
                "calibrated_static_per_channel" if scale_tensor.ndim > 0 else "calibrated_static_per_tensor"
            )
        self.in_features = weight.shape[1]
        self.out_features = weight.shape[0]

    @classmethod
    def from_float(cls, module: nn.Linear, activation_scale: Any | None = None) -> "W8A8Linear":
        return cls(module.weight, module.bias, activation_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.smooth_scale is not None:
            smooth_scale = self.smooth_scale.to(device=x.device, dtype=torch.float32)
            x = x.to(torch.float32) / smooth_scale
        if self.activation_scale is None:
            x_dequant = quantize_dequantize_activation(x).to(torch.bfloat16)
        elif self.activation_quantization == "calibrated_static_asymmetric_per_tensor":
            x_dequant = quantize_dequantize_activation_static_asymmetric(
                x, self.activation_scale, self.activation_zero_point
            ).to(torch.bfloat16)
        else:
            x_dequant = quantize_dequantize_activation_static(x, self.activation_scale).to(torch.bfloat16)
        return F.linear(x_dequant, self.weight, self.bias)


class W8A16Linear(nn.Module):
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None):
        super().__init__()
        qweight, weight_scale = quantize_weight_per_out_channel(weight.detach())
        self.register_buffer("weight", dequantize_weight(qweight, weight_scale).to(torch.bfloat16))
        if bias is None:
            self.register_buffer("bias", None)
        else:
            self.register_buffer("bias", bias.detach().to(torch.bfloat16))
        self.in_features = weight.shape[1]
        self.out_features = weight.shape[0]
        self.activation_quantization = "none"

    @classmethod
    def from_float(cls, module: nn.Linear) -> "W8A16Linear":
        return cls(module.weight, module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x.to(torch.bfloat16), self.weight, self.bias)


def quantize_weight_per_out_channel(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    weight_fp32 = weight.detach().to(torch.float32)
    scale = weight_fp32.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / 127.0
    qweight = torch.round(weight_fp32 / scale).clamp(-127, 127).to(torch.int8)
    return qweight, scale.squeeze(1)


def dequantize_weight(qweight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return qweight.to(torch.float32) * scale[:, None].to(torch.float32)


def quantize_dequantize_activation(x: torch.Tensor) -> torch.Tensor:
    x_fp32 = x.to(torch.float32)
    scale = x_fp32.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 127.0
    qx = torch.round(x_fp32 / scale).clamp(-127, 127).to(torch.int8)
    return qx.to(torch.float32) * scale


def quantize_dequantize_activation_static(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    x_fp32 = x.to(torch.float32)
    scale_fp32 = scale.to(device=x_fp32.device, dtype=torch.float32).clamp_min(1e-8)
    qx = torch.round(x_fp32 / scale_fp32).clamp(-128, 127).to(torch.int8)
    return qx.to(torch.float32) * scale_fp32


def quantize_dequantize_activation_static_asymmetric(
    x: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor
) -> torch.Tensor:
    x_fp32 = x.to(torch.float32)
    scale_fp32 = scale.to(device=x_fp32.device, dtype=torch.float32).clamp_min(1e-8)
    zp_fp32 = zero_point.to(device=x_fp32.device, dtype=torch.float32)
    qx = torch.round(x_fp32 / scale_fp32 + zp_fp32).clamp(-128, 127)
    return (qx - zp_fp32) * scale_fp32


def append_activation_scale(scales: dict[str, Any], module_name: str, value: Any) -> None:
    existing = scales.get(module_name)
    if existing is None:
        scales[module_name] = value
    elif isinstance(existing, list) and existing and isinstance(existing[0], dict):
        existing.append(value)
    else:
        scales[module_name] = [existing, value]


def activation_scale_channels(value: Any) -> int | None:
    if isinstance(value, dict):
        smooth_scale = value.get("smooth_scale")
        if isinstance(smooth_scale, list):
            return len(smooth_scale)
        scale = value.get("scale")
        if isinstance(scale, list):
            return len(scale)
        return None
    if isinstance(value, list):
        return len(value)
    return None


def select_activation_scale_for_linear(value: Any, in_features: int) -> Any:
    if isinstance(value, list) and value and isinstance(value[0], dict):
        for candidate in value:
            if activation_scale_channels(candidate) == in_features:
                return candidate
        return value[0]
    return value


def load_activation_scales_by_module(scales_json: str | Path, include_regexes: tuple[str, ...] | None = None) -> dict[str, Any]:
    with open(scales_json) as f:
        calibration = json.load(f)
    patterns = tuple(re.compile(pattern) for pattern in (include_regexes or ()))
    scales: dict[str, Any] = {}
    for entry in calibration.get("linear_call_scales", []):
        module_name = str(entry.get("module", ""))
        if patterns and not any(pattern.search(module_name) for pattern in patterns):
            continue
        if module_name:
            quantization = str(entry.get("quantization", ""))
            scale = entry.get("scale", 1.0e-8)
            if quantization == "asymmetric_static_per_tensor":
                append_activation_scale(scales, module_name, {
                    "quantization": quantization,
                    "scale": max(float(scale), 1.0e-8),
                    "zero_point": int(entry.get("zero_point", 0)),
                })
            elif quantization in {
                "smoothquant_symmetric_static_per_tensor",
                "smoothquant_asymmetric_static_per_tensor",
            }:
                smooth_scale = entry.get("smooth_scale")
                if not isinstance(smooth_scale, list):
                    raise ValueError(f"missing smooth_scale list for SmoothQuant module: {module_name}")
                value = {
                    "quantization": quantization,
                    "scale": max(float(scale), 1.0e-8),
                    "smooth_scale": [max(float(value), 1.0e-8) for value in smooth_scale],
                    "smoothquant_alpha": entry.get("smoothquant_alpha"),
                }
                if quantization == "smoothquant_asymmetric_static_per_tensor":
                    value["zero_point"] = int(entry.get("zero_point", 0))
                append_activation_scale(scales, module_name, value)
            elif isinstance(scale, list):
                append_activation_scale(scales, module_name, [max(float(value), 1.0e-8) for value in scale])
            else:
                append_activation_scale(scales, module_name, max(float(scale), 1.0e-8))
    return scales


def should_quantize_module(
    module_name: str,
    include_prefixes: tuple[str, ...] | None,
    include_regexes: tuple[str, ...] | None = None,
) -> bool:
    if include_regexes and any(re.search(pattern, module_name) for pattern in include_regexes):
        return True
    if not include_prefixes and not include_regexes:
        return True
    return any(module_name == prefix or module_name.startswith(f"{prefix}.") for prefix in (include_prefixes or ()))


def replace_linear_modules(
    module: nn.Module,
    include_prefixes: tuple[str, ...] | None = None,
    include_regexes: tuple[str, ...] | None = None,
    activation_scales_by_module: dict[str, Any] | None = None,
    quantization_kind: str = "w8a8",
    module_prefix: str = "",
) -> int:
    if quantization_kind not in {"w8a8", "w8a16"}:
        raise ValueError(f"unsupported quantization_kind: {quantization_kind}")
    replaced = 0
    for name, child in list(module.named_children()):
        child_name = f"{module_prefix}.{name}" if module_prefix else name
        if isinstance(child, nn.Linear):
            if should_quantize_module(child_name, include_prefixes, include_regexes):
                activation_scale = None
                if activation_scales_by_module is not None:
                    if child_name not in activation_scales_by_module:
                        raise KeyError(f"missing calibrated activation scale for quantized module: {child_name}")
                    activation_scale = select_activation_scale_for_linear(
                        activation_scales_by_module[child_name],
                        int(child.in_features),
                    )
                if quantization_kind == "w8a16":
                    setattr(module, name, W8A16Linear.from_float(child))
                else:
                    setattr(module, name, W8A8Linear.from_float(child, activation_scale))
                replaced += 1
        else:
            replaced += replace_linear_modules(
                child,
                include_prefixes,
                include_regexes,
                activation_scales_by_module,
                quantization_kind,
                child_name,
            )
    return replaced


def load_quantized_policy(
    policy_path: str,
    device: str,
    include_prefixes: tuple[str, ...] | None = None,
    include_regexes: tuple[str, ...] | None = None,
    activation_scales_by_module: dict[str, Any] | None = None,
    quantization_kind: str = "w8a8",
    extra_w8a16_regexes: tuple[str, ...] | None = None,
) -> tuple[nn.Module, Any, dict[str, Any], str]:
    cfg = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=[f"--device={device}"])
    policy_cls = get_policy_class(cfg.type)
    policy = policy_cls.from_pretrained(policy_path, config=cfg, strict=False)
    policy.eval()
    num_linear = replace_linear_modules(
        policy,
        include_prefixes=include_prefixes,
        include_regexes=include_regexes,
        activation_scales_by_module=activation_scales_by_module,
        quantization_kind=quantization_kind,
    )
    num_extra_w8a16 = 0
    if extra_w8a16_regexes:
        num_extra_w8a16 = replace_linear_modules(
            policy,
            include_prefixes=None,
            include_regexes=extra_w8a16_regexes,
            activation_scales_by_module=None,
            quantization_kind="w8a16",
        )
    effective_device = str(policy.config.device)
    policy.to(effective_device)
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=policy_path,
        preprocessor_overrides={"device_processor": {"device": effective_device}},
    )
    report = {
        "quantization": f"linear_only_{quantization_kind}_fake_quant",
        "weight_quantization": "int8 symmetric per-output-channel, dequantized/cache bf16",
        "activation_quantization": (
            "int8 symmetric calibrated static, dequantized bf16"
            if activation_scales_by_module is not None
            else "int8 symmetric dynamic per-token, dequantized bf16"
        ),
        "non_linear_ops": "bf16 autocast",
        "num_linear_replaced": num_linear,
        "num_extra_w8a16_replaced": num_extra_w8a16,
        "include_prefixes": list(include_prefixes) if include_prefixes else None,
        "include_regexes": list(include_regexes) if include_regexes else None,
        "extra_w8a16_regexes": list(extra_w8a16_regexes) if extra_w8a16_regexes else None,
        "requested_device": device,
        "effective_device": effective_device,
    }
    return policy, (preprocessor, postprocessor), report, effective_device


def make_raw_observation(config: PreTrainedConfig, task: str) -> dict[str, Any]:
    obs: dict[str, Any] = {"task": task}
    for key, feature in (config.input_features or {}).items():
        shape = tuple(int(dim) for dim in feature.shape)
        if feature.type is FeatureType.VISUAL:
            obs[key] = torch.rand(shape, dtype=torch.float32)
        elif feature.type is FeatureType.STATE or key == OBS_STATE:
            obs[key] = torch.zeros(shape, dtype=torch.float32)
        elif feature.type is FeatureType.ENV:
            obs[key] = torch.zeros(shape, dtype=torch.float32)
    if OBS_STATE not in obs and getattr(config, "max_state_dim", None):
        obs[OBS_STATE] = torch.zeros((int(config.max_state_dim),), dtype=torch.float32)
    return obs


def add_dummy_action(policy: nn.Module, batch: dict[str, Any]) -> dict[str, Any]:
    batch = {key: value.clone() if torch.is_tensor(value) else value for key, value in batch.items()}
    state = batch[OBS_STATE]
    batch_size = state.shape[0]
    action_dim = int(policy.config.action_feature.shape[0])
    chunk_size = int(policy.config.chunk_size)
    batch[ACTION] = torch.zeros((batch_size, chunk_size, action_dim), dtype=torch.float32, device=state.device)
    return batch


@contextmanager
def bf16_autocast(device: str):
    device_type = torch.device(device).type
    if device_type in {"cuda", "cpu"}:
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            yield
    else:
        yield


def sync(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def peak_memory_gb(device: str) -> float | None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024**3)
    return None


def summarize(values: list[float], prefix: str) -> dict[str, float]:
    sorted_values = sorted(values)
    return {
        f"{prefix}_mean_ms": statistics.fmean(values),
        f"{prefix}_median_ms": statistics.median(values),
        f"{prefix}_p95_ms": sorted_values[max(0, int(len(sorted_values) * 0.95) - 1)],
    }


def is_leaf_module(module: nn.Module) -> bool:
    return not any(module.children())


def add_leaf_hooks(
    module: nn.Module,
    group_name: str,
    timer_state: dict[str, Any],
    handles: list[Any],
) -> None:
    def pre_hook(_module, _inputs):
        if timer_state["use_cuda_events"]:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _module.__profile_start = (start, end)
        else:
            _module.__profile_start = time.perf_counter()

    def post_hook(_module, _inputs, _output):
        start = getattr(_module, "__profile_start", None)
        if start is None:
            return
        if timer_state["use_cuda_events"]:
            start_event, end_event = start
            end_event.record()
            timer_state["pending_events"].append((group_name, start_event, end_event))
        else:
            timer_state["groups"][group_name] += (time.perf_counter() - start) * 1000.0

    for child in module.modules():
        if is_leaf_module(child):
            handles.append(child.register_forward_pre_hook(pre_hook))
            handles.append(child.register_forward_hook(post_hook))


def add_direct_hook(module: nn.Module, group_name: str, timer_state: dict[str, Any], handles: list[Any]) -> None:
    def pre_hook(_module, _inputs):
        if timer_state["use_cuda_events"]:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _module.__profile_start = (start, end)
        else:
            _module.__profile_start = time.perf_counter()

    def post_hook(_module, _inputs, _output):
        start = getattr(_module, "__profile_start", None)
        if start is None:
            return
        if timer_state["use_cuda_events"]:
            start_event, end_event = start
            end_event.record()
            timer_state["pending_events"].append((group_name, start_event, end_event))
        else:
            timer_state["groups"][group_name] += (time.perf_counter() - start) * 1000.0

    handles.append(module.register_forward_pre_hook(pre_hook))
    handles.append(module.register_forward_hook(post_hook))


@contextmanager
def module_timer(policy: nn.Module, device: str):
    groups: dict[str, float] = defaultdict(float)
    timer_state: dict[str, Any] = {
        "groups": groups,
        "pending_events": [],
        "use_cuda_events": str(device).startswith("cuda") and torch.cuda.is_available(),
    }
    handles: list[Any] = []
    model = policy.model
    add_leaf_hooks(model.vlm_with_expert.vlm, "vlm", timer_state, handles)
    add_leaf_hooks(model.vlm_with_expert.lm_expert, "lm_expert", timer_state, handles)
    add_direct_hook(model.state_proj, "state_proj", timer_state, handles)
    add_direct_hook(model.action_in_proj, "action_in_proj", timer_state, handles)
    add_direct_hook(model.action_time_mlp_in, "action_time_mlp", timer_state, handles)
    add_direct_hook(model.action_time_mlp_out, "action_time_mlp", timer_state, handles)
    add_direct_hook(model.action_out_proj, "action_out_proj", timer_state, handles)
    try:
        yield groups
    finally:
        if timer_state["use_cuda_events"]:
            torch.cuda.synchronize()
            for group_name, start_event, end_event in timer_state["pending_events"]:
                groups[group_name] += start_event.elapsed_time(end_event)
        for handle in handles:
            handle.remove()


def run_profile(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    quantize_regexes = tuple(args.quantize_module_regex) if args.quantize_module_regex else None
    quantize_prefixes = None if quantize_regexes else tuple(args.quantize_module_prefix) if args.quantize_module_prefix else None
    activation_scales_by_module = None
    if args.fake_quant_activation_scale_mode == "calibrated":
        if not args.fake_quant_activation_scales_json:
            raise ValueError(
                "--fake-quant-activation-scales-json is required when "
                "--fake-quant-activation-scale-mode calibrated"
            )
        activation_scales_by_module = load_activation_scales_by_module(
            args.fake_quant_activation_scales_json,
            include_regexes=quantize_regexes,
        )
    policy, (preprocessor, _), report, device = load_quantized_policy(
        args.policy_path,
        args.device,
        include_prefixes=quantize_prefixes,
        include_regexes=quantize_regexes,
        activation_scales_by_module=activation_scales_by_module,
        quantization_kind=args.fake_quant_kind,
        extra_w8a16_regexes=tuple(args.extra_w8a16_module_regex) if args.extra_w8a16_module_regex else None,
    )
    raw_observation = make_raw_observation(policy.config, args.task)
    batch = add_dummy_action(policy, preprocessor(raw_observation))

    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    with torch.no_grad(), bf16_autocast(device):
        for _ in range(args.warmup):
            policy.forward({key: value.clone() if torch.is_tensor(value) else value for key, value in batch.items()})
    sync(device)

    rows = []
    with torch.no_grad(), bf16_autocast(device):
        for idx in range(args.iters):
            current = {key: value.clone() if torch.is_tensor(value) else value for key, value in batch.items()}
            with module_timer(policy, device) as module_ms:
                start = time.perf_counter()
                policy.forward(current)
                sync(device)
                e2e_ms = (time.perf_counter() - start) * 1000.0
            measured_modules_ms = sum(module_ms.values())
            rows.append(
                {
                    "iteration": idx,
                    "forward_e2e_ms": e2e_ms,
                    "vlm_ms": module_ms["vlm"],
                    "lm_expert_ms": module_ms["lm_expert"],
                    "state_proj_ms": module_ms["state_proj"],
                    "action_in_proj_ms": module_ms["action_in_proj"],
                    "action_time_mlp_ms": module_ms["action_time_mlp"],
                    "action_out_proj_ms": module_ms["action_out_proj"],
                    "unattributed_ms": max(0.0, e2e_ms - measured_modules_ms),
                }
            )

    metric_keys = [
        "forward_e2e_ms",
        "vlm_ms",
        "lm_expert_ms",
        "state_proj_ms",
        "action_in_proj_ms",
        "action_time_mlp_ms",
        "action_out_proj_ms",
        "unattributed_ms",
    ]
    latencies = [row["forward_e2e_ms"] for row in rows]
    summary = {
        "device": device,
        "peak_gpu_memory_gb": peak_memory_gb(device),
        "forward_fps": 1000.0 / statistics.fmean(latencies),
        **report,
    }
    for key in metric_keys:
        summary.update(summarize([float(row[key]) for row in rows], key.removesuffix("_ms")))

    with open(output_dir / "linear_only_profile_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(output_dir / "linear_only_profile_iterations.json", "w") as f:
        json.dump(rows, f, indent=2)
    with open(output_dir / "linear_only_profile_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)
    with open(output_dir / "linear_only_profile_iterations.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))


def run_eval(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    quantize_regexes = tuple(args.quantize_module_regex) if args.quantize_module_regex else None
    quantize_prefixes = None if quantize_regexes else tuple(args.quantize_module_prefix) if args.quantize_module_prefix else None
    activation_scales_by_module = None
    if args.fake_quant_activation_scale_mode == "calibrated":
        if not args.fake_quant_activation_scales_json:
            raise ValueError(
                "--fake-quant-activation-scales-json is required when "
                "--fake-quant-activation-scale-mode calibrated"
            )
        activation_scales_by_module = load_activation_scales_by_module(
            args.fake_quant_activation_scales_json,
            include_regexes=quantize_regexes,
        )
    policy, (preprocessor, postprocessor), report, device = load_quantized_policy(
        args.policy_path,
        args.device,
        include_prefixes=quantize_prefixes,
        include_regexes=quantize_regexes,
        activation_scales_by_module=activation_scales_by_module,
        quantization_kind=args.fake_quant_kind,
        extra_w8a16_regexes=tuple(args.extra_w8a16_module_regex) if args.extra_w8a16_module_regex else None,
    )
    env_cfg = make_env_config("libero", task=args.tasks, max_parallel_tasks=args.max_parallel_tasks)
    envs = make_env(env_cfg, n_envs=args.batch_size, use_async_envs=False, trust_remote_code=False)
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy.config)

    try:
        with torch.no_grad(), bf16_autocast(device):
            info = eval_policy_all(
                envs=envs,
                policy=policy,
                env_preprocessor=env_preprocessor,
                env_postprocessor=env_postprocessor,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                n_episodes=args.episodes,
                max_episodes_rendered=10,
                videos_dir=output_dir / "videos",
                start_seed=args.seed,
                max_parallel_tasks=args.max_parallel_tasks,
            )
        info["quantization"] = report
        with open(output_dir / "eval_info.json", "w") as f:
            json.dump(info, f, indent=2)
        print(json.dumps(info["overall"], indent=2))
    finally:
        close_envs(envs)


def add_fake_quant_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--quantize-module-prefix",
        action="append",
        default=["model.vlm_with_expert.vlm.model.text_model"],
        help="PyTorch fake-quant module prefix. Pass multiple times if needed.",
    )
    parser.add_argument(
        "--quantize-module-regex",
        action="append",
        default=None,
        help="Only PyTorch fake-quant nn.Linear modules whose full module name matches this regex.",
    )
    parser.add_argument(
        "--fake-quant-activation-scale-mode",
        choices=["dynamic", "calibrated"],
        default="dynamic",
        help="Activation scale mode for W8A8 fake quant.",
    )
    parser.add_argument(
        "--fake-quant-activation-scales-json",
        default=None,
        help="Calibration JSON used when --fake-quant-activation-scale-mode calibrated.",
    )
    parser.add_argument(
        "--fake-quant-kind",
        choices=["w8a8", "w8a16"],
        default="w8a8",
        help="Primary fake-quant kind.",
    )
    parser.add_argument(
        "--extra-w8a16-module-regex",
        action="append",
        default=None,
        help="Additional nn.Linear module regexes to replace with W8A16 after primary replacement.",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    profile = subparsers.add_parser("profile")
    profile.add_argument("--policy-path", required=True)
    profile.add_argument("--device", default="cuda")
    profile.add_argument("--output-dir", required=True)
    profile.add_argument("--task", default="libero_goal linear-only profile")
    profile.add_argument("--warmup", type=int, default=5)
    profile.add_argument("--iters", type=int, default=30)
    add_fake_quant_args(profile)

    eval_parser = subparsers.add_parser("eval")
    eval_parser.add_argument("--policy-path", required=True)
    eval_parser.add_argument("--device", default="cuda")
    eval_parser.add_argument("--output-dir", required=True)
    eval_parser.add_argument("--tasks", default="libero_goal")
    eval_parser.add_argument("--seed", type=int, default=1000)
    eval_parser.add_argument("--episodes", type=int, default=10)
    eval_parser.add_argument("--batch-size", type=int, default=1)
    eval_parser.add_argument("--max-parallel-tasks", type=int, default=1)
    add_fake_quant_args(eval_parser)

    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    if args.command == "profile":
        run_profile(args)
    elif args.command == "eval":
        run_eval(args)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
