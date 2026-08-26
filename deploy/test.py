#!/usr/bin/env python3

"""Compare original SmolVLA and quantized deployment outputs on identical inputs.

This script compares PyTorch original vs PyTorch W8A8 linear-quantized modules
with forward hooks, then optionally compares the final action chunk against a
TensorRT engine. TensorRT plan files normally expose only final outputs, so
per-module similarity is measured on the PyTorch quantized emulation.
"""

from __future__ import annotations

import argparse
import csv
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
from torch import nn

from lerobot.configs import FeatureType, PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE
from lerobot.utils.random_utils import set_seed

from linear_only_quant import replace_linear_modules


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", default="smolvla_libero")
    parser.add_argument("--engine-path", default=None, help="Optional TensorRT engine for final-output comparison.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--task", default="libero_goal quantization similarity test")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--output-dir", default="runs/deploy/3-3B1/similarity")
    parser.add_argument(
        "--capture",
        choices=["core", "leaf"],
        default="leaf",
        help="core captures main SmolVLA blocks; leaf captures every called leaf module plus core blocks.",
    )
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument(
        "--quantize-module-prefix",
        action="append",
        default=None,
        help=(
            "Only fake-quantize nn.Linear modules under this policy module prefix. "
            "Can be passed multiple times. Default quantizes all Linear modules."
        ),
    )
    return parser.parse_args()


def load_policy(policy_path: str, device: str) -> tuple[nn.Module, Any, str]:
    cfg = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=[f"--device={device}"])
    policy_cls = get_policy_class(cfg.type)
    policy = policy_cls.from_pretrained(policy_path, config=cfg, strict=False)
    policy.eval()
    effective_device = str(policy.config.device)
    policy.to(effective_device)
    preprocessor, _ = make_pre_post_processors(
        policy.config,
        pretrained_path=policy_path,
        preprocessor_overrides={"device_processor": {"device": effective_device}},
    )
    return policy, preprocessor, effective_device


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


def clone_batch(batch: dict[str, Any]) -> dict[str, Any]:
    return {key: value.clone() if torch.is_tensor(value) else value for key, value in batch.items()}


def flatten_tensors(value: Any) -> list[torch.Tensor]:
    if torch.is_tensor(value):
        return [value]
    if isinstance(value, (list, tuple)):
        tensors: list[torch.Tensor] = []
        for item in value:
            tensors.extend(flatten_tensors(item))
        return tensors
    if isinstance(value, dict):
        tensors = []
        for item in value.values():
            tensors.extend(flatten_tensors(item))
        return tensors
    return []


def tensor_vector(value: Any) -> torch.Tensor | None:
    tensors = [tensor.detach().to(torch.float32).flatten().cpu() for tensor in flatten_tensors(value)]
    tensors = [tensor for tensor in tensors if tensor.numel() > 0]
    if not tensors:
        return None
    return torch.cat(tensors)


def first_tensor(value: Any) -> torch.Tensor | None:
    tensors = flatten_tensors(value)
    if not tensors:
        return None
    return tensors[0].detach().to(torch.float32).cpu()


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    length = min(a.numel(), b.numel())
    if length == 0:
        return float("nan")
    a = a[:length]
    b = b[:length]
    denom = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    if denom.item() == 0.0:
        return 1.0 if torch.allclose(a, b) else 0.0
    return float(torch.dot(a, b) / denom)


def l2_relative_error(a: torch.Tensor, b: torch.Tensor) -> float:
    length = min(a.numel(), b.numel())
    if length == 0:
        return float("nan")
    a = a[:length]
    b = b[:length]
    denom = torch.linalg.vector_norm(a).clamp_min(1e-12)
    return float(torch.linalg.vector_norm(a - b) / denom)


def norm_metrics(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    length = min(a.numel(), b.numel())
    if length == 0:
        return {
            "original_l2_norm": float("nan"),
            "quantized_l2_norm": float("nan"),
            "l2_norm_ratio": float("nan"),
            "l2_norm_diff": float("nan"),
            "relative_l2_norm_error": float("nan"),
        }
    a = a[:length]
    b = b[:length]
    original_norm = torch.linalg.vector_norm(a).clamp_min(1e-12)
    quantized_norm = torch.linalg.vector_norm(b)
    norm_diff = quantized_norm - original_norm
    return {
        "original_l2_norm": float(original_norm),
        "quantized_l2_norm": float(quantized_norm),
        "l2_norm_ratio": float(quantized_norm / original_norm),
        "l2_norm_diff": float(norm_diff),
        "relative_l2_norm_error": float(torch.abs(norm_diff) / original_norm),
    }


def distribution_metrics(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    length = min(a.numel(), b.numel())
    if length == 0:
        return {
            "original_mean": float("nan"),
            "quantized_mean": float("nan"),
            "mean_diff": float("nan"),
            "relative_mean_error": float("nan"),
            "original_std": float("nan"),
            "quantized_std": float("nan"),
            "std_ratio": float("nan"),
            "std_diff": float("nan"),
            "relative_std_error": float("nan"),
        }
    a = a[:length]
    b = b[:length]
    original_mean = a.mean()
    quantized_mean = b.mean()
    original_std = a.std(unbiased=False)
    quantized_std = b.std(unbiased=False)
    mean_diff = quantized_mean - original_mean
    std_diff = quantized_std - original_std
    return {
        "original_mean": float(original_mean),
        "quantized_mean": float(quantized_mean),
        "mean_diff": float(mean_diff),
        "relative_mean_error": float(torch.abs(mean_diff) / original_mean.abs().clamp_min(1e-12)),
        "original_std": float(original_std),
        "quantized_std": float(quantized_std),
        "std_ratio": float(quantized_std / original_std.clamp_min(1e-12)),
        "std_diff": float(std_diff),
        "relative_std_error": float(torch.abs(std_diff) / original_std.clamp_min(1e-12)),
    }


def channel_metrics(original_value: Any, quantized_value: Any) -> dict[str, float | int]:
    original_tensor = first_tensor(original_value)
    quantized_tensor = first_tensor(quantized_value)
    if original_tensor is None or quantized_tensor is None or original_tensor.ndim == 0 or quantized_tensor.ndim == 0:
        return {
            "num_channels_compared": 0,
            "max_channel_abs_error": float("nan"),
            "max_channel_abs_error_index": -1,
            "max_channel_mean_abs_error_at_max": float("nan"),
            "max_channel_relative_l2_error": float("nan"),
            "max_channel_relative_l2_error_index": -1,
        }

    channels = min(int(original_tensor.shape[-1]), int(quantized_tensor.shape[-1]))
    if channels == 0:
        return {
            "num_channels_compared": 0,
            "max_channel_abs_error": float("nan"),
            "max_channel_abs_error_index": -1,
            "max_channel_mean_abs_error_at_max": float("nan"),
            "max_channel_relative_l2_error": float("nan"),
            "max_channel_relative_l2_error_index": -1,
        }

    original_2d = original_tensor[..., :channels].reshape(-1, channels)
    quantized_2d = quantized_tensor[..., :channels].reshape(-1, channels)
    rows = min(int(original_2d.shape[0]), int(quantized_2d.shape[0]))
    if rows == 0:
        return {
            "num_channels_compared": int(channels),
            "max_channel_abs_error": float("nan"),
            "max_channel_abs_error_index": -1,
            "max_channel_mean_abs_error_at_max": float("nan"),
            "max_channel_relative_l2_error": float("nan"),
            "max_channel_relative_l2_error_index": -1,
        }

    original_2d = original_2d[:rows]
    quantized_2d = quantized_2d[:rows]
    abs_error = (original_2d - quantized_2d).abs()
    per_channel_max_abs = abs_error.amax(dim=0)
    per_channel_mean_abs = abs_error.mean(dim=0)
    per_channel_relative_l2 = torch.linalg.vector_norm(original_2d - quantized_2d, dim=0) / torch.linalg.vector_norm(
        original_2d, dim=0
    ).clamp_min(1e-12)
    max_abs_index = int(torch.argmax(per_channel_max_abs))
    max_relative_l2_index = int(torch.argmax(per_channel_relative_l2))
    return {
        "num_channels_compared": int(channels),
        "max_channel_abs_error": float(per_channel_max_abs[max_abs_index]),
        "max_channel_abs_error_index": max_abs_index,
        "max_channel_mean_abs_error_at_max": float(per_channel_mean_abs[max_abs_index]),
        "max_channel_relative_l2_error": float(per_channel_relative_l2[max_relative_l2_index]),
        "max_channel_relative_l2_error_index": max_relative_l2_index,
    }


def comparison_row(module: str, original_value: Any, quantized_value: Any, note: str | None = None) -> dict[str, Any] | None:
    orig_vec = tensor_vector(original_value)
    quant_vec = tensor_vector(quantized_value)
    if orig_vec is None or quant_vec is None:
        return None
    length = min(orig_vec.numel(), quant_vec.numel())
    orig_slice = orig_vec[:length]
    quant_slice = quant_vec[:length]
    original_first = first_tensor(original_value)
    quantized_first = first_tensor(quantized_value)
    row: dict[str, Any] = {
        "module": module,
        "numel_compared": int(length),
        "cosine_similarity": cosine_similarity(orig_vec, quant_vec),
        "relative_l2_error": l2_relative_error(orig_vec, quant_vec),
        "max_abs_error": float((orig_slice - quant_slice).abs().max()) if length else float("nan"),
        "mean_abs_error": float((orig_slice - quant_slice).abs().mean()) if length else float("nan"),
        "original_shape": str(tuple(original_first.shape)) if original_first is not None else "",
        "quantized_shape": str(tuple(quantized_first.shape)) if quantized_first is not None else "",
    }
    if note:
        row["note"] = note
    row.update(norm_metrics(orig_vec, quant_vec))
    row.update(distribution_metrics(orig_vec, quant_vec))
    row.update(channel_metrics(original_value, quantized_value))
    return row


def is_leaf(module: nn.Module) -> bool:
    return not any(module.children())


def core_module_names(policy: nn.Module) -> set[str]:
    model = policy.model
    names: set[str] = set()
    for module_name in [
        "vlm_with_expert.vlm",
        "vlm_with_expert.lm_expert",
        "state_proj",
        "action_in_proj",
        "action_time_mlp_in",
        "action_time_mlp_out",
        "action_out_proj",
    ]:
        if dict(model.named_modules()).get(module_name) is not None:
            names.add(f"model.{module_name}")
    return names


@contextmanager
def capture_module_outputs(policy: nn.Module, mode: str):
    outputs: dict[str, Any] = {}
    handles: list[Any] = []
    core_names = core_module_names(policy)

    def should_capture(name: str, module: nn.Module) -> bool:
        if name in core_names:
            return True
        return mode == "leaf" and is_leaf(module)

    def make_hook(name: str):
        def hook(_module, _inputs, output):
            if name not in outputs:
                outputs[name] = output

        return hook

    for name, module in policy.named_modules():
        if name and should_capture(name, module):
            handles.append(module.register_forward_hook(make_hook(name)))
    try:
        yield outputs
    finally:
        for handle in handles:
            handle.remove()


def prepare_sample_inputs(policy: nn.Module, batch: dict[str, Any], noise: torch.Tensor):
    images, img_masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)
    lang_tokens = batch[OBS_LANGUAGE_TOKENS]
    lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
    return images, img_masks, lang_tokens, lang_masks, state, noise


def run_sample_actions(policy: nn.Module, batch: dict[str, Any], noise: torch.Tensor) -> torch.Tensor:
    inputs = prepare_sample_inputs(policy, batch, noise)
    actions = policy.model.sample_actions(*inputs[:-1], noise=inputs[-1])
    original_action_dim = int(policy.config.action_feature.shape[0])
    return actions[:, :, :original_action_dim]


def compare_outputs(original: dict[str, Any], quantized: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in sorted(set(original) & set(quantized)):
        row = comparison_row(name, original[name], quantized[name])
        if row is not None:
            rows.append(row)
    return rows


def run_trt_final_compare(policy: nn.Module, batch: dict[str, Any], noise: torch.Tensor, engine_path: str) -> tuple[dict[str, Any], dict[str, Any]]:
    from trt_sample_actions_core_deploy import TensorRTCoreRunner

    runner = TensorRTCoreRunner(engine_path)
    images, img_masks, lang_tokens, lang_masks, state, noise = prepare_sample_inputs(policy, batch, noise)
    with torch.no_grad():
        image_emb = policy.model.vlm_with_expert.embed_image(images[0])
        image2_emb = policy.model.vlm_with_expert.embed_image(images[1])

    def match_static_shape(tensor: torch.Tensor, name: str, pad_value: int | float | bool = 0) -> torch.Tensor:
        expected = runner.input_shapes[name]
        if tuple(tensor.shape) == expected:
            return tensor
        slices = tuple(slice(0, min(int(got), int(want))) for got, want in zip(tensor.shape, expected, strict=True))
        output = torch.full(expected, pad_value, dtype=tensor.dtype, device=tensor.device)
        output[slices] = tensor[slices]
        return output

    trt_actions = runner(
        image_emb=image_emb,
        image2_emb=image2_emb,
        image_mask=img_masks[0],
        image2_mask=img_masks[1],
        state=state,
        language_tokens=match_static_shape(lang_tokens, "language_tokens", 0),
        language_attention_mask=match_static_shape(lang_masks, "language_attention_mask", False),
        noise=noise,
    )
    pytorch_actions = run_sample_actions(policy, batch, noise)
    trt_actions = trt_actions[:, :, : policy.config.action_feature.shape[0]]
    final_row = comparison_row(
        "TensorRT.engine.action_chunk",
        pytorch_actions,
        trt_actions,
        note="TensorRT plan exposes final action_chunk only; per-module rows use PyTorch W8A8 emulation.",
    )
    first_action_row = comparison_row(
        "TensorRT.engine.action_chunk.first_action",
        pytorch_actions[:, 0, :],
        trt_actions[:, 0, :],
        note="First action in the chunk; this is the immediate action consumed by queue-based policy execution.",
    )
    assert final_row is not None and first_action_row is not None
    return final_row, first_action_row


def write_reports(rows: list[dict[str, Any]], summary: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "module_similarity.json", "w") as f:
        json.dump(rows, f, indent=2)
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    if rows:
        with open(output_dir / "module_similarity.csv", "w", newline="") as f:
            fieldnames = sorted({key for row in rows for key in row})
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(summary, indent=2))


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)

    original, preprocessor, device = load_policy(args.policy_path, args.device)
    quantized, _, _ = load_policy(args.policy_path, args.device)
    include_prefixes = tuple(args.quantize_module_prefix) if args.quantize_module_prefix else None
    num_linear = replace_linear_modules(quantized, include_prefixes=include_prefixes)
    quantized.to(device)
    quantized.eval()

    raw_observation = make_raw_observation(original.config, args.task)
    batch = preprocessor(raw_observation)
    bsize = batch[OBS_STATE].shape[0]
    noise_shape = (bsize, original.config.chunk_size, original.config.max_action_dim)
    noise = original.model.sample_noise(noise_shape, batch[OBS_STATE].device)

    with torch.no_grad(), capture_module_outputs(original, args.capture) as original_outputs:
        original_action = run_sample_actions(original, clone_batch(batch), noise.clone())
    with torch.no_grad(), torch.autocast(torch.device(device).type, dtype=torch.bfloat16), capture_module_outputs(
        quantized, args.capture
    ) as quantized_outputs:
        quantized_action = run_sample_actions(quantized, clone_batch(batch), noise.clone())

    rows = compare_outputs(original_outputs, quantized_outputs)
    final_row = comparison_row("PyTorch.quantized.action_chunk", original_action, quantized_action)
    first_action_row = comparison_row(
        "PyTorch.quantized.action_chunk.first_action",
        original_action[:, 0, :],
        quantized_action[:, 0, :],
        note="First action in the chunk; this is the immediate action consumed by queue-based policy execution.",
    )
    assert final_row is not None and first_action_row is not None
    rows.insert(0, final_row)
    rows.insert(1, first_action_row)

    trt_row = None
    trt_first_action_row = None
    if args.engine_path:
        trt_row, trt_first_action_row = run_trt_final_compare(original, clone_batch(batch), noise.clone(), args.engine_path)
        rows.insert(2, trt_row)
        rows.insert(3, trt_first_action_row)

    cosine_values = [
        float(row["cosine_similarity"])
        for row in rows
        if not str(row["module"]).startswith("TensorRT.engine.action_chunk")
    ]
    summary = {
        "policy_path": args.policy_path,
        "engine_path": args.engine_path,
        "device": device,
        "seed": args.seed,
        "capture": args.capture,
        "num_linear_replaced": num_linear,
        "quantize_module_prefixes": list(include_prefixes) if include_prefixes else None,
        "num_modules_compared": len(rows),
        "min_module_cosine_similarity": min(cosine_values) if cosine_values else None,
        "mean_module_cosine_similarity": sum(cosine_values) / len(cosine_values) if cosine_values else None,
        "final_pytorch_quantized_cosine_similarity": final_row["cosine_similarity"],
        "final_pytorch_quantized_relative_l2_error": final_row["relative_l2_error"],
        "final_trt_engine_cosine_similarity": trt_row["cosine_similarity"] if trt_row else None,
        "final_trt_engine_relative_l2_error": trt_row["relative_l2_error"] if trt_row else None,
        "final_pytorch_quantized_l2_norm_ratio": final_row["l2_norm_ratio"],
        "final_pytorch_quantized_relative_l2_norm_error": final_row["relative_l2_norm_error"],
        "final_trt_engine_l2_norm_ratio": trt_row["l2_norm_ratio"] if trt_row else None,
        "final_trt_engine_relative_l2_norm_error": trt_row["relative_l2_norm_error"] if trt_row else None,
        "final_pytorch_quantized_mean_diff": final_row["mean_diff"],
        "final_pytorch_quantized_std_ratio": final_row["std_ratio"],
        "final_pytorch_quantized_max_channel_abs_error": final_row["max_channel_abs_error"],
        "final_pytorch_quantized_max_channel_abs_error_index": final_row["max_channel_abs_error_index"],
        "final_pytorch_quantized_max_channel_relative_l2_error": final_row["max_channel_relative_l2_error"],
        "final_pytorch_quantized_max_channel_relative_l2_error_index": final_row[
            "max_channel_relative_l2_error_index"
        ],
        "final_trt_engine_mean_diff": trt_row["mean_diff"] if trt_row else None,
        "final_trt_engine_std_ratio": trt_row["std_ratio"] if trt_row else None,
        "final_trt_engine_max_channel_abs_error": trt_row["max_channel_abs_error"] if trt_row else None,
        "final_trt_engine_max_channel_abs_error_index": trt_row["max_channel_abs_error_index"] if trt_row else None,
        "final_trt_engine_max_channel_relative_l2_error": trt_row["max_channel_relative_l2_error"]
        if trt_row
        else None,
        "final_trt_engine_max_channel_relative_l2_error_index": trt_row["max_channel_relative_l2_error_index"]
        if trt_row
        else None,
        "first_action_pytorch_quantized_cosine_similarity": first_action_row["cosine_similarity"],
        "first_action_pytorch_quantized_relative_l2_error": first_action_row["relative_l2_error"],
        "first_action_pytorch_quantized_l2_norm_ratio": first_action_row["l2_norm_ratio"],
        "first_action_pytorch_quantized_mean_diff": first_action_row["mean_diff"],
        "first_action_pytorch_quantized_std_ratio": first_action_row["std_ratio"],
        "first_action_pytorch_quantized_max_channel_abs_error": first_action_row["max_channel_abs_error"],
        "first_action_pytorch_quantized_max_channel_abs_error_index": first_action_row[
            "max_channel_abs_error_index"
        ],
        "first_action_trt_engine_cosine_similarity": trt_first_action_row["cosine_similarity"]
        if trt_first_action_row
        else None,
        "first_action_trt_engine_relative_l2_error": trt_first_action_row["relative_l2_error"]
        if trt_first_action_row
        else None,
        "first_action_trt_engine_l2_norm_ratio": trt_first_action_row["l2_norm_ratio"]
        if trt_first_action_row
        else None,
        "first_action_trt_engine_mean_diff": trt_first_action_row["mean_diff"] if trt_first_action_row else None,
        "first_action_trt_engine_std_ratio": trt_first_action_row["std_ratio"] if trt_first_action_row else None,
        "first_action_trt_engine_max_channel_abs_error": trt_first_action_row["max_channel_abs_error"]
        if trt_first_action_row
        else None,
        "first_action_trt_engine_max_channel_abs_error_index": trt_first_action_row["max_channel_abs_error_index"]
        if trt_first_action_row
        else None,
        "note": "Per-module similarity compares original PyTorch vs W8A8 PyTorch emulation. TensorRT engine exposes final action_chunk only.",
    }
    write_reports(rows, summary, output_dir)


if __name__ == "__main__":
    main()
