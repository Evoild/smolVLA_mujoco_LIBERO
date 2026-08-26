#!/usr/bin/env python3

"""Collect per-channel Linear activation scales for selected SmolVLA modules."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from export_smolvla_sample_actions_onnx import SmolVLASampleActionsFromEmbedsWrapper, load_policy
from lerobot.envs import make_env, make_env_pre_post_processors, preprocess_observation
from lerobot.envs.factory import make_env_config
from lerobot.policies import make_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
from lerobot.utils.random_utils import set_seed


DEFAULT_REGEX = r"^model\.vlm_with_expert\.vlm\.model\.text_model\.layers\.[0-9]+\.mlp\.(gate_proj|up_proj|down_proj)$"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tasks", default="libero_goal")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--token-length", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-parallel-tasks", type=int, default=1)
    parser.add_argument("--percentile", type=float, default=99.99)
    parser.add_argument("--include-module-regex", default=DEFAULT_REGEX)
    return parser.parse_args()


def pad_or_truncate(tensor: torch.Tensor, shape: tuple[int, ...], pad_value: int | float | bool = 0) -> torch.Tensor:
    if tuple(tensor.shape) == shape:
        return tensor
    output = torch.full(shape, pad_value, dtype=tensor.dtype, device=tensor.device)
    slices = tuple(slice(0, min(got, want)) for got, want in zip(tensor.shape, shape, strict=True))
    output[slices] = tensor[slices]
    return output


class PerChannelLinearInputCollector:
    def __init__(self, percentile: float, include_module_regex: str):
        self.percentile = percentile
        self.include_pattern = re.compile(include_module_regex)
        self.include_module_regex = include_module_regex
        self.handles: list[Any] = []
        self.current: list[dict[str, Any]] = []
        self.calls: list[dict[str, Any]] = []

    def hook_modules(self, module: nn.Module) -> None:
        for name, child in module.named_modules():
            if isinstance(child, nn.Linear) and self.include_pattern.search(name):
                self.handles.append(child.register_forward_pre_hook(self._make_hook(name)))

    def _make_hook(self, name: str):
        def hook(_module, inputs):
            tensor = inputs[0].detach().to(torch.float32)
            if tensor.shape[-1] <= 0:
                raise RuntimeError(f"empty last dimension for {name}: {tuple(tensor.shape)}")
            flat = tensor.abs().reshape(-1, tensor.shape[-1])
            amax = flat.amax(dim=0).clamp_min(1.0e-8)
            percentile_amax = torch.quantile(flat, self.percentile / 100.0, dim=0).clamp_min(1.0e-8)
            self.current.append(
                {
                    "module": name,
                    "shape": list(tensor.shape),
                    "amax": amax.cpu(),
                    "percentile_amax": percentile_amax.cpu(),
                }
            )

        return hook

    def begin_sample(self) -> None:
        self.current = []

    def end_sample(self) -> None:
        if not self.calls:
            self.calls = [
                {
                    "index": idx,
                    "module": item["module"],
                    "shape": item["shape"],
                    "amax": item["amax"],
                    "percentile_amax": item["percentile_amax"],
                    "num_samples": 1,
                }
                for idx, item in enumerate(self.current)
            ]
            return
        if len(self.current) != len(self.calls):
            raise RuntimeError(f"Linear call count changed: got {len(self.current)}, expected {len(self.calls)}")
        for idx, item in enumerate(self.current):
            call = self.calls[idx]
            call["amax"] = torch.maximum(call["amax"], item["amax"])
            call["percentile_amax"] = torch.maximum(call["percentile_amax"], item["percentile_amax"])
            call["num_samples"] = int(call["num_samples"]) + 1

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()

    def scales(self) -> list[dict[str, Any]]:
        rows = []
        for call in self.calls:
            amax = call["amax"].to(torch.float32)
            percentile_amax = call["percentile_amax"].to(torch.float32)
            rows.append(
                {
                    "index": call["index"],
                    "module": call["module"],
                    "shape": call["shape"],
                    "channels": int(percentile_amax.numel()),
                    "amax_max": float(amax.max().item()),
                    "percentile_amax_max": float(percentile_amax.max().item()),
                    "num_samples": call["num_samples"],
                    "scale": (percentile_amax / 127.0).clamp_min(1.0e-8).tolist(),
                    "scale_source": f"per_channel_p{self.percentile}",
                    "amax_scale": (amax / 127.0).clamp_min(1.0e-8).tolist(),
                }
            )
        return rows


def make_core_inputs(policy: nn.Module, batch: dict[str, torch.Tensor], token_length: int):
    images, img_masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)
    lang_tokens = batch[OBS_LANGUAGE_TOKENS]
    lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
    bsize = state.shape[0]
    noise_shape = (bsize, policy.config.chunk_size, policy.config.max_action_dim)
    noise = policy.model.sample_noise(noise_shape, state.device)

    with torch.no_grad():
        image_emb = policy.model.vlm_with_expert.embed_image(images[0])
        image2_emb = policy.model.vlm_with_expert.embed_image(images[1])

    lang_tokens = pad_or_truncate(lang_tokens, (bsize, token_length), 0)
    lang_masks = pad_or_truncate(lang_masks, (bsize, token_length), False)
    return (
        image_emb.to(torch.bfloat16),
        image2_emb.to(torch.bfloat16),
        img_masks[0],
        img_masks[1],
        state.to(torch.bfloat16),
        lang_tokens,
        lang_masks,
        noise.to(torch.bfloat16),
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    policy, device = load_policy(args.policy_path, args.device, torch.bfloat16)
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=args.policy_path,
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    wrapper = SmolVLASampleActionsFromEmbedsWrapper(policy).to(device=device).eval()

    collector = PerChannelLinearInputCollector(args.percentile, args.include_module_regex)
    collector.hook_modules(wrapper)

    env_cfg = make_env_config("libero", task=args.tasks, max_parallel_tasks=args.max_parallel_tasks)
    envs = make_env(env_cfg, n_envs=args.batch_size, use_async_envs=False, trust_remote_code=False)
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy.config)
    task_envs = [(task_group, task_id, env) for task_group, group in envs.items() for task_id, env in group.items()]
    if not task_envs:
        raise RuntimeError("No LIBERO environments were created for calibration")
    samples_per_task = max(1, (args.samples + len(task_envs) - 1) // len(task_envs))
    samples_collected = 0
    try:
        for task_offset, (_task_group, _task_id, env) in enumerate(task_envs):
            if samples_collected >= args.samples:
                break
            policy.reset()
            seed_base = args.seed + task_offset * 1000
            observation, _info = env.reset(seed=[seed_base + idx for idx in range(args.batch_size)])
            done = np.array([False] * env.num_envs)
            max_steps = int(env.call("_max_episode_steps")[0])
            step = 0
            task_samples = 0
            while samples_collected < args.samples and task_samples < samples_per_task and step < max_steps:
                observation = preprocess_observation(observation)
                try:
                    observation["task"] = list(env.call("task_description"))
                except (AttributeError, NotImplementedError):
                    observation["task"] = list(env.call("task"))
                observation = env_preprocessor(observation)
                batch = preprocessor(observation)
                core_inputs = make_core_inputs(policy, batch, args.token_length)
                with torch.no_grad():
                    collector.begin_sample()
                    action_chunk = wrapper(*core_inputs)
                    collector.end_sample()
                samples_collected += 1
                task_samples += 1

                action = action_chunk[:, 0, : int(policy.config.action_feature.shape[0])].to(torch.float32)
                action = postprocessor(action)
                action_transition = env_postprocessor({ACTION: action})
                observation, _reward, terminated, truncated, _info = env.step(
                    action_transition[ACTION].to("cpu").numpy()
                )
                done = done | terminated | truncated
                if np.all(done):
                    policy.reset()
                    observation, _info = env.reset(
                        seed=[seed_base + samples_collected + idx for idx in range(args.batch_size)]
                    )
                    done = np.array([False] * env.num_envs)
                step += 1
        if samples_collected == 0:
            raise RuntimeError("Calibration did not collect any samples")
    finally:
        collector.close()
        for _task_group, _task_id, env in task_envs:
            try:
                env.close()
            except Exception:
                pass

    report = {
        "policy_path": args.policy_path,
        "tasks": args.tasks,
        "seed": args.seed,
        "samples_requested": args.samples,
        "samples_collected": samples_collected,
        "token_length": args.token_length,
        "percentile": args.percentile,
        "scale_granularity": "per_channel_last_dim",
        "include_module_regex": args.include_module_regex,
        "scale_formula": (
            "scale[channel] = max_over_samples(percentile(abs(linear_input[..., channel]))) / 127"
        ),
        "linear_call_scales": collector.scales(),
    }
    with open(output, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "linear_call_scales"}, indent=2))
    print(f"linear_call_scales: {len(report['linear_call_scales'])}")


if __name__ == "__main__":
    main()
