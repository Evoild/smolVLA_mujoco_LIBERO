#!/usr/bin/env python3

"""Calibrate asymmetric down_proj activation ranges after SmoothQuant smoothing."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from calibrate_smolvla_activation_scales import make_core_inputs
from export_smolvla_sample_actions_onnx import SmolVLASampleActionsFromEmbedsWrapper, load_policy
from lerobot.envs import make_env, make_env_pre_post_processors, preprocess_observation
from lerobot.envs.factory import make_env_config
from lerobot.policies import make_pre_post_processors
from lerobot.utils.constants import ACTION
from lerobot.utils.random_utils import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", required=True)
    parser.add_argument("--smoothquant-scales-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tasks", default="libero_spatial")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--token-length", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-parallel-tasks", type=int, default=1)
    parser.add_argument("--percentile", type=float, default=99.995)
    parser.add_argument(
        "--down-module-regex",
        default=r"^model\.vlm_with_expert\.vlm\.model\.text_model\.layers\.[0-9]+\.mlp\.down_proj$",
    )
    return parser.parse_args()


class SmoothedDownCollector:
    def __init__(self, smooth_entries: dict[str, dict[str, Any]], down_module_regex: str, percentile: float):
        self.smooth_entries = smooth_entries
        self.down_pattern = re.compile(down_module_regex)
        self.down_module_regex = down_module_regex
        self.percentile = percentile
        self.handles: list[Any] = []
        self.current: list[dict[str, Any]] = []
        self.calls: list[dict[str, Any]] = []

    def hook_modules(self, module: nn.Module) -> None:
        for name, child in module.named_modules():
            if isinstance(child, nn.Linear) and self.down_pattern.search(name):
                if name not in self.smooth_entries:
                    raise KeyError(f"missing SmoothQuant entry for down module: {name}")
                self.handles.append(child.register_forward_pre_hook(self._make_hook(name)))

    def _make_hook(self, name: str):
        smooth_scale = torch.as_tensor(self.smooth_entries[name]["smooth_scale"], dtype=torch.float32).clamp_min(1.0e-8)

        def hook(_module, inputs):
            tensor = inputs[0].detach().to(torch.float32)
            scale = smooth_scale.to(device=tensor.device, dtype=torch.float32)
            smoothed = tensor / scale
            flat = smoothed.flatten()
            if flat.numel() == 0:
                lower = 0.0
                upper = 1.0e-8
            else:
                lower = float(torch.quantile(flat, (100.0 - self.percentile) / 100.0).item())
                upper = float(torch.quantile(flat, self.percentile / 100.0).item())
            self.current.append({"module": name, "lower_percentile": lower, "upper_percentile": upper})

        return hook

    def begin_sample(self) -> None:
        self.current = []

    def end_sample(self) -> None:
        if not self.calls:
            self.calls = [
                {
                    "module": item["module"],
                    "lower_percentile": item["lower_percentile"],
                    "upper_percentile": item["upper_percentile"],
                    "num_samples": 1,
                }
                for item in self.current
            ]
            return
        if len(self.current) != len(self.calls):
            raise RuntimeError(f"Linear call count changed: got {len(self.current)}, expected {len(self.calls)}")
        for idx, item in enumerate(self.current):
            call = self.calls[idx]
            if call["module"] != item["module"]:
                raise RuntimeError(f"Linear call order changed: got {item['module']}, expected {call['module']}")
            call["lower_percentile"] = min(float(call["lower_percentile"]), item["lower_percentile"])
            call["upper_percentile"] = max(float(call["upper_percentile"]), item["upper_percentile"])
            call["num_samples"] = int(call["num_samples"]) + 1

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()

    def asymmetric_by_module(self) -> dict[str, dict[str, Any]]:
        rows = {}
        qmin = -128
        qmax = 127
        for call in self.calls:
            xmin = float(call["lower_percentile"])
            xmax = float(call["upper_percentile"])
            if xmax <= xmin:
                xmin = min(xmin, 0.0)
                xmax = max(xmax, 1.0e-8)
            scale = max((xmax - xmin) / float(qmax - qmin), 1.0e-8)
            zero_point = int(round(qmin - xmin / scale))
            zero_point = max(qmin, min(qmax, zero_point))
            rows[str(call["module"])] = {
                "scale": scale,
                "zero_point": zero_point,
                "qmin": qmin,
                "qmax": qmax,
                "calibrated_min": xmin,
                "calibrated_max": xmax,
                "num_samples": int(call["num_samples"]),
            }
        return rows


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(args.smoothquant_scales_json) as f:
        smoothquant = json.load(f)
    entries = smoothquant.get("linear_call_scales", [])
    smooth_entries = {str(entry["module"]): entry for entry in entries}

    policy, device = load_policy(args.policy_path, args.device, torch.bfloat16)
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=args.policy_path,
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    wrapper = SmolVLASampleActionsFromEmbedsWrapper(policy).to(device=device).eval()

    collector = SmoothedDownCollector(smooth_entries, args.down_module_regex, args.percentile)
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
                observation, _reward, terminated, truncated, _info = env.step(action_transition[ACTION].to("cpu").numpy())
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

    down_ranges = collector.asymmetric_by_module()
    down_pattern = re.compile(args.down_module_regex)
    merged_entries = []
    for entry in entries:
        module_name = str(entry.get("module", ""))
        if down_pattern.search(module_name):
            if module_name not in down_ranges:
                raise KeyError(f"missing calibrated down range for {module_name}")
            override = down_ranges[module_name]
            merged = {
                **entry,
                "scale": override["scale"],
                "scale_source": f"smoothquant_asymmetric_p{100.0 - args.percentile:g}_p{args.percentile:g}",
                "quantization": "smoothquant_asymmetric_static_per_tensor",
                "zero_point": override["zero_point"],
                "qmin": override["qmin"],
                "qmax": override["qmax"],
                "calibrated_min": override["calibrated_min"],
                "calibrated_max": override["calibrated_max"],
                "num_samples": override["num_samples"],
            }
            merged_entries.append(merged)
        else:
            merged_entries.append(entry)

    report = {
        **{key: value for key, value in smoothquant.items() if key != "linear_call_scales"},
        "source_smoothquant_scales_json": args.smoothquant_scales_json,
        "down_asymmetric_module_regex": args.down_module_regex,
        "down_asymmetric_percentile": args.percentile,
        "samples_requested": args.samples,
        "samples_collected": samples_collected,
        "scale_granularity": "smoothquant_per_tensor_activation_with_down_asymmetric_activation",
        "linear_call_scales": merged_entries,
    }
    with open(output, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "linear_call_scales"}, indent=2))
    print(f"linear_call_scales: {len(merged_entries)}")
    print(f"down_asymmetric_entries: {len(down_ranges)}")


if __name__ == "__main__":
    main()
