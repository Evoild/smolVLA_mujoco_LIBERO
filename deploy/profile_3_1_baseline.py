#!/usr/bin/env python3

"""Step 3-1 baseline profiler for SmolVLA forward latency, module timing, and peak GPU memory."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
from torch import nn

from lerobot.configs import FeatureType, PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_STATE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="runs/deploy/3-1baseline/profile")
    parser.add_argument("--task", default="libero_spatial baseline forward profile")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30)
    return parser.parse_args()


def sync(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def peak_memory_gb(device: str) -> float | None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024**3)
    return None


def summarize(values: list[float], prefix: str) -> dict[str, float]:
    if not values:
        return {
            f"{prefix}_mean_ms": 0.0,
            f"{prefix}_median_ms": 0.0,
            f"{prefix}_p95_ms": 0.0,
        }
    sorted_values = sorted(values)
    return {
        f"{prefix}_mean_ms": statistics.fmean(values),
        f"{prefix}_median_ms": statistics.median(values),
        f"{prefix}_p95_ms": sorted_values[max(0, int(len(sorted_values) * 0.95) - 1)],
    }


def load_policy(policy_path: str, device: str):
    cfg = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=[f"--device={device}"])
    policy_cls = get_policy_class(cfg.type)
    policy = policy_cls.from_pretrained(policy_path, config=cfg, strict=False)
    policy.eval()
    preprocessor, _ = make_pre_post_processors(policy.config, pretrained_path=policy_path)
    return policy, preprocessor


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


def add_dummy_action(policy: nn.Module, batch: dict[str, Any]) -> dict[str, Any]:
    batch = clone_batch(batch)
    state = batch[OBS_STATE]
    batch_size = state.shape[0]
    action_dim = int(policy.config.action_feature.shape[0])
    chunk_size = int(policy.config.chunk_size)
    batch[ACTION] = torch.zeros((batch_size, chunk_size, action_dim), dtype=torch.float32, device=state.device)
    return batch


def is_leaf_module(module: nn.Module) -> bool:
    return not any(module.children())


def add_leaf_hooks(
    module: nn.Module,
    group_name: str,
    timer_state: dict[str, Any],
    handles: list[Any],
    device: str,
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


def add_direct_hook(
    module: nn.Module,
    group_name: str,
    timer_state: dict[str, Any],
    handles: list[Any],
    device: str,
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
    add_leaf_hooks(model.vlm_with_expert.vlm, "vlm", timer_state, handles, device)
    add_leaf_hooks(model.vlm_with_expert.lm_expert, "lm_expert", timer_state, handles, device)
    add_direct_hook(model.state_proj, "state_proj", timer_state, handles, device)
    add_direct_hook(model.action_in_proj, "action_in_proj", timer_state, handles, device)
    add_direct_hook(model.action_time_mlp_in, "action_time_mlp", timer_state, handles, device)
    add_direct_hook(model.action_time_mlp_out, "action_time_mlp", timer_state, handles, device)
    add_direct_hook(model.action_out_proj, "action_out_proj", timer_state, handles, device)
    try:
        yield groups
    finally:
        if timer_state["use_cuda_events"]:
            torch.cuda.synchronize()
            for group_name, start_event, end_event in timer_state["pending_events"]:
                groups[group_name] += start_event.elapsed_time(end_event)
        for handle in handles:
            handle.remove()


def run_forward(policy: nn.Module, batch: dict[str, Any]) -> None:
    with torch.no_grad():
        policy.forward(batch)


def benchmark(policy: nn.Module, batch: dict[str, Any], device: str, warmup: int, iters: int) -> list[dict[str, Any]]:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    for _ in range(warmup):
        run_forward(policy, clone_batch(batch))
    sync(device)

    rows: list[dict[str, Any]] = []
    for idx in range(iters):
        with module_timer(policy, device) as module_ms:
            start = time.perf_counter()
            run_forward(policy, clone_batch(batch))
            sync(device)
            e2e_ms = (time.perf_counter() - start) * 1000.0

        measured_modules_ms = sum(module_ms.values())
        row = {
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
        rows.append(row)
    return rows


def write_reports(rows: list[dict[str, Any]], output_dir: Path, device: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

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
    summary: dict[str, Any] = {"device": device, "peak_gpu_memory_gb": peak_memory_gb(device)}
    for key in metric_keys:
        summary.update(summarize([float(row[key]) for row in rows], key.removesuffix("_ms")))
    if rows:
        summary["forward_fps"] = 1000.0 / summary["forward_e2e_mean_ms"]

    with open(output_dir / "forward_profile_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(output_dir / "forward_profile_iterations.json", "w") as f:
        json.dump(rows, f, indent=2)

    with open(output_dir / "forward_profile_iterations.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["iteration"])
        writer.writeheader()
        writer.writerows(rows)

    with open(output_dir / "forward_profile_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    print(json.dumps(summary, indent=2))


def main() -> None:
    args = parse_args()
    policy, preprocessor = load_policy(args.policy_path, args.device)
    raw_observation = make_raw_observation(policy.config, args.task)
    batch = add_dummy_action(policy, preprocessor(raw_observation))
    rows = benchmark(policy, batch, args.device, args.warmup, args.iters)
    write_reports(rows, Path(args.output_dir), args.device)


if __name__ == "__main__":
    main()
