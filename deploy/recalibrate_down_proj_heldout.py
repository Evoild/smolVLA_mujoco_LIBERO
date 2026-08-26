#!/usr/bin/env python3

"""Recalibrate down_proj activation scales and validate on held-out rollout samples."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch

from diagnose_3_4_numeric_baseline import SmolVLADebugCoreWrapper, core_inputs_from_policy, load_policy
from diagnose_down_proj_w8a8 import (
    DOWN_REGEX,
    action_rows,
    clipping_and_input_error_rows,
    flow_rows,
    output_error_rows,
    qdq_static,
    run_with_hooks,
    scale_for_module,
    vector_metrics,
)
from lerobot.envs import make_env, make_env_pre_post_processors, preprocess_observation
from lerobot.envs.factory import make_env_config
from lerobot.policies import make_pre_post_processors
from lerobot.utils.constants import ACTION
from lerobot.utils.random_utils import set_seed
from linear_only_quant import replace_linear_modules


PERCENTILES = [99.0, 99.9, 99.95, 99.99, 99.995, 99.999]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", default="smolvla_libero")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tasks", default="libero_goal")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--token-length", type=int, default=48)
    parser.add_argument("--input-dtype", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--calibration-sizes", default="32,64,128,256,512")
    parser.add_argument("--heldout-samples", type=int, default=50)
    parser.add_argument("--sample-stride", type=int, default=5)
    parser.add_argument("--max-parallel-tasks", type=int, default=1)
    parser.add_argument("--down-module-regex", default=DOWN_REGEX)
    parser.add_argument("--default-percentile", type=float, default=99.99)
    parser.add_argument("--percentile-sweep-sizes", default="256,512")
    parser.add_argument("--sign-flip-threshold", type=float, default=1.0e-3)
    parser.add_argument("--targeted-clip-threshold", type=float, default=0.05)
    parser.add_argument("--targeted-ratio-threshold", type=float, default=0.5)
    parser.add_argument("--targeted-factors", default="1.1,1.25,1.5,2.0")
    return parser.parse_args()


def parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def cpu_inputs(inputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    return tuple(t.detach().cpu() for t in inputs)


def device_inputs(inputs: tuple[torch.Tensor, ...], device: str) -> tuple[torch.Tensor, ...]:
    return tuple(t.to(device=device) for t in inputs)


def make_core_inputs_from_batch(policy, batch: dict[str, torch.Tensor], token_length: int):
    from calibrate_smolvla_activation_channel_scales import make_core_inputs

    return make_core_inputs(policy, batch, token_length)


def collect_rollout_inputs(args: argparse.Namespace, total_samples: int):
    set_seed(args.seed)
    policy, preprocessor, device = load_policy(args.policy_path, args.device)
    post_preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=args.policy_path,
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    del post_preprocessor

    env_cfg = make_env_config("libero", task=args.tasks, max_parallel_tasks=args.max_parallel_tasks)
    envs = make_env(env_cfg, n_envs=args.batch_size, use_async_envs=False, trust_remote_code=False)
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy.config)
    task_envs = [(task_group, task_id, env) for task_group, group in envs.items() for task_id, env in group.items()]
    if not task_envs:
        raise RuntimeError("No LIBERO environments were created")

    records = []
    try:
        while len(records) < total_samples:
            for task_offset, (task_group, task_id, env) in enumerate(task_envs):
                if len(records) >= total_samples:
                    break
                policy.reset()
                seed_base = args.seed + task_offset * 1000 + (len(records) // max(1, len(task_envs))) * 100000
                observation, _info = env.reset(seed=[seed_base + idx for idx in range(args.batch_size)])
                done = np.array([False] * env.num_envs)
                max_steps = int(env.call("_max_episode_steps")[0])
                for step in range(max_steps):
                    observation = preprocess_observation(observation)
                    try:
                        observation["task"] = list(env.call("task_description"))
                    except (AttributeError, NotImplementedError):
                        observation["task"] = list(env.call("task"))
                    observation = env_preprocessor(observation)
                    batch = preprocessor(observation)
                    if step % max(1, args.sample_stride) == 0 and len(records) < total_samples:
                        core_inputs = make_core_inputs_from_batch(policy, batch, args.token_length)
                        records.append(
                            {
                                "index": len(records),
                                "task_group": task_group,
                                "task_id": task_id,
                                "episode_seed": seed_base,
                                "timestep": step,
                                "core_inputs": cpu_inputs(core_inputs),
                            }
                        )
                    if len(records) >= total_samples:
                        break
                    with torch.no_grad():
                        core_inputs = make_core_inputs_from_batch(policy, batch, args.token_length)
                        action_chunk = SmolVLADebugCoreWrapper(policy).to(device=device).eval()(*core_inputs)[0]
                    action = action_chunk[:, 0, : int(policy.config.action_feature.shape[0])].to(torch.float32)
                    action = postprocessor(action)
                    action_transition = env_postprocessor({ACTION: action})
                    observation, _reward, terminated, truncated, _info = env.step(
                        action_transition[ACTION].to("cpu").numpy()
                    )
                    done = done | terminated | truncated
                    if np.all(done):
                        break
    finally:
        for _task_group, _task_id, env in task_envs:
            try:
                env.close()
            except Exception:
                pass
    return records, device


class DownStatsAccumulator:
    def __init__(self, percentiles: list[float]):
        self.percentiles = percentiles
        self.modules: dict[str, dict[str, Any]] = {}
        self.sample_count = 0

    def update(self, down_inputs: dict[str, torch.Tensor]) -> None:
        self.sample_count += 1
        for module, x in down_inputs.items():
            view = x.detach().to(torch.float32).reshape(-1, x.shape[-1]).abs()
            channels = view.shape[-1]
            if module not in self.modules:
                self.modules[module] = {
                    "channels": channels,
                    "count": 0,
                    "sum": torch.zeros(channels),
                    "sum_sq": torch.zeros(channels),
                    "max_abs": torch.zeros(channels),
                    "percentiles": {str(p): torch.zeros(channels) for p in self.percentiles},
                }
            item = self.modules[module]
            item["count"] += int(view.shape[0])
            item["sum"] += view.sum(dim=0).cpu()
            item["sum_sq"] += (view * view).sum(dim=0).cpu()
            item["max_abs"] = torch.maximum(item["max_abs"], view.amax(dim=0).cpu())
            for p in self.percentiles:
                q = torch.quantile(view, p / 100.0, dim=0).cpu()
                item["percentiles"][str(p)] = torch.maximum(item["percentiles"][str(p)], q)

    def snapshot(self) -> dict[str, Any]:
        modules = {}
        for module, item in self.modules.items():
            count = max(int(item["count"]), 1)
            mean = item["sum"] / count
            var = (item["sum_sq"] / count - mean * mean).clamp_min(0)
            modules[module] = {
                "channels": int(item["channels"]),
                "count": int(item["count"]),
                "mean_abs": mean.tolist(),
                "std_abs": torch.sqrt(var).tolist(),
                "max_abs": item["max_abs"].tolist(),
                "percentiles": {p: tensor.tolist() for p, tensor in item["percentiles"].items()},
            }
        return {"samples": self.sample_count, "modules": modules}


def scales_from_snapshot(snapshot: dict[str, Any], percentile: str) -> dict[str, Any]:
    scales = {}
    for module, item in snapshot["modules"].items():
        if percentile == "max":
            ranges = torch.tensor(item["max_abs"], dtype=torch.float32)
        else:
            ranges = torch.tensor(item["percentiles"][str(float(percentile))], dtype=torch.float32)
        scales[module] = (ranges.clamp_min(1.0e-8) / 127.0).tolist()
    return scales


def write_scale_json(path: Path, args: argparse.Namespace, scales: dict[str, Any], samples: int, percentile: str) -> None:
    rows = []
    for idx, module in enumerate(sorted(scales, key=layer_sort_key)):
        scale = torch.tensor(scales[module], dtype=torch.float32)
        rows.append(
            {
                "index": idx,
                "module": module,
                "call_index": 0,
                "shape": [None, None, int(scale.numel())],
                "channels": int(scale.numel()),
                "scale": scale.tolist(),
                "scale_source": f"down_recalibration_p{percentile}_samples_{samples}",
                "scale_min": float(scale.min()),
                "scale_max": float(scale.max()),
                "scale_median": float(scale.median()),
            }
        )
    report = {
        "policy_path": args.policy_path,
        "tasks": args.tasks,
        "seed": args.seed,
        "samples_collected": samples,
        "percentile": percentile,
        "scale_granularity": "per_channel_last_dim",
        "include_module_regex": args.down_module_regex,
        "scale_formula": "scale[channel] = max_over_calibration_samples(percentile(abs(linear_input[..., channel]))) / 127",
        "linear_call_scales": rows,
    }
    with open(path, "w") as f:
        json.dump(report, f, indent=2)


def layer_sort_key(module: str) -> int:
    match = re.search(r"\.layers\.([0-9]+)\.", module)
    return int(match.group(1)) if match else 10**9


def load_down_policy(policy_path: str, device: str, regex: str, scales: dict[str, Any]):
    policy, _, effective_device = load_policy(policy_path, device)
    replace_linear_modules(
        policy,
        include_prefixes=None,
        include_regexes=(regex,),
        activation_scales_by_module=scales,
        quantization_kind="w8a8",
    )
    return policy.to(device=effective_device).eval(), effective_device


def precompute_baseline(policy, records: list[dict[str, Any]], device: str, down_regex: str):
    outputs = []
    for record in records:
        core_inputs = device_inputs(record["core_inputs"], device)
        out, cap = run_with_hooks(policy, core_inputs, down_regex)
        outputs.append({"outputs": out, "capture": cap, "meta": {k: v for k, v in record.items() if k != "core_inputs"}})
    return outputs


def p95(values: list[float]) -> float:
    if not values:
        return float("nan")
    tensor = torch.tensor(values, dtype=torch.float32)
    return float(torch.quantile(tensor, 0.95))


def summarize_values(rows: list[dict[str, Any]], key: str) -> tuple[float, float, float]:
    values = [float(row[key]) for row in rows if key in row and math.isfinite(float(row[key]))]
    if not values:
        return float("nan"), float("nan"), float("nan")
    return float(sum(values) / len(values)), p95(values), float(max(values))


def validate_config(
    args: argparse.Namespace,
    label: str,
    scales: dict[str, Any],
    baseline: list[dict[str, Any]],
    heldout_records: list[dict[str, Any]],
    device: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    policy, effective_device = load_down_policy(args.policy_path, device, args.down_module_regex, scales)
    input_error_rows = []
    clipping_rows = []
    output_error_rows_all = []
    flow_error_rows = []
    action_l2 = []
    action_cos = []
    first_action_l2 = []
    timestep_l2 = []
    sign_flip = []
    mismatch_rows = []

    for idx, (base, record) in enumerate(zip(baseline, heldout_records, strict=True)):
        core_inputs = device_inputs(record["core_inputs"], effective_device)
        quant_outputs, quant_capture = run_with_hooks(policy, core_inputs, args.down_module_regex)
        fp_outputs = base["outputs"]
        fp_capture = base["capture"]

        clip, input_err, _ = clipping_and_input_error_rows(fp_capture.down_inputs, scales, topk=1)
        for row in clip:
            clipping_rows.append({"config": label, "heldout_index": idx, **base["meta"], **row})
        for row in input_err:
            input_error_rows.append({"config": label, "heldout_index": idx, **base["meta"], **row})
        output_err = output_error_rows(
            fp_capture,
            DummyCapture({}),
            quant_capture,
            {row["module"]: row["relative_l2_error"] for row in input_err},
        )
        for row in output_err:
            output_error_rows_all.append({"config": label, "heldout_index": idx, **base["meta"], **row})

        flow = flow_rows(fp_outputs, quant_outputs)
        for row in flow:
            flow_error_rows.append({"config": label, "heldout_index": idx, **base["meta"], **row})

        action = vector_metrics(fp_outputs["action_chunk"], quant_outputs["action_chunk"])
        action_l2.append(action["relative_l2_error"])
        action_cos.append(action["cosine_similarity"])
        first_action_l2.append(vector_metrics(fp_outputs["action_chunk"][:, 0], quant_outputs["action_chunk"][:, 0])["relative_l2_error"])
        t_rows, _dim_rows, action_summary = action_rows(
            fp_outputs["action_chunk"], quant_outputs["action_chunk"], args.sign_flip_threshold
        )
        timestep_l2.extend([row["relative_l2_error"] for row in t_rows])
        sign_flip.append(action_summary["meaningful_sign_flip_rate"])
        mismatch_rows.extend(scale_mismatch_rows(args, label, idx, fp_capture.down_inputs, scales, clip, input_err))

    down_in_mean, down_in_p95, down_in_max = summarize_values(input_error_rows, "relative_l2_error")
    down_out_mean, down_out_p95, down_out_max = summarize_values(output_error_rows_all, "relative_l2_error")
    mean_clip, p95_clip, max_clip = summarize_values(clipping_rows, "mean_channel_clipping_rate")
    max_channel_clip = max([float(row["max_channel_clipping_rate"]) for row in clipping_rows] or [float("nan")])
    vt9 = [row for row in flow_error_rows if row["step"] == 9]
    vt9_l2 = [float(row["v_t_relative_l2_error"]) for row in vt9]
    xt9_l2 = [float(row["x_t_relative_l2_error"]) for row in vt9]
    summary = {
        "config": label,
        "heldout_samples": len(heldout_records),
        "mean_channel_clipping_rate": mean_clip,
        "p95_channel_clipping_rate": p95_clip,
        "max_channel_clipping_rate": max_channel_clip,
        "down_input_relative_l2_mean": down_in_mean,
        "down_input_relative_l2_p95": down_in_p95,
        "down_input_relative_l2_max": down_in_max,
        "down_output_relative_l2_mean": down_out_mean,
        "down_output_relative_l2_p95": down_out_p95,
        "down_output_relative_l2_max": down_out_max,
        "v_t_step09_relative_l2": float(sum(vt9_l2) / len(vt9_l2)) if vt9_l2 else float("nan"),
        "x_t_step09_relative_l2": float(sum(xt9_l2) / len(xt9_l2)) if xt9_l2 else float("nan"),
        "action_chunk_cosine": float(sum(action_cos) / len(action_cos)) if action_cos else float("nan"),
        "action_chunk_relative_l2": float(sum(action_l2) / len(action_l2)) if action_l2 else float("nan"),
        "first_action_relative_l2": float(sum(first_action_l2) / len(first_action_l2)) if first_action_l2 else float("nan"),
        "mean_timestep_relative_l2": float(sum(timestep_l2) / len(timestep_l2)) if timestep_l2 else float("nan"),
        "p95_timestep_relative_l2": p95(timestep_l2),
        "max_timestep_relative_l2": float(max(timestep_l2)) if timestep_l2 else float("nan"),
        "meaningful_sign_flip_rate": float(sum(sign_flip) / len(sign_flip)) if sign_flip else float("nan"),
    }
    del policy
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary, clipping_rows, flow_error_rows, mismatch_rows


class DummyCapture:
    def __init__(self, down_outputs: dict[str, torch.Tensor]):
        self.down_outputs = down_outputs


def scale_mismatch_rows(
    args: argparse.Namespace,
    label: str,
    heldout_index: int,
    down_inputs: dict[str, torch.Tensor],
    offline_scales: dict[str, Any],
    clipping_rows: list[dict[str, Any]],
    input_error_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    clip_by_module = {row["module"]: row for row in clipping_rows}
    err_by_module = {row["module"]: row for row in input_error_rows}
    rows = []
    for module, x in down_inputs.items():
        view = x.detach().to(torch.float32).reshape(-1, x.shape[-1]).abs()
        current_scale = torch.quantile(view, args.default_percentile / 100.0, dim=0).clamp_min(1e-8) / 127.0
        offline = torch.tensor(offline_scales[module], dtype=torch.float32).clamp_min(1e-8)
        ratio = offline / current_scale
        qrange = offline * 127.0
        clipped = (view > qrange[None, :]).to(torch.float32).mean(dim=0)
        qview = qdq_static(x, offline).reshape(-1, x.shape[-1])
        fpview = x.detach().to(torch.float32).reshape(-1, x.shape[-1])
        channel_l2 = torch.linalg.vector_norm(fpview - qview, dim=0) / torch.linalg.vector_norm(fpview, dim=0).clamp_min(1e-12)
        worst = torch.topk(-ratio, k=min(20, ratio.numel())).indices
        for channel in worst.tolist():
            rows.append(
                {
                    "config": label,
                    "heldout_index": heldout_index,
                    "layer": layer_sort_key(module),
                    "pytorch_module": module,
                    "call_index": 0,
                    "channel": int(channel),
                    "offline_scale": float(offline[channel]),
                    "current_scale": float(current_scale[channel]),
                    "scale_ratio": float(ratio[channel]),
                    "offline_range": float(offline[channel] * 127.0),
                    "current_range": float(current_scale[channel] * 127.0),
                    "clipping_rate": float(clipped[channel]),
                    "input_relative_l2": float(channel_l2[channel]),
                    "module_max_channel_clipping_rate": clip_by_module[module]["max_channel_clipping_rate"],
                    "module_input_relative_l2": err_by_module[module]["relative_l2_error"],
                }
            )
    return rows


def write_mapping(path: Path, scales: dict[str, Any]) -> None:
    rows = []
    for idx, module in enumerate(sorted(scales, key=layer_sort_key)):
        scale = torch.tensor(scales[module], dtype=torch.float32)
        suffix = "" if idx == 0 else f"_{idx}"
        rows.append(
            {
                "pytorch_module": module,
                "call_index": 0,
                "onnx_node": f"/mlp/down_proj{suffix}/MatMul",
                "channel_count": int(scale.numel()),
                "scale_min": float(scale.min()),
                "scale_max": float(scale.max()),
                "scale_median": float(scale.median()),
                "mapping_status": "expected_vlm_prefix_order",
            }
        )
    write_csv(path, rows)


def adjusted_scales(base: dict[str, Any], mismatch_rows: list[dict[str, Any]], factor: float, args: argparse.Namespace):
    out = deepcopy(base)
    targets = set()
    for row in mismatch_rows:
        if float(row["clipping_rate"]) > args.targeted_clip_threshold or float(row["scale_ratio"]) < args.targeted_ratio_threshold:
            targets.add((row["pytorch_module"], int(row["channel"])))
    for module, channel in targets:
        if module in out:
            out[module][channel] = float(out[module][channel]) * factor
    return out, len(targets)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "calibration_raw_stats"
    raw_dir.mkdir(parents=True, exist_ok=True)
    sizes = parse_ints(args.calibration_sizes)
    max_calibration = max(sizes)
    total_samples = max_calibration + args.heldout_samples

    records, device = collect_rollout_inputs(args, total_samples)
    calibration_records = records[:max_calibration]
    heldout_records = records[max_calibration : max_calibration + args.heldout_samples]
    write_csv(
        output_dir / "sample_manifest.csv",
        [{k: v for k, v in row.items() if k != "core_inputs"} | {"split": "calibration" if idx < max_calibration else "heldout"} for idx, row in enumerate(records)],
    )

    fp_policy, _, effective_device = load_policy(args.policy_path, device)
    accumulator = DownStatsAccumulator(PERCENTILES)
    snapshots: dict[int, dict[str, Any]] = {}
    for idx, record in enumerate(calibration_records, start=1):
        core_inputs = device_inputs(record["core_inputs"], effective_device)
        _outputs, capture = run_with_hooks(fp_policy, core_inputs, args.down_module_regex)
        accumulator.update(capture.down_inputs)
        if idx in sizes:
            snapshots[idx] = accumulator.snapshot()
            with open(raw_dir / f"stats_{idx}.json", "w") as f:
                json.dump(snapshots[idx], f)

    baseline = precompute_baseline(fp_policy, heldout_records, effective_device, args.down_module_regex)

    all_validation_rows = []
    all_flow_rows = []
    all_mismatch_rows = []
    sample_sweep_rows = []
    scale_configs: dict[str, dict[str, Any]] = {}
    for size in sizes:
        scales = scales_from_snapshot(snapshots[size], str(args.default_percentile))
        scale_configs[f"samples_{size}_p{args.default_percentile}"] = scales
        write_scale_json(output_dir / f"activation_scales_{size}.json", args, scales, size, str(args.default_percentile))
        summary, clip_rows, flow, mismatch = validate_config(
            args, f"samples_{size}_p{args.default_percentile}", scales, baseline, heldout_records, effective_device
        )
        sample_sweep_rows.append({"samples": size, **summary})
        all_validation_rows.append(summary)
        all_flow_rows.extend(flow)
        all_mismatch_rows.extend(mismatch)

    percentile_rows = []
    percentile_sizes = parse_ints(args.percentile_sweep_sizes)
    for size in percentile_sizes:
        if size not in snapshots:
            continue
        for percentile in ["99.9", "99.95", "99.99", "99.995", "99.999", "max"]:
            scales = scales_from_snapshot(snapshots[size], percentile)
            label = f"samples_{size}_p{percentile}"
            scale_configs[label] = scales
            write_scale_json(output_dir / f"activation_scales_{size}_p{percentile}.json", args, scales, size, percentile)
            summary, _clip_rows, flow, mismatch = validate_config(
                args, label, scales, baseline, heldout_records, effective_device
            )
            percentile_rows.append({"samples": size, "percentile": percentile, **summary})
            all_validation_rows.append(summary)
            all_flow_rows.extend(flow)
            all_mismatch_rows.extend(mismatch)

    best = min(all_validation_rows, key=lambda row: float(row["action_chunk_relative_l2"]))
    best_scales = scale_configs[best["config"]]
    write_mapping(output_dir / "calibration_mapping.csv", best_scales)
    mismatch_sorted = sorted(all_mismatch_rows, key=lambda row: float(row["scale_ratio"]))[:20]
    write_csv(output_dir / "scale_mismatch_channels.csv", mismatch_sorted)

    targeted_rows = []
    factors = parse_floats(args.targeted_factors)
    for factor in factors:
        scales, target_count = adjusted_scales(best_scales, mismatch_sorted, factor, args)
        label = f"{best['config']}_targeted_x{factor}"
        summary, _clip_rows, flow, mismatch = validate_config(
            args, label, scales, baseline, heldout_records, effective_device
        )
        targeted_rows.append({"factor": factor, "target_channels": target_count, **summary})
        all_validation_rows.append(summary)
        all_flow_rows.extend(flow)

    write_csv(output_dir / "calibration_sample_sweep.csv", sample_sweep_rows)
    write_csv(output_dir / "calibration_percentile_sweep.csv", percentile_rows)
    write_csv(output_dir / "heldout_validation.csv", all_validation_rows)
    write_csv(output_dir / "flow_matching_error.csv", all_flow_rows)
    write_csv(output_dir / "targeted_range_correction.csv", targeted_rows)

    best_after_targeted = min(all_validation_rows, key=lambda row: float(row["action_chunk_relative_l2"]))
    summary = {
        "policy_path": args.policy_path,
        "tasks": args.tasks,
        "seed": args.seed,
        "device": effective_device,
        "calibration_sizes": sizes,
        "heldout_samples": len(heldout_records),
        "sample_stride": args.sample_stride,
        "default_percentile": args.default_percentile,
        "best_calibration_config": best,
        "best_overall_config": best_after_targeted,
        "sample_sweep_best": min(sample_sweep_rows, key=lambda row: float(row["action_chunk_relative_l2"])),
        "percentile_sweep_best": min(percentile_rows, key=lambda row: float(row["action_chunk_relative_l2"])) if percentile_rows else None,
        "targeted_correction_best": min(targeted_rows, key=lambda row: float(row["action_chunk_relative_l2"])) if targeted_rows else None,
        "automatic_answers": {
            "sample_size_improves": float(sample_sweep_rows[-1]["action_chunk_relative_l2"])
            < float(sample_sweep_rows[0]["action_chunk_relative_l2"]),
            "p99_99_best_on_heldout": (
                min(percentile_rows, key=lambda row: float(row["action_chunk_relative_l2"]))["percentile"] == "99.99"
                if percentile_rows
                else None
            ),
            "calibration_coverage_likely_issue": bool(mismatch_sorted and float(mismatch_sorted[0]["scale_ratio"]) < args.targeted_ratio_threshold),
            "scale_mapping_check": "pytorch module count and channel count validated; onnx mapping assumes VLM prefix order before action_in_proj",
            "deploy_readiness": "judge_by_best_overall_action_l2_and_flow_step09",
        },
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
