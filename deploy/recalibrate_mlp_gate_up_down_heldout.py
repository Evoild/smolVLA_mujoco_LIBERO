#!/usr/bin/env python3

"""Recalibrate VLM MLP gate/up/down W8A8 scales with one shared held-out split."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import re
from pathlib import Path
from typing import Any

import torch
from torch import nn

from diagnose_3_4_numeric_baseline import SmolVLADebugCoreWrapper, load_policy
from diagnose_down_proj_w8a8 import action_rows, flow_rows, qdq_static, vector_metrics
from linear_only_quant import replace_linear_modules
from recalibrate_down_proj_heldout import collect_rollout_inputs, cpu_inputs, device_inputs, p95, write_csv


MLP_LINEAR_REGEX = (
    r"^model\.vlm_with_expert\.vlm\.model\.text_model\.layers\.([0-9]+)"
    r"\.mlp\.(gate_proj|up_proj|down_proj)$"
)
PERCENTILES = ["99.0", "99.5", "99.9", "99.95", "99.99", "99.995", "max"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", default="smolvla_libero")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tasks", default="libero_spatial")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--token-length", type=int, default=48)
    parser.add_argument("--input-dtype", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--calibration-samples", type=int, default=512)
    parser.add_argument("--heldout-samples", type=int, default=50)
    parser.add_argument("--sample-stride", type=int, default=5)
    parser.add_argument("--max-parallel-tasks", type=int, default=1)
    parser.add_argument("--module-regex", default=MLP_LINEAR_REGEX)
    parser.add_argument("--percentiles", default=",".join(PERCENTILES))
    parser.add_argument("--default-percentile", default="99.99")
    parser.add_argument("--sign-flip-threshold", type=float, default=1.0e-3)
    parser.add_argument("--topk-modules", type=int, default=20)
    return parser.parse_args()


def parse_percentiles(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def layer_sublayer_sort_key(module: str) -> tuple[int, int]:
    match = re.search(r"\.layers\.([0-9]+)\.mlp\.(gate_proj|up_proj|down_proj)$", module)
    layer = int(match.group(1)) if match else 10**9
    order = {"gate_proj": 0, "up_proj": 1, "down_proj": 2}
    sublayer = order.get(match.group(2), 99) if match else 99
    return layer, sublayer


def sublayer_name(module: str) -> str:
    return module.rsplit(".", 1)[-1]


def module_layer(module: str) -> int:
    match = re.search(r"\.layers\.([0-9]+)\.", module)
    return int(match.group(1)) if match else -1


class LinearCapture:
    def __init__(self, module_regex: str):
        self.pattern = re.compile(module_regex)
        self.inputs: dict[str, list[torch.Tensor]] = {}
        self.outputs: dict[str, list[torch.Tensor]] = {}
        self.handles: list[Any] = []

    def install(self, module: nn.Module) -> None:
        for name, child in module.named_modules():
            if self.pattern.search(name):
                self.inputs[name] = []
                self.outputs[name] = []
                self.handles.append(child.register_forward_pre_hook(self._pre(name)))
                self.handles.append(child.register_forward_hook(self._post(name)))

    def _pre(self, name: str):
        def hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
            self.inputs[name].append(inputs[0].detach().to(torch.float32).cpu())

        return hook

    def _post(self, name: str):
        def hook(_module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            self.outputs[name].append(tensor_first(output).detach().to(torch.float32).cpu())

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


def concat_tensors(values: list[torch.Tensor]) -> torch.Tensor:
    if not values:
        return torch.empty(0, dtype=torch.float32)
    return torch.cat([value.reshape(-1, value.shape[-1]) for value in values], dim=0)


def run_with_linear_hooks(policy: nn.Module, inputs: tuple[torch.Tensor, ...], module_regex: str):
    wrapper = SmolVLADebugCoreWrapper(policy).to(device=str(policy.config.device)).eval()
    capture = LinearCapture(module_regex)
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


class MLPStatsAccumulator:
    def __init__(self, percentiles: list[str]):
        self.percentiles = percentiles
        self.modules: dict[str, dict[str, Any]] = {}
        self.sample_count = 0

    def update(self, inputs_by_module: dict[str, list[torch.Tensor]]) -> None:
        self.sample_count += 1
        for module, values in inputs_by_module.items():
            view = concat_tensors(values).abs()
            channels = view.shape[-1]
            if module not in self.modules:
                self.modules[module] = {
                    "channels": channels,
                    "count": 0,
                    "max_abs": torch.zeros(channels),
                    "percentiles": {p: torch.zeros(channels) for p in self.percentiles if p != "max"},
                }
            item = self.modules[module]
            item["count"] += int(view.shape[0])
            item["max_abs"] = torch.maximum(item["max_abs"], view.amax(dim=0).cpu())
            for p in item["percentiles"]:
                item["percentiles"][p] = torch.maximum(
                    item["percentiles"][p],
                    torch.quantile(view, float(p) / 100.0, dim=0).cpu(),
                )

    def snapshot(self) -> dict[str, Any]:
        return {
            "samples": self.sample_count,
            "modules": {
                module: {
                    "channels": int(item["channels"]),
                    "count": int(item["count"]),
                    "max_abs": item["max_abs"].tolist(),
                    "percentiles": {p: tensor.tolist() for p, tensor in item["percentiles"].items()},
                }
                for module, item in self.modules.items()
            },
        }


def scales_from_snapshot(snapshot: dict[str, Any], percentile: str) -> dict[str, Any]:
    scales = {}
    for module, item in snapshot["modules"].items():
        if percentile == "max":
            ranges = torch.tensor(item["max_abs"], dtype=torch.float32)
        else:
            ranges = torch.tensor(item["percentiles"][percentile], dtype=torch.float32)
        scales[module] = (ranges.clamp_min(1.0e-8) / 127.0).tolist()
    return scales


def write_scale_json(path: Path, args: argparse.Namespace, scales: dict[str, Any], percentile: str) -> None:
    rows = []
    for idx, module in enumerate(sorted(scales, key=layer_sublayer_sort_key)):
        scale = torch.tensor(scales[module], dtype=torch.float32)
        rows.append(
            {
                "index": idx,
                "module": module,
                "layer": module_layer(module),
                "sublayer": sublayer_name(module),
                "call_index": 0,
                "shape": [None, None, int(scale.numel())],
                "channels": int(scale.numel()),
                "scale": scale.tolist(),
                "scale_source": f"step4d_shared_samples_{args.calibration_samples}_p{percentile}",
                "scale_min": float(scale.min()),
                "scale_max": float(scale.max()),
                "scale_median": float(scale.median()),
            }
        )
    report = {
        "policy_path": args.policy_path,
        "tasks": args.tasks,
        "seed": args.seed,
        "samples_collected": args.calibration_samples,
        "percentile": percentile,
        "scale_granularity": "per_channel_last_dim",
        "include_module_regex": args.module_regex,
        "quantization": "vlm_mlp_gate_up_down_w8a8_static_per_channel_shared_calibration",
        "scale_formula": "scale[channel] = max_over_shared_calibration_samples(percentile(abs(linear_input[..., channel]))) / 127",
        "linear_call_scales": rows,
    }
    with open(path, "w") as f:
        json.dump(report, f, indent=2)


def load_quant_policy(policy_path: str, device: str, regex: str, scales: dict[str, Any]):
    policy, _, effective_device = load_policy(policy_path, device)
    replaced = replace_linear_modules(
        policy,
        include_prefixes=None,
        include_regexes=(regex,),
        activation_scales_by_module=scales,
        quantization_kind="w8a8",
    )
    return policy.to(device=effective_device).eval(), effective_device, replaced


def summarize_values(rows: list[dict[str, Any]], key: str) -> tuple[float, float, float]:
    values = [float(row[key]) for row in rows if key in row and math.isfinite(float(row[key]))]
    if not values:
        return float("nan"), float("nan"), float("nan")
    return float(sum(values) / len(values)), p95(values), float(max(values))


def clipping_and_input_rows(inputs_by_module: dict[str, list[torch.Tensor]], scales: dict[str, Any]):
    clipping_rows = []
    input_rows = []
    for module, values in sorted(inputs_by_module.items(), key=lambda item: layer_sublayer_sort_key(item[0])):
        x = concat_tensors(values)
        scale = torch.tensor(scales[module], dtype=torch.float32).clamp_min(1.0e-8)
        q = qdq_static(x, scale)
        metrics = vector_metrics(x, q)
        qrange_pos = scale * 127.0
        qrange_neg = scale * 128.0
        clipped = (x > qrange_pos[None, :]) | (x < -qrange_neg[None, :])
        per_channel = clipped.to(torch.float32).mean(dim=0)
        clipping_rows.append(
            {
                "module": module,
                "layer": module_layer(module),
                "sublayer": sublayer_name(module),
                "clipping_rate": float(clipped.to(torch.float32).mean()),
                "mean_channel_clipping_rate": float(per_channel.mean()),
                "p95_channel_clipping_rate": float(torch.quantile(per_channel, 0.95)),
                "max_channel_clipping_rate": float(per_channel.max()),
                "scale_min": float(scale.min()),
                "scale_max": float(scale.max()),
                "scale_median": float(scale.median()),
            }
        )
        input_rows.append(
            {
                "module": module,
                "layer": module_layer(module),
                "sublayer": sublayer_name(module),
                **metrics,
            }
        )
    return clipping_rows, input_rows


def output_rows(fp_outputs: dict[str, list[torch.Tensor]], quant_outputs: dict[str, list[torch.Tensor]]):
    rows = []
    for module, values in sorted(fp_outputs.items(), key=lambda item: layer_sublayer_sort_key(item[0])):
        if module not in quant_outputs:
            continue
        metrics = vector_metrics(concat_tensors(values), concat_tensors(quant_outputs[module]))
        rows.append(
            {
                "module": module,
                "layer": module_layer(module),
                "sublayer": sublayer_name(module),
                **metrics,
            }
        )
    return rows


def precompute_baseline(policy, records: list[dict[str, Any]], device: str, module_regex: str):
    outputs = []
    for record in records:
        core_inputs = device_inputs(record["core_inputs"], device)
        out, capture = run_with_linear_hooks(policy, core_inputs, module_regex)
        outputs.append(
            {
                "outputs": out,
                "capture": capture,
                "meta": {k: v for k, v in record.items() if k != "core_inputs"},
            }
        )
    return outputs


def aggregate_by_sublayer(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    result = {}
    for sublayer in ("gate_proj", "up_proj", "down_proj"):
        sub_rows = [row for row in rows if row.get("sublayer") == sublayer]
        mean, p95_value, max_value = summarize_values(sub_rows, key)
        worst = max(sub_rows, key=lambda row: float(row[key])) if sub_rows else None
        result[sublayer] = {
            f"mean_{key}": mean,
            f"p95_{key}": p95_value,
            f"max_{key}": max_value,
            "worst_layer": worst["layer"] if worst else None,
            "worst_module": worst["module"] if worst else None,
        }
    return result


def validate_config(
    args: argparse.Namespace,
    label: str,
    scales: dict[str, Any],
    baseline: list[dict[str, Any]],
    heldout_records: list[dict[str, Any]],
    device: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    policy, effective_device, replaced = load_quant_policy(args.policy_path, device, args.module_regex, scales)
    clipping_all = []
    input_all = []
    output_all = []
    flow_all = []
    action_l2 = []
    action_cos = []
    first_action_l2 = []
    timestep_l2 = []
    sign_flip = []

    for idx, (base, record) in enumerate(zip(baseline, heldout_records, strict=True)):
        core_inputs = device_inputs(record["core_inputs"], effective_device)
        quant_outputs, quant_capture = run_with_linear_hooks(policy, core_inputs, args.module_regex)
        fp_outputs = base["outputs"]
        fp_capture = base["capture"]
        meta = {"config": label, "heldout_index": idx, **base["meta"]}

        clip_rows, input_rows = clipping_and_input_rows(fp_capture.inputs, scales)
        out_rows = output_rows(fp_capture.outputs, quant_capture.outputs)
        for row in clip_rows:
            clipping_all.append({**meta, **row})
        for row in input_rows:
            input_all.append({**meta, **row})
        for row in out_rows:
            output_all.append({**meta, **row})
        for row in flow_rows(fp_outputs, quant_outputs):
            flow_all.append({**meta, **row})

        action = vector_metrics(fp_outputs["action_chunk"], quant_outputs["action_chunk"])
        action_l2.append(action["relative_l2_error"])
        action_cos.append(action["cosine_similarity"])
        first_action_l2.append(vector_metrics(fp_outputs["action_chunk"][:, 0], quant_outputs["action_chunk"][:, 0])["relative_l2_error"])
        t_rows, _dim_rows, action_summary = action_rows(
            fp_outputs["action_chunk"], quant_outputs["action_chunk"], args.sign_flip_threshold
        )
        timestep_l2.extend([row["relative_l2_error"] for row in t_rows])
        sign_flip.append(action_summary["meaningful_sign_flip_rate"])

    mean_clip, p95_clip, _max_clip_mean = summarize_values(clipping_all, "mean_channel_clipping_rate")
    max_channel_clip = max([float(row["max_channel_clipping_rate"]) for row in clipping_all] or [float("nan")])
    in_mean, in_p95, in_max = summarize_values(input_all, "relative_l2_error")
    out_mean, out_p95, out_max = summarize_values(output_all, "relative_l2_error")
    vt9 = [row for row in flow_all if row["step"] == 9]
    vt9_l2 = [float(row["v_t_relative_l2_error"]) for row in vt9]
    xt9_l2 = [float(row["x_t_relative_l2_error"]) for row in vt9]
    summary = {
        "config": label,
        "num_linear_replaced": replaced,
        "heldout_samples": len(heldout_records),
        "mean_channel_clipping_rate": mean_clip,
        "p95_channel_clipping_rate": p95_clip,
        "max_channel_clipping_rate": max_channel_clip,
        "input_relative_l2_mean": in_mean,
        "input_relative_l2_p95": in_p95,
        "input_relative_l2_max": in_max,
        "output_relative_l2_mean": out_mean,
        "output_relative_l2_p95": out_p95,
        "output_relative_l2_max": out_max,
        "input_by_sublayer": aggregate_by_sublayer(input_all, "relative_l2_error"),
        "output_by_sublayer": aggregate_by_sublayer(output_all, "relative_l2_error"),
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
    return summary, clipping_all, input_all, output_all, flow_all


def plot_percentile_sweep(rows: list[dict[str, Any]], output_dir: Path) -> dict[str, str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        return {"plot_status": f"skipped: {exc!r}"}

    x = list(range(len(rows)))
    labels = [row["percentile"] for row in rows]
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=160)
    ax.plot(x, [float(row["action_chunk_relative_l2"]) for row in rows], marker="o", label="action_chunk")
    ax.plot(x, [float(row["v_t_step09_relative_l2"]) for row in rows], marker="o", label="v_t_step09")
    ax.plot(x, [float(row["x_t_step09_relative_l2"]) for row in rows], marker="o", label="x_t_step09")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("percentile")
    ax.set_ylabel("held-out relative L2")
    ax.set_title("Step 4D Shared Calibration Percentile Sweep")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)
    ax.legend()
    fig.tight_layout()
    path = output_dir / "percentile_sweep_relative_l2.png"
    fig.savefig(path)
    plt.close(fig)
    return {"percentile_sweep_relative_l2_png": str(path)}


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "calibration_raw_stats"
    raw_dir.mkdir(parents=True, exist_ok=True)
    percentiles = parse_percentiles(args.percentiles)
    if args.default_percentile not in percentiles:
        percentiles.append(args.default_percentile)

    total_samples = args.calibration_samples + args.heldout_samples
    records, device = collect_rollout_inputs(args, total_samples)
    calibration_records = records[: args.calibration_samples]
    heldout_records = records[args.calibration_samples : total_samples]
    write_csv(
        output_dir / "sample_manifest.csv",
        [
            {k: v for k, v in row.items() if k != "core_inputs"}
            | {"split": "calibration" if idx < args.calibration_samples else "heldout"}
            for idx, row in enumerate(records)
        ],
    )

    fp_policy, _, effective_device = load_policy(args.policy_path, device)
    accumulator = MLPStatsAccumulator(percentiles)
    for record in calibration_records:
        core_inputs = device_inputs(record["core_inputs"], effective_device)
        _outputs, capture = run_with_linear_hooks(fp_policy, core_inputs, args.module_regex)
        accumulator.update(capture.inputs)
    snapshot = accumulator.snapshot()
    with open(raw_dir / f"stats_{args.calibration_samples}.json", "w") as f:
        json.dump(snapshot, f)

    baseline = precompute_baseline(fp_policy, heldout_records, effective_device, args.module_regex)
    validation_rows = []
    clipping_rows_all = []
    input_rows_all = []
    output_rows_all = []
    flow_rows_all = []
    scale_configs = {}

    for percentile in percentiles:
        scales = scales_from_snapshot(snapshot, percentile)
        label = f"samples_{args.calibration_samples}_p{percentile}"
        scale_configs[label] = scales
        write_scale_json(output_dir / f"activation_scales_{args.calibration_samples}_p{percentile}.json", args, scales, percentile)
        summary, clip_rows, input_rows, out_rows, flow = validate_config(
            args, label, scales, baseline, heldout_records, effective_device
        )
        validation_rows.append({"samples": args.calibration_samples, "percentile": percentile, **summary})
        clipping_rows_all.extend(clip_rows)
        input_rows_all.extend(input_rows)
        output_rows_all.extend(out_rows)
        flow_rows_all.extend(flow)

    best = min(validation_rows, key=lambda row: float(row["action_chunk_relative_l2"]))
    default_label = f"samples_{args.calibration_samples}_p{args.default_percentile}"
    if default_label in scale_configs:
        write_scale_json(output_dir / f"activation_scales_{args.calibration_samples}.json", args, scale_configs[default_label], args.default_percentile)
    write_scale_json(output_dir / "activation_scales_best.json", args, scale_configs[best["config"]], best["percentile"])

    write_csv(output_dir / "percentile_sweep.csv", validation_rows)
    write_csv(output_dir / "heldout_validation.csv", validation_rows)
    write_csv(output_dir / "activation_clipping.csv", clipping_rows_all)
    write_csv(output_dir / "activation_input_error.csv", input_rows_all)
    write_csv(output_dir / "linear_output_error.csv", output_rows_all)
    write_csv(output_dir / "flow_matching_error.csv", flow_rows_all)

    top_output = sorted(output_rows_all, key=lambda row: float(row["relative_l2_error"]), reverse=True)[: args.topk_modules]
    top_input = sorted(input_rows_all, key=lambda row: float(row["relative_l2_error"]), reverse=True)[: args.topk_modules]
    write_csv(output_dir / "top_output_error_modules.csv", top_output)
    write_csv(output_dir / "top_input_error_modules.csv", top_input)
    plots = plot_percentile_sweep(validation_rows, output_dir)

    summary = {
        "policy_path": args.policy_path,
        "tasks": args.tasks,
        "seed": args.seed,
        "device": effective_device,
        "calibration_samples": args.calibration_samples,
        "heldout_samples": len(heldout_records),
        "sample_stride": args.sample_stride,
        "percentiles": percentiles,
        "default_percentile": args.default_percentile,
        "module_regex": args.module_regex,
        "num_modules": len(snapshot["modules"]),
        "best_config": best,
        "default_config": next((row for row in validation_rows if row["config"] == default_label), None),
        "plots": plots,
        "automatic_answers": {
            "shared_calibration_samples": True,
            "gate_up_down_same_split": True,
            "deploy_readiness": "judge_by_best_config_action_l2_then_run_E_vs_D_before_ONNX_TensorRT",
        },
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
