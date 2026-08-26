#!/usr/bin/env python3

"""Trace where Text VLM down_proj quantization error amplifies inside attention."""

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
from lerobot.policies.smolvla.smolvlm_with_expert import apply_rope
from linear_only_quant import load_activation_scales_by_module, replace_linear_modules


DEFAULT_LAYERS = "10,18,24,26,27,28,29,30,31"
FULL_REGEX = (
    r"^model\.vlm_with_expert\.vlm\.model\.text_model\.layers\.[0-9]+"
    r"\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))$"
)
NO_DOWN_REGEX = (
    r"^model\.vlm_with_expert\.vlm\.model\.text_model\.layers\.[0-9]+"
    r"\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj))$"
)
PATH_STAGES = [
    "prev_down_output",
    "attention_input_hidden_state",
    "attention_norm_output",
    "qkv_mean",
    "attention_logits",
    "softmax_probability",
    "attention_context_o_input",
    "o_proj_output",
]
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
    parser.add_argument("--layers", default=DEFAULT_LAYERS)
    parser.add_argument("--full-quantize-module-regex", default=FULL_REGEX)
    parser.add_argument("--no-down-quantize-module-regex", default=NO_DOWN_REGEX)
    return parser.parse_args()


def parse_layers(value: str) -> list[int]:
    result = []
    for item in value.split(","):
        item = item.strip()
        if item:
            result.append(int(item))
    return sorted(set(result))


def cpu_float(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().to(torch.float32).cpu()


def tensor_metrics(ref: torch.Tensor, quant: torch.Tensor) -> dict[str, float]:
    lhs = ref.detach().to(torch.float32).flatten()
    rhs = quant.detach().to(torch.float32).flatten()
    if lhs.numel() == 0 or rhs.numel() == 0:
        return {
            "relative_l2": 0.0,
            "absolute_l2": 0.0,
            "cosine": 1.0,
            "ref_norm": 0.0,
            "quant_norm": 0.0,
            "norm_ratio": 1.0,
            "max_abs_error": 0.0,
        }
    diff = rhs - lhs
    ref_norm = torch.linalg.vector_norm(lhs)
    quant_norm = torch.linalg.vector_norm(rhs)
    abs_l2 = torch.linalg.vector_norm(diff)
    denom = ref_norm.clamp_min(EPS)
    cosine = torch.nn.functional.cosine_similarity(lhs[None], rhs[None], dim=1).item()
    return {
        "relative_l2": float((abs_l2 / denom).item()),
        "absolute_l2": float(abs_l2.item()),
        "cosine": float(cosine),
        "ref_norm": float(ref_norm.item()),
        "quant_norm": float(quant_norm.item()),
        "norm_ratio": float((quant_norm / denom).item()),
        "max_abs_error": float(diff.abs().max().item()),
    }


def probability_metrics(ref: torch.Tensor, quant: torch.Tensor) -> dict[str, float]:
    ref_f = ref.detach().to(torch.float32)
    quant_f = quant.detach().to(torch.float32)
    diff = quant_f - ref_f
    top1_ref = ref_f.argmax(dim=-1)
    top1_quant = quant_f.argmax(dim=-1)
    ref_entropy = -(ref_f.clamp_min(EPS) * ref_f.clamp_min(EPS).log()).sum(dim=-1)
    quant_entropy = -(quant_f.clamp_min(EPS) * quant_f.clamp_min(EPS).log()).sum(dim=-1)
    return {
        "prob_l1_distance": float(diff.abs().sum().item()),
        "prob_l2_distance": float(torch.linalg.vector_norm(diff.flatten()).item()),
        "prob_max_diff": float(diff.abs().max().item()),
        "top1_change_rate": float((top1_ref != top1_quant).to(torch.float32).mean().item()),
        "entropy_diff_mean": float((quant_entropy - ref_entropy).mean().item()),
        "entropy_abs_diff_mean": float((quant_entropy - ref_entropy).abs().mean().item()),
    }


def aggregate_numeric(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["config"], int(row["layer"]), row["stage"])].append(row)
    result = []
    for (config, layer, stage), values in grouped.items():
        out: dict[str, Any] = {"config": config, "layer": layer, "stage": stage, "num_samples": len(values)}
        numeric_keys = sorted(
            key
            for row in values
            for key, value in row.items()
            if key not in {"config", "layer", "stage", "sample_index"} and isinstance(value, (int, float))
        )
        for key in numeric_keys:
            nums = [float(row[key]) for row in values if isinstance(row.get(key), (int, float))]
            if nums:
                out[key] = sum(nums) / len(nums)
        result.append(out)
    return sorted(result, key=lambda row: (row["config"], row["layer"], stage_order(row["stage"])))


def stage_order(stage: str) -> int:
    order = {
        "prev_down_output": 0,
        "prev_ffn_residual_output": 1,
        "attention_input_hidden_state": 2,
        "attention_norm_output": 3,
        "q_proj_output": 4,
        "k_proj_output": 5,
        "v_proj_output": 6,
        "qkv_mean": 7,
        "attention_logits": 8,
        "softmax_probability": 9,
        "attention_context_o_input": 10,
        "o_proj_output": 11,
        "attention_residual_output": 12,
        "ffn_norm_output": 13,
        "down_proj_input": 14,
        "down_proj_output": 15,
        "ffn_residual_output": 16,
    }
    return order.get(stage, 999)


def build_stage_views(layer_capture: dict[int, dict[str, torch.Tensor]], layer: int) -> dict[str, torch.Tensor]:
    current = layer_capture[layer]
    previous = layer_capture.get(layer - 1, {})
    views = {
        "prev_down_output": previous.get("down_proj_output", current["attention_input_hidden_state"]),
        "prev_ffn_residual_output": previous.get("ffn_residual_output", current["attention_input_hidden_state"]),
        "attention_input_hidden_state": current["attention_input_hidden_state"],
        "attention_norm_output": current["attention_norm_output"],
        "q_proj_output": current["q_proj_output"],
        "k_proj_output": current["k_proj_output"],
        "v_proj_output": current["v_proj_output"],
        "attention_logits": current["attention_logits"],
        "softmax_probability": current["softmax_probability"],
        "attention_context_o_input": current["attention_context_o_input"],
        "o_proj_output": current["o_proj_output"],
        "attention_residual_output": current["attention_residual_output"],
        "ffn_norm_output": current["ffn_norm_output"],
        "down_proj_input": current["down_proj_input"],
        "down_proj_output": current["down_proj_output"],
        "ffn_residual_output": current["ffn_residual_output"],
    }
    qkv_values = [views["q_proj_output"].flatten(), views["k_proj_output"].flatten(), views["v_proj_output"].flatten()]
    views["qkv_mean"] = torch.cat(qkv_values, dim=0)
    return views


def expand_kv_states(
    model: nn.Module,
    batch_size: int,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_att_heads = model.num_attention_heads
    num_key_value_heads = model.num_key_value_heads
    num_key_value_groups = num_att_heads // num_key_value_heads
    sequence_length = key_states.shape[1]
    head_dim = key_states.shape[-1]
    key_states = key_states[:, :, :, None, :].expand(
        batch_size, sequence_length, num_key_value_heads, num_key_value_groups, head_dim
    )
    key_states = key_states.reshape(batch_size, sequence_length, num_key_value_heads * num_key_value_groups, head_dim)
    value_states = value_states[:, :, :, None, :].expand(
        batch_size, sequence_length, num_key_value_heads, num_key_value_groups, head_dim
    )
    value_states = value_states.reshape(
        batch_size, sequence_length, num_key_value_heads * num_key_value_groups, head_dim
    )
    return key_states, value_states


def run_prefix_trace(policy: nn.Module, inputs: tuple[torch.Tensor, ...], target_layers: set[int]) -> dict[int, dict[str, torch.Tensor]]:
    wrapper = SmolVLADebugCoreWrapper(policy).to(device=str(policy.config.device)).eval()
    model = wrapper.model
    prefix_embs, prefix_pad_masks, prefix_att_masks = wrapper.embed_prefix_from_image_embeds(*inputs[:-1])
    from diagnose_3_4_numeric_baseline import make_att_2d_masks_trt

    attention_mask = make_att_2d_masks_trt(prefix_pad_masks, prefix_att_masks)
    position_ids = torch.cumsum(prefix_pad_masks.to(torch.int64), dim=1) - 1

    vlm_with_expert = model.vlm_with_expert
    text_model = vlm_with_expert.get_vlm_model().text_model
    model_layers = vlm_with_expert.get_model_layers([text_model, vlm_with_expert.lm_expert])
    hidden_states = prefix_embs
    batch_size = hidden_states.shape[0]
    head_dim = vlm_with_expert.vlm.config.text_config.head_dim
    captures: dict[int, dict[str, torch.Tensor]] = {}

    for layer_idx, layer in enumerate(model_layers[0]):
        layer_capture: dict[str, torch.Tensor] = {}
        if layer_idx in target_layers or layer_idx + 1 in target_layers:
            layer_capture["attention_input_hidden_state"] = cpu_float(hidden_states)

        norm_states = layer.input_layernorm(hidden_states)
        if layer_idx in target_layers:
            layer_capture["attention_norm_output"] = cpu_float(norm_states)

        input_shape = norm_states.shape[:-1]
        hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
        proj_input = norm_states.to(dtype=layer.self_attn.q_proj.weight.dtype)
        q_raw = layer.self_attn.q_proj(proj_input).view(hidden_shape)
        k_raw = layer.self_attn.k_proj(proj_input).view(hidden_shape)
        v_raw = layer.self_attn.v_proj(proj_input).view(hidden_shape)
        if layer_idx in target_layers:
            layer_capture["q_proj_output"] = cpu_float(q_raw)
            layer_capture["k_proj_output"] = cpu_float(k_raw)
            layer_capture["v_proj_output"] = cpu_float(v_raw)

        q_states = apply_rope(q_raw, position_ids)
        k_states = apply_rope(k_raw, position_ids)
        k_expanded, v_expanded = expand_kv_states(vlm_with_expert, batch_size, k_states, v_raw)
        q_t = q_states.to(torch.float32).transpose(1, 2)
        k_t = k_expanded.to(torch.float32).transpose(1, 2)
        score = torch.matmul(q_t, k_t.transpose(2, 3)) * (head_dim**-0.5)
        big_neg = torch.finfo(score.dtype).min
        masked_score = torch.where(attention_mask[:, None, :, :], score, big_neg)
        probs = nn.functional.softmax(masked_score, dim=-1)
        context = torch.matmul(probs.to(dtype=v_expanded.dtype), v_expanded.permute(0, 2, 1, 3))
        context = context.permute(0, 2, 1, 3)
        context = context.reshape(batch_size, -1, vlm_with_expert.num_attention_heads * head_dim)
        if layer_idx in target_layers:
            layer_capture["attention_logits"] = cpu_float(score)
            layer_capture["softmax_probability"] = cpu_float(probs)
            layer_capture["attention_context_o_input"] = cpu_float(context)

        o_input = context.to(dtype=layer.self_attn.o_proj.weight.dtype)
        o_output = layer.self_attn.o_proj(o_input)
        attention_residual = o_output + hidden_states
        if layer_idx in target_layers:
            layer_capture["o_proj_output"] = cpu_float(o_output)
            layer_capture["attention_residual_output"] = cpu_float(attention_residual)

        ffn_norm = layer.post_attention_layernorm(attention_residual)
        if layer_idx in target_layers:
            layer_capture["ffn_norm_output"] = cpu_float(ffn_norm)

        down_holder: dict[str, torch.Tensor] = {}

        def down_pre_hook(_module: nn.Module, hook_inputs: tuple[torch.Tensor, ...]) -> None:
            down_holder["input"] = cpu_float(hook_inputs[0])

        def down_hook(_module: nn.Module, _hook_inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            down_holder["output"] = cpu_float(output)

        pre_handle = layer.mlp.down_proj.register_forward_pre_hook(down_pre_hook)
        out_handle = layer.mlp.down_proj.register_forward_hook(down_hook)
        try:
            mlp_output = layer.mlp(ffn_norm)
        finally:
            pre_handle.remove()
            out_handle.remove()

        ffn_residual = mlp_output + attention_residual
        if layer_idx in target_layers or layer_idx + 1 in target_layers:
            layer_capture["down_proj_input"] = down_holder.get("input", torch.empty(0))
            layer_capture["down_proj_output"] = down_holder.get("output", torch.empty(0))
            layer_capture["ffn_residual_output"] = cpu_float(ffn_residual)

        if layer_capture:
            captures[layer_idx] = layer_capture
        hidden_states = ffn_residual
    return captures


def make_input_records(args: argparse.Namespace, original: nn.Module, preprocessor: Any, device: str):
    input_dtype = torch.float32 if args.input_dtype == "fp32" else torch.bfloat16
    if args.input_source == "synthetic":
        inputs = core_inputs_from_policy(original, preprocessor, args.task, args.seed, input_dtype, args.token_length)
        return [
            {
                "sample_index": 0,
                "input_source": "synthetic",
                "core_inputs": tuple(tensor.detach().cpu() for tensor in inputs),
            }
        ]
    records, _rollout_device = collect_rollout_core_inputs(args, args.compare_samples)
    return records


def load_fake_quant_policy(
    args: argparse.Namespace,
    regex: str,
    activation_scales: dict[str, Any],
) -> nn.Module:
    policy, _, device = load_policy(args.policy_path, args.device, args.model_dtype)
    scales = {
        module: scale
        for module, scale in activation_scales.items()
        if re.search(regex, module)
    }
    replace_linear_modules(
        policy,
        include_prefixes=None,
        include_regexes=(regex,),
        activation_scales_by_module=scales,
        quantization_kind="w8a8",
    )
    policy.to(device=device).eval()
    return policy


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        if not rows:
            return
        fieldnames = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_rows_for_sample(
    sample_index: int,
    config: str,
    target_layers: list[int],
    ref_capture: dict[int, dict[str, torch.Tensor]],
    quant_capture: dict[int, dict[str, torch.Tensor]],
) -> list[dict[str, Any]]:
    rows = []
    for layer in target_layers:
        ref_views = build_stage_views(ref_capture, layer)
        quant_views = build_stage_views(quant_capture, layer)
        for stage in sorted(ref_views, key=stage_order):
            metrics = tensor_metrics(ref_views[stage], quant_views[stage])
            row = {"sample_index": sample_index, "config": config, "layer": layer, "stage": stage, **metrics}
            if stage == "softmax_probability":
                row.update(probability_metrics(ref_views[stage], quant_views[stage]))
            rows.append(row)
    return rows


def add_amplification(rows: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        grouped[(row["config"], int(row["layer"]))][row["stage"]] = row
    for (_config, _layer), by_stage in grouped.items():
        previous_error = None
        for stage in PATH_STAGES:
            row = by_stage.get(stage)
            if row is None:
                continue
            current_error = float(row.get("relative_l2", 0.0))
            if previous_error is None:
                row["amplification"] = None
            else:
                row["amplification"] = current_error / max(previous_error, EPS)
                row["amplification_delta"] = current_error - previous_error
            previous_error = current_error


def build_transition_rows(aggregate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in aggregate_rows:
        grouped[(row["config"], int(row["layer"]))][row["stage"]] = row

    transitions = []
    for (config, layer), by_stage in sorted(grouped.items()):
        previous_stage = None
        previous_error = None
        first_large_jump = None
        for stage in PATH_STAGES:
            row = by_stage.get(stage)
            if row is None:
                continue
            current_error = float(row.get("relative_l2", 0.0))
            if previous_stage is not None and previous_error is not None:
                ratio = current_error / max(previous_error, EPS)
                delta = current_error - previous_error
                is_large = ratio > 1.5 and delta > 0.05
                if first_large_jump is None and is_large:
                    first_large_jump = f"{previous_stage}->{stage}"
                transitions.append(
                    {
                        "config": config,
                        "layer": layer,
                        "from_stage": previous_stage,
                        "to_stage": stage,
                        "from_relative_l2": previous_error,
                        "to_relative_l2": current_error,
                        "amplification": ratio,
                        "delta": delta,
                        "is_large_jump": is_large,
                    }
                )
            previous_stage = stage
            previous_error = current_error
        for row in transitions:
            if row["config"] == config and row["layer"] == layer:
                row["first_large_jump"] = first_large_jump
    return transitions


def write_plot(aggregate_rows: list[dict[str, Any]], output_dir: Path) -> dict[str, str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return {"plot_status": f"skipped: {exc!r}"}

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    stage_labels = ["down_out", "residual", "norm", "QKV", "score", "softmax", "context", "o_proj"]
    x = list(range(len(PATH_STAGES)))
    for layer in sorted({int(row["layer"]) for row in aggregate_rows}):
        fig, ax = plt.subplots(figsize=(10.5, 5.4), dpi=160)
        for config, color in (("FULL", "#dc2626"), ("NO_DOWN", "#2563eb")):
            by_stage = {
                row["stage"]: row
                for row in aggregate_rows
                if row["config"] == config and int(row["layer"]) == layer
            }
            y = [by_stage.get(stage, {}).get("relative_l2") for stage in PATH_STAGES]
            ax.plot(x, y, marker="o", linewidth=1.8, markersize=4, color=color, label=config)
        ax.set_xticks(x, stage_labels, rotation=30, ha="right")
        ax.set_ylabel("Relative L2 vs BF16")
        ax.set_title(f"Layer {layer} attention error propagation")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)
        ax.legend()
        fig.tight_layout()
        path = output_dir / f"attention_error_propagation_layer_{layer:02d}.png"
        fig.savefig(path)
        plt.close(fig)
        paths[f"layer_{layer:02d}_png"] = str(path)

    fig, ax = plt.subplots(figsize=(11.5, 5.8), dpi=160)
    for config, color in (("FULL", "#dc2626"), ("NO_DOWN", "#2563eb")):
        means = []
        for stage in PATH_STAGES:
            vals = [
                float(row["relative_l2"])
                for row in aggregate_rows
                if row["config"] == config and row["stage"] == stage
            ]
            means.append(sum(vals) / len(vals) if vals else None)
        ax.plot(x, means, marker="o", linewidth=2.0, markersize=4.5, color=color, label=f"{config} mean")
    ax.set_xticks(x, stage_labels, rotation=30, ha="right")
    ax.set_ylabel("Mean relative L2 vs BF16")
    ax.set_title("Attention error propagation mean over selected layers")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)
    ax.legend()
    fig.tight_layout()
    path = output_dir / "attention_error_propagation.png"
    fig.savefig(path)
    plt.close(fig)
    paths["mean_png"] = str(path)
    return paths


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target_layers = parse_layers(args.layers)
    needed_layers = set(target_layers) | {layer - 1 for layer in target_layers if layer > 0}

    reference, preprocessor, device = load_policy(args.policy_path, args.device, args.model_dtype)
    input_records = make_input_records(args, reference, preprocessor, device)

    activation_scales = load_activation_scales_by_module(args.activation_scales_json)
    full_policy = load_fake_quant_policy(args, args.full_quantize_module_regex, activation_scales)
    no_down_policy = load_fake_quant_policy(args, args.no_down_quantize_module_regex, activation_scales)
    reference.to(device=device).eval()

    sample_rows: list[dict[str, Any]] = []
    for record in input_records:
        sample_index = int(record["sample_index"])
        inputs = device_inputs(record["core_inputs"], device)
        with torch.no_grad():
            ref_capture = run_prefix_trace(reference, inputs, needed_layers)
            full_capture = run_prefix_trace(full_policy, inputs, needed_layers)
            no_down_capture = run_prefix_trace(no_down_policy, inputs, needed_layers)
        sample_rows.extend(build_rows_for_sample(sample_index, "FULL", target_layers, ref_capture, full_capture))
        sample_rows.extend(build_rows_for_sample(sample_index, "NO_DOWN", target_layers, ref_capture, no_down_capture))

    add_amplification(sample_rows)
    aggregate_rows = aggregate_numeric(sample_rows)
    add_amplification(aggregate_rows)
    transition_rows = build_transition_rows(aggregate_rows)
    plots = write_plot(aggregate_rows, output_dir)

    first_jumps = {}
    for row in transition_rows:
        key = f"{row['config']}_layer_{int(row['layer']):02d}"
        if row.get("first_large_jump") and key not in first_jumps:
            first_jumps[key] = row["first_large_jump"]

    summary = {
        "policy_path": args.policy_path,
        "activation_scales_json": args.activation_scales_json,
        "task": args.task,
        "seed": args.seed,
        "input_source": args.input_source,
        "sample_count": len(input_records),
        "target_layers": target_layers,
        "path_stages": PATH_STAGES,
        "full_quantize_module_regex": args.full_quantize_module_regex,
        "no_down_quantize_module_regex": args.no_down_quantize_module_regex,
        "first_large_jumps": first_jumps,
        "plots": plots,
    }

    write_csv(output_dir / "attention_error_propagation_rows.csv", sample_rows)
    write_csv(output_dir / "attention_error_propagation.csv", aggregate_rows)
    write_csv(output_dir / "attention_error_amplification.csv", transition_rows)
    with open(output_dir / "attention_error_propagation_rows.json", "w") as f:
        json.dump(sample_rows, f, indent=2)
    with open(output_dir / "attention_error_propagation_summary.json", "w") as f:
        json.dump({**summary, "rows": aggregate_rows, "transitions": transition_rows}, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
