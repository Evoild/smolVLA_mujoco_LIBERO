#!/usr/bin/env python3

"""Fine-grained SmolVLA attention profiler for step 6.

This profiles the PyTorch action core and splits the eager attention body into
RoPE, QK^T, mask, softmax, PV, transpose/reshape, and related tensor movement.
It is intended to decide whether a TensorRT fused attention plugin is worth
building before implementing the plugin.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
from types import MethodType
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
from linear_only_quant import load_activation_scales_by_module, replace_linear_modules
from lerobot.policies.smolvla import smolvlm_with_expert
from lerobot.utils.random_utils import set_seed


ATTN_PROJ_SUFFIXES = (
    ".self_attn.q_proj",
    ".self_attn.k_proj",
    ".self_attn.v_proj",
    ".self_attn.o_proj",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", default="smolvla_libero")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--task", default="libero_spatial")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--token-length", type=int, default=48)
    parser.add_argument("--input-source", choices=["synthetic", "rollout"], default="rollout")
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--sample-stride", type=int, default=5)
    parser.add_argument("--max-parallel-tasks", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--model-dtype", choices=["native", "bf16"], default="bf16")
    parser.add_argument("--input-dtype", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument(
        "--quantize-module-regex",
        action="append",
        default=None,
        help="Optional regex for PyTorch fake quant modules.",
    )
    parser.add_argument(
        "--activation-scales-json",
        default=None,
        help="Optional SmoothQuant/calibrated activation scales for PyTorch fake quant.",
    )
    return parser.parse_args()


def sync(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


class SegmentTimer:
    def __init__(self, device: str):
        self.device = device
        self.rows: list[dict[str, Any]] = []
        self.current_sample = -1
        self.current_iter = -1
        self.current_context = "unknown"
        self.use_cuda_events = str(device).startswith("cuda") and torch.cuda.is_available()

    @contextmanager
    def time(self, segment: str, **meta: Any):
        if self.use_cuda_events:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            yield
            end.record()
            torch.cuda.synchronize()
            elapsed_ms = float(start.elapsed_time(end))
        else:
            start_time = time.perf_counter()
            yield
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        self.rows.append(
            {
                "sample_index": self.current_sample,
                "iteration": self.current_iter,
                "context": self.current_context,
                "segment": segment,
                "elapsed_ms": elapsed_ms,
                **meta,
            }
        )


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["context"]), str(row["segment"]))].append(float(row["elapsed_ms"]))
    out = []
    for (context, segment), values in sorted(grouped.items()):
        values_sorted = sorted(values)
        out.append(
            {
                "context": context,
                "segment": segment,
                "calls": len(values),
                "total_ms": float(sum(values)),
                "mean_ms": float(statistics.fmean(values)),
                "median_ms": float(statistics.median(values)),
                "p95_ms": values_sorted[max(0, int(len(values_sorted) * 0.95) - 1)],
            }
        )
    return out


def install_attention_body_profiler(model: nn.Module, timer: SegmentTimer) -> None:
    target = model.model.vlm_with_expert
    original_apply_rope = smolvlm_with_expert.apply_rope

    def timed_apply_rope(x, positions, max_wavelength=10_000):
        with timer.time("rope", shape=list(x.shape)):
            return original_apply_rope(x, positions, max_wavelength=max_wavelength)

    smolvlm_with_expert.apply_rope = timed_apply_rope

    original_forward_attn_layer = target.forward_attn_layer
    original_forward_cross_attn_layer = target.forward_cross_attn_layer

    def timed_forward_attn_layer(self, *args, **kwargs):
        previous = timer.current_context
        fill_kv_cache = bool(kwargs.get("fill_kv_cache", True))
        timer.current_context = "prefix_self_attn" if fill_kv_cache else "decode_self_attn"
        try:
            return original_forward_attn_layer(*args, **kwargs)
        finally:
            timer.current_context = previous

    def timed_forward_cross_attn_layer(self, *args, **kwargs):
        previous = timer.current_context
        fill_kv_cache = bool(kwargs.get("fill_kv_cache", True))
        timer.current_context = "prefix_cross_attn" if fill_kv_cache else "decode_cross_attn"
        try:
            return original_forward_cross_attn_layer(*args, **kwargs)
        finally:
            timer.current_context = previous

    def timed_eager_attention_forward(
        self, attention_mask, batch_size, head_dim, query_states, key_states, value_states
    ):
        num_att_heads = self.num_attention_heads
        num_key_value_heads = self.num_key_value_heads
        num_key_value_groups = num_att_heads // num_key_value_heads
        sequence_length = key_states.shape[1]

        with timer.time("kv_expand_reshape", seq_len=int(sequence_length), heads=int(num_att_heads), head_dim=int(head_dim)):
            key_states = key_states[:, :, :, None, :].expand(
                batch_size, sequence_length, num_key_value_heads, num_key_value_groups, head_dim
            )
            key_states = key_states.reshape(
                batch_size, sequence_length, num_key_value_heads * num_key_value_groups, head_dim
            )
            value_states = value_states[:, :, :, None, :].expand(
                batch_size, sequence_length, num_key_value_heads, num_key_value_groups, head_dim
            )
            value_states = value_states.reshape(
                batch_size, sequence_length, num_key_value_heads * num_key_value_groups, head_dim
            )

        with timer.time("qk_cast_fp32", seq_len=int(sequence_length)):
            query_states = query_states.to(dtype=torch.float32)
            key_states = key_states.to(dtype=torch.float32)

        with timer.time("qk_transpose", seq_len=int(sequence_length)):
            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)

        with timer.time("qk_matmul", seq_len=int(sequence_length)):
            att_weights = torch.matmul(query_states, key_states.transpose(2, 3))

        with timer.time("score_scale", seq_len=int(sequence_length)):
            att_weights *= head_dim**-0.5

        with timer.time("mask_where", seq_len=int(sequence_length)):
            att_weights = att_weights.to(dtype=torch.float32)
            big_neg = torch.finfo(att_weights.dtype).min
            masked_att_weights = torch.where(attention_mask[:, None, :, :], att_weights, big_neg)

        with timer.time("softmax", seq_len=int(sequence_length)):
            probs = nn.functional.softmax(masked_att_weights, dim=-1)

        with timer.time("probs_cast", seq_len=int(sequence_length)):
            probs = probs.to(dtype=value_states.dtype)

        with timer.time("pv_matmul", seq_len=int(sequence_length)):
            att_output = torch.matmul(probs, value_states.permute(0, 2, 1, 3))

        with timer.time("output_permute_reshape", seq_len=int(sequence_length)):
            att_output = att_output.permute(0, 2, 1, 3)
            att_output = att_output.reshape(batch_size, -1, num_key_value_heads * num_key_value_groups * head_dim)

        return att_output

    target.forward_attn_layer = MethodType(timed_forward_attn_layer, target)
    target.forward_cross_attn_layer = MethodType(timed_forward_cross_attn_layer, target)
    target.eager_attention_forward = MethodType(timed_eager_attention_forward, target)
    target._step6_original_apply_rope = original_apply_rope


def install_projection_hooks(model: nn.Module, timer: SegmentTimer) -> list[Any]:
    handles = []

    def projection_kind(name: str) -> str:
        for suffix in ATTN_PROJ_SUFFIXES:
            if name.endswith(suffix):
                return suffix.rsplit(".", 1)[-1]
        return ""

    def make_pre_hook(name: str):
        def pre_hook(module, _inputs):
            if timer.use_cuda_events:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                module.__step6_profile_start = (start, end)
            else:
                module.__step6_profile_start = time.perf_counter()

        return pre_hook

    def make_post_hook(name: str):
        def post_hook(module, inputs, output):
            start = getattr(module, "__step6_profile_start", None)
            if start is None:
                return
            if timer.use_cuda_events:
                start_event, end_event = start
                end_event.record()
                torch.cuda.synchronize()
                elapsed_ms = float(start_event.elapsed_time(end_event))
            else:
                elapsed_ms = (time.perf_counter() - start) * 1000.0
            timer.rows.append(
                {
                    "sample_index": timer.current_sample,
                    "iteration": timer.current_iter,
                    "context": timer.current_context,
                    "segment": f"projection_{projection_kind(name)}",
                    "module": name,
                    "elapsed_ms": elapsed_ms,
                    "input_shape": list(inputs[0].shape) if inputs else None,
                    "output_shape": list(output.shape) if torch.is_tensor(output) else None,
                }
            )

        return post_hook

    for name, child in model.named_modules():
        if any(name.endswith(suffix) for suffix in ATTN_PROJ_SUFFIXES):
            handles.append(child.register_forward_pre_hook(make_pre_hook(name)))
            handles.append(child.register_forward_hook(make_post_hook(name)))
    return handles


def input_dtype(args: argparse.Namespace) -> torch.dtype:
    return torch.float32 if args.input_dtype == "fp32" else torch.bfloat16


def collect_inputs(args: argparse.Namespace, policy: nn.Module, preprocessor: Any, device: str):
    if args.input_source == "synthetic":
        return [
            {
                "sample_index": 0,
                "core_inputs": tuple(t.detach().cpu() for t in core_inputs_from_policy(
                    policy, preprocessor, args.task, args.seed, input_dtype(args), args.token_length
                )),
            }
        ]
    records, _rollout_device = collect_rollout_core_inputs(args, args.samples)
    return records


def maybe_apply_fake_quant(policy: nn.Module, args: argparse.Namespace) -> int:
    if not args.quantize_module_regex:
        return 0
    activation_scales_by_module = None
    if args.activation_scales_json:
        activation_scales_by_module = load_activation_scales_by_module(
            args.activation_scales_json,
            include_regexes=tuple(args.quantize_module_regex),
        )
    return replace_linear_modules(
        policy,
        include_prefixes=None,
        include_regexes=tuple(args.quantize_module_regex),
        activation_scales_by_module=activation_scales_by_module,
        quantization_kind="w8a8",
    )


def write_reports(args: argparse.Namespace, timer: SegmentTimer, output_dir: Path, replaced: int, device: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = timer.rows
    summary_rows = summarize(rows)
    total_by_segment = Counter()
    total_ms = 0.0
    for row in rows:
        elapsed = float(row["elapsed_ms"])
        total_by_segment[str(row["segment"])] += elapsed
        total_ms += elapsed

    report = {
        "policy_path": args.policy_path,
        "device": device,
        "input_source": args.input_source,
        "samples": args.samples,
        "warmup": args.warmup,
        "iters": args.iters,
        "model_dtype": args.model_dtype,
        "input_dtype": args.input_dtype,
        "quantize_module_regex": args.quantize_module_regex,
        "activation_scales_json": args.activation_scales_json,
        "fake_quant_replaced_linear_modules": replaced,
        "total_profiled_segment_ms": total_ms,
        "total_by_segment_ms": dict(total_by_segment),
        "summary_rows": summary_rows,
    }
    (output_dir / "attention_breakdown_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    with open(output_dir / "attention_breakdown_rows.csv", "w", newline="") as f:
        fieldnames = sorted({key for row in rows for key in row}) if rows else ["segment"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with open(output_dir / "attention_breakdown_summary.csv", "w", newline="") as f:
        fieldnames = sorted({key for row in summary_rows for key in row}) if summary_rows else ["segment"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    lines = [
        "# Step 6B Attention Breakdown",
        "",
        "| segment | total ms | share |",
        "| --- | ---: | ---: |",
    ]
    for segment, value in sorted(total_by_segment.items(), key=lambda item: -item[1]):
        share = 100.0 * float(value) / total_ms if total_ms > 0 else 0.0
        lines.append(f"| `{segment}` | `{float(value):.3f}` | `{share:.2f}%` |")
    lines += [
        "",
        "## By context",
        "",
        "| context | segment | calls | mean ms | p95 ms | total ms |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(summary_rows, key=lambda item: (-float(item["total_ms"]), item["context"], item["segment"])):
        lines.append(
            f"| `{row['context']}` | `{row['segment']}` | `{row['calls']}` | "
            f"`{row['mean_ms']:.6f}` | `{row['p95_ms']:.6f}` | `{row['total_ms']:.3f}` |"
        )
    (output_dir / "attention_breakdown_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((output_dir / "attention_breakdown_summary.md").read_text(), end="")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    policy, preprocessor, device = load_policy(args.policy_path, args.device, args.model_dtype)
    records = collect_inputs(args, policy, preprocessor, device)

    replaced = maybe_apply_fake_quant(policy, args)
    wrapper = SmolVLADebugCoreWrapper(policy).to(device=device).eval()
    timer = SegmentTimer(device)
    install_attention_body_profiler(wrapper, timer)
    handles = install_projection_hooks(wrapper, timer)

    try:
        selected = records[: max(1, args.samples)]
        with torch.no_grad():
            for _ in range(args.warmup):
                inputs = device_inputs(selected[0]["core_inputs"], device)
                _ = wrapper(*inputs)
            sync(device)

            for iteration in range(args.iters):
                record = selected[iteration % len(selected)]
                timer.current_sample = int(record.get("sample_index", iteration))
                timer.current_iter = iteration
                inputs = device_inputs(record["core_inputs"], device)
                with timer.time("action_core_e2e"):
                    _ = wrapper(*inputs)
                sync(device)
    finally:
        for handle in handles:
            handle.remove()
        if hasattr(smolvlm_with_expert, "apply_rope") and hasattr(wrapper.model.vlm_with_expert, "_step6_original_apply_rope"):
            smolvlm_with_expert.apply_rope = wrapper.model.vlm_with_expert._step6_original_apply_rope

    write_reports(args, timer, Path(args.output_dir), replaced, device)


if __name__ == "__main__":
    main()
