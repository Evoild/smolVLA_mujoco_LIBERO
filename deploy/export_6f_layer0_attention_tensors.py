#!/usr/bin/env python3

"""Export real layer0 VLM attention tensors for Step 6F."""

from __future__ import annotations

import argparse
import json
import time
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
    make_att_2d_masks_trt,
)
from diagnose_5i_attention_error_propagation import expand_kv_states
from lerobot.policies.smolvla.smolvlm_with_expert import apply_rope
from linear_only_quant import load_activation_scales_by_module, replace_linear_modules
from lerobot.utils.random_utils import set_seed


FULL_W8A8_REGEX = (
    r"^model\.vlm_with_expert\.vlm\.model\.text_model\.layers\.[0-9]+"
    r"\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", default="smolvla_libero")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--task", default="libero_spatial")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--token-length", type=int, default=48)
    parser.add_argument("--input-source", choices=["synthetic", "rollout"], default="rollout")
    parser.add_argument("--sample-stride", type=int, default=5)
    parser.add_argument("--max-parallel-tasks", type=int, default=1)
    parser.add_argument("--model-dtype", choices=["native", "bf16"], default="bf16")
    parser.add_argument("--input-dtype", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--activation-scales-json", default=None)
    parser.add_argument("--quantize-module-regex", default=FULL_W8A8_REGEX)
    parser.add_argument("--disable-fake-quant", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    return parser.parse_args()


def input_dtype(args: argparse.Namespace) -> torch.dtype:
    return torch.float32 if args.input_dtype == "fp32" else torch.bfloat16


def tensor_metrics(ref: torch.Tensor, other: torch.Tensor) -> dict[str, float]:
    lhs = ref.detach().to(torch.float32).flatten()
    rhs = other.detach().to(torch.float32).flatten()
    diff = rhs - lhs
    ref_norm = torch.linalg.vector_norm(lhs).clamp_min(1.0e-12)
    return {
        "cosine": float(torch.nn.functional.cosine_similarity(lhs[None], rhs[None], dim=1).item()),
        "relative_l2": float((torch.linalg.vector_norm(diff) / ref_norm).item()),
        "max_abs": float(diff.abs().max().item()),
    }


def maybe_apply_fake_quant(policy: nn.Module, args: argparse.Namespace) -> int:
    if args.disable_fake_quant:
        return 0
    if not args.activation_scales_json:
        return 0
    scales = load_activation_scales_by_module(
        args.activation_scales_json,
        include_regexes=(args.quantize_module_regex,),
    )
    return replace_linear_modules(
        policy,
        include_prefixes=None,
        include_regexes=(args.quantize_module_regex,),
        activation_scales_by_module=scales,
        quantization_kind="w8a8",
    )


def collect_one_input(args: argparse.Namespace, policy: nn.Module, preprocessor: Any, device: str):
    if args.input_source == "synthetic":
        core_inputs = core_inputs_from_policy(policy, preprocessor, args.task, args.seed, input_dtype(args), args.token_length)
        return {"sample_index": 0, "input_source": "synthetic", "core_inputs": tuple(t.detach().cpu() for t in core_inputs)}
    records, _ = collect_rollout_core_inputs(args, 1)
    return records[0]


def reference_attention(
    vlm_with_expert: nn.Module,
    q_raw: torch.Tensor,
    k_raw: torch.Tensor,
    v_raw: torch.Tensor,
    position_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    batch_size = q_raw.shape[0]
    head_dim = vlm_with_expert.vlm.config.text_config.head_dim
    q_states = apply_rope(q_raw, position_ids)
    k_states = apply_rope(k_raw, position_ids)
    k_expanded, v_expanded = expand_kv_states(vlm_with_expert, batch_size, k_states, v_raw)
    q_t = q_states.to(torch.float32).transpose(1, 2)
    k_t = k_expanded.to(torch.float32).transpose(1, 2)
    scores = torch.matmul(q_t, k_t.transpose(2, 3)) * (head_dim**-0.5)
    scores = torch.where(attention_mask[:, None, :, :], scores, torch.finfo(scores.dtype).min)
    probs = nn.functional.softmax(scores, dim=-1)
    context = torch.matmul(probs.to(dtype=v_expanded.dtype), v_expanded.permute(0, 2, 1, 3))
    return context.to(torch.float32)


def time_reference(args: argparse.Namespace, fn) -> float:
    if str(args.device).startswith("cuda"):
        for _ in range(args.warmup):
            _ = fn()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.iters):
            _ = fn()
        end.record()
        torch.cuda.synchronize()
        return float(start.elapsed_time(end) / max(1, args.iters))
    for _ in range(args.warmup):
        _ = fn()
    start = time.perf_counter()
    for _ in range(args.iters):
        _ = fn()
    return float((time.perf_counter() - start) * 1000.0 / max(1, args.iters))


def write_binary(path: Path, tensor: torch.Tensor) -> None:
    array = tensor.detach().contiguous().cpu().numpy()
    array.tofile(path)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    policy, preprocessor, device = load_policy(args.policy_path, args.device, args.model_dtype)
    replaced = maybe_apply_fake_quant(policy, args)
    record = collect_one_input(args, policy, preprocessor, device)
    inputs = device_inputs(record["core_inputs"], device)

    wrapper = SmolVLADebugCoreWrapper(policy).to(device=device).eval()
    model = wrapper.model
    prefix_embs, prefix_pad_masks, prefix_att_masks = wrapper.embed_prefix_from_image_embeds(*inputs[:-1])
    attention_mask = make_att_2d_masks_trt(prefix_pad_masks, prefix_att_masks)
    position_ids = torch.cumsum(prefix_pad_masks.to(torch.int64), dim=1) - 1

    vlm_with_expert = model.vlm_with_expert
    text_model = vlm_with_expert.get_vlm_model().text_model
    layer0 = vlm_with_expert.get_model_layers([text_model, vlm_with_expert.lm_expert])[0][0]

    with torch.no_grad():
        hidden = prefix_embs
        norm_states = layer0.input_layernorm(hidden)
        input_shape = norm_states.shape[:-1]
        hidden_shape = (*input_shape, -1, layer0.self_attn.head_dim)
        proj_input = norm_states.to(dtype=layer0.self_attn.q_proj.weight.dtype)
        q_raw = layer0.self_attn.q_proj(proj_input).view(hidden_shape).to(torch.float32)
        k_raw = layer0.self_attn.k_proj(proj_input).view(hidden_shape).to(torch.float32)
        v_raw = layer0.self_attn.v_proj(proj_input).view(hidden_shape).to(torch.float32)
        context_ref = reference_attention(vlm_with_expert, q_raw, k_raw, v_raw, position_ids, attention_mask)
        context_ref_2 = reference_attention(vlm_with_expert, q_raw, k_raw, v_raw, position_ids, attention_mask)

    ref_latency_ms = time_reference(args, lambda: reference_attention(vlm_with_expert, q_raw, k_raw, v_raw, position_ids, attention_mask))
    ref_metrics = tensor_metrics(context_ref, context_ref_2)

    files = {
        "q_raw": "q_raw.fp32.bin",
        "k_raw": "k_raw.fp32.bin",
        "v_raw": "v_raw.fp32.bin",
        "position_ids": "position_ids.int64.bin",
        "attention_mask": "attention_mask.bool.bin",
        "context_ref": "context_ref.fp32.bin",
        "hidden": "layer0_hidden.fp32.bin",
        "norm": "layer0_norm.fp32.bin",
    }
    write_binary(output_dir / files["q_raw"], q_raw)
    write_binary(output_dir / files["k_raw"], k_raw)
    write_binary(output_dir / files["v_raw"], v_raw)
    write_binary(output_dir / files["position_ids"], position_ids.to(torch.int64))
    write_binary(output_dir / files["attention_mask"], attention_mask.to(torch.bool))
    write_binary(output_dir / files["context_ref"], context_ref)
    write_binary(output_dir / files["hidden"], hidden.to(torch.float32))
    write_binary(output_dir / files["norm"], norm_states.to(torch.float32))

    meta = {
        "policy_path": args.policy_path,
        "task": args.task,
        "seed": args.seed,
        "input_source": record.get("input_source"),
        "sample_index": record.get("sample_index"),
        "model_dtype": args.model_dtype,
        "input_dtype": args.input_dtype,
        "activation_scales_json": args.activation_scales_json,
        "quantize_module_regex": None if args.disable_fake_quant else args.quantize_module_regex,
        "fake_quant_replaced_linear_modules": replaced,
        "batch": int(q_raw.shape[0]),
        "seq_len": int(q_raw.shape[1]),
        "q_heads": int(vlm_with_expert.num_attention_heads),
        "kv_heads": int(vlm_with_expert.num_key_value_heads),
        "head_dim": int(layer0.self_attn.head_dim),
        "q_raw_shape": list(q_raw.shape),
        "k_raw_shape": list(k_raw.shape),
        "v_raw_shape": list(v_raw.shape),
        "position_ids_shape": list(position_ids.shape),
        "attention_mask_shape": list(attention_mask.shape),
        "context_ref_shape": list(context_ref.shape),
        "position_ids_unique": sorted(int(v) for v in torch.unique(position_ids).detach().cpu().tolist()),
        "attention_mask_true_ratio": float(attention_mask.to(torch.float32).mean().item()),
        "reference_self_metrics": ref_metrics,
        "reference_attention_latency_ms": ref_latency_ms,
        "reference_latency_iters": args.iters,
        "files": files,
    }
    (output_dir / "layer0_attention_tensors_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
