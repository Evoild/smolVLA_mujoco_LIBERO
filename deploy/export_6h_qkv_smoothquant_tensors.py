#!/usr/bin/env python3

"""Export real layer0 q/k/v SmoothQuant W8A8 tensors for Step 6H."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from export_6f_layer0_attention_tensors import collect_one_input, device_inputs, load_policy
from linear_only_quant import load_activation_scales_by_module, quantize_weight_per_out_channel
from profile_6b_attention_breakdown import SmolVLADebugCoreWrapper
from lerobot.utils.random_utils import set_seed


MODULES = {
    "q": "model.vlm_with_expert.vlm.model.text_model.layers.0.self_attn.q_proj",
    "k": "model.vlm_with_expert.vlm.model.text_model.layers.0.self_attn.k_proj",
    "v": "model.vlm_with_expert.vlm.model.text_model.layers.0.self_attn.v_proj",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", default="smolvla_libero")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--task", default="libero_spatial")
    parser.add_argument("--input-source", choices=["rollout", "synthetic"], default="rollout")
    parser.add_argument("--token-length", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument("--max-parallel-tasks", type=int, default=10)
    parser.add_argument("--model-dtype", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--input-dtype", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument(
        "--activation-scales-json",
        default="runs/deploy/5/O-smoothquant-alpha085-full-w8a8-deploy-cudagraph/smoothquant_alpha_0.85_text_and_lm_expert_activation_scales.json",
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=200)
    return parser.parse_args()


def write_binary(path: Path, tensor: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(tensor.detach().contiguous().cpu().numpy().tobytes())


def tensor_metrics(ref: torch.Tensor, got: torch.Tensor) -> dict[str, float]:
    ref_f = ref.detach().to(torch.float64).flatten()
    got_f = got.detach().to(torch.float64).flatten()
    diff = got_f - ref_f
    denom = torch.linalg.vector_norm(ref_f).clamp_min(1.0e-12)
    return {
        "cosine": float(F.cosine_similarity(ref_f, got_f, dim=0).item()),
        "relative_l2": float((torch.linalg.vector_norm(diff) / denom).item()),
        "max_abs": float(diff.abs().max().item()),
    }


def time_cuda(args: argparse.Namespace, fn) -> float:
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


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    policy, preprocessor, device = load_policy(args.policy_path, args.device, args.model_dtype)
    scales = load_activation_scales_by_module(args.activation_scales_json, tuple(rf"^{name}$" for name in MODULES.values()))
    record = collect_one_input(args, policy, preprocessor, device)
    inputs = device_inputs(record["core_inputs"], device)
    wrapper = SmolVLADebugCoreWrapper(policy).to(device=device).eval()
    model = wrapper.model
    prefix_embs, _, _ = wrapper.embed_prefix_from_image_embeds(*inputs[:-1])
    text_model = model.vlm_with_expert.get_vlm_model().text_model
    layer0 = model.vlm_with_expert.get_model_layers([text_model, model.vlm_with_expert.lm_expert])[0][0]

    with torch.no_grad():
        norm_states = layer0.input_layernorm(prefix_embs)
        x = norm_states.reshape(-1, norm_states.shape[-1]).to(torch.float32)
        write_binary(out / "x.fp32.bin", x)

        module_objs = {"q": layer0.self_attn.q_proj, "k": layer0.self_attn.k_proj, "v": layer0.self_attn.v_proj}
        entries = {}
        refs = []
        for short, module_name in MODULES.items():
            linear = module_objs[short]
            if not isinstance(linear, nn.Linear):
                raise TypeError(f"{short} is not nn.Linear: {type(linear)}")
            scale_entry = scales[module_name]
            if not isinstance(scale_entry, dict) or "smooth_scale" not in scale_entry:
                raise ValueError(f"missing SmoothQuant scale for {module_name}")
            activation_scale = float(scale_entry["scale"])
            smooth_scale = torch.as_tensor(scale_entry["smooth_scale"], device=device, dtype=torch.float32).clamp_min(1.0e-8)
            weight = linear.weight.detach().to(device=device, dtype=torch.float32)
            bias = linear.bias.detach().to(device=device, dtype=torch.float32) if linear.bias is not None else torch.zeros(weight.shape[0], device=device)
            qweight, weight_scale = quantize_weight_per_out_channel((weight * smooth_scale[None, :]).cpu())
            qweight = qweight.to(device=device)
            weight_scale = weight_scale.to(device=device, dtype=torch.float32)
            qx = torch.round((x / smooth_scale) / activation_scale).clamp(-128, 127).to(torch.int8)
            ref = F.linear(qx.to(torch.float32) * activation_scale, qweight.to(torch.float32) * weight_scale[:, None], bias)
            ref2 = F.linear(qx.to(torch.float32) * activation_scale, qweight.to(torch.float32) * weight_scale[:, None], bias)
            refs.append(ref)

            write_binary(out / f"{short}_qweight.int8.bin", qweight.to(torch.int8))
            write_binary(out / f"{short}_weight_scale.fp32.bin", weight_scale)
            write_binary(out / f"{short}_bias.fp32.bin", bias)
            write_binary(out / f"{short}_smooth_scale.fp32.bin", smooth_scale)
            write_binary(out / f"{short}_reference.fp32.bin", ref.to(torch.float32))
            entries[short] = {
                "module": module_name,
                "activation_scale": activation_scale,
                "smoothquant_alpha": scale_entry.get("smoothquant_alpha"),
                "quantization": scale_entry.get("quantization"),
                "shape": {"m": int(x.shape[0]), "k": int(x.shape[1]), "n": int(weight.shape[0])},
                "reference_self_metrics": tensor_metrics(ref, ref2),
            }

        latency_ms = time_cuda(args, lambda: tuple(ref.clone() for ref in refs)) if device.startswith("cuda") else 0.0

    meta = {
        "policy_path": args.policy_path,
        "task": args.task,
        "seed": args.seed,
        "input_source": args.input_source,
        "sample_index": int(record.get("sample_index", 0)),
        "activation_scales_json": args.activation_scales_json,
        "input_shape": {"m": int(x.shape[0]), "k": int(x.shape[1])},
        "modules": entries,
        "reference_clone_latency_ms": latency_ms,
        "files": {
            "x": "x.fp32.bin",
            "per_module": {
                short: {
                    "qweight": f"{short}_qweight.int8.bin",
                    "weight_scale": f"{short}_weight_scale.fp32.bin",
                    "bias": f"{short}_bias.fp32.bin",
                    "smooth_scale": f"{short}_smooth_scale.fp32.bin",
                    "reference": f"{short}_reference.fp32.bin",
                }
                for short in MODULES
            },
        },
    }
    (out / "qkv_smoothquant_tensors_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
