#!/usr/bin/env python3

"""Validate Step 6D fused RoPE/layout math against the PyTorch reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", default="runs/deploy/6/D-fused-rope-layout/fused_rope_layout_math_check.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="fp32")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=177)
    parser.add_argument("--q-heads", type=int, default=15)
    parser.add_argument("--kv-heads", type=int, default=5)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1000)
    return parser.parse_args()


def apply_rope_ref(x: torch.Tensor, positions: torch.Tensor, max_wavelength: float = 10_000.0) -> torch.Tensor:
    d_half = x.shape[-1] // 2
    dtype = x.dtype
    xf = x.to(torch.float32)
    freq_exponents = (2.0 / x.shape[-1]) * torch.arange(d_half, dtype=torch.float32, device=x.device)
    timescale = max_wavelength**freq_exponents
    radians = positions[..., None].to(torch.float32) / timescale[None, None, :]
    radians = radians[..., None, :]
    sin = torch.sin(radians)
    cos = torch.cos(radians)
    x1, x2 = xf.split(d_half, dim=-1)
    out = torch.empty_like(xf)
    out[..., :d_half] = x1 * cos - x2 * sin
    out[..., d_half:] = x2 * cos + x1 * sin
    return out.to(dtype)


def fused_ref(q_raw: torch.Tensor, k_raw: torch.Tensor, positions: torch.Tensor, q_heads: int, kv_heads: int, head_dim: int):
    bsz, seq_len, _ = q_raw.shape
    groups = q_heads // kv_heads
    q = q_raw.view(bsz, seq_len, q_heads, head_dim)
    k = k_raw.view(bsz, seq_len, kv_heads, head_dim)
    q = apply_rope_ref(q, positions)
    k = apply_rope_ref(k, positions)
    k = k[:, :, :, None, :].expand(bsz, seq_len, kv_heads, groups, head_dim)
    k = k.reshape(bsz, seq_len, q_heads, head_dim)
    return q.transpose(1, 2).contiguous(), k.transpose(1, 2).transpose(2, 3).contiguous()


def metrics(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    af = a.flatten().to(torch.float32)
    bf = b.flatten().to(torch.float32)
    diff = af - bf
    return {
        "cosine": float(torch.nn.functional.cosine_similarity(af, bf, dim=0).item()),
        "relative_l2": float((torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(af).clamp_min(1e-12)).item()),
        "max_abs": float(diff.abs().max().item()),
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    device = torch.device(args.device)
    q_raw = torch.randn(args.batch, args.seq_len, args.q_heads * args.head_dim, device=device, dtype=dtype)
    k_raw = torch.randn(args.batch, args.seq_len, args.kv_heads * args.head_dim, device=device, dtype=dtype)
    positions = torch.arange(args.seq_len, device=device, dtype=torch.int64).unsqueeze(0).expand(args.batch, -1)

    q_ref = apply_rope_ref(q_raw.view(args.batch, args.seq_len, args.q_heads, args.head_dim), positions)
    k_ref = apply_rope_ref(k_raw.view(args.batch, args.seq_len, args.kv_heads, args.head_dim), positions)
    k_ref = k_ref[:, :, :, None, :].expand(args.batch, args.seq_len, args.kv_heads, args.q_heads // args.kv_heads, args.head_dim)
    k_ref = k_ref.reshape(args.batch, args.seq_len, args.q_heads, args.head_dim)
    q_ref = q_ref.transpose(1, 2).contiguous()
    k_ref = k_ref.transpose(1, 2).transpose(2, 3).contiguous()

    q_fused, k_fused = fused_ref(q_raw, k_raw, positions, args.q_heads, args.kv_heads, args.head_dim)
    report = {
        "device": str(device),
        "dtype": args.dtype,
        "batch": args.batch,
        "seq_len": args.seq_len,
        "q_heads": args.q_heads,
        "kv_heads": args.kv_heads,
        "head_dim": args.head_dim,
        "q_output_shape": list(q_fused.shape),
        "k_output_shape": list(k_fused.shape),
        "q_metrics": metrics(q_ref, q_fused),
        "k_metrics": metrics(k_ref, k_fused),
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

