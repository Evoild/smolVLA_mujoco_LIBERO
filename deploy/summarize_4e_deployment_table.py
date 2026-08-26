#!/usr/bin/env python3

"""Summarize BF16 vs a quantized TensorRT deployment metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import onnx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--bf16-engine", required=True)
    parser.add_argument("--bf16-onnx", required=True)
    parser.add_argument("--bf16-profile", required=True)
    parser.add_argument("--bf16-eval", required=True)
    parser.add_argument("--mlp-engine", required=True)
    parser.add_argument("--mlp-onnx", required=True)
    parser.add_argument("--mlp-profile", required=True)
    parser.add_argument("--mlp-eval", required=True)
    parser.add_argument("--quant-config", default="MLP W8A8")
    return parser.parse_args()


def tensor_storage_bytes(tensor: onnx.TensorProto) -> int:
    if tensor.raw_data:
        return len(tensor.raw_data)
    elem_size = {
        onnx.TensorProto.FLOAT: 4,
        onnx.TensorProto.FLOAT16: 2,
        onnx.TensorProto.BFLOAT16: 2,
        onnx.TensorProto.DOUBLE: 8,
        onnx.TensorProto.INT8: 1,
        onnx.TensorProto.UINT8: 1,
        onnx.TensorProto.INT32: 4,
        onnx.TensorProto.INT64: 8,
        onnx.TensorProto.BOOL: 1,
    }.get(tensor.data_type, 4)
    count = 1
    for dim in tensor.dims:
        count *= int(dim)
    return count * elem_size


def onnx_initializer_mb(path: str | Path) -> float:
    model = onnx.load(path, load_external_data=False)
    used = {name for node in model.graph.node for name in node.input}
    return sum(tensor_storage_bytes(tensor) for tensor in model.graph.initializer if tensor.name in used) / (1024**2)


def load_json(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def success_rate(eval_info: dict[str, Any]) -> float | None:
    successes = []
    for task in eval_info.get("per_task", []):
        successes.extend(bool(item) for item in task.get("metrics", {}).get("successes", []))
    if not successes:
        overall = eval_info.get("overall", {})
        for key in ("success_rate", "mean_success_rate"):
            if key in overall:
                return float(overall[key])
        return None
    return sum(successes) / len(successes)


def row(config: str, engine: str, onnx_path: str, profile_path: str, eval_path: str) -> dict[str, Any]:
    profile = load_json(profile_path)
    eval_info = load_json(eval_path)
    sr = success_rate(eval_info)
    peak = profile.get("peak_gpu_memory_mb")
    if peak is None:
        peak = eval_info.get("peak_gpu_memory_mb")
    return {
        "配置": config,
        "Engine Size MB": Path(engine).stat().st_size / (1024**2),
        "Estimated Weight Memory MB": onnx_initializer_mb(onnx_path),
        "Peak GPU Memory MB": peak,
        "Latency ms": profile.get("policy_inference_e2e_mean_ms"),
        "Success Rate": sr,
        "FPS": profile.get("policy_inference_fps"),
        "engine": engine,
        "onnx": onnx_path,
        "profile": profile_path,
        "eval": eval_path,
    }


def fmt_mb(value: Any) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.2f} MB"


def fmt_gb(value: Any) -> str:
    if value is None:
        return "N/A"
    return f"{float(value) / 1024.0:.2f} GB"


def fmt_ms(value: Any) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.2f} ms"


def fmt_rate(value: Any) -> str:
    if value is None:
        return "N/A"
    return f"{float(value) * 100.0:.1f}%"


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    rows = [
        row("BF16", args.bf16_engine, args.bf16_onnx, args.bf16_profile, args.bf16_eval),
        row(args.quant_config, args.mlp_engine, args.mlp_onnx, args.mlp_profile, args.mlp_eval),
    ]
    with open(output_root / "deployment_table.json", "w") as f:
        json.dump({"rows": rows}, f, indent=2)
    with open(output_root / "deployment_table.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    markdown = [
        "| 配置 | Engine Size | Estimated Weight Memory | Peak GPU Memory | Latency | Success Rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in rows:
        markdown.append(
            "| {config} | {engine_size} | {weight_mem} | {peak_mem} | {latency} | {success} |".format(
                config=item["配置"],
                engine_size=fmt_mb(item["Engine Size MB"]),
                weight_mem=fmt_mb(item["Estimated Weight Memory MB"]),
                peak_mem=fmt_gb(item["Peak GPU Memory MB"]),
                latency=fmt_ms(item["Latency ms"]),
                success=fmt_rate(item["Success Rate"]),
            )
        )
    text = "\n".join(markdown) + "\n"
    (output_root / "deployment_table.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
