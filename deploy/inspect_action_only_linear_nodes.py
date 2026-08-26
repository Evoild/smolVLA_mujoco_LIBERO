#!/usr/bin/env python3

"""Inspect constant-weight Linear nodes in a SmolVLA action-only ONNX graph."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import onnx
from onnx import TensorProto, numpy_helper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--name-regex", default=None)
    return parser.parse_args()


def initializer_shape(tensor: TensorProto) -> list[int]:
    return [int(dim) for dim in tensor.dims]


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(args.name_regex) if args.name_regex else None

    model = onnx.load(args.onnx, load_external_data=False)
    initializers = {tensor.name: tensor for tensor in model.graph.initializer}

    rows: list[dict[str, object]] = []
    for idx, node in enumerate(model.graph.node):
        node_name = node.name or f"{node.op_type}_{idx}"
        if node.op_type not in {"Gemm", "MatMul"}:
            continue
        if pattern and not pattern.search(node_name):
            continue
        if len(node.input) < 2 or node.input[1] not in initializers:
            continue
        weight = initializers[node.input[1]]
        rows.append(
            {
                "node_index": idx,
                "node_name": node_name,
                "op_type": node.op_type,
                "input_name": node.input[0],
                "weight_name": node.input[1],
                "weight_shape": "x".join(str(dim) for dim in initializer_shape(weight)),
                "weight_dtype": TensorProto.DataType.Name(weight.data_type),
                "output_name": node.output[0] if node.output else "",
            }
        )

    summary = {
        "onnx": args.onnx,
        "name_regex": args.name_regex,
        "constant_weight_linear_nodes": len(rows),
        "weight_shape_counts": {},
    }
    for row in rows:
        key = str(row["weight_shape"])
        summary["weight_shape_counts"][key] = int(summary["weight_shape_counts"].get(key, 0)) + 1

    (output_dir / "linear_nodes_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "linear_nodes.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with (output_dir / "linear_nodes.csv").open("w", newline="") as f:
        fieldnames = ["node_index", "node_name", "op_type", "input_name", "weight_name", "weight_shape", "weight_dtype", "output_name"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
