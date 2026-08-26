#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import onnx


PROJECTIONS = ("q", "k", "v", "o", "gate", "up")
PROJ_TO_NODE = {
    "q": "q_proj",
    "k": "k_proj",
    "v": "v_proj",
    "o": "o_proj",
    "gate": "mlp/gate_proj",
    "up": "mlp/up_proj",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a strict 32x6 VLM W8A8 target table from ONNX Q/DQ and TensorRT layer_info."
    )
    parser.add_argument("--onnx", required=True, help="Q/DQ ONNX model.")
    parser.add_argument("--layer-info", required=True, help="TensorRT layer_info.json exported by trtexec.")
    parser.add_argument("--output-csv", required=True, help="Output CSV table.")
    parser.add_argument("--output-md", required=True, help="Output Markdown table.")
    parser.add_argument("--output-summary", required=True, help="Output JSON summary.")
    return parser.parse_args()


def find_layers(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if not isinstance(data, dict):
        return []
    for key in ("Layers", "layers", "layerInfo", "LayerInfo"):
        value = data.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def walk_strings(value: Any) -> list[str]:
    out: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            out.append(str(key))
            out.extend(walk_strings(item))
    elif isinstance(value, list):
        for item in value:
            out.extend(walk_strings(item))
    elif value is not None:
        out.append(str(value))
    return out


def layer_name(layer: dict[str, Any]) -> str:
    for key in ("Name", "name", "LayerName", "layerName"):
        if key in layer:
            return str(layer[key])
    strings = walk_strings(layer)
    return strings[0] if strings else ""


def tactic_name(layer: dict[str, Any]) -> str:
    for key in ("TacticName", "tacticName", "Tactic", "tactic"):
        if key in layer:
            return str(layer[key])
    return ""


def precision_from_tactic(tactic: str, layer: dict[str, Any]) -> str:
    # For GEMM proof, the tactic name is the source of truth. Metadata can contain
    # original Q/DQ node names even when the selected GEMM tactic is FP32/TF32.
    tactic_text = tactic.upper()
    tactic_lower = tactic.lower()
    if re.search(r"(^|_)i8(f32|i8|i32|_)", tactic_lower):
        return "INT8"
    if "F32F32_TF32" in tactic_text or "TF32" in tactic_text:
        return "FP32/TF32"
    if "BF16" in tactic_text:
        return "BF16"
    if "FP16" in tactic_text or "F16" in tactic_text:
        return "FP16"
    if "FP32" in tactic_text or "F32" in tactic_text:
        return "FP32"

    text = " ".join(walk_strings(layer)).upper()
    if "INT8" in text:
        return "INT8"
    if "BF16" in text or "BFLOAT16" in text:
        return "BF16"
    if "FP16" in text or "FLOAT16" in text or "HALF" in text:
        return "FP16"
    if "TF32" in text:
        return "FP32/TF32"
    if "FP32" in text or "FLOAT32" in text or re.search(r"\bFLOAT\b", text):
        return "FP32"
    return "unknown"


def is_gemm_like(layer: dict[str, Any]) -> bool:
    name = layer_name(layer).lower()
    tactic = tactic_name(layer).lower()
    return "gemm" in tactic or "matmul" in name or re.search(r"(^|_)fc", name) is not None


def target_node(layer_index: int, projection: str) -> str:
    base = PROJ_TO_NODE[projection]
    suffix = "" if layer_index == 0 else f"_{layer_index}"
    return f"/debug_core/{base}{suffix}/MatMul"


def short_node_name(node_name: str) -> str:
    return node_name.removeprefix("/debug_core")


def qdq_status(node: onnx.NodeProto | None, producers: dict[str, onnx.NodeProto]) -> str:
    if node is None:
        return "missing_node"
    input_producers = [producers.get(name) for name in node.input]
    dq_count = sum(1 for producer in input_producers if producer is not None and producer.op_type == "DequantizeLinear")
    if dq_count >= 2:
        return "yes"
    if dq_count == 1:
        return "partial"
    return "no"


def best_matching_layers(target: str, layers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for layer in layers:
        text = json.dumps(layer, ensure_ascii=False)
        if target in text:
            matches.append(layer)

    gemm_matches = [layer for layer in matches if is_gemm_like(layer)]
    if gemm_matches:
        return gemm_matches
    return matches


def compact_join(values: list[str], limit: int = 3) -> str:
    cleaned = [value for value in values if value]
    if len(cleaned) <= limit:
        return "; ".join(cleaned)
    return "; ".join(cleaned[:limit]) + f"; ...(+{len(cleaned) - limit})"


def write_markdown(rows: list[dict[str, str]], path: Path) -> None:
    headers = [
        "VLM layer",
        "projection",
        "ONNX node",
        "Q/DQ exists",
        "TRT fused layer",
        "tactic",
        "GEMM precision",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        values = [
            row["vlm_layer"],
            row["projection"],
            f"`{row['onnx_node']}`",
            row["qdq_exists"],
            f"`{row['trt_fused_layer']}`" if row["trt_fused_layer"] else "",
            f"`{row['tactic']}`" if row["tactic"] else "",
            row["gemm_precision"],
        ]
        lines.append("| " + " | ".join(value.replace("|", "\\|") for value in values) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    onnx_path = Path(args.onnx)
    layer_info_path = Path(args.layer_info)

    model = onnx.load(str(onnx_path), load_external_data=False)
    nodes_by_name = {node.name: node for node in model.graph.node}
    producers = {output: node for node in model.graph.node for output in node.output}

    layer_info = json.loads(layer_info_path.read_text())
    trt_layers = find_layers(layer_info)

    rows: list[dict[str, str]] = []
    for layer_index in range(32):
        for projection in PROJECTIONS:
            node_name = target_node(layer_index, projection)
            node = nodes_by_name.get(node_name)
            matched_layers = best_matching_layers(node_name, trt_layers)
            tactics = [tactic_name(layer) for layer in matched_layers]
            precisions = [precision_from_tactic(tactic_name(layer), layer) for layer in matched_layers]

            if "INT8" in precisions:
                precision = "INT8"
            elif any(item == "FP32/TF32" for item in precisions):
                precision = "FP32/TF32"
            elif precisions:
                precision = precisions[0]
            else:
                precision = "missing_trt_layer"

            rows.append(
                {
                    "vlm_layer": str(layer_index),
                    "projection": projection,
                    "onnx_node": short_node_name(node_name),
                    "qdq_exists": qdq_status(node, producers),
                    "trt_fused_layer": compact_join([layer_name(layer) for layer in matched_layers]),
                    "tactic": compact_join(tactics),
                    "gemm_precision": precision,
                    "matched_trt_layers": str(len(matched_layers)),
                }
            )

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    write_markdown(rows, Path(args.output_md))

    precision_counts = Counter(row["gemm_precision"] for row in rows)
    qdq_counts = Counter(row["qdq_exists"] for row in rows)
    by_projection = {
        projection: dict(Counter(row["gemm_precision"] for row in rows if row["projection"] == projection))
        for projection in PROJECTIONS
    }
    summary = {
        "onnx": str(onnx_path),
        "layer_info": str(layer_info_path),
        "num_strict_vlm_w8a8_targets": len(rows),
        "qdq_counts": dict(qdq_counts),
        "gemm_precision_counts": dict(precision_counts),
        "gemm_precision_by_projection": by_projection,
        "note": (
            "Strict target set is VLM text layers 0..31 and projections q/k/v/o/gate/up only. "
            "Action expert repeated q/k/v/o/mlp nodes and W8A16 down_proj are intentionally excluded."
        ),
    }
    output_summary = Path(args.output_summary)
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    output_summary.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
