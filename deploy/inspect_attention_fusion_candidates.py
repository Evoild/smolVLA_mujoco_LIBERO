#!/usr/bin/env python3

"""Inspect whether SmolVLA action-only ONNX keeps attention as fuseable pieces.

This is a deployment feasibility check for step 6. It does not create a fused
kernel. It reports whether q/k/v projections are still explicit Q/DQ MatMuls and
whether RoPE + attention remain decomposed into ONNX ops, which would require a
TensorRT plugin/custom kernel to keep INT8 inside attention.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

import onnx


ATTN_PROJ_RE = re.compile(
    r"^/(?:debug_core/)?(?P<proj>q_proj|k_proj|v_proj|o_proj)(?:_(?P<layer>[0-9]+))?/MatMul$"
)
MLP_PROJ_RE = re.compile(
    r"^/(?:debug_core/)?mlp/(?P<proj>gate_proj|up_proj|down_proj)(?:_(?P<layer>[0-9]+))?/MatMul$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", required=True, help="Action-only ONNX/QDQ ONNX to inspect.")
    parser.add_argument("--qdq-report", default=None, help="Optional insert_linear_w8a8_qdq report JSON.")
    parser.add_argument("--output-dir", required=True, help="Directory for JSON/Markdown reports.")
    parser.add_argument("--max-depth", type=int, default=10, help="Producer graph search depth from q/k/v outputs.")
    return parser.parse_args()


def node_name(node: onnx.NodeProto, index: int) -> str:
    return node.name or f"{node.op_type}_{index}"


def producer_map(graph: onnx.GraphProto) -> dict[str, onnx.NodeProto]:
    producers: dict[str, onnx.NodeProto] = {}
    for node in graph.node:
        for output in node.output:
            producers[output] = node
    return producers


def consumer_map(graph: onnx.GraphProto) -> dict[str, list[onnx.NodeProto]]:
    consumers: dict[str, list[onnx.NodeProto]] = defaultdict(list)
    for node in graph.node:
        for input_name in node.input:
            consumers[input_name].append(node)
    return consumers


def constant_inputs(graph: onnx.GraphProto) -> set[str]:
    return {tensor.name for tensor in graph.initializer}


def projection_rows(graph: onnx.GraphProto) -> list[dict[str, Any]]:
    constants = constant_inputs(graph)
    rows: list[dict[str, Any]] = []
    for index, node in enumerate(graph.node):
        if node.op_type != "MatMul":
            continue
        name = node_name(node, index)
        match = ATTN_PROJ_RE.search(name) or MLP_PROJ_RE.search(name)
        if not match:
            continue
        family = "attention" if ATTN_PROJ_RE.search(name) else "mlp"
        rows.append(
            {
                "node": name,
                "family": family,
                "projection": match.group("proj"),
                "layer": int(match.group("layer") or 0),
                "rhs_constant": len(node.input) > 1 and node.input[1] in constants,
                "input0": node.input[0] if node.input else "",
                "input1": node.input[1] if len(node.input) > 1 else "",
                "output": node.output[0] if node.output else "",
            }
        )
    return rows


def direct_qdq_status(graph: onnx.GraphProto, rows: list[dict[str, Any]]) -> dict[str, Any]:
    producers = producer_map(graph)
    consumers = consumer_map(graph)
    status = Counter()
    for row in rows:
        input0_producer = producers.get(row["input0"])
        input1_producer = producers.get(row["input1"])
        output_consumers = consumers.get(row["output"], [])
        if input0_producer and input0_producer.op_type == "DequantizeLinear":
            status[f"{row['family']}_{row['projection']}_activation_dq_input"] += 1
        if input1_producer and input1_producer.op_type == "DequantizeLinear":
            status[f"{row['family']}_{row['projection']}_weight_dq_input"] += 1
        if any(node.op_type == "Cast" for node in output_consumers):
            status[f"{row['family']}_{row['projection']}_output_cast_consumer"] += 1
    return dict(status)


def downstream_ops(graph: onnx.GraphProto, rows: list[dict[str, Any]], max_depth: int) -> dict[str, Any]:
    consumers = consumer_map(graph)
    interesting = {"q_proj", "k_proj", "v_proj"}
    by_projection: dict[str, Counter[str]] = {proj: Counter() for proj in interesting}
    examples: dict[str, list[str]] = {proj: [] for proj in interesting}

    for row in rows:
        if row["family"] != "attention" or row["projection"] not in interesting:
            continue
        seen_tensors = {row["output"]}
        queue = deque([(row["output"], 0)])
        while queue:
            tensor, depth = queue.popleft()
            if depth >= max_depth:
                continue
            for node in consumers.get(tensor, []):
                name = node.name or node.op_type
                by_projection[row["projection"]][node.op_type] += 1
                if len(examples[row["projection"]]) < 12:
                    examples[row["projection"]].append(f"{node.op_type}:{name}")
                for output in node.output:
                    if output not in seen_tensors:
                        seen_tensors.add(output)
                        queue.append((output, depth + 1))
    return {
        proj: {
            "op_counts": dict(counter),
            "examples": examples[proj],
        }
        for proj, counter in by_projection.items()
    }


def qdq_report_summary(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    report_path = Path(path)
    if not report_path.is_file():
        return {"error": f"missing qdq report: {report_path}"}
    report = json.loads(report_path.read_text())
    keys = [
        "rewritten_linear_nodes",
        "smoothquant_rewritten_linear_nodes",
        "rewritten_weight_only_linear_nodes",
        "output_cast_nodes",
        "activation_quantization",
        "weight_quantization",
        "cast_output_to",
    ]
    return {key: report.get(key) for key in keys}


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    projection_counts = summary["projection_counts"]
    op_counts = summary["op_counts"]
    qdq = summary.get("qdq_report") or {}
    lines = [
        "# Step 6A Attention Fusion Inspection",
        "",
        "## ONNX operator counts",
        "",
        "| op | count |",
        "| --- | ---: |",
    ]
    for op, count in sorted(op_counts.items(), key=lambda item: (-item[1], item[0]))[:30]:
        lines.append(f"| `{op}` | `{count}` |")

    lines += [
        "",
        "## Projection coverage",
        "",
        "| group | count |",
        "| --- | ---: |",
    ]
    for key, count in sorted(projection_counts.items()):
        lines.append(f"| `{key}` | `{count}` |")

    if qdq:
        lines += [
            "",
            "## Q/DQ report",
            "",
            "| field | value |",
            "| --- | ---: |",
        ]
        for key, value in qdq.items():
            lines.append(f"| `{key}` | `{value}` |")

    lines += [
        "",
        "## Fusion assessment",
        "",
        f"- attention projection MatMuls: `{summary['attention_projection_matmuls']}`",
        f"- non-constant RHS MatMuls, usually attention score/context MatMuls: `{summary['non_constant_rhs_matmuls']}`",
        f"- Softmax nodes: `{summary['softmax_nodes']}`",
        f"- Sin/Cos nodes for RoPE: `{summary['sin_nodes']}` / `{summary['cos_nodes']}`",
        f"- QuantizeLinear/DequantizeLinear nodes: `{summary['quantize_nodes']}` / `{summary['dequantize_nodes']}`",
        "",
        "Conclusion: q/k/v projections are quantized as separate explicit Q/DQ MatMuls, while RoPE, score MatMul, mask, softmax, and context MatMul remain decomposed ONNX ops. Keeping INT8 through the attention body requires replacing this subgraph with a TensorRT plugin/custom fused attention kernel.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = onnx.load(args.onnx, load_external_data=False)
    graph = model.graph
    op_counts = Counter(node.op_type for node in graph.node)
    rows = projection_rows(graph)
    projection_counts = Counter(f"{row['family']}.{row['projection']}" for row in rows)
    non_constant_rhs_matmuls = 0
    constants = constant_inputs(graph)
    for node in graph.node:
        if node.op_type == "MatMul" and (len(node.input) < 2 or node.input[1] not in constants):
            non_constant_rhs_matmuls += 1

    summary = {
        "onnx": args.onnx,
        "num_nodes": len(graph.node),
        "op_counts": dict(op_counts),
        "projection_counts": dict(projection_counts),
        "attention_projection_matmuls": sum(
            count for key, count in projection_counts.items() if key.startswith("attention.")
        ),
        "non_constant_rhs_matmuls": non_constant_rhs_matmuls,
        "softmax_nodes": op_counts.get("Softmax", 0),
        "sin_nodes": op_counts.get("Sin", 0),
        "cos_nodes": op_counts.get("Cos", 0),
        "quantize_nodes": op_counts.get("QuantizeLinear", 0),
        "dequantize_nodes": op_counts.get("DequantizeLinear", 0),
        "direct_qdq_status": direct_qdq_status(graph, rows),
        "downstream_ops_from_qkv": downstream_ops(graph, rows, args.max_depth),
        "qdq_report": qdq_report_summary(args.qdq_report),
    }

    (output_dir / "attention_fusion_inspection.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown(summary, output_dir / "attention_fusion_inspection.md")
    print((output_dir / "attention_fusion_inspection.md").read_text(), end="")


if __name__ == "__main__":
    main()
