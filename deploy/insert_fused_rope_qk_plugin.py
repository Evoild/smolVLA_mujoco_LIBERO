#!/usr/bin/env python3

"""Replace VLM text QK MatMul with a fused RoPE/layout/QK TensorRT plugin."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import onnx
from onnx import TensorProto, helper, shape_inference


PLUGIN_OP = "SmolVLAFusedRopeQK"
PLUGIN_SOFTMAX_OP = "SmolVLAFusedRopeQKSoftmax"
PLUGIN_ATTENTION_OP = "SmolVLAFusedRopeAttention"
PLUGIN_VERSION = "1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-onnx", required=True)
    parser.add_argument("--output-onnx", required=True)
    parser.add_argument("--report-json", required=True)
    parser.add_argument("--position-ids-tensor", default="/debug_core/Sub_output_0")
    parser.add_argument("--q-heads", type=int, default=15)
    parser.add_argument("--kv-heads", type=int, default=5)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--text-seq-len", type=int, default=177)
    parser.add_argument("--max-layer-index", type=int, default=31)
    parser.add_argument("--min-rewrites", type=int, default=1)
    parser.add_argument("--max-wavelength", type=float, default=10000.0)
    parser.add_argument("--mode", choices=["qk", "qk_softmax", "attention"], default="qk")
    return parser.parse_args()


def layer_suffix(layer: int) -> str:
    return "" if layer == 0 else f"_{layer}"


def qk_matmul_name(layer: int) -> str:
    return "/debug_core/MatMul" if layer == 0 else f"/debug_core/MatMul_{2 * layer}"


def projection_output(proj: str, layer: int) -> str:
    return f"/debug_core/{proj}{layer_suffix(layer)}/MatMul_output_0"


def tensor_shape(value: onnx.ValueInfoProto) -> list[int | str]:
    dims: list[int | str] = []
    for dim in value.type.tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            dims.append(dim.dim_value)
        elif dim.HasField("dim_param"):
            dims.append(dim.dim_param)
        else:
            dims.append("?")
    return dims


def build_shape_map(model: onnx.ModelProto) -> dict[str, list[int | str]]:
    inferred = shape_inference.infer_shapes(model, strict_mode=False)
    shapes: dict[str, list[int | str]] = {}
    for values in (inferred.graph.input, inferred.graph.value_info, inferred.graph.output):
        for value in values:
            shapes[value.name] = tensor_shape(value)
    return shapes


def prune_dead_nodes(model: onnx.ModelProto) -> int:
    producer_by_output: dict[str, onnx.NodeProto] = {}
    for node in model.graph.node:
        for output in node.output:
            if output:
                producer_by_output[output] = node

    required_tensors = {output.name for output in model.graph.output}
    required_nodes: set[str] = set()
    stack = list(required_tensors)
    while stack:
        tensor = stack.pop()
        node = producer_by_output.get(tensor)
        if node is None:
            continue
        node_key = node.name or "|".join(node.output)
        if node_key in required_nodes:
            continue
        required_nodes.add(node_key)
        stack.extend(inp for inp in node.input if inp)

    old_count = len(model.graph.node)
    kept_nodes = []
    for node in model.graph.node:
        node_key = node.name or "|".join(node.output)
        if node_key in required_nodes:
            kept_nodes.append(node)
    del model.graph.node[:]
    model.graph.node.extend(kept_nodes)
    return old_count - len(kept_nodes)


def main() -> None:
    args = parse_args()
    model = onnx.load(args.input_onnx)
    original_nodes = list(model.graph.node)
    nodes_by_name = {node.name: node for node in model.graph.node}
    consumers: dict[str, list[onnx.NodeProto]] = {}
    for node in model.graph.node:
        for inp in node.input:
            consumers.setdefault(inp, []).append(node)
    shapes = build_shape_map(model)
    tensor_names = {value.name for value in model.graph.value_info}
    tensor_names.update(value.name for value in model.graph.input)
    tensor_names.update(value.name for value in model.graph.output)
    tensor_names.update(init.name for init in model.graph.initializer)
    for node in model.graph.node:
        tensor_names.update(node.output)

    if args.position_ids_tensor not in tensor_names:
        raise ValueError(f"position tensor not found in graph: {args.position_ids_tensor}")

    expected_q_shape = [1, args.text_seq_len, args.q_heads * args.head_dim]
    expected_k_shape = [1, args.text_seq_len, args.kv_heads * args.head_dim]
    expected_q_attn_shape = [1, args.q_heads, args.text_seq_len, args.head_dim]
    expected_k_attn_shape = [1, args.q_heads, args.head_dim, args.text_seq_len]
    expected_qk_shape = [1, args.q_heads, args.text_seq_len, args.text_seq_len]

    insert_before: dict[str, list[onnx.NodeProto]] = {}
    remove_nodes: set[str] = set()
    inserted = []
    skipped = []

    for layer in range(args.max_layer_index + 1):
        matmul = nodes_by_name.get(qk_matmul_name(layer))
        if matmul is None:
            skipped.append({"layer": layer, "qk_matmul": qk_matmul_name(layer), "reason": "missing QK MatMul"})
            continue

        q_raw = projection_output("q_proj", layer)
        k_raw = projection_output("k_proj", layer)
        v_raw = projection_output("v_proj", layer)
        if q_raw not in tensor_names or k_raw not in tensor_names or (args.mode == "attention" and v_raw not in tensor_names):
            skipped.append({"layer": layer, "reason": "missing q/k/v projection output", "q_raw": q_raw, "k_raw": k_raw, "v_raw": v_raw})
            continue

        shape_report = {
            "q_raw_shape": shapes.get(q_raw),
            "k_raw_shape": shapes.get(k_raw),
            "old_q_attn_shape": shapes.get(matmul.input[0]),
            "old_k_attn_shape": shapes.get(matmul.input[1]),
            "qk_output_shape": shapes.get(matmul.output[0]),
        }
        if (
            shape_report["q_raw_shape"] != expected_q_shape
            or shape_report["k_raw_shape"] != expected_k_shape
            or shape_report["old_q_attn_shape"] != expected_q_attn_shape
            or shape_report["old_k_attn_shape"] != expected_k_attn_shape
            or shape_report["qk_output_shape"] != expected_qk_shape
        ):
            skipped.append(
                {
                    "layer": layer,
                    "qk_matmul": matmul.name,
                    "reason": "not an exported VLM text self-attention QK MatMul",
                    **shape_report,
                }
            )
            continue

        q_float = f"/debug_core/fused_rope_qk{layer_suffix(layer)}/q_raw_float"
        k_float = f"/debug_core/fused_rope_qk{layer_suffix(layer)}/k_raw_float"
        q_cast = helper.make_node(
            "Cast",
            inputs=[q_raw],
            outputs=[q_float],
            name=f"/debug_core/fused_rope_qk{layer_suffix(layer)}/q_raw_cast_float",
            to=TensorProto.FLOAT,
        )
        k_cast = helper.make_node(
            "Cast",
            inputs=[k_raw],
            outputs=[k_float],
            name=f"/debug_core/fused_rope_qk{layer_suffix(layer)}/k_raw_cast_float",
            to=TensorProto.FLOAT,
        )
        replaced_nodes = [matmul.name]
        plugin_output = matmul.output[0]
        plugin_inputs = [q_float, k_float, args.position_ids_tensor]
        plugin_op = PLUGIN_OP
        extra_casts = []
        if args.mode in {"qk_softmax", "attention"}:
            qk_users = consumers.get(matmul.output[0], [])
            if len(qk_users) != 1 or qk_users[0].op_type != "Mul":
                skipped.append({"layer": layer, "reason": "QK output does not feed single Mul", "qk_users": [n.name for n in qk_users]})
                continue
            mul = qk_users[0]
            cast_users = consumers.get(mul.output[0], [])
            if len(cast_users) != 1 or cast_users[0].op_type != "Cast":
                skipped.append({"layer": layer, "reason": "scaled QK does not feed single Cast", "mul": mul.name})
                continue
            cast = cast_users[0]
            where_users = consumers.get(cast.output[0], [])
            if len(where_users) != 1 or where_users[0].op_type != "Where":
                skipped.append({"layer": layer, "reason": "cast score does not feed single Where", "cast": cast.name})
                continue
            where = where_users[0]
            softmax_users = consumers.get(where.output[0], [])
            if len(softmax_users) != 1 or softmax_users[0].op_type != "Softmax":
                skipped.append({"layer": layer, "reason": "masked score does not feed single Softmax", "where": where.name})
                continue
            softmax = softmax_users[0]
            plugin_op = PLUGIN_SOFTMAX_OP
            plugin_inputs = [q_float, k_float, args.position_ids_tensor, where.input[0]]
            plugin_output = softmax.output[0]
            replaced_nodes = [matmul.name, mul.name, cast.name, where.name, softmax.name]
            if args.mode == "attention":
                softmax_cast_users = consumers.get(softmax.output[0], [])
                if len(softmax_cast_users) != 1 or softmax_cast_users[0].op_type != "Cast":
                    skipped.append({"layer": layer, "reason": "Softmax output does not feed single Cast", "softmax": softmax.name})
                    continue
                softmax_cast = softmax_cast_users[0]
                pv_users = consumers.get(softmax_cast.output[0], [])
                if len(pv_users) != 1 or pv_users[0].op_type != "MatMul":
                    skipped.append({"layer": layer, "reason": "Softmax Cast output does not feed single PV MatMul", "cast": softmax_cast.name})
                    continue
                pv = pv_users[0]
                v_float = f"/debug_core/fused_rope_qk{layer_suffix(layer)}/v_raw_float"
                v_cast = helper.make_node(
                    "Cast",
                    inputs=[v_raw],
                    outputs=[v_float],
                    name=f"/debug_core/fused_rope_qk{layer_suffix(layer)}/v_raw_cast_float",
                    to=TensorProto.FLOAT,
                )
                extra_casts.append(v_cast)
                plugin_op = PLUGIN_ATTENTION_OP
                plugin_inputs = [q_float, k_float, v_float, args.position_ids_tensor, where.input[0]]
                plugin_output = pv.output[0]
                replaced_nodes = [matmul.name, mul.name, cast.name, where.name, softmax.name, softmax_cast.name, pv.name]

        plugin = helper.make_node(
            plugin_op,
            inputs=plugin_inputs,
            outputs=[plugin_output],
            name=(
                f"/debug_core/fused_rope_qk{layer_suffix(layer)}"
                if args.mode == "qk"
                else f"/debug_core/fused_rope_attention{layer_suffix(layer)}"
                if args.mode == "attention"
                else f"/debug_core/fused_rope_qk_softmax{layer_suffix(layer)}"
            ),
            domain="",
            plugin_version=PLUGIN_VERSION,
            q_heads=args.q_heads,
            kv_heads=args.kv_heads,
            head_dim=args.head_dim,
            max_wavelength=float(args.max_wavelength),
        )
        insert_before[matmul.name] = [q_cast, k_cast, *extra_casts, plugin]
        remove_nodes.update(replaced_nodes)
        inserted.append(
            {
                "layer": layer,
                "plugin_node": plugin.name,
                "plugin_op": plugin_op,
                "q_input_cast_node": q_cast.name,
                "k_input_cast_node": k_cast.name,
                "q_raw": q_raw,
                "k_raw": k_raw,
                "v_raw": v_raw if args.mode == "attention" else None,
                "position_ids": args.position_ids_tensor,
                "replaced_nodes": replaced_nodes,
                "old_qk_inputs": list(matmul.input),
                "plugin_output": plugin_output,
                **shape_report,
            }
        )

    if len(inserted) < args.min_rewrites:
        raise ValueError(f"rewrote only {len(inserted)} QK MatMul nodes; min required is {args.min_rewrites}")

    reordered_nodes = []
    for node in original_nodes:
        reordered_nodes.extend(insert_before.get(node.name, []))
        if node.name not in remove_nodes:
            reordered_nodes.append(node)
    del model.graph.node[:]
    model.graph.node.extend(reordered_nodes)
    pruned_nodes = prune_dead_nodes(model)

    report = {
        "input_onnx": args.input_onnx,
        "output_onnx": args.output_onnx,
        "plugin_op": PLUGIN_ATTENTION_OP if args.mode == "attention" else PLUGIN_OP if args.mode == "qk" else PLUGIN_SOFTMAX_OP,
        "plugin_version": PLUGIN_VERSION,
        "mode": args.mode,
        "scope": "exported VLM text self-attention QK score MatMul only; action/expert attention is skipped by shape",
        "q_heads": args.q_heads,
        "kv_heads": args.kv_heads,
        "head_dim": args.head_dim,
        "text_seq_len": args.text_seq_len,
        "max_layer_index": args.max_layer_index,
        "max_wavelength": args.max_wavelength,
        "inserted_plugin_nodes": len(inserted),
        "inserted_cast_nodes": len(inserted) * 2,
        "removed_nodes": len(remove_nodes),
        "pruned_dead_nodes": pruned_nodes,
        "skipped_candidates": skipped,
        "entries": inserted,
        "note": "Dead nodes not contributing to graph outputs are explicitly pruned after QK replacement.",
    }
    out = Path(args.output_onnx)
    out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, out)
    Path(args.report_json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
