#!/usr/bin/env python3

"""Insert SmolVLA fused RoPE/layout TensorRT plugin nodes into action-only ONNX.

This is Step 6D's graph rewrite. It keeps the existing SmoothQuant/QDQ q_proj
and k_proj MatMul nodes unchanged, inserts one custom plugin for each exported
VLM text self-attention layer, and rewires the QK MatMul inputs to the plugin
outputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import onnx
from onnx import shape_inference
from onnx import TensorProto, helper


PLUGIN_OP = "SmolVLAFusedRopeLayout"
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
    parser.add_argument(
        "--plugin-input-dtype",
        choices=["fp32", "preserve"],
        default="fp32",
        help=(
            "fp32 inserts Cast(to=FLOAT) before the plugin and keeps plugin outputs FLOAT. "
            "preserve feeds q/k projection dtype to the plugin and inserts Cast(to=FLOAT) after it."
        ),
    )
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


def main() -> None:
    args = parse_args()
    model = onnx.load(args.input_onnx)
    original_nodes = list(model.graph.node)
    nodes_by_name = {node.name: node for node in model.graph.node}
    shapes = build_shape_map(model)
    tensor_names = {value.name for value in model.graph.value_info}
    tensor_names.update(value.name for value in model.graph.input)
    tensor_names.update(value.name for value in model.graph.output)
    tensor_names.update(init.name for init in model.graph.initializer)
    for node in model.graph.node:
        tensor_names.update(node.output)

    if args.position_ids_tensor not in tensor_names:
        raise ValueError(f"position tensor not found in graph: {args.position_ids_tensor}")

    inserted = []
    skipped = []
    insert_before: dict[str, list[onnx.NodeProto]] = {}
    expected_q_shape = [1, args.text_seq_len, args.q_heads * args.head_dim]
    expected_k_shape = [1, args.text_seq_len, args.kv_heads * args.head_dim]
    expected_q_attn_shape = [1, args.q_heads, args.text_seq_len, args.head_dim]
    expected_k_attn_shape = [1, args.q_heads, args.head_dim, args.text_seq_len]
    expected_qk_shape = [1, args.q_heads, args.text_seq_len, args.text_seq_len]

    for layer in range(args.max_layer_index + 1):
        matmul = nodes_by_name.get(qk_matmul_name(layer))
        if matmul is None:
            skipped.append(
                {
                    "layer": layer,
                    "qk_matmul": qk_matmul_name(layer),
                    "reason": "missing expected QK MatMul node",
                }
            )
            continue
        q_raw = projection_output("q_proj", layer)
        k_raw = projection_output("k_proj", layer)
        if q_raw not in tensor_names:
            skipped.append({"layer": layer, "q_raw": q_raw, "reason": "missing q projection output"})
            continue
        if k_raw not in tensor_names:
            skipped.append({"layer": layer, "k_raw": k_raw, "reason": "missing k projection output"})
            continue

        q_shape = shapes.get(q_raw)
        k_shape = shapes.get(k_raw)
        q_attn_shape = shapes.get(matmul.input[0])
        k_attn_shape = shapes.get(matmul.input[1])
        qk_shape = shapes.get(matmul.output[0])
        shape_report = {
            "q_raw_shape": q_shape,
            "k_raw_shape": k_shape,
            "old_q_attn_shape": q_attn_shape,
            "old_k_attn_shape": k_attn_shape,
            "qk_output_shape": qk_shape,
        }
        if (
            q_shape != expected_q_shape
            or k_shape != expected_k_shape
            or q_attn_shape != expected_q_attn_shape
            or k_attn_shape != expected_k_attn_shape
            or qk_shape != expected_qk_shape
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

        q_cast_in_node = None
        k_cast_in_node = None
        q_cast_out_node = None
        k_cast_out_node = None
        if args.plugin_input_dtype == "fp32":
            q_plugin_in = f"/debug_core/fused_rope_layout{layer_suffix(layer)}/q_raw_float"
            k_plugin_in = f"/debug_core/fused_rope_layout{layer_suffix(layer)}/k_raw_float"
            q_out = f"/debug_core/fused_rope_layout{layer_suffix(layer)}/q_attn_float"
            k_out = f"/debug_core/fused_rope_layout{layer_suffix(layer)}/k_attn_float"
            plugin_outputs = [q_out, k_out]
            q_cast_in_node = helper.make_node(
                "Cast",
                inputs=[q_raw],
                outputs=[q_plugin_in],
                name=f"/debug_core/fused_rope_layout{layer_suffix(layer)}/q_raw_cast_float",
                to=TensorProto.FLOAT,
            )
            k_cast_in_node = helper.make_node(
                "Cast",
                inputs=[k_raw],
                outputs=[k_plugin_in],
                name=f"/debug_core/fused_rope_layout{layer_suffix(layer)}/k_raw_cast_float",
                to=TensorProto.FLOAT,
            )
        else:
            q_plugin_in = q_raw
            k_plugin_in = k_raw
            q_layout = f"/debug_core/fused_rope_layout{layer_suffix(layer)}/q_layout"
            k_layout = f"/debug_core/fused_rope_layout{layer_suffix(layer)}/k_layout"
            q_out = f"/debug_core/fused_rope_layout{layer_suffix(layer)}/q_attn_float"
            k_out = f"/debug_core/fused_rope_layout{layer_suffix(layer)}/k_attn_float"
            plugin_outputs = [q_layout, k_layout]
            q_cast_out_node = helper.make_node(
                "Cast",
                inputs=[q_layout],
                outputs=[q_out],
                name=f"/debug_core/fused_rope_layout{layer_suffix(layer)}/q_cast_float",
                to=TensorProto.FLOAT,
            )
            k_cast_out_node = helper.make_node(
                "Cast",
                inputs=[k_layout],
                outputs=[k_out],
                name=f"/debug_core/fused_rope_layout{layer_suffix(layer)}/k_cast_float",
                to=TensorProto.FLOAT,
            )
        plugin = helper.make_node(
            PLUGIN_OP,
            inputs=[q_plugin_in, k_plugin_in, args.position_ids_tensor],
            outputs=plugin_outputs,
            name=f"/debug_core/fused_rope_layout{layer_suffix(layer)}",
            domain="",
            plugin_version=PLUGIN_VERSION,
            q_heads=args.q_heads,
            kv_heads=args.kv_heads,
            head_dim=args.head_dim,
            max_wavelength=float(args.max_wavelength),
        )
        new_nodes = []
        if q_cast_in_node is not None:
            new_nodes.extend([q_cast_in_node, k_cast_in_node])
        new_nodes.append(plugin)
        if q_cast_out_node is not None:
            new_nodes.extend([q_cast_out_node, k_cast_out_node])
        insert_before[matmul.name] = new_nodes
        old_inputs = list(matmul.input)
        matmul.input[0] = q_out
        matmul.input[1] = k_out
        inserted.append(
            {
                "layer": layer,
                "plugin_node": plugin.name,
                "q_input_cast_node": q_cast_in_node.name if q_cast_in_node is not None else None,
                "k_input_cast_node": k_cast_in_node.name if k_cast_in_node is not None else None,
                "q_output_cast_node": q_cast_out_node.name if q_cast_out_node is not None else None,
                "k_output_cast_node": k_cast_out_node.name if k_cast_out_node is not None else None,
                "q_raw": q_raw,
                "k_raw": k_raw,
                "q_plugin_input": q_plugin_in,
                "k_plugin_input": k_plugin_in,
                "position_ids": args.position_ids_tensor,
                "qk_matmul": matmul.name,
                "old_qk_inputs": old_inputs,
                "new_qk_inputs": [q_out, k_out],
                "plugin_outputs": plugin_outputs,
                **shape_report,
            }
        )

    if len(inserted) < args.min_rewrites:
        raise ValueError(
            f"rewrote only {len(inserted)} VLM text self-attention layers; "
            f"min required is {args.min_rewrites}. See skipped entries for shape details."
        )

    reordered_nodes = []
    for node in original_nodes:
        reordered_nodes.extend(insert_before.get(node.name, []))
        reordered_nodes.append(node)
    del model.graph.node[:]
    model.graph.node.extend(reordered_nodes)

    report = {
        "input_onnx": args.input_onnx,
        "output_onnx": args.output_onnx,
        "plugin_op": PLUGIN_OP,
        "plugin_version": PLUGIN_VERSION,
        "scope": "exported VLM text self-attention layers only; action/expert attention is skipped by shape",
        "q_heads": args.q_heads,
        "kv_heads": args.kv_heads,
        "head_dim": args.head_dim,
        "text_seq_len": args.text_seq_len,
        "max_layer_index": args.max_layer_index,
        "plugin_input_dtype": args.plugin_input_dtype,
        "max_wavelength": args.max_wavelength,
        "inserted_plugin_nodes": len(inserted),
        "inserted_cast_nodes": len(inserted) * 2,
        "rewired_qk_matmul_nodes": len(inserted),
        "skipped_candidates": skipped,
        "entries": inserted,
        "note": "Old RoPE/layout subgraphs are left disconnected; TensorRT should prune them during build.",
    }
    out = Path(args.output_onnx)
    out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, out)
    Path(args.report_json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
