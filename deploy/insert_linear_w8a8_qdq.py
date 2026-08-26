#!/usr/bin/env python3

"""Insert explicit W8A8 Q/DQ pairs around ONNX Linear Gemm/MatMul nodes.

The input ONNX should already be exported in bf16/fp32 form. This script keeps
non-Linear operators unchanged and rewrites Linear weights to int8 initializers
with DequantizeLinear, while inserting QuantizeLinear/DequantizeLinear on each
Linear activation input.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input ONNX path.")
    parser.add_argument("--output", required=True, help="Output ONNX path with explicit Q/DQ.")
    parser.add_argument(
        "--activation-scale",
        type=float,
        default=1.0 / 127.0,
        help="Static symmetric int8 activation scale inserted before every Linear node.",
    )
    parser.add_argument(
        "--activation-scale-mode",
        choices=["static", "dynamic", "calibrated"],
        default="static",
        help="static uses --activation-scale; dynamic computes abs(input).max()/127 at runtime; calibrated reads per-node scales from JSON.",
    )
    parser.add_argument("--activation-scales-json", default=None, help="Calibration JSON for calibrated mode.")
    parser.add_argument(
        "--include-module-regex",
        default=None,
        help=(
            "Only rewrite Linear calls whose calibration module name matches this regex. "
            "This requires --activation-scale-mode calibrated."
        ),
    )
    parser.add_argument(
        "--include-node-regex",
        default=None,
        help="Only rewrite ONNX Gemm/MatMul nodes whose node name matches this regex.",
    )
    parser.add_argument(
        "--stop-before-node-regex",
        default=None,
        help="Stop rewriting when an ONNX node name matches this regex; later Linear nodes are left unchanged.",
    )
    parser.add_argument(
        "--weight-only-include-node-regex",
        default=None,
        help="Rewrite matching ONNX Gemm/MatMul weights to W8 only, without activation Q/DQ.",
    )
    parser.add_argument(
        "--vlm-node-module-prefix",
        default="model.vlm_with_expert.vlm.model.text_model.layers",
        help=(
            "PyTorch module prefix used when mapping ONNX names like /q_proj_3/MatMul to calibration modules. "
            "Use model.vlm_with_expert.lm_expert.layers for action-only expert graphs."
        ),
    )
    parser.add_argument(
        "--alternate-vlm-node-module-prefix",
        action="append",
        default=[],
        help="Additional module prefixes to try when mapping ONNX VLM/expert node names to calibration modules.",
    )
    parser.add_argument(
        "--cast-output-to",
        choices=["none", "bf16", "fp16", "fp32"],
        default="none",
        help=(
            "Optionally insert a Cast after each rewritten Linear output. "
            "Use bf16 when Q/DQ MatMul outputs must rejoin a BF16 residual stream."
        ),
    )
    parser.add_argument("--check", action="store_true", help="Run onnx.checker after rewriting.")
    return parser.parse_args()


def bf16_to_float32(tensor: TensorProto) -> np.ndarray:
    if tensor.raw_data:
        data = np.frombuffer(tensor.raw_data, dtype=np.uint16)
    else:
        data = np.asarray(tensor.int32_data, dtype=np.uint16)
    return (data.astype(np.uint32) << 16).view(np.float32).reshape(tuple(tensor.dims))


def tensor_to_float32(tensor: TensorProto) -> np.ndarray:
    if tensor.data_type == TensorProto.BFLOAT16:
        return bf16_to_float32(tensor)
    array = numpy_helper.to_array(tensor)
    if array.dtype == np.float32:
        return array
    return array.astype(np.float32)


def make_scalar_initializer(name: str, value: float, data_type: int = TensorProto.FLOAT) -> TensorProto:
    if data_type == TensorProto.INT8:
        return helper.make_tensor(name, data_type, [], [int(value)])
    return helper.make_tensor(name, data_type, [], [float(value)])


def make_scale_initializer(name: str, scale: float | list[float]) -> tuple[TensorProto, bool]:
    if isinstance(scale, list):
        values = np.asarray(scale, dtype=np.float32)
        return numpy_helper.from_array(values, name=name), True
    return make_scalar_initializer(name, float(scale), TensorProto.FLOAT), False


def make_zero_point_initializer(name: str, per_channel: bool, channels: int | None = None) -> TensorProto:
    if per_channel:
        if channels is None:
            raise ValueError("channels is required for per-channel zero point")
        return numpy_helper.from_array(np.zeros((channels,), dtype=np.int8), name=name)
    return make_scalar_initializer(name, 0, TensorProto.INT8)


def quantize_weight_per_channel(weight: np.ndarray, axis: int) -> tuple[np.ndarray, np.ndarray]:
    reduce_axes = tuple(idx for idx in range(weight.ndim) if idx != axis)
    scale = np.max(np.abs(weight), axis=reduce_axes, keepdims=False)
    scale = np.maximum(scale, 1e-8).astype(np.float32) / 127.0
    reshape = [1] * weight.ndim
    reshape[axis] = scale.shape[0]
    qweight = np.round(weight / scale.reshape(reshape)).clip(-127, 127).astype(np.int8)
    return qweight, scale


def get_int_attr(node: onnx.NodeProto, name: str, default: int) -> int:
    for attr in node.attribute:
        if attr.name == name:
            return int(attr.i)
    return default


def tensorproto_for_cast_output(dtype: str) -> int | None:
    if dtype == "none":
        return None
    if dtype == "bf16":
        return TensorProto.BFLOAT16
    if dtype == "fp16":
        return TensorProto.FLOAT16
    if dtype == "fp32":
        return TensorProto.FLOAT
    raise ValueError(f"unsupported cast output dtype: {dtype}")


def add_output_cast(
    graph: onnx.GraphProto,
    node: onnx.NodeProto,
    node_idx: int,
    cast_to: int | None,
) -> int:
    if cast_to is None:
        return 0
    if len(node.output) != 1:
        raise ValueError(f"expected one output for {node.name or node.op_type}, got {len(node.output)}")

    original_output = node.output[0]
    pre_cast_output = f"{original_output}_pre_cast"
    node.output[0] = pre_cast_output
    cast_node = helper.make_node(
        "Cast",
        [pre_cast_output],
        [original_output],
        name=f"{node.name or node.op_type}_{node_idx}_CastOutput",
        to=cast_to,
    )
    graph.node.insert(node_idx + 1, cast_node)
    return 1


def add_activation_qdq(
    graph: onnx.GraphProto,
    node: onnx.NodeProto,
    node_idx: int,
    input_idx: int,
    activation_scale: float | list[float],
    activation_scale_mode: str,
    input_name: str | None = None,
) -> str:
    original = input_name or node.input[input_idx]
    prefix = f"{node.name or node.op_type}_{node_idx}_act{input_idx}"
    scale_name = f"{prefix}_scale"
    zp_name = f"{prefix}_zero_point"
    q_name = f"{prefix}_quantized"
    dq_name = f"{prefix}_dequantized"

    if activation_scale_mode == "static":
        scale_initializer, per_channel = make_scale_initializer(scale_name, activation_scale)
        channels = len(activation_scale) if isinstance(activation_scale, list) else None
        graph.initializer.append(scale_initializer)
        graph.initializer.append(make_zero_point_initializer(zp_name, per_channel, channels))
        pre_nodes = []
    else:
        graph.initializer.append(make_scalar_initializer(zp_name, 0, TensorProto.INT8))
        cast_name = f"{prefix}_scale_input_fp32"
        abs_name = f"{prefix}_abs"
        max_name = f"{prefix}_amax"
        floor_name = f"{prefix}_scale_floor"
        clamped_name = f"{prefix}_scale_clamped"
        graph.initializer.append(make_scalar_initializer(floor_name, 1.0e-8, TensorProto.FLOAT))
        divisor_name = f"{prefix}_scale_divisor"
        graph.initializer.append(make_scalar_initializer(divisor_name, 127.0, TensorProto.FLOAT))
        pre_nodes = [
            helper.make_node("Cast", [original], [cast_name], name=f"{prefix}_CastScaleInput", to=TensorProto.FLOAT),
            helper.make_node("Abs", [cast_name], [abs_name], name=f"{prefix}_Abs"),
            helper.make_node(
                "ReduceMax",
                [abs_name],
                [max_name],
                name=f"{prefix}_ReduceMax",
                keepdims=0,
            ),
            helper.make_node("Max", [max_name, floor_name], [clamped_name], name=f"{prefix}_MaxScaleFloor"),
            helper.make_node("Div", [clamped_name, divisor_name], [scale_name], name=f"{prefix}_Div127"),
        ]
    qdq_attrs = {"axis": -1} if activation_scale_mode == "static" and isinstance(activation_scale, list) else {}
    q_node = helper.make_node(
        "QuantizeLinear",
        [original, scale_name, zp_name],
        [q_name],
        name=f"{prefix}_QuantizeLinear",
        **qdq_attrs,
    )
    dq_node = helper.make_node(
        "DequantizeLinear",
        [q_name, scale_name, zp_name],
        [dq_name],
        name=f"{prefix}_DequantizeLinear",
        **qdq_attrs,
    )
    for offset, pre_node in enumerate(pre_nodes):
        graph.node.insert(node_idx + offset, pre_node)
    qdq_idx = node_idx + len(pre_nodes)
    graph.node.insert(qdq_idx, q_node)
    graph.node.insert(qdq_idx + 1, dq_node)
    return dq_name


def smooth_scale_for_entry(entry: dict[str, Any] | None) -> np.ndarray | None:
    if entry is None:
        return None
    quantization = str(entry.get("quantization", ""))
    value = entry.get("smooth_scale")
    if not quantization.startswith("smoothquant_") or value is None:
        return None
    smooth_scale = np.asarray(value, dtype=np.float32)
    if smooth_scale.ndim != 1:
        raise ValueError(f"smooth_scale must be 1-D, got shape={smooth_scale.shape}")
    return np.maximum(smooth_scale, 1.0e-8).astype(np.float32)


def input_axis_for_weight(node: onnx.NodeProto, weight: np.ndarray, channel_axis: int) -> int:
    if weight.ndim != 2:
        raise ValueError(f"SmoothQuant currently expects 2-D Linear weights, got shape={weight.shape}")
    if node.op_type == "MatMul":
        return 0
    if node.op_type == "Gemm":
        return 1 if get_int_attr(node, "transB", 0) else 0
    return 1 - channel_axis


def fold_smooth_scale_into_weight(
    node: onnx.NodeProto,
    weight: np.ndarray,
    channel_axis: int,
    smooth_scale: np.ndarray,
) -> tuple[np.ndarray, int]:
    matching_axes = [idx for idx, size in enumerate(weight.shape) if size == smooth_scale.shape[0]]
    input_axis = matching_axes[0] if len(matching_axes) == 1 else input_axis_for_weight(node, weight, channel_axis)
    if weight.shape[input_axis] != smooth_scale.shape[0]:
        raise ValueError(
            f"SmoothQuant scale length mismatch for {node.name or node.op_type}: "
            f"weight shape={weight.shape}, input_axis={input_axis}, smooth_scale={smooth_scale.shape}"
        )
    reshape = [1] * weight.ndim
    reshape[input_axis] = smooth_scale.shape[0]
    output_axis = 1 - input_axis if weight.ndim == 2 else channel_axis
    return weight * smooth_scale.reshape(reshape), output_axis


def add_smooth_scale_div(
    graph: onnx.GraphProto,
    node: onnx.NodeProto,
    node_idx: int,
    input_idx: int,
    smooth_scale: np.ndarray,
) -> tuple[str, int]:
    original = node.input[input_idx]
    prefix = f"{node.name or node.op_type}_{node_idx}_smooth{input_idx}"
    scale_name = f"{prefix}_scale"
    cast_name = f"{prefix}_input_fp32"
    div_name = f"{prefix}_divided"
    graph.initializer.append(numpy_helper.from_array(smooth_scale.astype(np.float32), name=scale_name))
    graph.node.insert(
        node_idx,
        helper.make_node("Cast", [original], [cast_name], name=f"{prefix}_CastInput", to=TensorProto.FLOAT),
    )
    graph.node.insert(
        node_idx + 1,
        helper.make_node("Div", [cast_name, scale_name], [div_name], name=f"{prefix}_DivSmoothScale"),
    )
    return div_name, 2


def calibration_entry_for_index(calibration: dict[str, Any] | None, index: int) -> dict[str, Any] | None:
    if calibration is None:
        return None
    scales = calibration.get("linear_call_scales", [])
    if index >= len(scales):
        return None
    return scales[index]


def calibration_entries_by_module(calibration: dict[str, Any] | None) -> dict[str, list[dict[str, Any]]]:
    if calibration is None:
        return {}
    entries: dict[str, list[dict[str, Any]]] = {}
    for entry in calibration.get("linear_call_scales", []):
        module = entry.get("module", "")
        if module:
            entries.setdefault(module, []).append(entry)
    return entries


def entry_channel_count(entry: dict[str, Any]) -> int | None:
    smooth_scale = entry.get("smooth_scale")
    if isinstance(smooth_scale, list):
        return len(smooth_scale)
    scale = entry.get("scale")
    if isinstance(scale, list):
        return len(scale)
    channels = entry.get("channels")
    return int(channels) if channels is not None else None


def select_calibration_entry(
    candidates: list[dict[str, Any]],
    expected_input_channels: int,
) -> dict[str, Any] | None:
    for entry in candidates:
        if entry_channel_count(entry) == expected_input_channels:
            return entry
    return candidates[0] if candidates else None


def vlm_text_module_from_node_name(node_name: str, module_prefix: str) -> str | None:
    prefix = r"^/(?:debug_core/)?"
    suffix = r"(?:_([0-9]+))?/MatMul$"
    match = re.search(prefix + r"(q_proj|k_proj|v_proj|o_proj)" + suffix, node_name)
    if match:
        proj = match.group(1)
        layer = int(match.group(2) or 0)
        return f"{module_prefix}.{layer}.self_attn.{proj}"
    match = re.search(prefix + r"mlp/(gate_proj|up_proj|down_proj)" + suffix, node_name)
    if match:
        proj = match.group(1)
        layer = int(match.group(2) or 0)
        return f"{module_prefix}.{layer}.mlp.{proj}"
    return None


def calibration_scale_for_entry(entry: dict[str, Any] | None, fallback: float) -> float | list[float]:
    if entry is None:
        return fallback
    value = entry.get("scale", fallback)
    if isinstance(value, list):
        return [max(float(item), 1.0e-8) for item in value]
    return max(float(value), 1.0e-8)


def add_weight_dq(
    graph: onnx.GraphProto,
    weight_name: str,
    weight: np.ndarray,
    axis: int,
    prefix: str,
) -> tuple[onnx.NodeProto, str]:
    qweight, scale = quantize_weight_per_channel(weight, axis=axis)
    qweight_name = f"{prefix}_{weight_name}_int8"
    scale_name = f"{prefix}_{weight_name}_scale"
    zp_name = f"{prefix}_{weight_name}_zero_point"
    dq_name = f"{prefix}_{weight_name}_dequantized"

    graph.initializer.append(numpy_helper.from_array(qweight, name=qweight_name))
    graph.initializer.append(numpy_helper.from_array(scale.astype(np.float32), name=scale_name))
    graph.initializer.append(numpy_helper.from_array(np.zeros_like(scale, dtype=np.int8), name=zp_name))
    dq_node = helper.make_node(
        "DequantizeLinear",
        [qweight_name, scale_name, zp_name],
        [dq_name],
        name=f"{prefix}_{weight_name}_DequantizeLinear",
        axis=axis,
    )
    return dq_node, dq_name


def is_supported_weight(tensor: TensorProto) -> bool:
    return tensor.data_type in {
        TensorProto.FLOAT,
        TensorProto.FLOAT16,
        TensorProto.BFLOAT16,
        TensorProto.DOUBLE,
    }


def rewrite_graph(
    model: onnx.ModelProto,
    activation_scale: float,
    activation_scale_mode: str,
    calibration: dict[str, Any] | None = None,
    include_module_regex: str | None = None,
    include_node_regex: str | None = None,
    stop_before_node_regex: str | None = None,
    weight_only_include_node_regex: str | None = None,
    vlm_node_module_prefix: str = "model.vlm_with_expert.vlm.model.text_model.layers",
    alternate_vlm_node_module_prefixes: list[str] | None = None,
    cast_output_to: str = "none",
) -> dict[str, Any]:
    graph = model.graph
    initializers = {tensor.name: tensor for tensor in graph.initializer}
    rewritten = 0
    rewritten_weight_only = 0
    smoothquant_rewritten = 0
    skipped: list[dict[str, str]] = []
    excluded: list[dict[str, str]] = []
    linear_call_index = 0
    activation_qdq_call_index = 0
    cast_to = tensorproto_for_cast_output(cast_output_to)
    output_cast_nodes = 0
    calibration_by_module = calibration_entries_by_module(calibration)
    include_pattern = re.compile(include_module_regex) if include_module_regex else None
    include_node_pattern = re.compile(include_node_regex) if include_node_regex else None
    stop_before_node_pattern = re.compile(stop_before_node_regex) if stop_before_node_regex else None
    weight_only_node_pattern = re.compile(weight_only_include_node_regex) if weight_only_include_node_regex else None
    module_prefixes = [vlm_node_module_prefix] + list(alternate_vlm_node_module_prefixes or [])
    stop_rewriting = False

    node_idx = 0
    while node_idx < len(graph.node):
        node = graph.node[node_idx]
        node_name = node.name or f"{node.op_type}_{node_idx}"
        if stop_before_node_pattern is not None and stop_before_node_pattern.search(node_name):
            stop_rewriting = True
        if node.op_type not in {"Gemm", "MatMul"}:
            node_idx += 1
            continue

        if node.op_type == "Gemm":
            if len(node.input) < 2 or node.input[1] not in initializers:
                skipped.append({"node": node.name or f"Gemm_{node_idx}", "reason": "non_constant_weight"})
                node_idx += 1
                continue
            weight_input_idx = 1
            weight_name = node.input[weight_input_idx]
            trans_b = get_int_attr(node, "transB", 0)
            channel_axis = 0 if trans_b else 1
        else:
            if len(node.input) < 2 or node.input[1] not in initializers:
                skipped.append({"node": node.name or f"MatMul_{node_idx}", "reason": "non_constant_rhs"})
                node_idx += 1
                continue
            weight_input_idx = 1
            weight_name = node.input[weight_input_idx]
            channel_axis = 1

        weight_tensor = initializers[weight_name]
        if not is_supported_weight(weight_tensor):
            skipped.append({"node": node.name or f"{node.op_type}_{node_idx}", "reason": "unsupported_weight_dtype"})
            node_idx += 1
            continue

        weight = tensor_to_float32(weight_tensor)
        if weight.ndim < 2 or channel_axis >= weight.ndim:
            skipped.append({"node": node.name or f"{node.op_type}_{node_idx}", "reason": "unsupported_weight_shape"})
            node_idx += 1
            continue

        calibration_entry = None
        calibration_module = ""
        linear_call_index += 1
        if stop_rewriting:
            excluded.append(
                {
                    "node": node.name or f"{node.op_type}_{node_idx}",
                    "reason": "after_stop_before_node_filter",
                    "module": calibration_module,
                }
            )
            node_idx += 1
            continue
        calibration_entry = calibration_entry_for_index(calibration, activation_qdq_call_index)
        calibration_module = calibration_entry.get("module", "") if calibration_entry else ""
        activation_qdq_call_index += 1
        expected_input_channels = weight.shape[input_axis_for_weight(node, weight, channel_axis)] if weight.ndim == 2 else -1
        for module_prefix in module_prefixes:
            node_module = vlm_text_module_from_node_name(node_name, module_prefix)
            if not node_module or node_module not in calibration_by_module:
                continue
            selected_entry = select_calibration_entry(calibration_by_module[node_module], int(expected_input_channels))
            if selected_entry is not None and entry_channel_count(selected_entry) == int(expected_input_channels):
                calibration_entry = selected_entry
                calibration_module = node_module
                break
            if calibration_module == "":
                calibration_entry = selected_entry
                calibration_module = node_module
        activation_qdq_candidate = include_node_pattern is None or include_node_pattern.search(node_name)
        weight_only_candidate = (
            weight_only_node_pattern is not None and weight_only_node_pattern.search(node_name)
        )
        if not activation_qdq_candidate and not weight_only_candidate:
            excluded.append(
                {
                    "node": node.name or f"{node.op_type}_{node_idx}",
                    "reason": "excluded_by_node_filter",
                    "module": calibration_module,
                }
            )
            node_idx += 1
            continue

        if weight_only_candidate:
            prefix = f"{node.name or node.op_type}_{node_idx}_w8a16"
            weight_dq_node, weight_dq = add_weight_dq(graph, weight_name, weight, channel_axis, prefix)
            graph.node.insert(node_idx, weight_dq_node)
            node_idx += 1
            node = graph.node[node_idx]
            node.input[weight_input_idx] = weight_dq
            rewritten_weight_only += 1
            output_cast_nodes += add_output_cast(graph, node, node_idx, cast_to)
            node_idx += 1
            continue

        if include_pattern is not None and not include_pattern.search(calibration_module):
            excluded.append(
                {
                    "node": node.name or f"{node.op_type}_{node_idx}",
                    "reason": "excluded_by_module_filter",
                    "module": calibration_module,
                }
            )
            node_idx += 1
            continue

        current_activation_scale = calibration_scale_for_entry(calibration_entry, activation_scale)
        smooth_scale = smooth_scale_for_entry(calibration_entry)
        qdq_input = node.input[0]
        if smooth_scale is not None:
            weight, channel_axis = fold_smooth_scale_into_weight(node, weight, channel_axis, smooth_scale)
            qdq_input, inserted = add_smooth_scale_div(graph, node, node_idx, 0, smooth_scale)
            node_idx += inserted
            smoothquant_rewritten += 1
            node = graph.node[node_idx]
        qdq_mode = "static" if activation_scale_mode == "calibrated" else activation_scale_mode
        dq_input = add_activation_qdq(graph, node, node_idx, 0, current_activation_scale, qdq_mode, qdq_input)
        node_idx += 2 if qdq_mode == "static" else 7
        node = graph.node[node_idx]
        node.input[0] = dq_input

        prefix = f"{node.name or node.op_type}_{node_idx}_w8a8"
        weight_dq_node, weight_dq = add_weight_dq(graph, weight_name, weight, channel_axis, prefix)
        graph.node.insert(node_idx, weight_dq_node)
        node_idx += 1
        node = graph.node[node_idx]
        node.input[weight_input_idx] = weight_dq
        rewritten += 1
        output_cast_nodes += add_output_cast(graph, node, node_idx, cast_to)
        node_idx += 1

    return {
        "quantization": "explicit_qdq_linear_w8a8",
        "rewritten_linear_nodes": rewritten,
        "rewritten_weight_only_linear_nodes": rewritten_weight_only,
        "smoothquant_rewritten_linear_nodes": smoothquant_rewritten,
        "activation_quantization": (
            "static symmetric int8 per tensor"
            if activation_scale_mode == "static"
            else "dynamic symmetric int8 per tensor, scale = max(abs(input))/127 at runtime"
            if activation_scale_mode == "dynamic"
            else "calibrated static symmetric int8 per Linear call; scalar or per-channel from calibration JSON"
        ),
        "activation_scale_mode": activation_scale_mode,
        "activation_scale": activation_scale if activation_scale_mode == "static" else None,
        "calibration_scales_used": min(rewritten, len(calibration.get("linear_call_scales", [])))
        if calibration
        else 0,
        "linear_calls_seen": linear_call_index,
        "include_module_regex": include_module_regex,
        "include_node_regex": include_node_regex,
        "stop_before_node_regex": stop_before_node_regex,
        "weight_only_include_node_regex": weight_only_include_node_regex,
        "vlm_node_module_prefix": vlm_node_module_prefix,
        "alternate_vlm_node_module_prefixes": alternate_vlm_node_module_prefixes or [],
        "cast_output_to": cast_output_to,
        "output_cast_nodes": output_cast_nodes,
        "weight_quantization": "static symmetric int8 per output channel",
        "skipped_nodes": skipped,
        "excluded_nodes": excluded,
    }


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = onnx.load(input_path)
    calibration = None
    if args.activation_scale_mode == "calibrated":
        if not args.activation_scales_json:
            raise ValueError("--activation-scales-json is required when --activation-scale-mode calibrated")
        with open(args.activation_scales_json) as f:
            calibration = json.load(f)
    if args.include_module_regex and args.activation_scale_mode != "calibrated":
        raise ValueError("--include-module-regex requires --activation-scale-mode calibrated")
    report = rewrite_graph(
        model,
        args.activation_scale,
        args.activation_scale_mode,
        calibration,
        include_module_regex=args.include_module_regex,
        include_node_regex=args.include_node_regex,
        stop_before_node_regex=args.stop_before_node_regex,
        weight_only_include_node_regex=args.weight_only_include_node_regex,
        vlm_node_module_prefix=args.vlm_node_module_prefix,
        alternate_vlm_node_module_prefixes=args.alternate_vlm_node_module_prefix,
        cast_output_to=args.cast_output_to,
    )
    if args.check:
        onnx.checker.check_model(model)
        report["onnx_check"] = "ok"
    else:
        report["onnx_check"] = "skipped"
    onnx.save(model, output_path)

    report["input_onnx"] = str(input_path)
    report["output_onnx"] = str(output_path)
    report_path = output_path.with_suffix(".qdq_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
