#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECTIONS_QKVO = ("q_proj", "k_proj", "v_proj", "o_proj")
PROJECTIONS_GATE_UP = ("mlp/gate_proj", "mlp/up_proj")
PROJECTION_DOWN = "mlp/down_proj"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize 5M TensorRT per-layer profile by SmolVLA categories.")
    parser.add_argument("--profile", required=True, help="trtexec --exportProfile JSON.")
    parser.add_argument("--layer-info", required=True, help="trtexec --exportLayerInfo JSON.")
    parser.add_argument("--engine-latency-ms", type=float, required=True, help="Observed engine latency to normalize to.")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
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


def layer_name(layer: dict[str, Any]) -> str:
    return str(layer.get("Name") or layer.get("name") or "")


def tactic_name(layer: dict[str, Any]) -> str:
    return str(layer.get("TacticName") or layer.get("tacticName") or "")


def layer_text(layer: dict[str, Any]) -> str:
    return json.dumps(layer, ensure_ascii=False)


def vlm_node(layer_index: int, projection: str) -> str:
    suffix = "" if layer_index == 0 else f"_{layer_index}"
    return f"/debug_core/{projection}{suffix}/MatMul"


def strict_target_nodes(projections: tuple[str, ...]) -> set[str]:
    return {vlm_node(index, projection) for index in range(32) for projection in projections}


STRICT_QKVO = strict_target_nodes(PROJECTIONS_QKVO)
STRICT_GATE_UP = strict_target_nodes(PROJECTIONS_GATE_UP)
STRICT_DOWN = strict_target_nodes((PROJECTION_DOWN,))
STRICT_W8A8 = STRICT_QKVO | STRICT_GATE_UP


def has_any(text: str, nodes: set[str]) -> bool:
    return any(node in text for node in nodes)


def is_gemm(layer: dict[str, Any]) -> bool:
    name = layer_name(layer).lower()
    tactic = tactic_name(layer).lower()
    layer_type = str(layer.get("LayerType", "")).lower()
    return "gemm" in tactic or layer_type in {"gemm", "fusion"} or "matmul" in name or re.search(r"(^|_)fc", name) is not None


def is_action_expert_or_head(text: str) -> bool:
    if any(token in text for token in ("action_in_proj", "action_time_mlp", "action_out_proj", "state_proj")):
        return True

    # Action expert / ODE unroll reuses VLM-like names with suffixes >= 32.
    for match in re.finditer(
        r"/debug_core/(?:q_proj|k_proj|v_proj|o_proj|input_layernorm|post_attention_layernorm|mlp/(?:gate_proj|up_proj|down_proj|act_fn))_(\d+)",
        text,
    ):
        if int(match.group(1)) >= 32:
            return True

    # VLM attention inner MatMul uses roughly MatMul[_1] ... MatMul_63.
    # MatMul_64+ belongs to the unrolled expert path in this exported graph.
    for match in re.finditer(r"/debug_core/MatMul_(\d+)", text):
        if int(match.group(1)) >= 64:
            return True

    for match in re.finditer(r"/debug_core/Softmax_(\d+)", text):
        if int(match.group(1)) >= 32:
            return True

    return False


def classify_layer(layer: dict[str, Any]) -> str:
    text = layer_text(layer)
    name = layer_name(layer)
    tactic = tactic_name(layer)
    combined = f"{name} {tactic} {text}"

    if "fused_rope_layout" in combined or "SmolVLAFusedRopeLayout" in combined:
        return "Attention / Softmax / RoPE"

    if has_any(text, STRICT_QKVO) and is_gemm(layer):
        return "q/k/v/o INT8 GEMM"
    if has_any(text, STRICT_GATE_UP) and is_gemm(layer):
        return "gate/up INT8 GEMM"
    if has_any(text, STRICT_DOWN) and is_gemm(layer):
        return "down W8A16"

    if has_any(text, STRICT_W8A8) and re.search(r"QuantizeLinear|DequantizeLinear|Cast|Roun|MinMax|Reformat|Tran", combined):
        return "Q/DQ / Cast / Reformat"

    if is_action_expert_or_head(text):
        return "Action Expert"

    if re.search(r"Softmax|MatMul(?:_|\])|Cos|Sin|Cums|RoPE|Where|LessOrEqual|ScatterND|Split|Slic|Tran|Repl|Resh", combined):
        return "Attention / Softmax / RoPE"

    if re.search(r"LayerNorm|layernorm|Add|Mul|Mean|Sqrt|Div|Pow|Silu|Sigmoid|Sub|Exp|Sum|Maxr|Minr|Sele|Residual", combined):
        return "Norm / Residual / Elementwise"

    return "Other"


def read_profile(path: Path) -> tuple[int, list[dict[str, Any]]]:
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise TypeError(f"expected trtexec profile list: {path}")
    count = int(data[0].get("count", 1)) if data and "count" in data[0] else 1
    rows = [row for row in data[1:] if isinstance(row, dict) and "name" in row]
    return count, rows


def format_ms(value: float) -> str:
    return f"{value:.3f} ms"


def format_pct(value: float) -> str:
    return f"{value:.2f}%"


def main() -> None:
    args = parse_args()
    profile_path = Path(args.profile)
    layer_info_path = Path(args.layer_info)
    count, profile_rows = read_profile(profile_path)
    layer_info = find_layers(json.loads(layer_info_path.read_text()))
    info_by_name = {layer_name(layer): layer for layer in layer_info}

    raw_total = sum(float(row.get("averageMs", 0.0)) for row in profile_rows)
    scale = args.engine_latency_ms / raw_total if raw_total else 0.0

    category_raw: dict[str, float] = defaultdict(float)
    category_layers: dict[str, list[dict[str, Any]]] = defaultdict(list)
    missing_info = 0
    for row in profile_rows:
        name = str(row["name"])
        layer = info_by_name.get(name)
        if layer is None:
            missing_info += 1
            layer = {"Name": name, "LayerType": "", "TacticName": "", "Metadata": ""}
        category = classify_layer(layer)
        raw_ms = float(row.get("averageMs", 0.0))
        category_raw[category] += raw_ms
        category_layers[category].append({"name": name, "raw_average_ms": raw_ms, "normalized_ms": raw_ms * scale})

    vlm_categories = [
        "q/k/v/o INT8 GEMM",
        "gate/up INT8 GEMM",
        "down W8A16",
        "Attention / Softmax / RoPE",
        "Q/DQ / Cast / Reformat",
        "Norm / Residual / Elementwise",
    ]
    vlm_total = sum(category_raw.get(category, 0.0) for category in vlm_categories)
    action_total = category_raw.get("Action Expert", 0.0)
    other_total = raw_total - vlm_total - action_total

    def norm(value: float) -> float:
        return value * scale

    vlm_rows = []
    for category in vlm_categories:
        raw = category_raw.get(category, 0.0)
        vlm_rows.append(
            {
                "category": category,
                "raw_profile_ms": raw,
                "normalized_ms": norm(raw),
                "pct_of_vlm": 100.0 * raw / vlm_total if vlm_total else 0.0,
                "num_layers": len(category_layers.get(category, [])),
            }
        )

    top_layers = {}
    for category, rows in category_layers.items():
        top_layers[category] = sorted(rows, key=lambda item: item["raw_average_ms"], reverse=True)[:10]

    summary = {
        "profile": str(profile_path),
        "layer_info": str(layer_info_path),
        "profile_count": count,
        "engine_latency_ms": args.engine_latency_ms,
        "raw_profile_sum_average_ms": raw_total,
        "normalization_scale": scale,
        "missing_layer_info_rows": missing_info,
        "vlm_categories": vlm_rows,
        "totals": {
            "vlm_raw_profile_ms": vlm_total,
            "vlm_normalized_ms": norm(vlm_total),
            "action_expert_raw_profile_ms": action_total,
            "action_expert_normalized_ms": norm(action_total),
            "other_raw_profile_ms": other_total,
            "other_normalized_ms": norm(other_total),
            "engine_normalized_ms": args.engine_latency_ms,
        },
        "category_counts": dict(Counter(classify_layer(info_by_name.get(str(row["name"]), {"Name": str(row["name"])})) for row in profile_rows)),
        "top_layers_by_category": top_layers,
        "notes": [
            "trtexec per-layer averageMs sums can exceed observed latency because TensorRT may use auxiliary streams and overlapping execution.",
            "normalized_ms scales raw per-layer averageMs so category totals sum to the CUDA Graph ON engine latency.",
            "Strict VLM W8A8 targets are layers 0..31 q/k/v/o/gate/up. Action expert repeated same-name nodes and W8A16 down are classified separately.",
        ],
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# 5M TensorRT Per-Layer Profile Summary",
        "",
        f"- Engine latency used for normalization: `{args.engine_latency_ms:.4f} ms`",
        f"- Raw trtexec per-layer averageMs sum: `{raw_total:.4f} ms`",
        f"- Normalization scale: `{scale:.6f}`",
        "",
        "| VLM category | Time | VLM share | Layers |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in vlm_rows:
        lines.append(
            f"| {row['category']} | `{format_ms(row['normalized_ms'])}` | `{format_pct(row['pct_of_vlm'])}` | `{row['num_layers']}` |"
        )
    lines.extend(
        [
            "",
            "| Bucket | Time | Engine share |",
            "| --- | ---: | ---: |",
        ]
    )
    for label, value in (
        ("VLM total", norm(vlm_total)),
        ("Action Expert", norm(action_total)),
        ("Other", norm(other_total)),
        ("Engine", args.engine_latency_ms),
    ):
        lines.append(f"| {label} | `{format_ms(value)}` | `{format_pct(100.0 * value / args.engine_latency_ms if args.engine_latency_ms else 0.0)}` |")

    lines.extend(
        [
            "",
            "Notes:",
            "",
            "- The table uses the selected trtexec profile data and normalizes per-layer profile sums to the measured engine latency.",
            "- Raw per-layer sums exceed latency when TensorRT runs layers concurrently on auxiliary streams.",
            "- Action Expert includes action/state projections and repeated q/k/v/o/mlp nodes from the Flow Matching unroll.",
        ]
    )
    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
