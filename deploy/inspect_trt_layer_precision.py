#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


TARGET_RE = re.compile(
    r"(?:^|/)(q_proj|k_proj|v_proj|o_proj|mlp/(?:gate_proj|up_proj|down_proj))(?:_[0-9]+)?/MatMul"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize TensorRT layer_info precision coverage.")
    parser.add_argument("--layer-info", required=True, help="JSON exported by trtexec --exportLayerInfo.")
    parser.add_argument("--output", default=None, help="Optional JSON summary path.")
    return parser.parse_args()


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


def find_layers(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if not isinstance(data, dict):
        return []
    for key in ("Layers", "layers", "layerInfo", "LayerInfo"):
        value = data.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    layers: list[dict[str, Any]] = []
    for value in data.values():
        if isinstance(value, list) and value and all(isinstance(item, dict) for item in value):
            if any("Layer" in "".join(item.keys()) or "Name" in item or "name" in item for item in value):
                layers = value
                break
    return layers


def layer_name(layer: dict[str, Any]) -> str:
    for key in ("Name", "name", "LayerName", "layerName"):
        if key in layer:
            return str(layer[key])
    strings = walk_strings(layer)
    return strings[0] if strings else ""


def precision_class(layer: dict[str, Any]) -> str:
    text = " ".join(walk_strings(layer)).upper()
    has_int8 = "INT8" in text
    has_bf16 = "BF16" in text or "BFLOAT16" in text
    has_fp16 = "FP16" in text or "FLOAT16" in text or "HALF" in text
    has_fp32 = "FP32" in text or "FLOAT32" in text or re.search(r"\bFLOAT\b", text) is not None
    if has_int8:
        return "INT8"
    if has_bf16:
        return "BF16"
    if has_fp16:
        return "FP16"
    if has_fp32:
        return "FP32"
    return "unknown"


def target_group(name: str) -> str | None:
    match = TARGET_RE.search(name)
    if not match:
        return None
    group = match.group(1)
    if group.startswith("mlp/"):
        return group
    return group


def target_groups(layer: dict[str, Any]) -> list[str]:
    groups: list[str] = []
    seen: set[str] = set()
    for text in walk_strings(layer):
        for match in TARGET_RE.finditer(text):
            group = match.group(1)
            if group not in seen:
                groups.append(group)
                seen.add(group)
    return groups


def main() -> None:
    args = parse_args()
    path = Path(args.layer_info)
    data = json.loads(path.read_text())
    layers = find_layers(data)
    rows = []
    for layer in layers:
        name = layer_name(layer)
        group = target_group(name)
        groups = target_groups(layer)
        pclass = precision_class(layer)
        rows.append({"name": name, "target_group": group, "target_groups": groups, "precision_class": pclass})

    all_counts = Counter(row["precision_class"] for row in rows)
    target_rows = [row for row in rows if row["target_group"]]
    target_counts = Counter(row["precision_class"] for row in target_rows)
    by_group = {
        group: dict(Counter(row["precision_class"] for row in target_rows if row["target_group"] == group))
        for group in sorted({row["target_group"] for row in target_rows})
    }
    metadata_target_rows = [row for row in rows if row["target_groups"]]
    metadata_target_counts = Counter(row["precision_class"] for row in metadata_target_rows)
    by_metadata_group = {
        group: dict(Counter(row["precision_class"] for row in metadata_target_rows if group in row["target_groups"]))
        for group in sorted({group for row in metadata_target_rows for group in row["target_groups"]})
    }
    non_int8_targets = [
        row for row in target_rows if row["precision_class"] != "INT8"
    ][:100]

    summary = {
        "layer_info": str(path),
        "num_layers": len(rows),
        "precision_counts_all_layers": dict(all_counts),
        "num_target_matmul_layers": len(target_rows),
        "precision_counts_target_matmul_layers": dict(target_counts),
        "precision_counts_by_target_group": by_group,
        "num_non_int8_target_matmul_layers": sum(1 for row in target_rows if row["precision_class"] != "INT8"),
        "non_int8_target_matmul_layer_samples": non_int8_targets,
        "num_metadata_target_layers": len(metadata_target_rows),
        "precision_counts_metadata_target_layers": dict(metadata_target_counts),
        "precision_counts_by_metadata_target_group": by_metadata_group,
        "note": (
            "This parser classifies precision by strings present in TensorRT layer_info. "
            "precision_counts_target_matmul_layers matches target patterns in TensorRT layer names only. "
            "precision_counts_metadata_target_layers also scans metadata, so it can capture fused layers whose names "
            "do not contain the original ONNX target node."
        ),
    }

    print(json.dumps(summary, indent=2))
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
