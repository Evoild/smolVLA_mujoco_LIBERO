#!/usr/bin/env python3

"""TensorRT engine-level VLM attention breakdown for Step 6C.

This script refines the existing 5M TensorRT per-layer bucket
``Attention / Softmax / RoPE`` into engine-level attention buckets using
trtexec detailed layer metadata. It does not rebuild the engine and does not
change the quantization strategy.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import summarize_5m_trt_profile as coarse


BUCKET_ORDER = [
    "rope",
    "qk_matmul",
    "score_scale_mask",
    "softmax",
    "pv_matmul",
    "layout_reshape_transpose",
    "attention_fused_other",
    "unknown_attention",
]

BUCKET_LABELS = {
    "rope": "RoPE",
    "qk_matmul": "QK matmul",
    "score_scale_mask": "scale/mask",
    "softmax": "Softmax",
    "pv_matmul": "PV matmul",
    "layout_reshape_transpose": "reshape/transpose/layout",
    "attention_fused_other": "fused other",
    "unknown_attention": "unknown",
}

DEFAULT_PROFILE = "runs/deploy/5/N-cuda-graph-on-off-5m/cuda_graph_on/profile.json"
DEFAULT_LAYER_INFO = "runs/deploy/6/A-trt-kernel-tactic-inspect-5m/layer_info.json"
DEFAULT_CATEGORY_SUMMARY = "runs/deploy/6/B-5m-final-profile/per_layer_category_summary.json"
DEFAULT_OUTPUT_DIR = "runs/deploy/6/C-trt-attention-breakdown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default=DEFAULT_PROFILE, help="trtexec --exportProfile JSON.")
    parser.add_argument("--layer-info", default=DEFAULT_LAYER_INFO, help="trtexec --exportLayerInfo JSON.")
    parser.add_argument("--category-summary", default=DEFAULT_CATEGORY_SUMMARY)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--engine-latency-ms", type=float, default=None)
    parser.add_argument("--vlm-latency-ms", type=float, default=None)
    parser.add_argument("--attention-latency-ms", type=float, default=None)
    parser.add_argument(
        "--pytorch-native-summary",
        default="runs/deploy/6/B-attention-breakdown/native/attention_breakdown_summary.json",
    )
    parser.add_argument(
        "--pytorch-smoothquant-summary",
        default="runs/deploy/6/B-attention-breakdown/smoothquant_alpha085_full_a8w8_fake_quant/attention_breakdown_summary.json",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_profile(path: Path) -> list[dict[str, Any]]:
    data = load_json(path)
    if not isinstance(data, list):
        raise TypeError(f"expected trtexec profile list: {path}")
    return [row for row in data[1:] if isinstance(row, dict) and "name" in row]


def metadata_text(layer: dict[str, Any]) -> str:
    return str(layer.get("Metadata") or layer.get("metadata") or "")


def origin_nodes(layer: dict[str, Any]) -> list[str]:
    text = metadata_text(layer)
    nodes = re.findall(r"\[ONNX Layer: ([^\]]+)\]", text)
    if nodes:
        return nodes
    name = coarse.layer_name(layer)
    if name.startswith("/debug_core/"):
        return [re.sub(r"_myl\d+_\d+$", "", name)]
    return []


def node_ops(nodes: list[str]) -> set[str]:
    ops = set()
    for node in nodes:
        match = re.search(r"/debug_core/([A-Za-z]+)", node)
        if match:
            ops.add(match.group(1))
    return ops


def layer_blob(layer: dict[str, Any]) -> str:
    return " ".join(
        [
            coarse.layer_name(layer),
            coarse.tactic_name(layer),
            metadata_text(layer),
            json.dumps(layer.get("Inputs", []), ensure_ascii=False),
            json.dumps(layer.get("Outputs", []), ensure_ascii=False),
        ]
    )


def metadata_blob(layer: dict[str, Any]) -> str:
    return " ".join([coarse.layer_name(layer), coarse.tactic_name(layer), metadata_text(layer)])


def contains_op(nodes: list[str], *ops: str) -> bool:
    present = node_ops(nodes)
    return any(op in present for op in ops)


def is_attention_matmul(layer: dict[str, Any], nodes: list[str]) -> bool:
    return coarse.layer_name(layer).startswith("/debug_core/MatMul") or (
        contains_op(nodes, "MatMul") and str(layer.get("LayerType", "")).lower() == "gemm"
    )


def contains_pv(layer: dict[str, Any], nodes: list[str]) -> bool:
    if not is_attention_matmul(layer, nodes):
        return False
    blob = layer_blob(layer)
    if "Softmax" in blob:
        return True
    for node in nodes:
        match = re.search(r"/debug_core/MatMul(?:_(\d+))?$", node)
        if match:
            idx = int(match.group(1) or 0)
            return idx % 2 == 1
    return False


def contains_qk(layer: dict[str, Any], nodes: list[str]) -> bool:
    if not is_attention_matmul(layer, nodes):
        return False
    if contains_pv(layer, nodes):
        return False
    blob = layer_blob(layer)
    if re.search(r"\[\s*15,\s*177,\s*177\s*\]", blob):
        return True
    for node in nodes:
        match = re.search(r"/debug_core/MatMul(?:_(\d+))?$", node)
        if match:
            idx = int(match.group(1) or 0)
            return idx % 2 == 0
    return True


def contains_rope(layer: dict[str, Any], nodes: list[str]) -> bool:
    blob = metadata_blob(layer)
    if "fused_rope_layout" in blob or "SmolVLAFusedRopeLayout" in blob:
        return True
    ops = node_ops(nodes)
    if {"Cos", "Sin"} & ops:
        return True
    if {"Split", "Add", "Sub"} <= ops and "Mul" in ops:
        return True
    return bool(re.search(r"SlicMulMulSlicMulAddMulSub|CosSin|RoPE", blob))


def contains_softmax(layer: dict[str, Any], nodes: list[str]) -> bool:
    return contains_op(nodes, "Softmax") or "Softmax" in metadata_blob(layer)


def contains_mask_scale(layer: dict[str, Any], nodes: list[str]) -> bool:
    ops = node_ops(nodes)
    blob = metadata_blob(layer)
    if {"Where", "LessOrEqual", "And", "Equal", "Or"} & ops:
        return True
    if contains_softmax(layer, nodes) and ({"Mul", "Div", "Max", "Sub"} & ops):
        return True
    return bool(re.search(r"Sele|Lt|Less|Where|MaxrSubExpSumDivMul|MulSele", blob))


def contains_layout(layer: dict[str, Any], nodes: list[str]) -> bool:
    ops = node_ops(nodes)
    blob = metadata_blob(layer)
    if "fused_rope_layout" in blob or "SmolVLAFusedRopeLayout" in blob:
        return True
    if {"Transpose", "Reshape", "Expand", "Unsqueeze", "Concat", "Slice", "ScatterND"} & ops:
        return True
    return bool(re.search(r"Resh|Tran|Repl|Conc|Slic|Scat|Move", blob))


def precision(layer: dict[str, Any]) -> str:
    values = []
    for field in ("Inputs", "Outputs"):
        for item in layer.get(field, []) or []:
            dtype = item.get("Format/Datatype") if isinstance(item, dict) else None
            if dtype:
                values.append(str(dtype))
    return "; ".join(sorted(set(values)))


def origin_module(nodes: list[str]) -> str:
    text = " ".join(nodes)
    for module in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
        if module in text:
            return module
    if "Softmax" in text:
        return "attention.softmax"
    if "MatMul" in text:
        return "attention.matmul"
    if any(op in text for op in ("Cos", "Sin", "Split", "ScatterND")):
        return "attention.rope_or_layout"
    return "attention"


def classify_attention_bucket(flags: dict[str, bool]) -> str:
    stage_count = sum(
        bool(flags[key])
        for key in ("contains_rope", "contains_qk", "contains_softmax", "contains_pv", "contains_layout", "contains_mask_scale")
    )
    if flags["contains_qk"] and not any(flags[k] for k in ("contains_softmax", "contains_pv", "contains_rope", "contains_mask_scale")):
        return "qk_matmul"
    if flags["contains_pv"] and not any(flags[k] for k in ("contains_softmax", "contains_qk", "contains_rope", "contains_mask_scale")):
        return "pv_matmul"
    if flags["contains_rope"]:
        # RoPE often fuses reshape/scatter; keep it in RoPE unless it also
        # includes attention score/probability work.
        if not any(flags[k] for k in ("contains_qk", "contains_softmax", "contains_pv", "contains_mask_scale")):
            return "rope"
        return "attention_fused_other"
    if flags["contains_softmax"] and flags["contains_mask_scale"]:
        return "attention_fused_other"
    if flags["contains_softmax"]:
        return "softmax"
    if flags["contains_mask_scale"]:
        return "score_scale_mask"
    if flags["contains_layout"]:
        return "layout_reshape_transpose"
    if stage_count > 1:
        return "attention_fused_other"
    return "unknown_attention"


def format_ms(value: float) -> str:
    return f"{value:.3f} ms"


def format_pct(value: float) -> str:
    return f"{value:.2f}%"


def pct(num: float, den: float) -> float:
    return 100.0 * num / den if den else 0.0


def summarize_buckets(rows: list[dict[str, Any]], attention_norm_ms: float, vlm_ms: float, engine_ms: float) -> list[dict[str, Any]]:
    raw_by_bucket: dict[str, float] = defaultdict(float)
    count_by_bucket: Counter[str] = Counter()
    raw_attention = sum(float(row["average_ms_raw"]) for row in rows)
    attention_scale = attention_norm_ms / raw_attention if raw_attention else 0.0
    for row in rows:
        raw_by_bucket[str(row["bucket"])] += float(row["average_ms_raw"])
        count_by_bucket[str(row["bucket"])] += 1
    out = []
    for bucket in BUCKET_ORDER:
        raw = raw_by_bucket.get(bucket, 0.0)
        norm = raw * attention_scale
        out.append(
            {
                "bucket": bucket,
                "label": BUCKET_LABELS[bucket],
                "raw_ms": raw,
                "normalized_ms": norm,
                "attention_share_pct": pct(norm, attention_norm_ms),
                "vlm_share_pct": pct(norm, vlm_ms),
                "engine_share_pct": pct(norm, engine_ms),
                "layer_count": count_by_bucket.get(bucket, 0),
            }
        )
    total_raw = sum(row["raw_ms"] for row in out)
    out.append(
        {
            "bucket": "total",
            "label": "Total",
            "raw_ms": total_raw,
            "normalized_ms": total_raw * attention_scale,
            "attention_share_pct": 100.0 if attention_norm_ms else 0.0,
            "vlm_share_pct": pct(total_raw * attention_scale, vlm_ms),
            "engine_share_pct": pct(total_raw * attention_scale, engine_ms),
            "layer_count": len(rows),
        }
    )
    return out


def top_rows(rows: list[dict[str, Any]], n: int = 20) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: float(row["average_ms_raw"]), reverse=True)[:n]


def read_optional_py_summary(path: Path) -> dict[str, float] | None:
    if not path.is_file():
        return None
    data = load_json(path)
    values = data.get("total_by_segment_ms", {})
    return {str(k): float(v) for k, v in values.items()}


def write_pytorch_vs_trt(output_dir: Path, args: argparse.Namespace, bucket_rows: list[dict[str, Any]]) -> None:
    native = read_optional_py_summary(Path(args.pytorch_native_summary))
    sq = read_optional_py_summary(Path(args.pytorch_smoothquant_summary))
    trt = {row["bucket"]: row for row in bucket_rows if row["bucket"] != "total"}
    mapping = [
        ("rope", "rope"),
        ("qk_cast_fp32 + qk_transpose + qk_matmul + score_scale", "qk_matmul / score_scale_mask / fused other"),
        ("mask_where + softmax + probs_cast", "softmax / score_scale_mask / fused other"),
        ("pv_matmul", "pv_matmul"),
        ("kv_expand_reshape + output_permute_reshape", "layout_reshape_transpose"),
        ("projection_q/k/v/o", "not in attention bucket; already counted as INT8 projection GEMM"),
    ]
    lines = [
        "# PyTorch 6B vs TensorRT 6C Attention Breakdown",
        "",
        "PyTorch 6B is source-level forward segmentation. TensorRT 6C is final engine/kernel-level segmentation after Myelin fusion and tactic selection.",
        "These columns are structural correspondences only; they are not per-op speedup ratios.",
        "",
        "| PyTorch forward segment | TensorRT engine bucket | TensorRT normalized ms | note |",
        "| --- | --- | ---: | --- |",
    ]
    for py_seg, trt_bucket in mapping:
        trt_ms = ""
        note = ""
        if trt_bucket == "layout_reshape_transpose":
            trt_ms = format_ms(trt[trt_bucket]["normalized_ms"])
        elif trt_bucket == "pv_matmul":
            trt_ms = format_ms(trt[trt_bucket]["normalized_ms"])
        elif trt_bucket == "rope":
            trt_ms = format_ms(trt[trt_bucket]["normalized_ms"])
        elif "qk_matmul" in trt_bucket:
            trt_ms = format_ms(
                trt["qk_matmul"]["normalized_ms"]
                + trt["score_scale_mask"]["normalized_ms"]
                + trt["attention_fused_other"]["normalized_ms"]
            )
            note = "includes fused QK/scale/mask/softmax-like Myelin kernels when metadata spans stages"
        elif "softmax" in trt_bucket:
            trt_ms = format_ms(
                trt["softmax"]["normalized_ms"]
                + trt["score_scale_mask"]["normalized_ms"]
                + trt["attention_fused_other"]["normalized_ms"]
            )
            note = "TensorRT commonly fuses mask/scale and softmax internals"
        else:
            trt_ms = "N/A"
            note = "outside the 5.429 ms attention bucket"
        lines.append(f"| `{py_seg}` | `{trt_bucket}` | `{trt_ms}` | {note} |")

    lines += [
        "",
        "PyTorch summary availability:",
        "",
        f"- native: `{'available' if native is not None else 'missing'}` at `{args.pytorch_native_summary}`",
        f"- SmoothQuant fake quant: `{'available' if sq is not None else 'missing'}` at `{args.pytorch_smoothquant_summary}`",
        "",
        "Decision about a TensorRT plugin should use the TensorRT engine-level 6C profile as the primary signal.",
    ]
    (output_dir / "pytorch_vs_trt_attention_breakdown.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def decision(chain_norm_ms: float, attention_ms: float, vlm_ms: float, engine_ms: float, counters: dict[str, int], bucket_rows: list[dict[str, Any]]) -> tuple[str, list[str], list[str]]:
    chain_engine_share = pct(chain_norm_ms, engine_ms)
    chain_vlm_share = pct(chain_norm_ms, vlm_ms)
    attention_vlm_share = pct(attention_ms, vlm_ms)
    unfused_layers = counters.get("still_separate_layer_count", 0) + counters.get("layout_or_reformat_boundary_count", 0)
    proceed = attention_vlm_share > 50.0 and chain_engine_share >= 5.0 and unfused_layers > 0
    if proceed:
        stages = []
        by_bucket = {row["bucket"]: row for row in bucket_rows}
        if by_bucket["rope"]["normalized_ms"] + by_bucket["layout_reshape_transpose"]["normalized_ms"] > 0.5:
            stages.append("RoPE + transpose/layout")
        if by_bucket["qk_matmul"]["normalized_ms"] + by_bucket["score_scale_mask"]["normalized_ms"] > 0.5:
            stages.append("QK + scale/mask")
        if by_bucket["softmax"]["normalized_ms"] + by_bucket["attention_fused_other"]["normalized_ms"] + by_bucket["pv_matmul"]["normalized_ms"] > 0.5:
            stages.append("softmax + PV")
        if by_bucket["layout_reshape_transpose"]["normalized_ms"] > 0.5:
            stages.append("output layout")
        reasons = [
            f"attention bucket is {attention_vlm_share:.2f}% of VLM",
            f"candidate chain is {chain_engine_share:.2f}% of engine and {chain_vlm_share:.2f}% of VLM",
            "q/k/v/o projection GEMM is already outside the bottleneck and was verified as INT8",
        ]
        return "PROCEED_WITH_FUSED_ATTENTION", stages, reasons
    reasons = [
        f"candidate chain engine share is {chain_engine_share:.2f}%",
        f"attention VLM share is {attention_vlm_share:.2f}%",
        f"separate/materialization boundary layers counted: {unfused_layers}",
    ]
    return "DO_NOT_PROCEED", [], reasons


def main() -> None:
    args = parse_args()
    profile_path = Path(args.profile)
    layer_info_path = Path(args.layer_info)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    category_summary = load_json(Path(args.category_summary))
    engine_ms = float(args.engine_latency_ms or category_summary["engine_latency_ms"])
    vlm_ms = float(args.vlm_latency_ms or category_summary["totals"]["vlm_normalized_ms"])
    attention_ms = float(
        args.attention_latency_ms
        or next(row["normalized_ms"] for row in category_summary["vlm_categories"] if row["category"] == "Attention / Softmax / RoPE")
    )

    profile_rows = read_profile(profile_path)
    layer_info = coarse.find_layers(load_json(layer_info_path))
    info_by_name = {coarse.layer_name(layer): layer for layer in layer_info}

    raw_attention_rows: list[dict[str, Any]] = []
    for prof in profile_rows:
        name = str(prof["name"])
        layer = info_by_name.get(name, {"Name": name, "LayerType": "", "TacticName": "", "Metadata": ""})
        if coarse.classify_layer(layer) != "Attention / Softmax / RoPE":
            continue
        raw_ms = float(prof.get("averageMs", 0.0))
        raw_attention_rows.append({"profile": prof, "layer": layer, "raw_ms": raw_ms})

    raw_attention_ms = sum(item["raw_ms"] for item in raw_attention_rows)
    attention_scale = attention_ms / raw_attention_ms if raw_attention_ms else 0.0

    output_rows: list[dict[str, Any]] = []
    already_fused_layer_count = 0
    still_separate_layer_count = 0
    layout_or_reformat_boundary_count = 0
    for item in raw_attention_rows:
        layer = item["layer"]
        nodes = origin_nodes(layer)
        flags = {
            "contains_rope": contains_rope(layer, nodes),
            "contains_qk": contains_qk(layer, nodes),
            "contains_softmax": contains_softmax(layer, nodes),
            "contains_pv": contains_pv(layer, nodes),
            "contains_layout": contains_layout(layer, nodes),
            "contains_mask_scale": contains_mask_scale(layer, nodes),
        }
        bucket = classify_attention_bucket(flags)
        flag_count = sum(flags.values())
        if bucket == "attention_fused_other" or flag_count > 1:
            already_fused_layer_count += 1
        else:
            still_separate_layer_count += 1
        if flags["contains_layout"] and not flags["contains_rope"]:
            layout_or_reformat_boundary_count += 1

        output_rows.append(
            {
                "trt_layer_name": coarse.layer_name(layer),
                "bucket": bucket,
                "average_ms_raw": item["raw_ms"],
                "average_ms_normalized": item["raw_ms"] * attention_scale,
                "origin_onnx_nodes": ";".join(nodes),
                "origin_module": origin_module(nodes),
                "tactic": coarse.tactic_name(layer),
                "precision": precision(layer),
                **flags,
                "layer_type": str(layer.get("LayerType", "")),
                "stream_id": layer.get("StreamId", ""),
            }
        )

    bucket_rows = summarize_buckets(output_rows, attention_ms, vlm_ms, engine_ms)
    candidate_buckets = {"rope", "qk_matmul", "score_scale_mask", "softmax", "pv_matmul", "layout_reshape_transpose", "attention_fused_other"}
    fused_candidate_chain_ms = sum(row["normalized_ms"] for row in bucket_rows if row["bucket"] in candidate_buckets)
    counters = {
        "already_fused_layer_count": already_fused_layer_count,
        "still_separate_layer_count": still_separate_layer_count,
        "layout_or_reformat_boundary_count": layout_or_reformat_boundary_count,
    }
    decision_label, recommended_stages, decision_reasons = decision(
        fused_candidate_chain_ms, attention_ms, vlm_ms, engine_ms, counters, bucket_rows
    )

    with open(output_dir / "trt_attention_kernel_rows.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "trt_layer_name",
            "bucket",
            "average_ms_raw",
            "average_ms_normalized",
            "origin_onnx_nodes",
            "origin_module",
            "tactic",
            "precision",
            "contains_rope",
            "contains_qk",
            "contains_softmax",
            "contains_pv",
            "contains_layout",
            "contains_mask_scale",
            "layer_type",
            "stream_id",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    with open(output_dir / "trt_attention_breakdown_summary.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = ["bucket", "label", "raw_ms", "normalized_ms", "attention_share_pct", "vlm_share_pct", "engine_share_pct", "layer_count"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(bucket_rows)

    summary = {
        "profile": str(profile_path),
        "layer_info": str(layer_info_path),
        "category_summary": str(args.category_summary),
        "engine_latency_ms": engine_ms,
        "vlm_latency_ms": vlm_ms,
        "attention_latency_ms": attention_ms,
        "raw_attention_ms": raw_attention_ms,
        "attention_normalization_scale": attention_scale,
        "bucket_rows": bucket_rows,
        "fused_candidate_chain": {
            "normalized_ms": fused_candidate_chain_ms,
            "share_of_attention_pct": pct(fused_candidate_chain_ms, attention_ms),
            "share_of_vlm_pct": pct(fused_candidate_chain_ms, vlm_ms),
            "share_of_engine_pct": pct(fused_candidate_chain_ms, engine_ms),
            **counters,
        },
        "decision": decision_label,
        "recommended_fusion_stages": recommended_stages,
        "decision_reasons": decision_reasons,
        "top_attention_layers": top_rows(output_rows),
        "notes": [
            "Raw trtexec per-layer averageMs can exceed observed engine latency because TensorRT auxiliary streams and overlapping execution are present.",
            "normalized_ms scales attention raw rows so the attention bucket sums to the existing 5.429 ms attribution.",
            "normalized bucket values are for bottleneck attribution only and are not true serial additive kernel latency.",
            "Fused Myelin layers with multiple logical stages are counted once and retain per-stage flags.",
        ],
    }
    (output_dir / "trt_attention_breakdown_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# TensorRT Attention Kernel Breakdown",
        "",
        f"- Engine latency: `{engine_ms:.4f} ms`",
        f"- VLM latency: `{vlm_ms:.4f} ms`",
        f"- Attention / Softmax / RoPE latency: `{attention_ms:.4f} ms`",
        f"- Raw attention per-layer sum: `{raw_attention_ms:.4f} ms`",
        f"- Attention normalization scale: `{attention_scale:.6f}`",
        "",
        "| TensorRT attention bucket | raw ms | normalized ms | attention share | VLM share | engine share | layer count |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in bucket_rows:
        lines.append(
            f"| {row['label']} | `{format_ms(row['raw_ms'])}` | `{format_ms(row['normalized_ms'])}` | "
            f"`{format_pct(row['attention_share_pct'])}` | `{format_pct(row['vlm_share_pct'])}` | "
            f"`{format_pct(row['engine_share_pct'])}` | `{row['layer_count']}` |"
        )
    chain = summary["fused_candidate_chain"]
    lines += [
        "",
        "## Fused Candidate Chain",
        "",
        f"- fused_candidate_chain_ms: `{chain['normalized_ms']:.4f} ms`",
        f"- share of attention: `{chain['share_of_attention_pct']:.2f}%`",
        f"- share of VLM: `{chain['share_of_vlm_pct']:.2f}%`",
        f"- share of engine: `{chain['share_of_engine_pct']:.2f}%`",
        f"- already_fused_layer_count: `{chain['already_fused_layer_count']}`",
        f"- still_separate_layer_count: `{chain['still_separate_layer_count']}`",
        f"- layout_or_reformat_boundary_count: `{chain['layout_or_reformat_boundary_count']}`",
        "",
        "## Decision",
        "",
        f"`Decision: {decision_label}`",
        "",
        "Reasons:",
    ]
    lines.extend(f"- {reason}" for reason in decision_reasons)
    if recommended_stages:
        lines += ["", "Recommended next-stage fusion targets:"]
        lines.extend(f"- {stage}" for stage in recommended_stages)
    lines += [
        "",
        "## Top Attention Layers",
        "",
        "| bucket | normalized ms | raw ms | layer | flags |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    for row in top_rows(output_rows, 20):
        flags = ",".join(
            key.replace("contains_", "")
            for key in ("contains_rope", "contains_qk", "contains_softmax", "contains_pv", "contains_layout", "contains_mask_scale")
            if row[key]
        )
        lines.append(
            f"| `{row['bucket']}` | `{row['average_ms_normalized']:.6f}` | `{row['average_ms_raw']:.6f}` | "
            f"`{row['trt_layer_name']}` | `{flags}` |"
        )
    lines += [
        "",
        "Limitations: raw per-layer sums are affected by TensorRT auxiliary streams and overlap; normalized values are attribution values only.",
        "A Myelin fused layer can cover multiple ONNX stages, so flags describe coverage but the layer is counted once.",
    ]
    (output_dir / "trt_attention_breakdown_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    write_pytorch_vs_trt(output_dir, args, bucket_rows)
    print((output_dir / "trt_attention_breakdown_summary.md").read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
