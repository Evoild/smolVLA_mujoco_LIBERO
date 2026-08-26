#!/usr/bin/env python3

"""Compare static fake-quant impact of each VLM text MLP Linear individually."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import torch
from torch import nn

from diagnose_3_4_numeric_baseline import (
    SmolVLADebugCoreWrapper,
    compare_tensors,
    core_inputs_from_policy,
    load_policy,
    tensor_dict_from_outputs,
)
from linear_only_quant import W8A8Linear, load_activation_scales_by_module


DEFAULT_REGEX = r"^model\.vlm_with_expert\.vlm\.model\.text_model\.layers\.([0-9]+)\.mlp\.(gate_proj|up_proj|down_proj)$"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", default="smolvla_libero")
    parser.add_argument("--activation-scales-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--task", default="libero_goal step 3-4 numeric diagnosis")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--token-length", type=int, default=48)
    parser.add_argument("--input-dtype", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--module-regex", default=DEFAULT_REGEX)
    return parser.parse_args()


def split_parent(module_name: str) -> tuple[str, str]:
    if "." not in module_name:
        return "", module_name
    parent_name, child_name = module_name.rsplit(".", 1)
    return parent_name, child_name


def run_wrapper(wrapper: SmolVLADebugCoreWrapper, inputs: tuple[torch.Tensor, ...]) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        outputs = wrapper(*inputs)
    return tensor_dict_from_outputs(wrapper.output_names, outputs)


def main() -> None:
    args = parse_args()
    input_dtype = torch.float32 if args.input_dtype == "fp32" else torch.bfloat16
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    policy, preprocessor, device = load_policy(args.policy_path, args.device)
    wrapper = SmolVLADebugCoreWrapper(policy).to(device=device).eval()
    inputs = core_inputs_from_policy(policy, preprocessor, args.task, args.seed, input_dtype, args.token_length)
    baseline = run_wrapper(wrapper, inputs)

    pattern = re.compile(args.module_regex)
    scales = load_activation_scales_by_module(args.activation_scales_json, (args.module_regex,))
    module_map = dict(wrapper.named_modules())
    targets: list[tuple[int, str, str, nn.Linear]] = []
    for module_name, module in module_map.items():
        match = pattern.search(module_name)
        if match and isinstance(module, nn.Linear):
            layer_idx = int(match.group(1))
            sublayer = match.group(2)
            targets.append((layer_idx, sublayer, module_name, module))
    targets.sort(key=lambda item: (item[0], item[1]))
    if not targets:
        raise RuntimeError(f"no nn.Linear modules matched --module-regex: {args.module_regex}")

    rows: list[dict[str, Any]] = []
    for layer_idx, sublayer, module_name, module in targets:
        if module_name not in scales:
            raise KeyError(f"missing calibrated activation scale for {module_name}")
        parent_name, child_name = split_parent(module_name)
        parent = module_map[parent_name] if parent_name else wrapper
        original = getattr(parent, child_name)
        quantized = W8A8Linear.from_float(module, scales[module_name]).to(device=device).eval()
        setattr(parent, child_name, quantized)
        try:
            candidate = run_wrapper(wrapper, inputs)
        finally:
            setattr(parent, child_name, original)

        for output_name in ("action_chunk", "v_t_step_09", "x_t_step_09"):
            row = compare_tensors("A_vs_E_single", output_name, baseline[output_name], candidate[output_name])
            row.update(
                {
                    "layer": layer_idx,
                    "sublayer": sublayer,
                    "module": module_name,
                    "scale": scales[module_name],
                }
            )
            rows.append(row)

    rows.sort(
        key=lambda row: (
            row["output"] != "action_chunk",
            -float(row["relative_l2_error"]) if row["output"] == "action_chunk" else 0.0,
            int(row["layer"]),
            str(row["sublayer"]),
        )
    )
    summary_rows = [row for row in rows if row["output"] == "action_chunk"]
    summary_rows = sorted(summary_rows, key=lambda row: row["relative_l2_error"], reverse=True)
    summary = {
        "policy_path": args.policy_path,
        "activation_scales_json": args.activation_scales_json,
        "seed": args.seed,
        "module_regex": args.module_regex,
        "num_modules": len(targets),
        "top_action_relative_l2_error": summary_rows[:20],
        "rows": summary_rows,
    }
    with open(output_dir / "single_mlp_module_compare_rows.json", "w") as f:
        json.dump(rows, f, indent=2)
    with open(output_dir / "single_mlp_module_compare_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(output_dir / "single_mlp_module_compare_rows.csv", "w", newline="") as f:
        fieldnames = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
