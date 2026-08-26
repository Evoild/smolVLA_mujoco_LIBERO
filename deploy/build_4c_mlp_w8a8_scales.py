#!/usr/bin/env python3

"""Build combined gate/up/down W8A8 activation scales for Step 4C."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-up-scales", required=True)
    parser.add_argument("--down-scales", required=True)
    parser.add_argument("--mismatch-csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--targeted-factor", type=float, default=1.1)
    parser.add_argument("--targeted-ratio-threshold", type=float, default=0.5)
    parser.add_argument("--targeted-clip-threshold", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.gate_up_scales) as f:
        gate_up = json.load(f)
    with open(args.down_scales) as f:
        down = json.load(f)

    entries = []
    for entry in gate_up.get("linear_call_scales", []):
        module = str(entry.get("module", ""))
        if module.endswith(".gate_proj") or module.endswith(".up_proj"):
            item = dict(entry)
            item["scale_source"] = f"{item.get('scale_source', 'unknown')};step4c_gate_up"
            entries.append(item)

    down_entries_by_module = {}
    for entry in down.get("linear_call_scales", []):
        module = str(entry.get("module", ""))
        if module.endswith(".down_proj"):
            item = dict(entry)
            item["scale"] = [float(value) for value in item["scale"]]
            item["scale_source"] = f"{item.get('scale_source', 'unknown')};step4c_down_512_p99.99"
            down_entries_by_module[module] = item

    targeted = set()
    with open(args.mismatch_csv) as f:
        for row in csv.DictReader(f):
            ratio = float(row.get("scale_ratio", "inf"))
            clip = float(row.get("clipping_rate", "0"))
            if ratio < args.targeted_ratio_threshold or clip > args.targeted_clip_threshold:
                targeted.add((row["pytorch_module"], int(row["channel"])))

    for module, channel in sorted(targeted):
        if module in down_entries_by_module:
            scale = down_entries_by_module[module]["scale"]
            if 0 <= channel < len(scale):
                scale[channel] *= args.targeted_factor
                down_entries_by_module[module]["scale_source"] += f";targeted_ch{channel}_x{args.targeted_factor}"

    entries.extend(down_entries_by_module[module] for module in sorted(down_entries_by_module))
    for idx, entry in enumerate(entries):
        entry["index"] = idx

    report = {
        "policy_path": gate_up.get("policy_path") or down.get("policy_path"),
        "tasks": {
            "gate_up": gate_up.get("tasks"),
            "down": down.get("tasks"),
        },
        "sources": {
            "gate_up_scales": str(args.gate_up_scales),
            "down_scales": str(args.down_scales),
            "mismatch_csv": str(args.mismatch_csv),
        },
        "quantization": "step4c_vlm_mlp_gate_up_down_w8a8_static_per_channel",
        "targeted_factor": args.targeted_factor,
        "targeted_channels": [
            {"module": module, "channel": channel}
            for module, channel in sorted(targeted)
            if module in down_entries_by_module
        ],
        "linear_call_scales": entries,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(report, f, indent=2)
    print(
        json.dumps(
            {
                "output": str(output),
                "num_linear_call_scales": len(entries),
                "targeted_channels": report["targeted_channels"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
