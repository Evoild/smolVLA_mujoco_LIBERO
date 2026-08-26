#!/usr/bin/env python3

"""Merge scalar and per-channel activation scale files for hybrid fake quant."""

from __future__ import annotations

import argparse
import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scalar-scales", required=True, help="Base calibration JSON with scalar per-call scales.")
    parser.add_argument("--channel-scales", required=True, help="Calibration JSON with per-channel scales.")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--per-channel-module-regex",
        action="append",
        required=True,
        help="Module regex whose entries should be taken from --channel-scales. Can be repeated.",
    )
    return parser.parse_args()


def entries_by_module(calibration: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result = {}
    for entry in calibration.get("linear_call_scales", []):
        module = str(entry.get("module", ""))
        if module:
            result[module] = entry
    return result


def scale_kind(scale: Any) -> str:
    return "per_channel" if isinstance(scale, list) else "per_tensor"


def main() -> None:
    args = parse_args()
    scalar_path = Path(args.scalar_scales)
    channel_path = Path(args.channel_scales)
    output_path = Path(args.output)

    scalar = json.loads(scalar_path.read_text())
    channel = json.loads(channel_path.read_text())
    channel_by_module = entries_by_module(channel)
    patterns = [re.compile(pattern) for pattern in args.per_channel_module_regex]

    merged = deepcopy(scalar)
    replaced = []
    kept_scalar = []
    for entry in merged.get("linear_call_scales", []):
        module = str(entry.get("module", ""))
        use_channel = bool(module) and any(pattern.search(module) for pattern in patterns)
        if not use_channel:
            entry["scale_granularity"] = scale_kind(entry.get("scale"))
            kept_scalar.append(module)
            continue
        if module not in channel_by_module:
            raise KeyError(f"missing per-channel scale for module: {module}")
        channel_entry = channel_by_module[module]
        if not isinstance(channel_entry.get("scale"), list):
            raise TypeError(f"selected module does not have list scale: {module}")
        entry["scale"] = channel_entry["scale"]
        entry["scale_source"] = channel_entry.get("scale_source", "per_channel")
        entry["scale_granularity"] = "per_channel_last_dim"
        for key in ("channels", "amax_max", "percentile_amax_max", "amax_scale"):
            if key in channel_entry:
                entry[key] = channel_entry[key]
        replaced.append(module)

    report = {
        "hybrid_scale_source": "scalar_base_with_selected_per_channel_overrides",
        "scalar_scales": str(scalar_path),
        "channel_scales": str(channel_path),
        "per_channel_module_regex": args.per_channel_module_regex,
        "num_linear_call_scales": len(merged.get("linear_call_scales", [])),
        "num_per_channel_overrides": len(replaced),
        "num_scalar_kept": len(kept_scalar),
        "per_channel_modules": replaced,
    }
    merged["hybrid_report"] = report
    merged["scale_granularity"] = "hybrid_per_tensor_scalar_plus_selected_per_channel"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(merged, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "per_channel_modules"}, indent=2))


if __name__ == "__main__":
    main()
