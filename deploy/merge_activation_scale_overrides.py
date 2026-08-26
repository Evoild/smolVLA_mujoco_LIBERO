#!/usr/bin/env python3

"""Merge selected module calibration entries from an override JSON into a base JSON."""

from __future__ import annotations

import argparse
import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="Base activation scale JSON.")
    parser.add_argument("--override", required=True, help="Activation scale JSON containing replacement entries.")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--override-module-regex",
        action="append",
        required=True,
        help="Module regex whose entries should be taken from --override. Can be repeated.",
    )
    return parser.parse_args()


def entries_by_module(calibration: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result = {}
    for entry in calibration.get("linear_call_scales", []):
        module = str(entry.get("module", ""))
        if module:
            result[module] = entry
    return result


def main() -> None:
    args = parse_args()
    base_path = Path(args.base)
    override_path = Path(args.override)
    output_path = Path(args.output)

    base = json.loads(base_path.read_text())
    override = json.loads(override_path.read_text())
    override_by_module = entries_by_module(override)
    patterns = [re.compile(pattern) for pattern in args.override_module_regex]

    merged = deepcopy(base)
    replaced = []
    kept = []
    for idx, entry in enumerate(merged.get("linear_call_scales", [])):
        module = str(entry.get("module", ""))
        should_override = bool(module) and any(pattern.search(module) for pattern in patterns)
        if not should_override:
            kept.append(module)
            continue
        if module not in override_by_module:
            raise KeyError(f"missing override scale for module: {module}")
        replacement = deepcopy(override_by_module[module])
        replacement["index"] = entry.get("index", idx)
        merged["linear_call_scales"][idx] = replacement
        replaced.append(module)

    report = {
        "merge_source": "base_with_selected_module_overrides",
        "base": str(base_path),
        "override": str(override_path),
        "override_module_regex": args.override_module_regex,
        "num_linear_call_scales": len(merged.get("linear_call_scales", [])),
        "num_replaced": len(replaced),
        "num_kept": len(kept),
        "replaced_modules": replaced,
    }
    merged["merge_report"] = report

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(merged, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "replaced_modules"}, indent=2))


if __name__ == "__main__":
    main()
