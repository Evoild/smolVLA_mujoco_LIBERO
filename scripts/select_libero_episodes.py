#!/usr/bin/env python3
"""Select episode indices for one suite in HuggingFaceVLA/libero v3."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq


SUITE_TASK_RANGES = {
    "libero_10": range(0, 10),
    "libero_goal": range(10, 20),
    "libero_object": range(20, 30),
    "libero_spatial": range(30, 40),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--suite", required=True, choices=sorted(SUITE_TASK_RANGES))
    args = parser.parse_args()

    tasks_path = args.dataset_root / "meta/tasks.parquet"
    episodes_path = args.dataset_root / "meta/episodes/chunk-000/file-000.parquet"
    if not tasks_path.is_file() or not episodes_path.is_file():
        parser.error(f"invalid LIBERO v3 dataset root: {args.dataset_root}")

    task_rows = pq.read_table(tasks_path).to_pylist()
    if len(task_rows) != 40:
        parser.error(f"expected 40 LIBERO tasks, found {len(task_rows)}")

    name_column = "__index_level_0__"
    selected_task_names = {
        row[name_column]
        for row in task_rows
        if int(row["task_index"]) in SUITE_TASK_RANGES[args.suite]
    }
    if len(selected_task_names) != 10:
        parser.error(f"expected 10 tasks for {args.suite}, found {len(selected_task_names)}")

    episode_rows = pq.read_table(episodes_path, columns=["episode_index", "tasks"]).to_pylist()
    selected_episodes = [
        int(row["episode_index"])
        for row in episode_rows
        if row["tasks"] and set(row["tasks"]).issubset(selected_task_names)
    ]
    if not selected_episodes:
        parser.error(f"no episodes matched {args.suite}")

    print(
        f"Selected {len(selected_episodes)} episodes from {args.suite} "
        f"({len(selected_task_names)} tasks)",
        file=sys.stderr,
    )
    print(json.dumps(selected_episodes, separators=(",", ":")))


if __name__ == "__main__":
    main()
