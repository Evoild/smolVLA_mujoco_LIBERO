#!/usr/bin/env python3
"""Turn LeRobot eval_info.json files into compact metrics and replay indexes."""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path, help="eval_info.json file(s) or directories")
    parser.add_argument("--output-dir", type=Path, default=Path("results/eval_report"))
    parser.add_argument("--plot-suite", help="Only include this suite in success-rate curves")
    return parser.parse_args()


def discover(inputs: Iterable[Path]) -> list[Path]:
    files: set[Path] = set()
    for item in inputs:
        if item.is_file():
            files.add(item.resolve())
        elif item.is_dir():
            files.update(path.resolve() for path in item.rglob("eval_info.json"))
        else:
            raise FileNotFoundError(item)
    if not files:
        raise FileNotFoundError("no eval_info.json found")
    return sorted(files)


def infer_seed(path: Path) -> str:
    for part in reversed(path.parts):
        match = re.fullmatch(r"seed[_-]?(\d+)", part)
        if match:
            return match.group(1)
    return "unknown"


def infer_run(path: Path) -> str:
    if re.fullmatch(r"seed[_-]?\d+", path.parent.name):
        return path.parent.parent.name
    return path.parent.name


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def success_count(values: list[Any]) -> int:
    return sum(bool(value) for value in values)


def write_success_curve_svg(path: Path, plot_values: dict[tuple[str, str, int], list[bool]]) -> None:
    series = sorted({(run, suite) for run, suite, _ in plot_values})
    runs = {run for run, _ in series}
    suites = {suite for _, suite in series}
    width, height = 900, 560
    left, right, top = 80, 30, 50
    bottom = max(90, 60 + 20 * len(series))
    plot_width, plot_height = width - left - right, height - top - bottom
    task_ids = sorted({task_id for _, _, task_id in plot_values})
    max_task = max(task_ids, default=1)
    colors = ["#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed"]

    def series_label(run: str, suite: str) -> str:
        if len(runs) > 1 and len(suites) == 1:
            return run
        if len(runs) == 1:
            return suite
        return f"{run} / {suite}"

    def x(task_id: int) -> float:
        return left + (task_id / max(max_task, 1)) * plot_width

    def y(rate: float) -> float:
        return top + (100 - rate) / 100 * plot_height

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="450" y="28" text-anchor="middle" font-family="sans-serif" font-size="19">LIBERO per-task success rate</text>',
    ]
    for rate in range(0, 101, 20):
        y_pos = y(rate)
        lines.append(f'<line x1="{left}" y1="{y_pos:.1f}" x2="{width-right}" y2="{y_pos:.1f}" stroke="#d1d5db"/>')
        lines.append(f'<text x="{left-12}" y="{y_pos+5:.1f}" text-anchor="end" font-family="sans-serif" font-size="12">{rate}</text>')
    for task_id in task_ids:
        x_pos = x(task_id)
        lines.append(f'<text x="{x_pos:.1f}" y="{height-bottom+22}" text-anchor="middle" font-family="sans-serif" font-size="12">{task_id}</text>')
    lines.extend(
        [
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#111827"/>',
            f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#111827"/>',
            f'<text x="{left+plot_width/2:.1f}" y="{height-18}" text-anchor="middle" font-family="sans-serif" font-size="14">Task ID</text>',
            f'<text x="20" y="{top+plot_height/2:.1f}" transform="rotate(-90 20 {top+plot_height/2:.1f})" text-anchor="middle" font-family="sans-serif" font-size="14">Success rate (%)</text>',
        ]
    )
    for series_index, (run, suite) in enumerate(series):
        color = colors[series_index % len(colors)]
        points = sorted(
            (task_id, plot_values[(run, suite, task_id)])
            for group_run, group_suite, task_id in plot_values
            if group_run == run and group_suite == suite
        )
        coordinates = []
        for task_id, values in points:
            rate = 100 * success_count(values) / len(values)
            coordinates.append(f"{x(task_id):.1f},{y(rate):.1f}")
        lines.append(f'<polyline points="{" ".join(coordinates)}" fill="none" stroke="{color}" stroke-width="2"/>')
        for coordinate in coordinates:
            cx, cy = coordinate.split(",")
            lines.append(f'<circle cx="{cx}" cy="{cy}" r="4" fill="{color}"/>')
        legend_y = height - bottom + 42 + series_index * 20
        lines.append(f'<line x1="{left}" y1="{legend_y}" x2="{left+22}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>')
        lines.append(f'<text x="{left+28}" y="{legend_y+5}" font-family="sans-serif" font-size="12">{html.escape(series_label(run, suite))}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    files = discover(args.inputs)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    task_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    suite_values: dict[tuple[str, str, str], list[bool]] = defaultdict(list)
    plot_values: dict[tuple[str, str, int], list[bool]] = defaultdict(list)

    for path in files:
        with path.open(encoding="utf-8") as handle:
            report = json.load(handle)
        seed = infer_seed(path)
        run = infer_run(path)
        for task in report.get("per_task", []):
            suite = str(task["task_group"])
            task_id = int(task["task_id"])
            metrics = task["metrics"]
            successes = [bool(value) for value in metrics.get("successes", [])]
            videos = metrics.get("video_paths", [])
            succeeded = success_count(successes)
            task_rows.append(
                {
                    "run": run,
                    "seed": seed,
                    "suite": suite,
                    "task_id": task_id,
                    "successes": succeeded,
                    "episodes": len(successes),
                    "success_rate_pct": round(100 * succeeded / len(successes), 4) if successes else 0,
                    "source": str(path),
                }
            )
            suite_values[(run, seed, suite)].extend(successes)
            if args.plot_suite is None or suite == args.plot_suite:
                plot_values[(run, suite, task_id)].extend(successes)
            for episode, succeeded_flag in enumerate(successes):
                if succeeded_flag:
                    continue
                video = str(videos[episode]) if episode < len(videos) else ""
                failure_rows.append(
                    {
                        "run": run,
                        "seed": seed,
                        "suite": suite,
                        "task_id": task_id,
                        "episode": episode,
                        "video_path": video,
                        "video_exists": Path(video).is_file() if video else False,
                        "source": str(path),
                    }
                )

    summary_rows: list[dict[str, Any]] = []
    overall_by_run: dict[tuple[str, str], list[bool]] = defaultdict(list)
    for (run, seed, suite), values in sorted(suite_values.items()):
        succeeded = success_count(values)
        overall_by_run[(run, seed)].extend(values)
        summary_rows.append(
            {
                "run": run,
                "seed": seed,
                "suite": suite,
                "successes": succeeded,
                "episodes": len(values),
                "success_rate_pct": round(100 * succeeded / len(values), 4),
                "seed_mean_pct": "",
                "seed_std_pct": "",
            }
        )
    for (run, seed), values in sorted(overall_by_run.items()):
        succeeded = success_count(values)
        summary_rows.append(
            {
                "run": run,
                "seed": seed,
                "suite": "OVERALL",
                "successes": succeeded,
                "episodes": len(values),
                "success_rate_pct": round(100 * succeeded / len(values), 4),
                "seed_mean_pct": "",
                "seed_std_pct": "",
            }
        )

    rates_by_run_suite: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    for (run, _seed, suite), values in suite_values.items():
        rates_by_run_suite[(run, suite)].append((success_count(values), len(values)))
    for (run, _seed), values in overall_by_run.items():
        rates_by_run_suite[(run, "OVERALL")].append((success_count(values), len(values)))
    for (run, suite), counts in sorted(rates_by_run_suite.items()):
        if len(counts) < 2:
            continue
        successes = sum(item[0] for item in counts)
        episodes = sum(item[1] for item in counts)
        seed_rates = [100 * item[0] / item[1] for item in counts]
        summary_rows.append(
            {
                "run": run,
                "seed": "ALL",
                "suite": suite,
                "successes": successes,
                "episodes": episodes,
                "success_rate_pct": round(100 * successes / episodes, 4),
                "seed_mean_pct": round(statistics.mean(seed_rates), 4),
                "seed_std_pct": round(statistics.pstdev(seed_rates), 4),
            }
        )

    write_csv(
        args.output_dir / "summary.csv",
        [
            "run",
            "seed",
            "suite",
            "successes",
            "episodes",
            "success_rate_pct",
            "seed_mean_pct",
            "seed_std_pct",
        ],
        summary_rows,
    )
    write_csv(
        args.output_dir / "per_task.csv",
        ["run", "seed", "suite", "task_id", "successes", "episodes", "success_rate_pct", "source"],
        sorted(task_rows, key=lambda row: (row["suite"], row["task_id"], row["seed"])),
    )
    write_csv(
        args.output_dir / "failures.csv",
        ["run", "seed", "suite", "task_id", "episode", "video_path", "video_exists", "source"],
        failure_rows,
    )

    if not plot_values:
        raise ValueError(f"no plotting data matched --plot-suite={args.plot_suite!r}")

    write_success_curve_svg(args.output_dir / "success_curve.svg", plot_values)
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        pass
    else:
        figure, axis = plt.subplots(figsize=(9, 5))
        series = sorted({(run, suite) for run, suite, _ in plot_values})
        runs = {run for run, _ in series}
        suites = {suite for _, suite in series}
        for run, suite in series:
            points = sorted(
                (task_id, plot_values[(run, suite, task_id)])
                for group_run, group_suite, task_id in plot_values
                if group_run == run and group_suite == suite
            )
            label = run if len(runs) > 1 and len(suites) == 1 else suite
            if len(runs) > 1 and len(suites) > 1:
                label = f"{run} / {suite}"
            axis.plot(
                [task_id for task_id, _ in points],
                [100 * success_count(values) / len(values) for _, values in points],
                marker="o",
                label=label,
            )
        axis.set(xlabel="Task ID", ylabel="Success rate (%)", title="LIBERO per-task success rate")
        axis.set_ylim(0, 105)
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(args.output_dir / "success_curve.png", dpi=180)
        plt.close(figure)

    print(f"analyzed {len(files)} eval file(s)")
    print(f"wrote report to {args.output_dir}")


if __name__ == "__main__":
    main()
