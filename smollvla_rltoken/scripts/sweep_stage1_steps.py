"""Sweep Stage 1 RL-token reconstruction steps and plot final losses."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path


def parse_steps(raw: str) -> list[int]:
    steps = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not steps:
        raise ValueError("--steps-list must contain at least one integer")
    if any(s <= 0 for s in steps):
        raise ValueError("--steps-list values must be positive")
    return sorted(set(steps))


def last_losses(metrics_path: Path) -> tuple[float | None, float | None]:
    train_loss = None
    val_loss = None
    if not metrics_path.is_file():
        return train_loss, val_loss
    with metrics_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if "loss_ro" in row:
                train_loss = float(row["loss_ro"])
            if "val_loss_ro" in row:
                val_loss = float(row["val_loss_ro"])
    return train_loss, val_loss


def run_one(args: argparse.Namespace, steps: int, out_dir: Path) -> None:
    cmd = [
        sys.executable,
        "-m",
        "rlt.train_rl_token",
        "--checkpoint",
        args.checkpoint,
        "--dataset",
        args.dataset,
        "--dataset-root",
        args.dataset_root,
        "--dataset-suite",
        args.dataset_suite,
        "--device",
        args.device,
        "--steps",
        str(steps),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--vla-sft-alpha",
        str(args.vla_sft_alpha),
        "--val-ratio",
        str(args.val_ratio),
        "--val-freq",
        str(args.val_freq),
        "--val-batches",
        str(args.val_batches),
        "--log-freq",
        str(args.log_freq),
        "--save-freq",
        str(args.save_freq),
        "--seed",
        str(args.seed),
        "--out",
        str(out_dir),
    ]
    if args.dtype:
        cmd.extend(["--dtype", args.dtype])
    if args.max_train_episodes is not None:
        cmd.extend(["--max-train-episodes", str(args.max_train_episodes)])
    if args.dataset_task_ids:
        cmd.extend(["--dataset-task-ids", args.dataset_task_ids])

    print(f"[sweep] running steps={steps} out={out_dir}", flush=True)
    subprocess.run(cmd, check=True, env=os.environ.copy())


def write_plot(csv_path: Path, png_path: Path) -> None:
    import matplotlib.pyplot as plt

    rows = []
    with csv_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    if not rows:
        return

    x = [int(r["steps"]) for r in rows]
    train = [float(r["final_train_loss_ro"]) for r in rows]
    val = [float(r["final_val_loss_ro"]) for r in rows]

    plt.figure(figsize=(7, 4.5))
    plt.plot(x, train, marker="o", label="final train loss_ro")
    plt.plot(x, val, marker="o", label="final val_loss_ro")
    plt.xlabel("Stage 1 training steps")
    plt.ylabel("reconstruction loss")
    plt.title("RL Token Stage 1 steps sweep")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(png_path, dpi=160)
    plt.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="/home/evoild/program/smolVLA_mujoco_LIBERO/smolvla_libero")
    p.add_argument("--dataset", default="HuggingFaceVLA/libero")
    p.add_argument("--dataset-root", default="/home/evoild/program/smolVLA_mujoco_LIBERO/libero")
    p.add_argument("--dataset-suite", default="libero_goal")
    p.add_argument("--dataset-task-ids", default=None)
    p.add_argument("--steps-list", default="500,1000,2000,3000,5000")
    p.add_argument("--out-root", default="/home/evoild/program/smolVLA_mujoco_LIBERO/outputs/rlt_libero/stage1_goal_steps_sweep")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default=None, choices=[None, "float32", "bfloat16"])
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--vla-sft-alpha", type=float, default=0.0)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--val-freq", type=int, default=100)
    p.add_argument("--val-batches", type=int, default=4)
    p.add_argument("--log-freq", type=int, default=20)
    p.add_argument("--save-freq", type=int, default=500)
    p.add_argument("--max-train-episodes", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for steps in parse_steps(args.steps_list):
        out_dir = out_root / f"steps_{steps}"
        summary_path = out_dir / "summary.json"
        metrics_path = out_dir / "metrics.jsonl"
        if args.force or not summary_path.is_file() or not metrics_path.is_file():
            run_one(args, steps, out_dir)
        else:
            print(f"[sweep] reuse existing steps={steps} out={out_dir}", flush=True)

        train_loss, val_loss = last_losses(metrics_path)
        if train_loss is None or val_loss is None:
            raise RuntimeError(f"missing final train/val losses in {metrics_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "steps": steps,
                "final_train_loss_ro": train_loss,
                "final_val_loss_ro": val_loss,
                "elapsed_s": float(summary.get("elapsed_s", 0.0)),
                "gpu_peak_mem_mb": float(summary.get("gpu_peak_mem_mb", 0.0)),
                "out": str(out_dir),
            }
        )

    csv_path = out_root / "sweep_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "steps",
                "final_train_loss_ro",
                "final_val_loss_ro",
                "elapsed_s",
                "gpu_peak_mem_mb",
                "out",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    json_path = out_root / "sweep_summary.json"
    json_path.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    png_path = out_root / "loss_vs_steps.png"
    write_plot(csv_path, png_path)

    print(f"[sweep] wrote {csv_path}", flush=True)
    print(f"[sweep] wrote {json_path}", flush=True)
    print(f"[sweep] wrote {png_path}", flush=True)


if __name__ == "__main__":
    main()
