"""Plot Stage 1 train/validation reconstruction loss curves from metrics.jsonl."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def moving_average(values: list[float], window: int) -> list[float]:
    if window <= 1:
        return values
    out = []
    total = 0.0
    queue = []
    for value in values:
        queue.append(value)
        total += value
        if len(queue) > window:
            total -= queue.pop(0)
        out.append(total / len(queue))
    return out


def load_metrics(metrics_path: Path) -> tuple[list[int], list[float], list[int], list[float]]:
    train_steps = []
    train_losses = []
    val_steps = []
    val_losses = []
    with metrics_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            step = int(row["step"])
            if "loss_ro" in row:
                train_steps.append(step)
                train_losses.append(float(row["loss_ro"]))
            if "val_loss_ro" in row:
                val_steps.append(step)
                val_losses.append(float(row["val_loss_ro"]))
    if not train_steps:
        raise ValueError(f"no loss_ro records found in {metrics_path}")
    if not val_steps:
        raise ValueError(f"no val_loss_ro records found in {metrics_path}")
    return train_steps, train_losses, val_steps, val_losses


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--smooth-window", type=int, default=5)
    args = p.parse_args()

    import matplotlib.pyplot as plt

    run_dir = Path(args.run_dir)
    metrics_path = run_dir / "metrics.jsonl"
    out_path = Path(args.out) if args.out else run_dir / "loss_curve_train_val.png"
    train_steps, train_losses, val_steps, val_losses = load_metrics(metrics_path)
    smoothed_train = moving_average(train_losses, args.smooth_window)

    plt.figure(figsize=(8, 4.8))
    plt.plot(train_steps, train_losses, color="#8aa0b8", alpha=0.35, linewidth=1.0, label="train loss_ro")
    plt.plot(
        train_steps,
        smoothed_train,
        color="#1f77b4",
        linewidth=2.0,
        label=f"train loss_ro smoothed ({args.smooth_window})",
    )
    plt.plot(val_steps, val_losses, color="#d62728", marker="o", linewidth=2.0, label="val_loss_ro")
    plt.xlabel("training step")
    plt.ylabel("reconstruction loss")
    plt.title("Stage 1 RL Token Reconstruction Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()

    print(f"[plot] train points={len(train_steps)} val points={len(val_steps)}")
    print(f"[plot] first train={train_losses[0]:.6f} final train={train_losses[-1]:.6f}")
    print(f"[plot] first val={val_losses[0]:.6f} final val={val_losses[-1]:.6f}")
    print(f"[plot] wrote {out_path}")


if __name__ == "__main__":
    main()
