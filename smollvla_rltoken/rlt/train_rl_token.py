"""Stage 1: train the RL token encoder/decoder on demonstration data (Eq. 2).

Follows Algorithm 1 lines 1-3: with the VLA frozen (alpha = 0 by default),
minimize the autoregressive reconstruction loss L_ro on task demos; with
--vla-sft-alpha > 0 the VLA is jointly fine-tuned with its flow-matching loss.

Example (LIBERO-10 demos, frozen VLA):
  python -m rlt.train_rl_token \
    --checkpoint smolvla_base \
    --dataset lerobot/libero_10 --dataset-root ~/pi05_reproduce/data/lerobot/libero_10 \
    --steps 5000 --batch-size 16 --out outputs/rl_token
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .configs import RLTokenConfig
from .rl_token import RLTokenModule, SmolVLAPrefixExtractor
from .smolvla_compat import load_smolvla_policy


SUITE_TASK_RANGES = {
    "libero_goal": range(0, 10),
    "libero_10": range(10, 20),
    "libero_object": range(20, 30),
    "libero_spatial": range(30, 40),
}


def _parse_task_ids(task_ids: str | None) -> set[int] | None:
    if task_ids is None or task_ids.strip() == "":
        return None
    ids = set()
    for raw in task_ids.split(","):
        raw = raw.strip()
        if not raw:
            continue
        ids.add(int(raw))
    return ids


def _select_suite_episodes(
    dataset_root: str | None,
    suite: str | None,
    task_ids: str | None = None,
) -> tuple[list[int] | None, list[int] | None]:
    explicit_task_ids = _parse_task_ids(task_ids)
    if (suite is None or suite == "all") and explicit_task_ids is None:
        return None, None
    if suite not in SUITE_TASK_RANGES:
        if suite != "all":
            raise ValueError(f"unknown LIBERO suite {suite!r}; expected one of {sorted(SUITE_TASK_RANGES)}")
    if dataset_root is None:
        raise ValueError("--dataset-root is required when --dataset-suite is not 'all' or --dataset-task-ids is set")

    import pyarrow.parquet as pq

    root = Path(dataset_root)
    tasks_path = root / "meta/tasks.parquet"
    episodes_path = root / "meta/episodes/chunk-000/file-000.parquet"
    if not tasks_path.is_file() or not episodes_path.is_file():
        raise FileNotFoundError(f"invalid LeRobot LIBERO dataset root: {root}")

    task_rows = pq.read_table(tasks_path).to_pylist()
    name_column = "__index_level_0__"
    selected_indices = set(range(40)) if suite == "all" else set(SUITE_TASK_RANGES[suite])
    if explicit_task_ids is not None:
        selected_indices &= explicit_task_ids
    if not selected_indices:
        raise ValueError(f"empty task selection for suite={suite!r}, task_ids={task_ids!r}")
    selected_task_names = {
        row[name_column]
        for row in task_rows
        if int(row["task_index"]) in selected_indices
    }
    episode_rows = pq.read_table(episodes_path, columns=["episode_index", "tasks"]).to_pylist()
    episodes = [
        int(row["episode_index"])
        for row in episode_rows
        if row["tasks"] and set(row["tasks"]).issubset(selected_task_names)
    ]
    if not episodes:
        raise ValueError(f"no episodes matched suite={suite!r}, task_ids={task_ids!r} under {root}")
    return episodes, sorted(selected_indices)


def _split_episodes(
    episodes: list[int] | None,
    val_ratio: float,
    seed: int,
    max_train_episodes: int | None = None,
) -> tuple[list[int] | None, list[int] | None]:
    if episodes is None:
        return None, None
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(episodes), generator=gen).tolist()
    shuffled = [episodes[i] for i in perm]
    n_val = int(round(len(shuffled) * val_ratio))
    if val_ratio > 0 and n_val == 0 and len(shuffled) > 1:
        n_val = 1
    val_eps = sorted(shuffled[:n_val]) if n_val else []
    train_eps = sorted(shuffled[n_val:])
    if max_train_episodes is not None:
        train_eps = train_eps[:max_train_episodes]
    return train_eps, val_eps or None


def build_dataset_and_processors(
    policy_config,
    dataset_repo: str,
    dataset_root: str | None,
    train_episodes: list[int] | None = None,
    val_episodes: list[int] | None = None,
):
    from lerobot.configs.types import FeatureType
    from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
    try:
        from lerobot.utils.feature_utils import dataset_to_policy_features
    except ImportError:
        from lerobot.datasets.feature_utils import dataset_to_policy_features
    from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors

    meta = LeRobotDatasetMetadata(dataset_repo, root=dataset_root)
    features = dataset_to_policy_features(meta.features)
    output_features = {k: f for k, f in features.items() if f.type is FeatureType.ACTION}
    input_features = {k: f for k, f in features.items() if k not in output_features}
    policy_config.input_features = input_features
    policy_config.output_features = output_features
    policy_config.validate_features()

    preprocessor, postprocessor = make_smolvla_pre_post_processors(
        policy_config, dataset_stats=meta.stats
    )

    delta_timestamps = {k: [0.0] for k in policy_config.input_features if k.startswith("observation.")}
    for k in policy_config.output_features:
        if k.startswith("action"):
            delta_timestamps[k] = [i / meta.fps for i in policy_config.action_delta_indices]

    train_dataset = LeRobotDataset(
        dataset_repo,
        root=dataset_root,
        episodes=train_episodes,
        delta_timestamps=delta_timestamps,
        video_backend="pyav",
    )
    val_dataset = None
    if val_episodes:
        val_dataset = LeRobotDataset(
            dataset_repo,
            root=dataset_root,
            episodes=val_episodes,
            delta_timestamps=delta_timestamps,
            video_backend="pyav",
        )
    return train_dataset, val_dataset, preprocessor, postprocessor


@torch.no_grad()
def evaluate_reconstruction(loader, preprocessor, extractor, rl_token, cfg, max_batches: int) -> float | None:
    if loader is None:
        return None
    rl_token.eval()
    losses = []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = preprocessor(batch)
        feats = extractor.extract(batch)
        z, mask = extractor.select_tokens(feats, cfg.use_image_tokens_only)
        loss_ro, _ = rl_token.reconstruction_loss(z, mask)
        losses.append(loss_ro.item())
    rl_token.train()
    if not losses:
        return None
    return sum(losses) / len(losses)


def train(args):
    device = args.device
    torch.manual_seed(args.seed)
    cfg = RLTokenConfig(
        d_model=args.d_model,
        n_encoder_layers=args.n_layers,
        n_decoder_layers=args.n_layers,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        vla_sft_alpha=args.vla_sft_alpha,
    )

    print(f"[stage1] loading SmolVLA from {args.checkpoint} ...")
    policy = load_smolvla_policy(args.checkpoint, device=device, dtype=args.dtype)
    selected_episodes, selected_task_indices = _select_suite_episodes(
        args.dataset_root, args.dataset_suite, args.dataset_task_ids
    )
    train_episodes, val_episodes = _split_episodes(
        selected_episodes,
        val_ratio=args.val_ratio,
        seed=args.seed,
        max_train_episodes=args.max_train_episodes,
    )
    dataset, val_dataset, preprocessor, _ = build_dataset_and_processors(
        policy.config,
        args.dataset,
        args.dataset_root,
        train_episodes=train_episodes,
        val_episodes=val_episodes,
    )
    print(
        f"[stage1] dataset={args.dataset} root={args.dataset_root} suite={args.dataset_suite} "
        f"task_indices={selected_task_indices if selected_task_indices is not None else 'all'} "
        f"train_episodes={dataset.num_episodes} train_frames={dataset.num_frames} "
        f"val_episodes={0 if val_dataset is None else val_dataset.num_episodes}",
        flush=True,
    )
    extractor = SmolVLAPrefixExtractor(policy)

    finetune_vla = cfg.vla_sft_alpha > 0
    policy.requires_grad_(finetune_vla)
    policy.train(finetune_vla)

    rl_token = RLTokenModule(cfg).to(device)
    n_params = sum(p.numel() for p in rl_token.parameters())
    print(f"[stage1] RL token module: {n_params / 1e6:.1f}M params, d_model={cfg.d_model}")

    params = list(rl_token.parameters())
    if finetune_vla:
        params += [p for p in policy.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.startswith("cuda"),
        drop_last=True,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.startswith("cuda"),
            drop_last=False,
        )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:
        raise ImportError(
            "TensorBoard logging requires tensorboard. Install it in the LIBERO-smolvla env "
            "or run with an environment that provides torch.utils.tensorboard."
        ) from exc
    tb_dir = out_dir / "tensorboard"
    writer = SummaryWriter(log_dir=str(tb_dir))

    step, t0 = 0, time.time()
    data_iter = iter(loader)
    try:
        while step < cfg.steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)
            batch = preprocessor(batch)

            # L_ro treats VLA embeddings with stop-gradient (Eq. 2), so extraction
            # never needs the VLA graph even when alpha > 0; the SFT gradient path
            # goes through policy.forward below.
            feats = extractor.extract(batch)
            z, mask = extractor.select_tokens(feats, cfg.use_image_tokens_only)
            loss_ro, _ = rl_token.reconstruction_loss(z, mask)
            with torch.no_grad():
                z_rl_for_log = rl_token.rl_token(z, mask)
                z_rl_norm = z_rl_for_log.norm(dim=-1).mean().item()

            loss = loss_ro
            log = {"loss_ro": loss_ro.item(), "z_rl_norm": z_rl_norm}
            if finetune_vla:
                loss_vla, _ = policy.forward(batch)
                loss = loss + cfg.vla_sft_alpha * loss_vla
                log["loss_vla"] = loss_vla.item()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip_norm)
            opt.step()
            log["grad_norm"] = float(grad_norm)
            if device.startswith("cuda") and torch.cuda.is_available():
                log["gpu_mem_mb"] = torch.cuda.max_memory_allocated() / (1024**2)
            step += 1

            if step % args.log_freq == 0 or step == 1:
                if val_loader is not None and (step % args.val_freq == 0 or step == 1):
                    val_loss = evaluate_reconstruction(
                        val_loader, preprocessor, extractor, rl_token, cfg, args.val_batches
                    )
                    if val_loss is not None:
                        log["val_loss_ro"] = val_loss
                elapsed_s = time.time() - t0
                speed = step / elapsed_s
                log["step"] = step
                log["elapsed_s"] = elapsed_s
                log["it_per_sec"] = speed
                with metrics_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(log, sort_keys=True) + "\n")

                writer.add_scalar("train/loss_ro", log["loss_ro"], step)
                writer.add_scalar("train/z_rl_norm", log["z_rl_norm"], step)
                writer.add_scalar("train/grad_norm", log["grad_norm"], step)
                writer.add_scalar("system/it_per_sec", speed, step)
                writer.add_scalar("system/elapsed_s", elapsed_s, step)
                if "gpu_mem_mb" in log:
                    writer.add_scalar("system/gpu_mem_mb", log["gpu_mem_mb"], step)
                if "loss_vla" in log:
                    writer.add_scalar("train/loss_vla", log["loss_vla"], step)
                if "val_loss_ro" in log:
                    writer.add_scalar("val/loss_ro", log["val_loss_ro"], step)
                writer.flush()

                msg = " ".join(f"{k}={v:.5f}" for k, v in log.items())
                print(f"[stage1] step {step}/{cfg.steps} {msg} ({speed:.2f} it/s)", flush=True)

            if step % args.save_freq == 0 or step == cfg.steps:
                ckpt = {
                    "rl_token": rl_token.state_dict(),
                    "config": vars(cfg),
                    "step": step,
                    "dataset": {
                        "repo": args.dataset,
                        "root": args.dataset_root,
                        "suite": args.dataset_suite,
                        "task_indices": selected_task_indices,
                        "train_episodes": train_episodes,
                        "val_episodes": val_episodes,
                    },
                }
                torch.save(ckpt, out_dir / "rl_token.pt")
                if finetune_vla:
                    torch.save(policy.state_dict(), out_dir / "smolvla_sft.pt")
    finally:
        writer.close()

    summary = {
        "steps": step,
        "elapsed_s": time.time() - t0,
        "checkpoint": args.checkpoint,
        "dataset": args.dataset,
        "dataset_root": args.dataset_root,
        "dataset_suite": args.dataset_suite,
        "dataset_task_indices": selected_task_indices,
        "train_episodes": dataset.num_episodes,
        "val_episodes": 0 if val_dataset is None else val_dataset.num_episodes,
        "out": str(out_dir),
        "tensorboard_dir": str(tb_dir),
    }
    if device.startswith("cuda") and torch.cuda.is_available():
        summary["gpu_peak_mem_mb"] = torch.cuda.max_memory_allocated() / (1024**2)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[stage1] done; saved to {out_dir/'rl_token.pt'}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="smolvla_base")
    p.add_argument("--dataset", default="lerobot/libero_10")
    p.add_argument("--dataset-root", default=None)
    p.add_argument("--dataset-suite", default="all", choices=["all", *sorted(SUITE_TASK_RANGES)])
    p.add_argument(
        "--dataset-task-ids",
        default=None,
        help="Optional comma-separated global LIBERO task_index values from meta/tasks.parquet.",
    )
    p.add_argument("--out", default="outputs/rl_token")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default=None, choices=[None, "float32", "bfloat16"])
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--vla-sft-alpha", type=float, default=0.0)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--val-freq", type=int, default=100)
    p.add_argument("--val-batches", type=int, default=4)
    p.add_argument("--max-train-episodes", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-freq", type=int, default=20)
    p.add_argument("--save-freq", type=int, default=500)
    train(p.parse_args())


if __name__ == "__main__":
    main()
