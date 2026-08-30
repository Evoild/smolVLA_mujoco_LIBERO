"""Evaluate frozen SmolVLA or trained RLT actor on LIBERO guardrail tasks.

This script performs fixed-policy evaluation only. It never updates the
RL-token, actor, critic, replay buffer, or SmolVLA weights.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch
from lerobot.utils.constants import OBS_STATE

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from rlt.actor_critic import RLTAgent  # noqa: E402
from rlt.configs import ActorCriticConfig, OnlineRLConfig  # noqa: E402
from rlt.libero_env import LiberoChunkEnv  # noqa: E402
from rlt.rlt_policy import RLTController  # noqa: E402
from rlt.smolvla_compat import load_smolvla_policy  # noqa: E402
from rlt.train_online import load_rl_token  # noqa: E402


def parse_ids(raw: str) -> list[int]:
    ids = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not ids:
        raise ValueError("expected at least one task id")
    return ids


def load_agent(path: str | None, cfg: OnlineRLConfig, device: str) -> RLTAgent | None:
    if path is None:
        return None
    agent = RLTAgent(cfg, device=device)
    state = torch.load(path, map_location=device, weights_only=False)
    agent.load_state_dict(state)
    agent.actor.eval()
    agent.critic.eval()
    return agent


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def maybe_write_video(path: Path, frames: list, fps: int) -> str:
    if not frames:
        return ""
    try:
        import imageio.v2 as imageio
    except ImportError:
        return ""
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, fps=fps)
    return str(path)


@torch.no_grad()
def run_episode(
    env: LiberoChunkEnv,
    controller: RLTController,
    use_actor: bool,
    device: str,
    save_video: bool,
) -> tuple[int, float, bool, float, float, list]:
    obs = env.reset()
    frames = []
    if save_video:
        frames.append(env.env.render())

    ep_steps = 0
    ep_return = 0.0
    deltas = []
    while ep_steps < env.max_episode_steps:
        batch = env.obs_to_batch([obs], device)
        plan = controller.plan_chunk(batch, use_actor=use_actor, deterministic=True)
        chunk = plan["action_chunk"][0]
        deltas.append(float(plan["actor_ref_l2"].detach().cpu()))

        for i in range(chunk.shape[0]):
            obs, reward, done = env.step(chunk[i])
            ep_return += reward
            ep_steps += 1
            if save_video:
                frames.append(env.env.render())
            if done or ep_steps >= env.max_episode_steps:
                return (
                    ep_steps,
                    ep_return,
                    bool(env.last_success),
                    sum(deltas) / max(len(deltas), 1),
                    max(deltas) if deltas else 0.0,
                    frames,
                )

    return (
        ep_steps,
        ep_return,
        bool(env.last_success),
        sum(deltas) / max(len(deltas), 1),
        max(deltas) if deltas else 0.0,
        frames,
    )


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=str(REPO.parent / "smolvla_libero"))
    p.add_argument("--rl-token", required=True)
    p.add_argument("--agent", default=None, help="Trained rlt_agent.pt. Omit for frozen SmolVLA reference eval.")
    p.add_argument("--suite", default="libero_goal")
    p.add_argument("--task-ids", default="0,1,2,3,4,5,6,7,8,9")
    p.add_argument("--episodes-per-task", type=int, default=10)
    p.add_argument("--max-episode-steps", type=int, default=400)
    p.add_argument("--chunk-len", type=int, default=10)
    p.add_argument("--num-inference-steps", type=int, default=2)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default=None, choices=[None, "float32", "bfloat16"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--save-videos", action="store_true")
    p.add_argument("--video-fps", type=int, default=20)
    p.add_argument("--observation-height", type=int, default=360)
    p.add_argument("--observation-width", type=int, default=360)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--action-std", type=float, default=0.05)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    task_ids = parse_ids(args.task_ids)
    mode = "rlt_actor" if args.agent else "frozen_reference"
    t0 = time.time()

    print(f"[eval] loading SmolVLA from {args.checkpoint} ...", flush=True)
    policy = load_smolvla_policy(args.checkpoint, device=args.device, dtype=args.dtype)
    policy.requires_grad_(False)
    rl_token, rt_cfg = load_rl_token(args.rl_token, args.device)

    action_dim = int(policy.config.action_feature.shape[0])
    state_dim = int(policy.config.input_features[OBS_STATE].shape[0])
    cfg = OnlineRLConfig(
        ac=ActorCriticConfig(
            chunk_len=args.chunk_len,
            action_dim=action_dim,
            proprio_dim=state_dim,
            rl_token_dim=rt_cfg.d_model,
            hidden_dim=args.hidden_dim,
            n_layers=args.n_layers,
            action_std=args.action_std,
        ),
        device=args.device,
        seed=args.seed,
    )
    agent = load_agent(args.agent, cfg, args.device)
    controller = RLTController(
        policy,
        rl_token,
        agent,
        chunk_len=args.chunk_len,
        action_dim=action_dim,
        proprio_dim=state_dim,
        num_inference_steps=args.num_inference_steps,
    )

    episode_rows: list[dict] = []
    per_task_rows: list[dict] = []
    try:
        for task_id in task_ids:
            env = LiberoChunkEnv(
                policy=policy,
                checkpoint=args.checkpoint,
                suite_name=args.suite,
                task_id=task_id,
                max_episode_steps=args.max_episode_steps,
                seed=args.seed,
                device=args.device,
                observation_height=args.observation_height,
                observation_width=args.observation_width,
            )
            successes = []
            try:
                print(f"[eval] task={task_id} desc={env.task_description!r} mode={mode}", flush=True)
                for ep in range(args.episodes_per_task):
                    steps, ret, success, delta_mean, delta_max, frames = run_episode(
                        env,
                        controller,
                        use_actor=agent is not None,
                        device=args.device,
                        save_video=args.save_videos,
                    )
                    successes.append(1.0 if success else 0.0)
                    video_path = ""
                    if args.save_videos:
                        video_path = maybe_write_video(
                            out_dir / "videos" / f"{args.suite}_{task_id}" / f"eval_episode_{ep}.mp4",
                            frames,
                            fps=args.video_fps,
                        )
                    row = {
                        "suite": args.suite,
                        "task_id": task_id,
                        "task": env.task_description,
                        "episode": ep,
                        "success": int(success),
                        "return": ret,
                        "steps": steps,
                        "mode": mode,
                        "actor_ref_l2_mean": delta_mean,
                        "actor_ref_l2_max": delta_max,
                        "video_path": video_path,
                    }
                    episode_rows.append(row)
                    print(
                        f"[episode] task={task_id} ep={ep} success={success} "
                        f"steps={steps} sr={sum(successes) / len(successes):.3f}",
                        flush=True,
                    )
            finally:
                env.close()

            per_task_rows.append(
                {
                    "suite": args.suite,
                    "task_id": task_id,
                    "task": episode_rows[-1]["task"] if episode_rows else "",
                    "successes": int(sum(successes)),
                    "episodes": len(successes),
                    "success_rate_pct": 100.0 * sum(successes) / max(len(successes), 1),
                    "mode": mode,
                }
            )

    finally:
        write_csv(
            out_dir / "episodes.csv",
            episode_rows,
            [
                "suite",
                "task_id",
                "task",
                "episode",
                "success",
                "return",
                "steps",
                "mode",
                "actor_ref_l2_mean",
                "actor_ref_l2_max",
                "video_path",
            ],
        )
        write_csv(
            out_dir / "per_task.csv",
            per_task_rows,
            ["suite", "task_id", "task", "successes", "episodes", "success_rate_pct", "mode"],
        )
        total_successes = sum(row["successes"] for row in per_task_rows)
        total_episodes = sum(row["episodes"] for row in per_task_rows)
        summary = {
            "mode": mode,
            "checkpoint": args.checkpoint,
            "rl_token": args.rl_token,
            "agent": args.agent,
            "suite": args.suite,
            "task_ids": task_ids,
            "episodes_per_task": args.episodes_per_task,
            "successes": int(total_successes),
            "episodes": int(total_episodes),
            "success_rate_pct": 100.0 * total_successes / max(total_episodes, 1),
            "out": str(out_dir),
            "elapsed_s": time.time() - t0,
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(
        f"[done] mode={mode} successes={total_successes}/{total_episodes} "
        f"success_rate={summary['success_rate_pct']:.1f}% out={out_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
