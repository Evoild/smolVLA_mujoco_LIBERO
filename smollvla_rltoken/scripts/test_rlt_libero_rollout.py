"""Frozen SmolVLA + RLT inference smoke test on real LIBERO.

This script validates the rollout path only. It does not train the RL-token,
actor, or critic:

LIBERO obs -> SmolVLA prefix -> z_rl -> reference chunk -> actor chunk ->
LIBERO env.step
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from rlt.actor_critic import RLTAgent  # noqa: E402
from rlt.configs import ActorCriticConfig, OnlineRLConfig, RLTokenConfig  # noqa: E402
from rlt.libero_env import LiberoChunkEnv  # noqa: E402
from rlt.rl_token import RLTokenModule  # noqa: E402
from rlt.rlt_policy import RLTController  # noqa: E402
from rlt.smolvla_compat import load_smolvla_policy  # noqa: E402
from rlt.train_online import load_rl_token  # noqa: E402


def check(name: str, cond: bool, extra: str = "") -> None:
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name} {extra}", flush=True)
    if not cond:
        raise SystemExit(f"RLT LIBERO rollout smoke failed at: {name}")


def load_or_init_rl_token(path: str | None, device: str) -> tuple[RLTokenModule, RLTokenConfig]:
    if path:
        return load_rl_token(path, device)
    cfg = RLTokenConfig()
    module = RLTokenModule(cfg).to(device).eval().requires_grad_(False)
    print("[warn] --rl-token not provided; using a randomly initialized RLTokenModule for plumbing smoke.")
    return module, cfg


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=str(REPO.parent / "smolvla_libero"))
    p.add_argument("--rl-token", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default=None, choices=[None, "float32", "bfloat16"])
    p.add_argument("--suite", default="libero_spatial")
    p.add_argument("--task-id", type=int, default=0)
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--max-episode-steps", type=int, default=60)
    p.add_argument("--chunk-len", type=int, default=10)
    p.add_argument("--num-inference-steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--execute", choices=["actor", "reference"], default="actor")
    p.add_argument("--stochastic-actor", action="store_true")
    p.add_argument("--observation-height", type=int, default=256)
    p.add_argument("--observation-width", type=int, default=256)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    t0 = time.time()

    policy = load_smolvla_policy(args.checkpoint, device=args.device, dtype=args.dtype)
    policy.requires_grad_(False)
    rl_token, rt_cfg = load_or_init_rl_token(args.rl_token, args.device)

    action_dim = int(policy.config.action_feature.shape[0])
    state_dim = int(policy.config.input_features["observation.state"].shape[0])
    cfg = OnlineRLConfig(
        ac=ActorCriticConfig(
            chunk_len=args.chunk_len,
            action_dim=action_dim,
            proprio_dim=state_dim,
            rl_token_dim=rt_cfg.d_model,
        ),
        device=args.device,
        seed=args.seed,
    )
    agent = RLTAgent(cfg, device=args.device)
    controller = RLTController(
        policy,
        rl_token,
        agent,
        chunk_len=args.chunk_len,
        action_dim=action_dim,
        proprio_dim=state_dim,
        num_inference_steps=args.num_inference_steps,
    )
    env = LiberoChunkEnv(
        policy=policy,
        checkpoint=args.checkpoint,
        suite_name=args.suite,
        task_id=args.task_id,
        max_episode_steps=args.max_episode_steps,
        seed=args.seed,
        device=args.device,
        observation_height=args.observation_height,
        observation_width=args.observation_width,
    )

    print(
        f"[info] suite={args.suite} task_id={args.task_id} task={env.task_description!r} "
        f"execute={args.execute} action_dim={action_dim} state_dim={state_dim}",
        flush=True,
    )

    first_chunk_checked = False
    successes = []
    try:
        for ep in range(args.episodes):
            obs = env.reset()
            ep_return = 0.0
            ep_steps = 0
            ep_success = False

            while ep_steps < env.max_episode_steps:
                batch = env.obs_to_batch([obs], args.device)
                feats = controller.extractor.extract(batch)
                z, mask = controller.extractor.select_tokens(feats, controller.image_only)
                z_rl = controller.rl_token.rl_token(z, mask)
                ref_full = controller.extractor.sample_reference_chunk(
                    feats, num_steps=args.num_inference_steps
                )[:, :, :action_dim]
                ref_chunk = ref_full[:, : args.chunk_len]
                actor_chunk = agent.act(
                    torch.cat([z_rl, controller._proprio(batch)], dim=-1),
                    ref_chunk,
                    deterministic=not args.stochastic_actor,
                )

                if not first_chunk_checked:
                    check("prefix hidden shape", z.ndim == 3 and z.shape[0] == 1, f"z={tuple(z.shape)}")
                    check("prefix hidden finite", torch.isfinite(z).all().item())
                    check("z_rl shape", z_rl.shape == (1, rt_cfg.d_model), f"z_rl={tuple(z_rl.shape)}")
                    check(
                        "reference action shape",
                        ref_full.shape == (1, policy.config.chunk_size, action_dim),
                        f"ref={tuple(ref_full.shape)}",
                    )
                    check(
                        "actor action shape",
                        actor_chunk.shape == (1, args.chunk_len, action_dim),
                        f"actor={tuple(actor_chunk.shape)}",
                    )
                    check("reference action finite", torch.isfinite(ref_full).all().item())
                    check("actor action finite", torch.isfinite(actor_chunk).all().item())
                    print(
                        "[info] normalized ranges "
                        f"ref=[{ref_full.min().item():.3f}, {ref_full.max().item():.3f}] "
                        f"actor=[{actor_chunk.min().item():.3f}, {actor_chunk.max().item():.3f}]",
                        flush=True,
                    )
                    first_chunk_checked = True

                chunk = actor_chunk[0] if args.execute == "actor" else ref_chunk[0]
                for i in range(chunk.shape[0]):
                    obs, reward, done = env.step(chunk[i])
                    ep_return += reward
                    ep_steps += 1
                    if done or ep_steps >= env.max_episode_steps:
                        ep_success = bool(env.last_success)
                        break
                if done or ep_steps >= env.max_episode_steps:
                    break

            successes.append(1.0 if ep_success else 0.0)
            print(
                f"[episode] ep={ep} steps={ep_steps} return={ep_return:.1f} "
                f"success={ep_success}",
                flush=True,
            )
    finally:
        env.close()

    success_rate = sum(successes) / max(len(successes), 1)
    print(
        f"[done] episodes={len(successes)} success_rate={success_rate:.3f} "
        f"elapsed={time.time() - t0:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
