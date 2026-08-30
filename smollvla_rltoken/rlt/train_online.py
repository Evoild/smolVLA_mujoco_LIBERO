"""Stage 2: online RL with the RL token (Algorithm 1).

Rollout-and-update loop:
  * warmup: execute the base VLA reference chunks for N_warm env steps,
  * afterwards the actor refines the VLA chunks; every executed chunk is
    stored with stride-2 subsampling, and `utd * (C / stride)` gradient steps
    are performed per chunk (2 critic updates per actor update),
  * optional human interventions replace both the executed action and the
    stored reference (paper Sec. V).

Example (mock env smoke run):
  python -m rlt.train_online --checkpoint smolvla_base \
    --rl-token outputs/rl_token/rl_token.pt --mock-env \
    --total-env-steps 2000 --warmup-env-steps 400
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from lerobot.utils.constants import OBS_STATE

from .actor_critic import RLTAgent
from .configs import ActorCriticConfig, OnlineRLConfig, RLTokenConfig
from .envs import MockManipEnv
from .libero_env import LiberoChunkEnv
from .replay_buffer import ChunkRecord, ChunkReplayBuffer
from .rl_token import RLTokenModule
from .rlt_policy import RLTController
from .smolvla_compat import load_smolvla_policy


def _parse_task_ids(task_ids: str | None) -> list[int] | None:
    if task_ids is None or task_ids.strip() == "":
        return None
    parsed = []
    for raw in task_ids.split(","):
        raw = raw.strip()
        if raw:
            parsed.append(int(raw))
    if not parsed:
        return None
    return parsed


def load_rl_token(path: str | Path, device: str) -> tuple[RLTokenModule, RLTokenConfig]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = RLTokenConfig(**{
        k: v for k, v in ckpt["config"].items() if k in RLTokenConfig.__dataclass_fields__
    })
    module = RLTokenModule(cfg)
    module.load_state_dict(ckpt["rl_token"])
    module.to(device).eval().requires_grad_(False)
    return module, cfg


class RolloutWorker:
    """Executes chunks in a single env and assembles ChunkRecords."""

    def __init__(self, env, controller: RLTController, buffer: ChunkReplayBuffer, stride: int):
        self.env = env
        self.controller = controller
        self.buffer = buffer
        self.stride = stride
        self.device = next(controller.rl_token.parameters()).device
        self.obs = None
        self.ep_steps = 0
        self.ep_return = 0.0
        self.last_plan_metrics: dict[str, float] = {}

    def reset(self) -> None:
        self.obs = self.env.reset()
        self.ep_steps = 0
        self.ep_return = 0.0
        self.buffer.start_episode()

    def run_chunk(self, use_actor: bool, deterministic: bool = False):
        """Plan at the chunk boundary, execute up to C steps, store the record.

        Returns (n_steps, episode_done, episode_success).
        """
        c = self.controller.chunk_len
        batch = self.env.obs_to_batch([self.obs], self.device)
        plan = self.controller.plan_chunk(batch, use_actor=use_actor, deterministic=deterministic)
        self.last_plan_metrics = {
            "rollout_delta_l2": float(plan["delta_l2"].detach().cpu()),
            "rollout_actor_ref_l2": float(plan["actor_ref_l2"].detach().cpu()),
        }

        intervention = self.env.get_intervention()
        if intervention is not None:
            actions = intervention.to(plan["action_chunk"][0])
            ref_full = actions[-1:].repeat(plan["ref_full"].shape[1], 1).clone()
            ref_full[:c] = actions
        else:
            actions = plan["action_chunk"][0]
            ref_full = plan["ref_full"][0]

        rewards = torch.zeros(c)
        inter_obs: list[dict] = []
        success, env_done, truncated, done_step = False, False, False, None
        for j in range(c):
            self.obs, r, step_done = self.env.step(actions[j])
            step_success = bool(getattr(self.env, "last_success", step_done))
            rewards[j] = r
            self.ep_steps += 1
            self.ep_return += r
            if step_done:
                env_done = True
                success = step_success
                done_step = j + 1
                break
            if self.ep_steps >= self.env.max_episode_steps:
                truncated = True
                done_step = j + 1
                break
            if (j + 1) % self.stride == 0 and (j + 1) < c:
                inter_obs.append(self.obs)

        xs = [plan["x"][0].cpu()]
        if inter_obs:
            xb = self.env.obs_to_batch(inter_obs, self.device)
            xs.extend(self.controller.compute_x(xb).cpu())

        rec = ChunkRecord(
            xs=torch.stack(xs),
            actions=actions.detach().cpu(),
            rewards=rewards,
            ref_full=ref_full.detach().cpu(),
            done=env_done,
            done_step=done_step,
        )
        self.buffer.add_chunk(rec)
        if (truncated or env_done) and not success:
            self.buffer.end_episode()

        n_steps = done_step if done_step is not None else c
        return n_steps, env_done or truncated, success


class MultiTaskLiberoChunkEnv:
    """Cycles LIBERO single-task envs at episode boundaries."""

    def __init__(self, task_ids: list[int], **env_kwargs):
        if not task_ids:
            raise ValueError("MultiTaskLiberoChunkEnv requires at least one task id")
        self.task_ids = [int(t) for t in task_ids]
        self.env_kwargs = env_kwargs
        self._task_cursor = -1
        self._env: LiberoChunkEnv | None = None
        self.task_id = self.task_ids[0]

    def _make_env(self, task_id: int) -> LiberoChunkEnv:
        kwargs = dict(self.env_kwargs)
        kwargs["task_id"] = task_id
        return LiberoChunkEnv(**kwargs)

    def reset(self) -> dict:
        self._task_cursor = (self._task_cursor + 1) % len(self.task_ids)
        next_task_id = self.task_ids[self._task_cursor]
        if self._env is None or self.task_id != next_task_id:
            if self._env is not None:
                self._env.close()
            self._env = self._make_env(next_task_id)
            self.task_id = next_task_id
        return self._env.reset()

    def obs_to_batch(self, obs_list: list[dict], device):
        return self._require_env().obs_to_batch(obs_list, device)

    def step(self, action):
        return self._require_env().step(action)

    def get_intervention(self):
        return self._require_env().get_intervention()

    @property
    def max_episode_steps(self) -> int:
        return self._require_env().max_episode_steps

    @property
    def last_success(self) -> bool:
        return self._require_env().last_success

    @property
    def last_info(self) -> dict:
        return self._require_env().last_info

    def _require_env(self) -> LiberoChunkEnv:
        if self._env is None:
            raise RuntimeError("environment is not initialized; call reset() first")
        return self._env

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None


def train(args):
    device = args.device
    cfg = OnlineRLConfig(
        ac=ActorCriticConfig(
            chunk_len=args.chunk_len,
            action_dim=args.action_dim or ActorCriticConfig.action_dim,
            proprio_dim=args.proprio_dim or ActorCriticConfig.proprio_dim,
            hidden_dim=args.hidden_dim,
            n_layers=args.n_layers,
            action_std=args.action_std,
        ),
        bc_beta=args.bc_beta,
        utd=args.utd,
        batch_size=args.batch_size,
        warmup_env_steps=args.warmup_env_steps,
        total_env_steps=args.total_env_steps,
        max_episode_steps=args.max_episode_steps,
        device=device,
        seed=args.seed,
    )
    torch.manual_seed(cfg.seed)

    print(f"[stage2] loading SmolVLA from {args.checkpoint} ...")
    policy = load_smolvla_policy(args.checkpoint, device=device, dtype=args.dtype)
    rl_token, rt_cfg = load_rl_token(args.rl_token, device)
    cfg.ac.rl_token_dim = rt_cfg.d_model
    state_dim = policy.config.input_features[OBS_STATE].shape[0]
    cfg.ac.action_dim = args.action_dim or policy.config.action_feature.shape[0]
    cfg.ac.proprio_dim = args.proprio_dim or state_dim

    agent = RLTAgent(cfg, device=device)
    controller = RLTController(
        policy,
        rl_token,
        agent,
        chunk_len=cfg.ac.chunk_len,
        action_dim=cfg.ac.action_dim,
        proprio_dim=cfg.ac.proprio_dim,
        num_inference_steps=args.num_inference_steps,
    )

    selected_task_ids = _parse_task_ids(args.libero_task_ids)
    if args.libero_env:
        libero_env_kwargs = dict(
            policy=policy,
            checkpoint=args.checkpoint,
            suite_name=args.libero_suite,
            task_id=args.libero_task_id,
            max_episode_steps=cfg.max_episode_steps,
            seed=cfg.seed,
            device=device,
            init_states=not args.libero_no_init_states,
            control_mode=args.libero_control_mode,
            observation_height=args.libero_observation_height,
            observation_width=args.libero_observation_width,
        )
        if selected_task_ids is not None:
            libero_env_kwargs.pop("task_id")
            env = MultiTaskLiberoChunkEnv(selected_task_ids, **libero_env_kwargs)
        else:
            env = LiberoChunkEnv(**libero_env_kwargs)
    elif args.mock_env:
        env = MockManipEnv(
            action_dim=cfg.ac.action_dim,
            state_dim=state_dim,
            camera_keys=tuple(policy.config.image_features),
            max_episode_steps=cfg.max_episode_steps,
            seed=cfg.seed,
        )
    else:
        raise ValueError("Select one environment backend: --mock-env or --libero-env")

    x_dim = cfg.ac.rl_token_dim + cfg.ac.proprio_dim
    buffer = ChunkReplayBuffer(
        capacity=cfg.buffer_capacity,
        x_dim=x_dim,
        chunk_len=cfg.ac.chunk_len,
        action_dim=cfg.ac.action_dim,
        discount=cfg.discount,
        stride=cfg.subsample_stride,
        device=device,
    )
    worker = RolloutWorker(env, controller, buffer, stride=cfg.subsample_stride)

    updates_per_chunk = cfg.utd * (cfg.ac.chunk_len // cfg.subsample_stride)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"

    env_steps, episodes, ep_results = 0, 0, []
    worker.reset()
    t0 = time.time()
    metrics: dict[str, float] = {}

    while env_steps < cfg.total_env_steps:
        warmup = env_steps < cfg.warmup_env_steps
        prev_steps = env_steps
        n_steps, ep_done, ep_success = worker.run_chunk(use_actor=not warmup)
        env_steps += n_steps

        if ep_done:
            episodes += 1
            ep_results.append(1.0 if ep_success else 0.0)
            ep_log = {
                "event": "episode",
                "episode": episodes,
                "env_steps": env_steps,
                "task_id": int(getattr(env, "task_id", args.libero_task_id)),
                "return": float(worker.ep_return),
                "success": bool(ep_success),
                "buffer_size": len(buffer),
                **worker.last_plan_metrics,
            }
            with metrics_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(ep_log, sort_keys=True) + "\n")
            worker.reset()

        if not warmup and len(buffer) >= cfg.batch_size:
            for _ in range(updates_per_chunk):
                metrics = agent.update(lambda: buffer.sample(cfg.batch_size))

        if env_steps // args.log_freq != prev_steps // args.log_freq:
            recent = ep_results[-20:]
            sr = sum(recent) / max(len(recent), 1)
            speed = env_steps / (time.time() - t0)
            m = " ".join(f"{k}={v:.4f}" for k, v in metrics.items())
            print(
                f"[stage2] steps={env_steps} eps={episodes} buffer={len(buffer)} "
                f"task={getattr(env, 'task_id', args.libero_task_id)} "
                f"success20={sr:.2f} {m} ({speed:.1f} steps/s)",
                flush=True,
            )
            step_log = {
                "event": "train",
                "env_steps": env_steps,
                "episodes": episodes,
                "task_id": int(getattr(env, "task_id", args.libero_task_id)),
                "buffer_size": len(buffer),
                "success20": sr,
                "steps_per_sec": speed,
                **worker.last_plan_metrics,
                **metrics,
            }
            with metrics_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(step_log, sort_keys=True) + "\n")

        if env_steps // args.save_freq != prev_steps // args.save_freq:
            torch.save(agent.state_dict(), out_dir / "rlt_agent.pt")

    torch.save(agent.state_dict(), out_dir / "rlt_agent.pt")
    summary = {
        "checkpoint": args.checkpoint,
        "rl_token": args.rl_token,
        "out": str(out_dir),
        "env_steps": env_steps,
        "episodes": episodes,
        "success_rate": sum(ep_results) / max(len(ep_results), 1),
        "libero_suite": args.libero_suite,
        "libero_task_ids": selected_task_ids or [args.libero_task_id],
        "warmup_env_steps": cfg.warmup_env_steps,
        "total_env_steps": cfg.total_env_steps,
        "elapsed_s": time.time() - t0,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[stage2] done; {episodes} episodes, saved to {out_dir/'rlt_agent.pt'}")
    if hasattr(env, "close"):
        env.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="smolvla_base")
    p.add_argument("--rl-token", default="outputs/rl_token/rl_token.pt")
    p.add_argument("--out", default="outputs/rlt_online")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default=None, choices=[None, "float32", "bfloat16"])
    p.add_argument("--mock-env", action="store_true")
    p.add_argument("--libero-env", action="store_true")
    p.add_argument("--libero-suite", default="libero_spatial")
    p.add_argument("--libero-task-id", type=int, default=0)
    p.add_argument("--libero-task-ids", default=None, help="Comma-separated LIBERO task ids cycled per episode.")
    p.add_argument("--libero-control-mode", default="relative", choices=["relative", "absolute"])
    p.add_argument("--libero-no-init-states", action="store_true")
    p.add_argument("--libero-observation-height", type=int, default=360)
    p.add_argument("--libero-observation-width", type=int, default=360)
    p.add_argument("--chunk-len", type=int, default=10)
    p.add_argument("--action-dim", type=int, default=None)
    p.add_argument("--proprio-dim", type=int, default=None)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--action-std", type=float, default=0.05)
    p.add_argument("--bc-beta", type=float, default=1.0)
    p.add_argument("--utd", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--warmup-env-steps", type=int, default=2000)
    p.add_argument("--total-env-steps", type=int, default=100_000)
    p.add_argument("--max-episode-steps", type=int, default=400)
    p.add_argument("--num-inference-steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-freq", type=int, default=200)
    p.add_argument("--save-freq", type=int, default=2000)
    train(p.parse_args())


if __name__ == "__main__":
    main()
