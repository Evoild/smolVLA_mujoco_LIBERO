"""End-to-end smoke test for the RLT-on-SmolVLA reproduction.

Exercises, on GPU, with the real smolvla_base checkpoint:
  1. smolvla_compat: load checkpoint weights (SmolVLM2 backbone + policy)
  2. SmolVLAPrefixExtractor: prefix forward -> z_{1:M} + KV cache
  3. reference chunk sampling from the cached prefix
  4. RLTokenModule: Eq.1 encode + Eq.2 reconstruction loss + backward
  5. actor-critic: Eq.3/5 updates on synthetic transitions
  6. RLTController.plan_chunk + a few mock-env chunks through the
     replay-buffer assembler and agent updates (mini Algorithm 1)

Run:
  cd /mnt/sdb/feng/smollvla_rltoken && \
  /home/feng/pi05_reproduce/.venv/bin/python scripts/smoke_test.py [--device cuda:0]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from lerobot.utils.constants import OBS_LANGUAGE_TOKENS  # noqa: E402
from lerobot.utils.constants import OBS_STATE  # noqa: E402

from rlt.actor_critic import RLTAgent  # noqa: E402
from rlt.configs import ActorCriticConfig, OnlineRLConfig, RLTokenConfig  # noqa: E402
from rlt.envs import MockManipEnv  # noqa: E402
from rlt.replay_buffer import ChunkReplayBuffer  # noqa: E402
from rlt.rl_token import RLTokenModule, SmolVLAPrefixExtractor  # noqa: E402
from rlt.rlt_policy import RLTController  # noqa: E402
from rlt.smolvla_compat import load_smolvla_policy  # noqa: E402
from rlt.train_online import RolloutWorker  # noqa: E402

VLA_WIDTH = 960  # SmolVLM2-500M text hidden size


def check(name: str, cond: bool, extra: str = ""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name} {extra}")
    if not cond:
        raise SystemExit(f"smoke test failed at: {name}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--checkpoint", default=str(REPO / "smolvla_base"))
    p.add_argument("--dtype", default=None, choices=[None, "float32", "bfloat16"])
    args = p.parse_args()
    device = args.device
    torch.manual_seed(0)

    # ---------------------------------------------------------- 1. load VLA
    t0 = time.time()
    policy = load_smolvla_policy(args.checkpoint, device=device, dtype=args.dtype)
    check("load smolvla checkpoint", True, f"({time.time()-t0:.1f}s, dtype={args.dtype})")
    policy.requires_grad_(False)
    action_dim = policy.config.action_feature.shape[0]
    state_dim = policy.config.input_features[OBS_STATE].shape[0]
    camera_keys = tuple(policy.config.image_features)

    env = MockManipEnv(
        action_dim=action_dim,
        state_dim=state_dim,
        camera_keys=camera_keys,
        max_episode_steps=60,
        seed=0,
    )
    obs = env.reset()
    batch = env.obs_to_batch([obs, env.reset()], device)  # B=2

    # ------------------------------------------------- 2. prefix extraction
    extractor = SmolVLAPrefixExtractor(policy)
    t0 = time.time()
    feats = extractor.extract(batch)
    z, mask = extractor.select_tokens(feats, image_only=True)
    check(
        "prefix extraction",
        z.ndim == 3 and z.shape[0] == 2 and z.shape[2] == VLA_WIDTH
        and feats["n_img_tokens"] == z.shape[1] and feats["n_img_tokens"] > 0,
        f"z={tuple(z.shape)} n_img={feats['n_img_tokens']} ({time.time()-t0:.1f}s)",
    )
    check("embeddings finite", torch.isfinite(z).all().item())
    check(
        "prefix layout (img + lang + state)",
        feats["z"].shape[1] == feats["n_img_tokens"] + batch[OBS_LANGUAGE_TOKENS].shape[1] + 1,
        f"M_total={feats['z'].shape[1]} n_lang={batch[OBS_LANGUAGE_TOKENS].shape[1]}",
    )

    # ------------------------------------------------- 3. reference sampling
    t0 = time.time()
    ref = extractor.sample_reference_chunk(feats)
    check(
        "reference chunk sampling",
        ref.shape == (2, policy.config.chunk_size, policy.config.max_action_dim)
        and torch.isfinite(ref).all().item(),
        f"ref={tuple(ref.shape)} ({time.time()-t0:.1f}s)",
    )

    # ----------------------------------------------------- 4. RL token module
    rt_cfg = RLTokenConfig(d_model=512, n_encoder_layers=2, n_decoder_layers=2)
    rl_token = RLTokenModule(rt_cfg).to(device)
    loss, z_rl = rl_token.reconstruction_loss(z, mask)
    loss.backward()
    grad_ok = all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in rl_token.parameters()
    )
    check(
        "RL token loss + backward",
        torch.isfinite(loss).item() and z_rl.shape == (2, 512) and grad_ok,
        f"L_ro={loss.item():.4f} z_rl={tuple(z_rl.shape)}",
    )
    rl_token.zero_grad(set_to_none=True)
    rl_token.eval().requires_grad_(False)

    # ----------------------------------------------------- 5. actor-critic
    cfg = OnlineRLConfig(
        ac=ActorCriticConfig(
            chunk_len=10, action_dim=action_dim, proprio_dim=action_dim, rl_token_dim=512
        ),
        batch_size=64,
    )
    agent = RLTAgent(cfg, device=device)
    x_dim = cfg.ac.rl_token_dim + cfg.ac.proprio_dim
    fake = {
        "x": torch.randn(64, x_dim, device=device),
        "action": torch.randn(64, 10, action_dim, device=device),
        "ref": torch.randn(64, 10, action_dim, device=device),
        "reward_disc": torch.rand(64, device=device),
        "x_next": torch.randn(64, x_dim, device=device),
        "ref_next": torch.randn(64, 10, action_dim, device=device),
        "done": torch.zeros(64, device=device),
    }
    m1 = agent.update_critic(fake)
    m2 = agent.update_actor(fake)
    check(
        "actor-critic updates",
        all(torch.isfinite(torch.tensor(v)) for v in {**m1, **m2}.values()),
        f"critic_loss={m1['critic_loss']:.4f} actor_loss={m2['actor_loss']:.4f}",
    )

    # --------------------------------------------- 6. mini online loop (Alg 1)
    controller = RLTController(
        policy, rl_token, agent, chunk_len=10, action_dim=action_dim, proprio_dim=action_dim,
        num_inference_steps=3,  # fewer denoise steps to keep the test fast
    )
    buffer = ChunkReplayBuffer(
        capacity=1000, x_dim=x_dim, chunk_len=10, action_dim=action_dim,
        discount=cfg.discount, stride=2, device=device,
    )
    worker = RolloutWorker(env, controller, buffer, stride=2)
    worker.reset()
    t0 = time.time()
    n_chunks = 4
    for k in range(n_chunks):
        n_steps, ep_done, _ = worker.run_chunk(use_actor=(k >= 2))
        if ep_done:
            worker.reset()
    # chunks 1..3 emit 5 transitions each once their successor arrives
    check(
        "rollout -> replay buffer",
        len(buffer) >= (n_chunks - 1) * (10 // 2),
        f"buffer={len(buffer)} after {n_chunks} chunks ({time.time()-t0:.1f}s)",
    )

    b = buffer.sample(32)
    m = agent.update_critic(b)
    m.update(agent.update_actor(b))
    check(
        "update from real rollout data",
        all(torch.isfinite(torch.tensor(v)) for v in m.values()),
        f"critic_loss={m['critic_loss']:.4f} bc_dist={m['bc_dist']:.4f}",
    )

    plan = controller.plan_chunk(env.obs_to_batch([env.reset()], device), use_actor=True)
    check(
        "RLTController.plan_chunk",
        plan["action_chunk"].shape == (1, 10, action_dim)
        and torch.isfinite(plan["action_chunk"]).all().item(),
        f"action_chunk={tuple(plan['action_chunk'].shape)}",
    )

    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    main()
