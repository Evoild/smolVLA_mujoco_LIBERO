"""LIBERO ChunkEnv adapter for RLT online training.

This module keeps the RLT environment boundary aligned with LeRobot's LIBERO
evaluation path:

raw LIBERO obs -> preprocess_observation -> LiberoProcessorStep ->
SmolVLA policy preprocessor -> RLTController

Actions stay in SmolVLA's normalized action space inside RLT/replay and are
unnormalized by the checkpoint postprocessor immediately before env.step().
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
from lerobot.envs.factory import make_env_pre_post_processors
from lerobot.envs.libero import LiberoEnv as LeRobotLiberoEnv
from lerobot.envs.libero import _get_suite
from lerobot.envs.utils import preprocess_observation
from lerobot.policies import make_pre_post_processors
from lerobot.utils.constants import OBS_STATE


def _stack_nested(values: list[Any]) -> Any:
    """Stack a list of LIBERO raw observation leaves into a batched structure."""
    first = values[0]
    if isinstance(first, Mapping):
        return {key: _stack_nested([value[key] for value in values]) for key in first}
    if isinstance(first, np.ndarray):
        return np.stack(values, axis=0)
    if torch.is_tensor(first):
        return torch.stack(values, dim=0)
    return np.asarray(values)


class LiberoChunkEnv:
    """Single-task LIBERO adapter implementing the existing ChunkEnv protocol.

    `step()` executes exactly one normalized action. Chunk execution remains in
    `RolloutWorker`, which already loops over the actor chunk one action at a
    time and records stride observations for replay.
    """

    def __init__(
        self,
        policy,
        checkpoint: str | Path,
        suite_name: str = "libero_spatial",
        task_id: int = 0,
        max_episode_steps: int | None = None,
        seed: int | None = None,
        device: str | torch.device | None = None,
        init_states: bool = True,
        control_mode: str = "relative",
        obs_type: str = "pixels_agent_pos",
        render_mode: str = "rgb_array",
        observation_height: int = 360,
        observation_width: int = 360,
        camera_name: str = "agentview_image,robot0_eye_in_hand_image",
        num_steps_wait: int = 10,
    ):
        self.policy = policy
        self.checkpoint = str(checkpoint)
        self.suite_name = suite_name
        self.task_id = int(task_id)
        self.device = str(device or policy.config.device)
        self.action_dim = int(policy.config.action_feature.shape[0])
        self.state_dim = int(policy.config.input_features[OBS_STATE].shape[0])
        self.last_success = False
        self.last_info: dict[str, Any] = {}
        self._seed = seed

        suite = _get_suite(suite_name)
        env_cfg = LiberoEnvConfig(
            task=suite_name,
            task_ids=[self.task_id],
            episode_length=max_episode_steps,
            obs_type=obs_type,
            render_mode=render_mode,
            camera_name=camera_name,
            init_states=init_states,
            observation_height=observation_height,
            observation_width=observation_width,
            control_mode=control_mode,
        )
        self.env_preprocessor, _ = make_env_pre_post_processors(env_cfg, policy.config)
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=policy.config,
            pretrained_path=self.checkpoint,
            preprocessor_overrides={
                "device_processor": {"device": self.device},
                "rename_observations_processor": {"rename_map": {}},
            },
        )

        self.env = LeRobotLiberoEnv(
            task_suite=suite,
            task_id=self.task_id,
            task_suite_name=suite_name,
            episode_length=max_episode_steps,
            camera_name=camera_name,
            obs_type=obs_type,
            render_mode=render_mode,
            observation_width=observation_width,
            observation_height=observation_height,
            init_states=init_states,
            episode_index=0,
            n_envs=1,
            num_steps_wait=num_steps_wait,
            control_mode=control_mode,
        )
        self.task_description = self.env.task_description
        self.task = self.env.task
        self.max_episode_steps = int(self.env._max_episode_steps)

    def reset(self) -> dict:
        obs, info = self.env.reset(seed=self._seed)
        # Use the seed only for the first reset; LIBERO init-state cycling then
        # provides deterministic episode variation through the wrapped env.
        self._seed = None
        self.last_success = False
        self.last_info = dict(info)
        return obs

    def obs_to_batch(self, obs_list: list[dict], device) -> dict[str, Tensor]:
        if not obs_list:
            raise ValueError("obs_to_batch requires at least one observation")

        raw_obs = _stack_nested(obs_list)
        batch = preprocess_observation(raw_obs)
        batch["task"] = [self.task_description] * len(obs_list)
        batch = self.env_preprocessor(batch)
        batch = self.preprocessor(batch)

        target_device = torch.device(device)
        for key, value in list(batch.items()):
            if torch.is_tensor(value) and value.device != target_device:
                batch[key] = value.to(target_device)
        return batch

    def step(self, action: Tensor) -> tuple[dict, float, bool]:
        if action.ndim != 1 or action.shape[0] != self.action_dim:
            raise ValueError(f"expected action shape ({self.action_dim},), got {tuple(action.shape)}")
        if not torch.isfinite(action).all().item():
            raise ValueError("RLT actor produced NaN/Inf action before postprocessing")

        action_batch = action.detach().unsqueeze(0)
        env_action = self.postprocessor(action_batch)
        if env_action.ndim == 2:
            env_action = env_action.squeeze(0)
        if env_action.ndim != 1 or env_action.shape[0] != self.action_dim:
            raise ValueError(
                f"postprocessed action must have shape ({self.action_dim},), got {tuple(env_action.shape)}"
            )
        if not torch.isfinite(env_action).all().item():
            raise ValueError("postprocessed LIBERO action contains NaN/Inf")

        action_np = env_action.detach().cpu().numpy().astype(np.float32)
        obs, _env_reward, terminated, _truncated, info = self.env.step(action_np)
        self.last_info = dict(info)
        self.last_success = bool(info.get("is_success", False))

        reward = 1.0 if self.last_success else 0.0
        done = bool(terminated)
        return obs, reward, done

    def get_intervention(self) -> Tensor | None:
        return None

    def close(self) -> None:
        self.env.close()
