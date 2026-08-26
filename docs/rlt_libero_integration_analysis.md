# RLT LIBERO Integration Analysis

目标：把 `smollvla_rltoken` 的 Stage 1 / Stage 2 接到当前 `SmolVLA + LIBERO` 项目，先明确现有 LeRobot/LIBERO/SmolVLA/RLT 的接口边界，后续实现 `LiberoChunkEnv` 时不改 Algorithm 1 主体。

## 总数据流

```text
LIBERO observation
        ↓
SmolVLA processor
        ↓
prefix hidden z_{1:M}
        ↓
RL Token Encoder
        ↓
z_rl
        ↓
Actor(z_rl, proprio, reference_action)
        ↓
RL action chunk
        ↓
LIBERO env.step
        ↓
reward / done / next_obs
        ↓
Replay Buffer
        ↓
Actor-Critic update
```

## 当前 LIBERO Env 创建方式

当前 benchmark 入口是 `lerobot-eval`，命令中使用：

```bash
--env.type=libero
--env.task=libero_spatial,libero_object,libero_goal,libero_10
--env.max_parallel_tasks=1
```

创建链路：

```text
lerobot_eval.py
  -> make_env(cfg.env)
  -> LiberoEnv.create_envs(...)
  -> create_libero_envs(...)
  -> gym.vector.SyncVectorEnv / AsyncVectorEnv
  -> lerobot.envs.libero.LiberoEnv
  -> libero.libero.envs.OffScreenRenderEnv
```

`LiberoEnv` 每个 suite/task_id 构建一个 vector env。默认相机为：

```text
agentview_image
robot0_eye_in_hand_image
```

默认映射到 LeRobot policy key：

```text
agentview_image          -> observation.images.image
robot0_eye_in_hand_image -> observation.images.image2
```

当前 `smolvla_libero/config.json` 与此一致：

```text
observation.images.image  : (3, 256, 256)
observation.images.image2 : (3, 256, 256)
observation.state         : (8,)
action                    : (7,)
```

LIBERO wrapper 的 action space 是 `Box(-1, 1, shape=(7,), dtype=float32)`。默认 `control_mode="relative"`，reset 后会设置 underlying controller `use_delta=True`。

## Observation 格式和 preprocessing

### Raw Gym observation

`lerobot.envs.libero.LiberoEnv.reset/step()` 返回 nested Gym observation：

```text
{
  "pixels": {
    "image": uint8 HWC,
    "image2": uint8 HWC,
  },
  "robot_state": {
    "eef": {
      "pos": (3,),
      "quat": (4,),
      "mat": (3, 3),
    },
    "gripper": {
      "qpos": (2,),
      "qvel": (2,),
    },
    "joints": {
      "pos": (7,),
      "vel": (7,),
    },
  },
}
```

For vector env with batch size 1, image tensors are batched as `(B, H, W, C)` before LeRobot preprocessing.

### `preprocess_observation()`

`lerobot.envs.utils.preprocess_observation()` converts raw Gym observations to LeRobot tensor keys:

```text
pixels.image  uint8 (B,H,W,C) -> observation.images.image  float32 (B,C,H,W), range [0,1]
pixels.image2 uint8 (B,H,W,C) -> observation.images.image2 float32 (B,C,H,W), range [0,1]
robot_state nested dict       -> observation.robot_state nested torch tensors
```

### `LiberoProcessorStep`

`make_env_pre_post_processors(env_cfg, policy_cfg)` returns `env_cfg.get_env_processors()` for SmolVLA. For LIBERO this is:

```text
env_preprocessor  = PolicyProcessorPipeline([LiberoProcessorStep()])
env_postprocessor = PolicyProcessorPipeline([])
```

`LiberoProcessorStep` does two important things:

1. Flips image tensors on height and width dimensions, matching `HuggingFaceVLA/libero` camera orientation.
2. Converts nested `observation.robot_state` into flat `observation.state`:

```text
observation.state = concat(
  eef_pos(3),
  quat_to_axis_angle(eef_quat)(3),
  gripper_qpos(2),
)  -> shape (B, 8), dtype float32
```

`joints.pos`, `joints.vel`, `eef.mat`, and `gripper.qvel` exist in raw obs but are not used by this SmolVLA policy input.

### SmolVLA policy preprocessor

`make_pre_post_processors(policy_cfg, pretrained_path=smolvla_libero)` loads the saved policy processors from the checkpoint. In evaluation the preprocessor override sets:

```text
device_processor.device = policy.config.device
rename_observations_processor.rename_map = cfg.rename_map
```

The SmolVLA preprocessor pipeline includes:

```text
RenameObservationsProcessorStep
AddBatchDimensionProcessorStep
NewLineTaskProcessorStep
TokenizerProcessorStep
DeviceProcessorStep
NormalizerProcessorStep
```

For LIBERO rollout, `lerobot_eval.py` injects task language by calling:

```python
observation["task"] = list(env.call("task_description"))
```

`NewLineTaskProcessorStep` ensures a trailing newline, and `TokenizerProcessorStep` creates:

```text
observation.language.tokens
observation.language.attention_mask
```

`NormalizerProcessorStep` applies checkpoint/dataset stats. Current normalization mapping:

```text
VISUAL -> IDENTITY
STATE  -> MEAN_STD
ACTION -> MEAN_STD
```

This means `obs_to_batch()` in `LiberoChunkEnv` should reuse the exact same `env_preprocessor` and `preprocessor`. It must not manually tokenize language or normalize state.

## SmolVLA action and prefix calling chain

### Standard eval path

LeRobot eval uses single-step action selection:

```text
preprocessed observation
  -> policy.select_action(observation)
  -> if action queue empty:
       _get_action_chunk(...)
       model.sample_actions(...)
       queue first n_action_steps
  -> one action
  -> postprocessor(action)
  -> env.step(action_numpy)
```

For current checkpoint:

```text
chunk_size = 50
n_action_steps = 1
num_steps = 10
max_action_dim = 32
action_dim = 7
```

`policy.predict_action_chunk(batch)` returns the whole unpadded action chunk `(B, 50, 7)`. Internally, `model.sample_actions()` samples padded actions `(B, 50, 32)` and `SmolVLAPolicy._get_action_chunk()` slices to original action dim.

### RLT prefix extractor path

`smollvla_rltoken/rlt/rl_token.py::SmolVLAPrefixExtractor.extract()` mirrors the prefix half of `model.sample_actions()`:

```text
policy.prepare_images(batch)
policy.prepare_state(batch)
tokens = batch[observation.language.tokens]
masks  = batch[observation.language.attention_mask]
model.embed_prefix(...)
model.vlm_with_expert.forward(... fill_kv_cache=True)
```

It returns:

```text
z                 : prefix final hidden, float32, (B, M_total, 960)
pad_mask          : (B, M_total)
n_img_tokens      : image-token prefix length
past_key_values   : KV cache reused for denoising reference actions
prefix_pad_masks  : prefix masks for denoise_step
```

Current smoke test with `smolvla_libero` produced:

```text
image tokens: 128   # 2 cameras x 64 connector tokens
language tokens: 48
state tokens: 1
M_total: 177
z image-only shape: (B, 128, 960)
```

`RLTokenModule` defaults to `use_image_tokens_only=True`, so RLT compresses only the image-token portion of `z`.

`SmolVLAPrefixExtractor.sample_reference_chunk(feats)` reuses the cached prefix and runs Flow Matching denoising:

```text
x_t ~ noise, shape (B, 50, 32)
for num_steps=10:
  denoise_step(prefix_pad_masks, past_key_values, x_t, timestep)
return x_t
```

`RLTController.plan_chunk()` then slices:

```text
ref_full  = sample_reference_chunk(... )[:, :, :action_dim]  -> (B, 50, 7)
ref_chunk = ref_full[:, :C]                                  -> (B, 10, 7)
```

## Action normalization / unnormalization

Inside RLT:

```text
reference action
actor action
replay action/ref
```

are in SmolVLA normalized action space, because they come directly from the frozen policy before postprocessing.

Before sending actions to LIBERO `env.step`, `LiberoChunkEnv` must run the same policy postprocessor used by `lerobot_eval.py`:

```text
normalized action tensor
  -> policy postprocessor / UnnormalizerProcessorStep
  -> CPU tensor / numpy
  -> LIBERO env.step(action_np)
```

For current checkpoint, action normalization stats come from `libero/meta/stats.json` / saved checkpoint processor; action dim is 7 and the env action space is nominally `[-1, 1]`.

Important boundary:

- Store normalized actions in replay buffer, matching current `RLTAgent` and BC loss.
- Execute unnormalized actions in LIBERO.
- If human intervention is added later, convert intervention actions into the same normalized action space before storing them as `action` and `ref`.

## Episode termination, truncation, success

`lerobot.envs.libero.LiberoEnv.step(action)` calls underlying LIBERO:

```python
raw_obs, reward, done, info = self._env.step(action)
is_success = self._env.check_success()
terminated = done or is_success
info["is_success"] = is_success
truncated = False
```

If `terminated`, the wrapper currently calls `self.reset()` before returning the observation. For RLT training this is a problem: the returned observation after success may already be a reset observation. `LiberoChunkEnv` should treat success/terminated as episode boundary and should not use the returned post-reset observation as a bootstrap state for a successful terminal transition.

Time-limit truncation is handled outside the LIBERO wrapper in LeRobot eval by reading `env.call("_max_episode_steps")` and forcing done at max steps. In RLT, `RolloutWorker` already has:

```text
if ep_steps >= env.max_episode_steps:
    truncated = True
```

For replay semantics:

- success termination should be stored as `done=True` so TD bootstrap is zeroed.
- time-limit truncation should end the episode but be stored as `done=False`, preserving bootstrap; this matches current `RolloutWorker`.

Sparse RL reward should be:

```text
reward = 1.0 if is_success else 0.0
```

Do not use dense simulator reward until separately validated.

## RLT environment contract

Current `ChunkEnv` protocol in `smollvla_rltoken/rlt/envs.py`:

```python
action_dim: int
max_episode_steps: int

reset() -> dict
step(action: Tensor) -> tuple[dict, float, bool]
obs_to_batch(obs_list: list[dict], device) -> dict[str, Tensor]
get_intervention() -> Tensor | None
```

Current `RolloutWorker.run_chunk()` assumes:

- `env.step(actions[j])` applies one action step, not a whole chunk.
- `env.step()` returns `(obs, reward, success)`.
- truncation is checked by `env.max_episode_steps`.
- inter-step observations are collected every replay stride for `compute_x()`.

Therefore `LiberoChunkEnv` should expose a single-step `step(action)` method even though its name in the user plan says `step(action_chunk)`. The chunk loop is already in `RolloutWorker`; duplicating chunk execution inside env would double-loop actions. If a higher-level chunk API is desired later, it should be a separate method and not replace the current `ChunkEnv.step(action)` contract unless `RolloutWorker` is changed.

## Proposed `LiberoChunkEnv` construction

`LiberoChunkEnv` should wrap one non-vectorized LIBERO task, or a vector env with `num_envs=1`. To match existing LeRobot eval with minimal risk, reuse:

```text
LiberoEnv / create_libero_envs
env_preprocessor = make_env_pre_post_processors(...)
preprocessor, postprocessor = make_pre_post_processors(...)
```

Suggested fields:

```text
suite_name
task_id
task_description
raw_env or vec_env
policy
env_preprocessor
policy_preprocessor
policy_postprocessor
action_dim = 7
state_dim = 8
max_episode_steps = suite default or override
```

`reset()`:

```text
raw_obs, info = env.reset(seed=optional)
cache task_description
return raw_obs
```

`obs_to_batch(obs_list, device)`:

```text
for obs in obs_list:
  make a batch-compatible raw observation
  add task language
  preprocess_observation(raw_obs)
  env_preprocessor(...)
  policy_preprocessor(...)
return batch on device
```

For a first implementation, keep `obs_list` length 1 unless batching intermediate stride observations is carefully tested. The current `RolloutWorker` may call `obs_to_batch(inter_obs, device)` with multiple observations; therefore `LiberoChunkEnv` should either implement robust stacking or explicitly adapt `RolloutWorker` to process inter_obs one by one.

`step(action_normalized)`:

```text
action = action_normalized[None, :] or action_normalized
action = policy_postprocessor(action)
action_np = action.squeeze(0).cpu().numpy().astype(np.float32)
next_obs, env_reward, terminated, truncated, info = env.step(action_np)
success = bool(info.get("is_success", terminated))
reward = 1.0 if success else 0.0
return next_obs, reward, success
```

Must guard:

- action finite check before postprocessor and before `env.step`.
- action shape exactly `(7,)`.
- do not pass `(C, 7)` directly to `env.step`.

## What can be reused unchanged

- `SmolVLAPrefixExtractor`: already matches SmolVLA prefix and cached reference sampling. Smoke test passes with LIBERO checkpoint.
- `RLTokenModule`: VLA-width 960 and image-only token path match current checkpoint.
- `RLTController`: already computes `z_rl`, `x=(z_rl, proprio)`, reference chunk `(H=50)`, actor chunk `(C=10)`.
- `ChunkReplayBuffer`: chunk-level storage and stride-2 transition assembly are independent of LIBERO.
- `RLTAgent`: actor/critic update logic is environment-agnostic.
- `RolloutWorker` chunk loop: already executes one action at a time and stops early on success/truncation.

## What must be modified or added

1. Add `smollvla_rltoken/rlt/libero_env.py` implementing `ChunkEnv` with real LIBERO reset/step/preprocessing/postprocessing.
2. Add a `train_online.py` path or factory flag to instantiate `LiberoChunkEnv` instead of `MockManipEnv`.
3. Make `obs_to_batch()` reuse `preprocess_observation`, `LiberoProcessorStep`, and the checkpoint policy preprocessor.
4. Make action execution use the checkpoint policy postprocessor before `env.step`.
5. Handle LIBERO wrapper auto-reset on terminal states so terminal next observations are not accidentally used for successful bootstrap.
6. Add `scripts/test_rlt_libero_rollout.py` for frozen VLA + RLT inference rollout before actor/critic training.
7. Later, implement residual actor zero-init before allowing non-warmup actor control in LIBERO.

## Open risks for implementation

- The current LIBERO wrapper auto-resets inside `step()` when `terminated=True`; `LiberoChunkEnv` must mark terminal immediately and avoid treating the returned observation as same-episode next state.
- Current `RolloutWorker` expects `obs_to_batch()` to handle a list of observations for stride states. Implementing batch collation for nested LIBERO observations must be tested directly.
- The exact saved pre/postprocessor files should be used from `smolvla_libero`; recreating processors from config may miss checkpoint-specific stats or future migration details.
- Stage 1 currently uses all dataset episodes unless filtering is added. For the first real RLT experiment, add suite/task filtering so `libero_spatial` data and selected online tasks match.
