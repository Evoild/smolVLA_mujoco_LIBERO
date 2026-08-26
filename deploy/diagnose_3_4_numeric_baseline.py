#!/usr/bin/env python3

"""Strict numeric baseline for SmolVLA sample_actions core.

The debug graph returns named intermediate tensors so PyTorch native mixed
precision, ONNX Runtime, TensorRT BF16, TensorRT INT8 Q/DQ, and PyTorch fake
quant can be compared on identical core inputs and identical denoising noise.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from lerobot.configs import FeatureType, PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE
from lerobot.utils.random_utils import set_seed

from linear_only_quant import load_activation_scales_by_module, replace_linear_modules

INPUT_NAMES = [
    "image_emb",
    "image2_emb",
    "image_mask",
    "image2_mask",
    "state",
    "language_tokens",
    "language_attention_mask",
    "noise",
]


def make_att_2d_masks_trt(pad_masks: torch.Tensor, att_masks: torch.Tensor) -> torch.Tensor:
    cumsum = torch.cumsum(att_masks.to(torch.int64), dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] & pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


class SmolVLADebugCoreWrapper(nn.Module):
    """sample_actions core with explicit debug outputs.

    Outputs:
    - action_chunk
    - prefix_embs and prefix_out
    - suffix_out_step_N before action_out_proj
    - v_t_step_N velocity predicted by action_out_proj
    - x_t_step_N after the Euler update
    """

    def __init__(self, policy: nn.Module):
        super().__init__()
        self.model = policy.model
        self.action_dim = int(policy.config.action_feature.shape[0])
        self.chunk_size = int(policy.config.chunk_size)
        self.num_steps = int(policy.config.num_steps)
        self.output_names = ["action_chunk", "prefix_embs", "prefix_out"]
        for step in range(self.num_steps):
            self.output_names.extend(
                [f"suffix_out_step_{step:02d}", f"v_t_step_{step:02d}", f"x_t_step_{step:02d}"]
            )

    def embed_prefix_from_image_embeds(
        self,
        image_emb: torch.Tensor,
        image2_emb: torch.Tensor,
        image_mask: torch.Tensor,
        image2_mask: torch.Tensor,
        state: torch.Tensor,
        language_tokens: torch.Tensor,
        language_attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state = state.to(self.model.state_proj.weight.dtype)
        embs = []
        pad_masks = []
        att_masks = []

        for img_emb, img_mask in ((image_emb, image_mask), (image2_emb, image2_mask)):
            img_emb_dim = img_emb.shape[-1]
            img_emb = img_emb * torch.tensor(img_emb_dim**0.5, dtype=img_emb.dtype, device=img_emb.device)
            bsize, num_img_embs = img_emb.shape[:2]
            img_mask = img_mask[:, None].expand(bsize, num_img_embs)
            embs.append(img_emb)
            pad_masks.append(img_mask)
            att_masks += [0] * num_img_embs

        lang_emb = self.model.vlm_with_expert.embed_language_tokens(language_tokens)
        lang_emb_dim = lang_emb.shape[-1]
        lang_emb = lang_emb * math.sqrt(lang_emb_dim)
        embs.append(lang_emb)
        pad_masks.append(language_attention_mask)
        att_masks += [0] * lang_emb.shape[1]

        state_emb = self.model.state_proj(state)
        state_emb = state_emb[:, None, :] if state_emb.ndim == 2 else state_emb
        embs.append(state_emb)
        bsize = state_emb.shape[0]
        state_mask = torch.ones(bsize, state_emb.shape[1], dtype=torch.bool, device=state_emb.device)
        pad_masks.append(state_mask)
        att_masks += [1] * state_emb.shape[1]

        prefix_embs = torch.cat(embs, dim=1)
        prefix_pad_masks = torch.cat(pad_masks, dim=1)
        prefix_att_masks = torch.tensor(att_masks, dtype=torch.bool, device=prefix_pad_masks.device)
        prefix_att_masks = prefix_att_masks[None, :].expand(bsize, -1)
        return prefix_embs, prefix_pad_masks, prefix_att_masks

    def denoise_step_debug(
        self,
        prefix_pad_masks: torch.Tensor,
        past_key_values: dict,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.model.embed_suffix(x_t, timestep)
        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks_trt(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        prefix_offsets = torch.sum(prefix_pad_masks.to(torch.int64), dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks.to(torch.int64), dim=1) - 1

        outputs_embeds, _ = self.model.vlm_with_expert.forward(
            attention_mask=full_att_2d_masks,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=self.model.config.use_cache,
            fill_kv_cache=False,
        )
        suffix_out = outputs_embeds[1][:, -self.chunk_size :]
        suffix_out = suffix_out.to(dtype=self.model.action_out_proj.weight.dtype)
        v_t = self.model.action_out_proj(suffix_out)
        return suffix_out, v_t

    def forward(
        self,
        image_emb: torch.Tensor,
        image2_emb: torch.Tensor,
        image_mask: torch.Tensor,
        image2_mask: torch.Tensor,
        state: torch.Tensor,
        language_tokens: torch.Tensor,
        language_attention_mask: torch.Tensor,
        noise: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        bsize = state.shape[0]
        device = state.device
        noise = noise.to(self.model.action_in_proj.weight.dtype)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix_from_image_embeds(
            image_emb,
            image2_emb,
            image_mask,
            image2_mask,
            state,
            language_tokens,
            language_attention_mask,
        )
        prefix_att_2d_masks = make_att_2d_masks_trt(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks.to(torch.int64), dim=1) - 1

        prefix_outputs, past_key_values = self.model.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.model.config.use_cache,
            fill_kv_cache=True,
        )

        dt = -1.0 / self.num_steps
        x_t = noise
        suffix_outs = []
        velocities = []
        states = []
        for step in range(self.num_steps):
            time = 1.0 + step * dt
            time_tensor = torch.tensor(time, dtype=x_t.dtype, device=device).expand(bsize)
            suffix_out, v_t = self.denoise_step_debug(
                x_t=x_t,
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values,
                timestep=time_tensor,
            )
            x_t = x_t + dt * v_t
            suffix_outs.append(suffix_out)
            velocities.append(v_t)
            states.append(x_t)

        action_chunk = x_t[:, :, : self.action_dim]
        outputs: list[torch.Tensor] = [action_chunk, prefix_embs, prefix_outputs[0]]
        for suffix_out, v_t, x_step in zip(suffix_outs, velocities, states, strict=True):
            outputs.extend([suffix_out, v_t, x_step])
        return tuple(outputs)


class SmolVLAActionOnlyCoreWrapper(nn.Module):
    """sample_actions core for deployment profiling; returns only action_chunk."""

    def __init__(self, policy: nn.Module):
        super().__init__()
        self.debug_core = SmolVLADebugCoreWrapper(policy)
        self.output_names = ["action_chunk"]

    def forward(
        self,
        image_emb: torch.Tensor,
        image2_emb: torch.Tensor,
        image_mask: torch.Tensor,
        image2_mask: torch.Tensor,
        state: torch.Tensor,
        language_tokens: torch.Tensor,
        language_attention_mask: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        return self.debug_core(
            image_emb,
            image2_emb,
            image_mask,
            image2_mask,
            state,
            language_tokens,
            language_attention_mask,
            noise,
        )[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--policy-path", default="smolvla_libero")
    common.add_argument("--device", default="cuda")
    common.add_argument("--seed", type=int, default=1000)
    common.add_argument("--task", default="libero_goal step 3-4 numeric diagnosis")
    common.add_argument("--batch-size", type=int, default=1)
    common.add_argument("--token-length", type=int, default=48)
    common.add_argument("--input-dtype", choices=["fp32", "bf16"], default="fp32")
    common.add_argument(
        "--model-dtype",
        choices=["native", "bf16"],
        default="native",
        help=(
            "Model dtype used by PyTorch baselines and ONNX export. native preserves checkpoint dtypes; "
            "bf16 casts all non-quantized modules to BF16 before fake quant or export."
        ),
    )
    common.add_argument(
        "--quantize-module-prefix",
        action="append",
        default=["model.vlm_with_expert.vlm.model.text_model"],
        help="PyTorch fake-quant module prefix. Pass multiple times if needed.",
    )
    common.add_argument(
        "--quantize-module-regex",
        action="append",
        default=None,
        help="Only PyTorch fake-quant nn.Linear modules whose full module name matches this regex.",
    )
    common.add_argument(
        "--fake-quant-activation-scale-mode",
        choices=["dynamic", "calibrated"],
        default="dynamic",
        help="Activation scale mode for PyTorch fake quant. dynamic is the historical per-token fake quant.",
    )
    common.add_argument(
        "--fake-quant-activation-scales-json",
        default=None,
        help="Calibration JSON used when --fake-quant-activation-scale-mode calibrated.",
    )
    common.add_argument(
        "--fake-quant-kind",
        choices=["w8a8", "w8a16"],
        default="w8a8",
        help="PyTorch fake-quant kind. w8a16 quantizes weights only and leaves activations unquantized.",
    )
    common.add_argument(
        "--extra-w8a16-module-regex",
        action="append",
        default=None,
        help=(
            "Additional PyTorch nn.Linear module regexes to replace with W8A16 after the primary fake quant "
            "replacement. Useful for mixed fake-quant experiments such as gate/up W8A8 plus down W8A16."
        ),
    )

    export_parser = subparsers.add_parser("export", parents=[common])
    export_parser.add_argument("--output", required=True)
    export_parser.add_argument(
        "--output-mode",
        choices=["debug", "action-only"],
        default="debug",
        help="debug exports intermediate tensors for numeric diagnosis; action-only exports only action_chunk for deployment profiling.",
    )
    export_parser.add_argument("--opset", type=int, default=18)
    export_parser.add_argument("--check", action="store_true")

    ort_parser = subparsers.add_parser("make-ort-compatible", parents=[common])
    ort_parser.add_argument("--input", required=True)
    ort_parser.add_argument("--output", required=True)
    ort_parser.add_argument("--check", action="store_true")

    compare_parser = subparsers.add_parser("compare", parents=[common])
    compare_parser.add_argument("--bf16-onnx", default=None)
    compare_parser.add_argument("--int8-onnx", default=None)
    compare_parser.add_argument("--ort-bf16-onnx", default=None)
    compare_parser.add_argument("--ort-int8-onnx", default=None)
    compare_parser.add_argument("--bf16-engine", default=None)
    compare_parser.add_argument("--int8-engine", default=None)
    compare_parser.add_argument("--output-dir", required=True)
    compare_parser.add_argument("--ort-provider", default="CUDAExecutionProvider")
    compare_parser.add_argument(
        "--input-source",
        choices=["synthetic", "rollout"],
        default="synthetic",
        help="synthetic uses the historical random observation; rollout uses real LIBERO rollout samples.",
    )
    compare_parser.add_argument("--compare-samples", type=int, default=1)
    compare_parser.add_argument("--sample-stride", type=int, default=5)
    compare_parser.add_argument("--max-parallel-tasks", type=int, default=1)
    compare_parser.add_argument(
        "--comparison-set",
        choices=["legacy", "py-fake-trt"],
        default="legacy",
        help=(
            "legacy keeps the historical A/B/C/D/E matrix. "
            "py-fake-trt reports only A=PyTorch native, B=PyTorch fake quant, C=the requested TensorRT engine."
        ),
    )
    return parser.parse_args()


def load_policy(policy_path: str, device: str, model_dtype: str = "native") -> tuple[nn.Module, Any, str]:
    cfg = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=[f"--device={device}"])
    policy_cls = get_policy_class(cfg.type)
    policy = policy_cls.from_pretrained(policy_path, config=cfg, strict=False)
    policy.eval()
    effective_device = str(policy.config.device)
    policy.to(device=effective_device)
    if model_dtype == "bf16":
        policy.to(dtype=torch.bfloat16)
    elif model_dtype != "native":
        raise ValueError(f"unsupported model dtype: {model_dtype}")
    preprocessor, _ = make_pre_post_processors(
        policy.config,
        pretrained_path=policy_path,
        preprocessor_overrides={"device_processor": {"device": effective_device}},
    )
    return policy, preprocessor, effective_device


def make_raw_observation(config: PreTrainedConfig, task: str) -> dict[str, Any]:
    obs: dict[str, Any] = {"task": task}
    for key, feature in (config.input_features or {}).items():
        shape = tuple(int(dim) for dim in feature.shape)
        if feature.type is FeatureType.VISUAL:
            obs[key] = torch.rand(shape, dtype=torch.float32)
        elif feature.type is FeatureType.STATE or key == OBS_STATE:
            obs[key] = torch.zeros(shape, dtype=torch.float32)
        elif feature.type is FeatureType.ENV:
            obs[key] = torch.zeros(shape, dtype=torch.float32)
    if OBS_STATE not in obs and getattr(config, "max_state_dim", None):
        obs[OBS_STATE] = torch.zeros((int(config.max_state_dim),), dtype=torch.float32)
    return obs


def core_inputs_from_policy(
    policy: nn.Module,
    preprocessor: Any,
    task: str,
    seed: int,
    input_dtype: torch.dtype,
    token_length: int,
) -> tuple[torch.Tensor, ...]:
    set_seed(seed)
    raw_observation = make_raw_observation(policy.config, task)
    batch = preprocessor(raw_observation)
    images, img_masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)
    lang_tokens = batch[OBS_LANGUAGE_TOKENS]
    lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
    if lang_tokens.shape[1] != token_length:
        lang_tokens = match_static_shape(lang_tokens, (lang_tokens.shape[0], token_length), 0)
        lang_masks = match_static_shape(lang_masks, (lang_masks.shape[0], token_length), False)
    bsize = state.shape[0]
    noise_shape = (bsize, policy.config.chunk_size, policy.config.max_action_dim)
    noise = policy.model.sample_noise(noise_shape, state.device)
    with torch.no_grad():
        image_emb = policy.model.vlm_with_expert.embed_image(images[0])
        image2_emb = policy.model.vlm_with_expert.embed_image(images[1])
    return (
        image_emb.to(input_dtype),
        image2_emb.to(input_dtype),
        img_masks[0],
        img_masks[1],
        state.to(input_dtype),
        lang_tokens,
        lang_masks,
        noise.to(input_dtype),
    )


def cpu_inputs(inputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    return tuple(tensor.detach().cpu() for tensor in inputs)


def device_inputs(inputs: tuple[torch.Tensor, ...], device: str) -> tuple[torch.Tensor, ...]:
    return tuple(tensor.to(device=device) for tensor in inputs)


def core_inputs_from_batch(policy: nn.Module, batch: dict[str, torch.Tensor], token_length: int) -> tuple[torch.Tensor, ...]:
    from calibrate_smolvla_activation_channel_scales import make_core_inputs

    return make_core_inputs(policy, batch, token_length)


def collect_rollout_core_inputs(args: argparse.Namespace, total_samples: int) -> tuple[list[dict[str, Any]], str]:
    from lerobot.envs import make_env, make_env_pre_post_processors, preprocess_observation
    from lerobot.envs.factory import make_env_config
    from lerobot.policies import make_pre_post_processors as make_policy_pre_post_processors
    from lerobot.utils.constants import ACTION

    set_seed(args.seed)
    policy, preprocessor, device = load_policy(args.policy_path, args.device)
    post_preprocessor, postprocessor = make_policy_pre_post_processors(
        policy.config,
        pretrained_path=args.policy_path,
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    del post_preprocessor

    env_cfg = make_env_config("libero", task=args.task, max_parallel_tasks=args.max_parallel_tasks)
    envs = make_env(env_cfg, n_envs=args.batch_size, use_async_envs=False, trust_remote_code=False)
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy.config)
    task_envs = [(task_group, task_id, env) for task_group, group in envs.items() for task_id, env in group.items()]
    if not task_envs:
        raise RuntimeError("No LIBERO environments were created for rollout numeric diagnosis")

    records: list[dict[str, Any]] = []
    wrapper = SmolVLADebugCoreWrapper(policy).to(device=device).eval()
    try:
        while len(records) < total_samples:
            for task_offset, (task_group, task_id, env) in enumerate(task_envs):
                if len(records) >= total_samples:
                    break
                policy.reset()
                seed_base = args.seed + task_offset * 1000 + (len(records) // max(1, len(task_envs))) * 100000
                observation, _info = env.reset(seed=[seed_base + idx for idx in range(args.batch_size)])
                done = np.array([False] * env.num_envs)
                max_steps = int(env.call("_max_episode_steps")[0])
                for step in range(max_steps):
                    observation = preprocess_observation(observation)
                    try:
                        observation["task"] = list(env.call("task_description"))
                    except (AttributeError, NotImplementedError):
                        observation["task"] = list(env.call("task"))
                    observation = env_preprocessor(observation)
                    batch = preprocessor(observation)
                    if step % max(1, args.sample_stride) == 0 and len(records) < total_samples:
                        core_inputs = core_inputs_from_batch(policy, batch, args.token_length)
                        records.append(
                            {
                                "sample_index": len(records),
                                "input_source": "rollout",
                                "task_group": task_group,
                                "task_id": int(task_id),
                                "episode_seed": int(seed_base),
                                "timestep": int(step),
                                "core_inputs": cpu_inputs(core_inputs),
                            }
                        )
                    if len(records) >= total_samples:
                        break
                    with torch.no_grad():
                        core_inputs = core_inputs_from_batch(policy, batch, args.token_length)
                        action_chunk = wrapper(*core_inputs)[0]
                    action = action_chunk[:, 0, : int(policy.config.action_feature.shape[0])].to(torch.float32)
                    action = postprocessor(action)
                    action_transition = env_postprocessor({ACTION: action})
                    observation, _reward, terminated, truncated, _info = env.step(
                        action_transition[ACTION].to("cpu").numpy()
                    )
                    done = done | terminated | truncated
                    if np.all(done):
                        break
    finally:
        for _task_group, _task_id, env in task_envs:
            try:
                env.close()
            except Exception:
                pass
    return records, device


def match_static_shape(tensor: torch.Tensor, expected: tuple[int, ...], pad_value: int | float | bool = 0) -> torch.Tensor:
    if tuple(tensor.shape) == expected:
        return tensor
    slices = tuple(slice(0, min(int(got), int(want))) for got, want in zip(tensor.shape, expected, strict=True))
    output = torch.full(expected, pad_value, dtype=tensor.dtype, device=tensor.device)
    output[slices] = tensor[slices]
    return output


def export_debug_onnx(args: argparse.Namespace) -> None:
    input_dtype = torch.float32 if args.input_dtype == "fp32" else torch.bfloat16
    policy, preprocessor, device = load_policy(args.policy_path, args.device, args.model_dtype)
    if args.output_mode == "action-only":
        wrapper = SmolVLAActionOnlyCoreWrapper(policy).to(device=device).eval()
    else:
        wrapper = SmolVLADebugCoreWrapper(policy).to(device=device).eval()
    inputs = core_inputs_from_policy(policy, preprocessor, args.task, args.seed, input_dtype, args.token_length)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        reference = wrapper(*inputs)
    if isinstance(reference, torch.Tensor):
        reference_outputs = (reference,)
    else:
        reference_outputs = reference

    torch.onnx.export(
        wrapper,
        inputs,
        output,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
        input_names=INPUT_NAMES,
        output_names=wrapper.output_names,
        dynamic_axes=None,
    )

    report = {
        "onnx_path": str(output),
        "precision": "native_mixed_precision",
        "model_dtype": args.model_dtype,
        "input_dtype": args.input_dtype,
        "device": device,
        "seed": args.seed,
        "token_length": args.token_length,
        "output_mode": args.output_mode,
        "input_shapes": {name: list(tensor.shape) for name, tensor in zip(INPUT_NAMES, inputs, strict=True)},
        "output_shapes": {
            name: list(tensor.shape) for name, tensor in zip(wrapper.output_names, reference_outputs, strict=True)
        },
    }
    if args.check:
        import onnx

        model = onnx.load(output)
        onnx.checker.check_model(model)
        report["onnx_check"] = "ok"
    with open(output.with_suffix(".export_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


def bf16_to_float32_array(tensor) -> np.ndarray:
    if tensor.raw_data:
        data = np.frombuffer(tensor.raw_data, dtype=np.uint16)
    else:
        data = np.asarray(tensor.int32_data, dtype=np.uint16)
    return (data.astype(np.uint32) << 16).view(np.float32).reshape(tuple(tensor.dims))


def make_ort_compatible_onnx(args: argparse.Namespace) -> None:
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model = onnx.load(input_path)
    converted_initializers = 0
    converted_value_infos = 0
    converted_casts = 0
    converted_constants = 0
    converted_double = 0

    for initializer in model.graph.initializer:
        if initializer.data_type == TensorProto.BFLOAT16:
            array = bf16_to_float32_array(initializer)
            initializer.CopyFrom(numpy_helper.from_array(array, initializer.name))
            converted_initializers += 1
        elif initializer.data_type == TensorProto.DOUBLE:
            array = numpy_helper.to_array(initializer).astype(np.float32)
            initializer.CopyFrom(numpy_helper.from_array(array, initializer.name))
            converted_initializers += 1
            converted_double += 1

    for value_info in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
        tensor_type = value_info.type.tensor_type
        if tensor_type.elem_type in {TensorProto.BFLOAT16, TensorProto.DOUBLE}:
            if tensor_type.elem_type == TensorProto.DOUBLE:
                converted_double += 1
            tensor_type.elem_type = TensorProto.FLOAT
            converted_value_infos += 1

    for node in model.graph.node:
        if node.op_type == "Cast":
            for attr in node.attribute:
                if attr.name == "to" and attr.i in {TensorProto.BFLOAT16, TensorProto.DOUBLE}:
                    if attr.i == TensorProto.DOUBLE:
                        converted_double += 1
                    attr.i = TensorProto.FLOAT
                    converted_casts += 1
        if node.op_type in {"Constant", "ConstantOfShape"}:
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.TENSOR and attr.t.data_type == TensorProto.BFLOAT16:
                    array = bf16_to_float32_array(attr.t)
                    attr.t.CopyFrom(numpy_helper.from_array(array, attr.t.name))
                    converted_constants += 1
                elif attr.type == onnx.AttributeProto.TENSOR and attr.t.data_type == TensorProto.DOUBLE:
                    array = numpy_helper.to_array(attr.t).astype(np.float32)
                    attr.t.CopyFrom(numpy_helper.from_array(array, attr.t.name))
                    converted_constants += 1
                    converted_double += 1
                elif attr.type == onnx.AttributeProto.TENSORS:
                    for tensor in attr.tensors:
                        if tensor.data_type == TensorProto.BFLOAT16:
                            array = bf16_to_float32_array(tensor)
                            tensor.CopyFrom(numpy_helper.from_array(array, tensor.name))
                            converted_constants += 1
                        elif tensor.data_type == TensorProto.DOUBLE:
                            array = numpy_helper.to_array(tensor).astype(np.float32)
                            tensor.CopyFrom(numpy_helper.from_array(array, tensor.name))
                            converted_constants += 1
                            converted_double += 1

    onnx.save_model(
        model,
        output_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=f"{output_path.name}.data",
        size_threshold=1024,
    )
    if args.check:
        onnx.checker.check_model(str(output_path))
    report = {
        "input_onnx": str(input_path),
        "output_onnx": str(output_path),
        "purpose": "ONNX Runtime compatibility shadow graph; BF16 tensors are represented/executed as FP32.",
        "converted_initializers": converted_initializers,
        "converted_value_infos": converted_value_infos,
        "converted_casts_to_bf16": converted_casts,
        "converted_bf16_constants": converted_constants,
        "converted_double_to_float": converted_double,
        "onnx_check": "ok" if args.check else "skipped",
    }
    with open(output_path.with_suffix(".ort_compat_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


def tensor_dict_from_outputs(names: list[str], outputs: tuple[torch.Tensor, ...]) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().to(torch.float32).cpu() for name, tensor in zip(names, outputs, strict=True)}


def run_pytorch(policy: nn.Module, inputs: tuple[torch.Tensor, ...]) -> dict[str, torch.Tensor]:
    wrapper = SmolVLADebugCoreWrapper(policy).to(device=str(policy.config.device)).eval()
    with torch.no_grad():
        outputs = wrapper(*inputs)
    return tensor_dict_from_outputs(wrapper.output_names, outputs)


def torch_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.to(torch.float32)
    return tensor.detach().cpu().numpy()


def run_onnxruntime(onnx_path: str, inputs: tuple[torch.Tensor, ...], provider: str) -> dict[str, torch.Tensor]:
    import onnxruntime as ort

    runner = ONNXRuntimeRunner(onnx_path, provider)
    return runner(inputs)


class ONNXRuntimeRunner:
    def __init__(self, onnx_path: str, provider: str):
        import onnxruntime as ort

        available = ort.get_available_providers()
        providers = [provider] if provider in available else []
        if "CPUExecutionProvider" in available:
            providers.append("CPUExecutionProvider")
        self.session = ort.InferenceSession(onnx_path, providers=providers or available)
        self.output_names = [output.name for output in self.session.get_outputs()]

    def __call__(self, inputs: tuple[torch.Tensor, ...]) -> dict[str, torch.Tensor]:
        feed = {name: torch_to_numpy(tensor) for name, tensor in zip(INPUT_NAMES, inputs, strict=True)}
        outputs = self.session.run(self.output_names, feed)
        return {
            name: torch.from_numpy(np.asarray(value)).to(torch.float32).cpu()
            for name, value in zip(self.output_names, outputs, strict=True)
        }


def torch_dtype_from_trt(dtype: Any) -> torch.dtype:
    import tensorrt as trt

    if dtype == trt.float32:
        return torch.float32
    if dtype == trt.float16:
        return torch.float16
    if hasattr(trt, "bfloat16") and dtype == trt.bfloat16:
        return torch.bfloat16
    if dtype == trt.int64:
        return torch.int64
    if dtype == trt.int32:
        return torch.int32
    if dtype == trt.int8:
        return torch.int8
    if dtype == trt.bool:
        return torch.bool
    raise TypeError(f"unsupported TensorRT dtype: {dtype}")


class TensorRTDebugRunner:
    def __init__(self, engine_path: str):
        import tensorrt as trt

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(engine_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        if engine is None:
            raise RuntimeError(f"failed to deserialize TensorRT engine: {engine_path}")
        self.trt = trt
        self.engine = engine
        self.context = engine.create_execution_context()
        self.tensor_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
        self.input_names = [
            name for name in self.tensor_names if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
        ]
        self.output_names = [
            name for name in self.tensor_names if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT
        ]
        self.input_shapes = {
            name: tuple(int(dim) for dim in engine.get_tensor_shape(name)) for name in self.input_names
        }

    def __call__(self, **inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        prepared = {}
        for name in self.input_names:
            expected = self.input_shapes[name]
            tensor = inputs[name]
            if tuple(tensor.shape) != expected:
                pad_value = False if tensor.dtype == torch.bool else 0
                tensor = match_static_shape(tensor, expected, pad_value)
            expected_dtype = torch_dtype_from_trt(self.engine.get_tensor_dtype(name))
            prepared[name] = tensor.to(device="cuda", dtype=expected_dtype).contiguous()
        for name, tensor in prepared.items():
            self.context.set_input_shape(name, tuple(tensor.shape))
            self.context.set_tensor_address(name, tensor.data_ptr())

        outputs = {}
        for name in self.output_names:
            output_shape = tuple(int(dim) for dim in self.context.get_tensor_shape(name))
            output_dtype = torch_dtype_from_trt(self.engine.get_tensor_dtype(name))
            output = torch.empty(output_shape, device="cuda", dtype=output_dtype)
            self.context.set_tensor_address(name, output.data_ptr())
            outputs[name] = output
        ok = self.context.execute_async_v3(stream_handle=torch.cuda.current_stream().cuda_stream)
        if not ok:
            raise RuntimeError("TensorRT execute_async_v3 returned false")
        torch.cuda.synchronize()
        return {name: tensor.detach().to(torch.float32).cpu() for name, tensor in outputs.items()}


def run_tensorrt(engine_path: str, inputs: tuple[torch.Tensor, ...]) -> dict[str, torch.Tensor]:
    if not torch.cuda.is_available():
        raise RuntimeError("TensorRT comparison requires CUDA, but torch.cuda.is_available() is false")
    runner = TensorRTDebugRunner(engine_path)
    return runner(**{name: tensor for name, tensor in zip(INPUT_NAMES, inputs, strict=True)})


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    length = min(a.numel(), b.numel())
    if length == 0:
        return float("nan")
    a = a.flatten()[:length]
    b = b.flatten()[:length]
    denom = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    if denom.item() == 0.0:
        return 1.0 if torch.allclose(a, b) else 0.0
    return float(torch.dot(a, b) / denom)


def compare_tensors(pair: str, name: str, a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    a = a.to(torch.float32).flatten()
    b = b.to(torch.float32).flatten()
    length = min(a.numel(), b.numel())
    a = a[:length]
    b = b[:length]
    diff = a - b
    a_norm = torch.linalg.vector_norm(a).clamp_min(1e-12)
    b_norm = torch.linalg.vector_norm(b)
    a_std = a.std(unbiased=False)
    b_std = b.std(unbiased=False)
    return {
        "pair": pair,
        "output": name,
        "numel_compared": int(length),
        "cosine_similarity": cosine_similarity(a, b),
        "relative_l2_error": float(torch.linalg.vector_norm(diff) / a_norm),
        "max_abs_error": float(diff.abs().max()) if length else float("nan"),
        "mean_abs_error": float(diff.abs().mean()) if length else float("nan"),
        "lhs_l2_norm": float(a_norm),
        "rhs_l2_norm": float(b_norm),
        "l2_norm_ratio": float(b_norm / a_norm),
        "lhs_mean": float(a.mean()) if length else float("nan"),
        "rhs_mean": float(b.mean()) if length else float("nan"),
        "mean_diff": float(b.mean() - a.mean()) if length else float("nan"),
        "lhs_std": float(a_std),
        "rhs_std": float(b_std),
        "std_ratio": float(b_std / a_std.clamp_min(1e-12)),
    }


def compare_outputs(pair: str, lhs: dict[str, torch.Tensor], rhs: dict[str, torch.Tensor]) -> list[dict[str, Any]]:
    rows = []
    for name in sorted(set(lhs) & set(rhs)):
        rows.append(compare_tensors(pair, name, lhs[name], rhs[name]))
    return rows


def p95(values: list[float]) -> float:
    finite = sorted(value for value in values if math.isfinite(value))
    if not finite:
        return float("nan")
    index = min(len(finite) - 1, int(math.ceil(0.95 * len(finite))) - 1)
    return float(finite[index])


def aggregate_metric_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = [
        "cosine_similarity",
        "relative_l2_error",
        "max_abs_error",
        "mean_abs_error",
        "l2_norm_ratio",
        "mean_diff",
        "std_ratio",
    ]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("pair") == "errors":
            continue
        grouped.setdefault((str(row["pair"]), str(row["output"])), []).append(row)

    aggregate_rows = []
    for (pair, output), group in sorted(grouped.items()):
        aggregate: dict[str, Any] = {
            "pair": pair,
            "output": output,
            "num_samples": len(group),
        }
        for metric in metrics:
            values = [float(row[metric]) for row in group if metric in row and math.isfinite(float(row[metric]))]
            if not values:
                continue
            aggregate[f"{metric}_mean"] = float(sum(values) / len(values))
            aggregate[f"{metric}_p95"] = p95(values)
            aggregate[f"{metric}_max"] = float(max(values))
            aggregate[f"{metric}_min"] = float(min(values))
        aggregate_rows.append(aggregate)
    return aggregate_rows


def write_reports(rows: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregate_rows = aggregate_metric_rows(rows)
    summary_rows = [
        row
        for row in aggregate_rows
        if row["output"] in {"action_chunk", "x_t_step_09", "v_t_step_09", "prefix_out"}
    ]
    summary = {
        "num_rows": len(rows),
        "num_aggregate_rows": len(aggregate_rows),
        "critical_outputs": summary_rows,
        "pairs": sorted({row["pair"] for row in rows}),
        "sample_count": max([int(row.get("sample_index", -1)) for row in rows if row.get("pair") != "errors"] or [-1])
        + 1,
    }
    with open(output_dir / "numeric_baseline_rows.json", "w") as f:
        json.dump(rows, f, indent=2)
    with open(output_dir / "numeric_baseline_aggregate_rows.json", "w") as f:
        json.dump(aggregate_rows, f, indent=2)
    with open(output_dir / "numeric_baseline_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(output_dir / "numeric_baseline_rows.csv", "w", newline="") as f:
        fieldnames = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with open(output_dir / "numeric_baseline_aggregate_rows.csv", "w", newline="") as f:
        fieldnames = sorted({key for row in aggregate_rows for key in row})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(aggregate_rows)
    print(json.dumps(summary, indent=2))


def compare_baselines(args: argparse.Namespace) -> None:
    input_dtype = torch.float32 if args.input_dtype == "fp32" else torch.bfloat16
    original, preprocessor, device = load_policy(args.policy_path, args.device, args.model_dtype)

    fake_quant, _, _ = load_policy(args.policy_path, args.device, args.model_dtype)
    quantize_regexes = tuple(args.quantize_module_regex) if args.quantize_module_regex else None
    quantize_prefixes = None if quantize_regexes else tuple(args.quantize_module_prefix) if args.quantize_module_prefix else None
    activation_scales_by_module = None
    if args.fake_quant_activation_scale_mode == "calibrated":
        if not args.fake_quant_activation_scales_json:
            raise ValueError(
                "--fake-quant-activation-scales-json is required when "
                "--fake-quant-activation-scale-mode calibrated"
            )
        activation_scales_by_module = load_activation_scales_by_module(
            args.fake_quant_activation_scales_json,
            include_regexes=quantize_regexes,
        )
    replace_linear_modules(
        fake_quant,
        include_prefixes=quantize_prefixes,
        include_regexes=quantize_regexes,
        activation_scales_by_module=activation_scales_by_module,
        quantization_kind=args.fake_quant_kind,
    )
    if args.extra_w8a16_module_regex:
        replace_linear_modules(
            fake_quant,
            include_prefixes=None,
            include_regexes=tuple(args.extra_w8a16_module_regex),
            activation_scales_by_module=None,
            quantization_kind="w8a16",
        )
    fake_quant.to(device=device).eval()

    if args.input_source == "synthetic":
        inputs = core_inputs_from_policy(original, preprocessor, args.task, args.seed, input_dtype, args.token_length)
        input_records = [
            {
                "sample_index": 0,
                "input_source": "synthetic",
                "task_group": args.task,
                "task_id": -1,
                "episode_seed": args.seed,
                "timestep": -1,
                "core_inputs": cpu_inputs(inputs),
            }
        ]
    else:
        input_records, rollout_device = collect_rollout_core_inputs(args, args.compare_samples)
        device = rollout_device

    runners: dict[str, Any] = {}
    optional_errors = {}
    ort_bf16_onnx = args.ort_bf16_onnx or args.bf16_onnx
    ort_int8_onnx = args.ort_int8_onnx or args.int8_onnx
    if args.comparison_set == "legacy":
        for label, path in [
            ("B_onnxruntime_native_mixed", ort_bf16_onnx),
            ("D_onnxruntime_int8_qdq", ort_int8_onnx),
        ]:
            if not path:
                continue
            try:
                runners[label] = ONNXRuntimeRunner(path, args.ort_provider)
            except Exception as exc:  # pragma: no cover - depends on local ORT build
                optional_errors[label] = repr(exc)

        for label, path in [("C_tensorrt_bf16", args.bf16_engine), ("D_tensorrt_int8_qdq", args.int8_engine)]:
            if not path:
                continue
            try:
                if not torch.cuda.is_available():
                    raise RuntimeError("TensorRT comparison requires CUDA, but torch.cuda.is_available() is false")
                runners[label] = TensorRTDebugRunner(path)
            except Exception as exc:  # pragma: no cover - depends on local TensorRT build
                optional_errors[label] = repr(exc)

        fake_quant_label = "E_pytorch_fake_quant"
        pair_defs = [
            ("A_vs_B", "A_pytorch_native_mixed", "B_onnxruntime_native_mixed"),
            ("B_vs_C", "B_onnxruntime_native_mixed", "C_tensorrt_bf16"),
            ("A_vs_E", "A_pytorch_native_mixed", "E_pytorch_fake_quant"),
            ("E_vs_D", "E_pytorch_fake_quant", "D_tensorrt_int8_qdq"),
            ("A_vs_D", "A_pytorch_native_mixed", "D_tensorrt_int8_qdq"),
            ("B_vs_D_ORT", "B_onnxruntime_native_mixed", "D_onnxruntime_int8_qdq"),
        ]
    else:
        if args.int8_engine:
            try:
                if not torch.cuda.is_available():
                    raise RuntimeError("TensorRT comparison requires CUDA, but torch.cuda.is_available() is false")
                runners["C_tensorrt_quant_engine"] = TensorRTDebugRunner(args.int8_engine)
            except Exception as exc:  # pragma: no cover - depends on local TensorRT build
                optional_errors["C_tensorrt_quant_engine"] = repr(exc)

        fake_quant_label = "B_pytorch_fake_quant"
        pair_defs = [
            ("A_vs_B", "A_pytorch_native_mixed", "B_pytorch_fake_quant"),
            ("B_vs_C", "B_pytorch_fake_quant", "C_tensorrt_quant_engine"),
            ("A_vs_C", "A_pytorch_native_mixed", "C_tensorrt_quant_engine"),
        ]

    rows: list[dict[str, Any]] = []

    for record in input_records:
        inputs = device_inputs(record["core_inputs"], device)
        baselines: dict[str, dict[str, torch.Tensor]] = {
            "A_pytorch_native_mixed": run_pytorch(original, inputs),
            fake_quant_label: run_pytorch(fake_quant, inputs),
        }
        for label, runner in runners.items():
            try:
                if isinstance(runner, TensorRTDebugRunner):
                    baselines[label] = runner(**{name: tensor for name, tensor in zip(INPUT_NAMES, inputs, strict=True)})
                else:
                    baselines[label] = runner(inputs)
            except Exception as exc:  # pragma: no cover - depends on backend/runtime
                optional_errors[f"{label}_sample_{record['sample_index']}"] = repr(exc)

        meta = {k: v for k, v in record.items() if k != "core_inputs"}
        for pair, lhs, rhs in pair_defs:
            if lhs in baselines and rhs in baselines:
                for row in compare_outputs(pair, baselines[lhs], baselines[rhs]):
                    rows.append({**meta, **row})

    if optional_errors:
        rows.append({"pair": "errors", "output": "optional_backend_errors", "errors": json.dumps(optional_errors)})
    write_reports(rows, Path(args.output_dir))


def main() -> None:
    args = parse_args()
    if args.command == "export":
        export_debug_onnx(args)
    elif args.command == "make-ort-compatible":
        make_ort_compatible_onnx(args)
    elif args.command == "compare":
        compare_baselines(args)
    else:
        raise ValueError(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()
