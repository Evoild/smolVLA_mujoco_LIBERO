#!/usr/bin/env python3

"""Export SmolVLA sample_actions to a static-shape ONNX graph for TensorRT smoke tests."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import torch
from torch import nn

from lerobot.configs import PreTrainedConfig
from lerobot.policies.factory import get_policy_class


def make_att_2d_masks_trt(pad_masks: torch.Tensor, att_masks: torch.Tensor) -> torch.Tensor:
    cumsum = torch.cumsum(att_masks.to(torch.int64), dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] & pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


class SmolVLASampleActionsWrapper(nn.Module):
    """Thin tensor-only wrapper around VLAFlowMatching.sample_actions.

    Inputs are already preprocessed:
    - images are resized and normalized to the policy's VLM range
    - state is padded to max_state_dim
    - language tokens/masks are padded to token_length
    """

    def __init__(self, policy: nn.Module):
        super().__init__()
        self.model = policy.model
        self.action_dim = int(policy.config.action_feature.shape[0])

    def forward(
        self,
        image: torch.Tensor,
        image2: torch.Tensor,
        image_mask: torch.Tensor,
        image2_mask: torch.Tensor,
        state: torch.Tensor,
        language_tokens: torch.Tensor,
        language_attention_mask: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        actions = self.model.sample_actions(
            images=[image, image2],
            img_masks=[image_mask, image2_mask],
            lang_tokens=language_tokens,
            lang_masks=language_attention_mask,
            state=state,
            noise=noise,
        )
        return actions[:, :, : self.action_dim]


class SmolVLASampleActionsFromEmbedsWrapper(nn.Module):
    """Tensor-only sample_actions core with image embeddings as static inputs."""

    def __init__(self, policy: nn.Module):
        super().__init__()
        self.model = policy.model
        self.action_dim = int(policy.config.action_feature.shape[0])
        self.chunk_size = int(policy.config.chunk_size)

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
        bsize = state.shape[0]
        device = state.device
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

        _, past_key_values = self.model.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.model.config.use_cache,
            fill_kv_cache=True,
        )

        num_steps = self.model.config.num_steps
        dt = -1.0 / num_steps
        x_t = noise
        for step in range(num_steps):
            time = 1.0 + step * dt
            time_tensor = torch.tensor(time, dtype=x_t.dtype, device=device).expand(bsize)
            v_t = self.denoise_step_trt(
                x_t=x_t,
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values,
                timestep=time_tensor,
            )
            x_t = x_t + dt * v_t
        return x_t[:, :, : self.action_dim]

    def denoise_step_trt(
        self,
        prefix_pad_masks: torch.Tensor,
        past_key_values: dict,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
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
        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.chunk_size :]
        suffix_out = suffix_out.to(dtype=self.model.action_out_proj.weight.dtype)
        return self.model.action_out_proj(suffix_out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", required=True)
    parser.add_argument("--output", default="runs/deploy/tensorrt/smolvla_sample_actions.onnx")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument(
        "--input-mode",
        choices=["image-embeds", "images"],
        default="image-embeds",
        help="Export from image embeddings by default; images mode includes the vision encoder.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--token-length", type=int, default=None)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--check", action="store_true", help="Run onnx.checker if onnx is installed.")
    parser.add_argument("--build-engine", action="store_true", help="Run trtexec if available after export.")
    parser.add_argument("--engine-output", default=None)
    return parser.parse_args()


def load_policy(policy_path: str, device: str, dtype: torch.dtype):
    cfg = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=[f"--device={device}"])
    policy_cls = get_policy_class(cfg.type)
    policy = policy_cls.from_pretrained(policy_path, config=cfg, strict=False)
    effective_device = str(policy.config.device)
    policy.to(device=effective_device)
    policy.to(dtype=dtype)
    policy.eval()
    return policy, effective_device


def make_inputs(policy: nn.Module, batch_size: int, token_length: int, device: str, dtype: torch.dtype):
    config = policy.config
    image_h, image_w = tuple(int(v) for v in config.resize_imgs_with_padding)
    max_state_dim = int(config.max_state_dim)
    chunk_size = int(config.chunk_size)
    max_action_dim = int(config.max_action_dim)
    pad_id = int(policy.model.vlm_with_expert.processor.tokenizer.pad_token_id or 0)

    image = torch.zeros((batch_size, 3, image_h, image_w), dtype=dtype, device=device)
    image2 = torch.zeros((batch_size, 3, image_h, image_w), dtype=dtype, device=device)
    image_mask = torch.ones((batch_size,), dtype=torch.bool, device=device)
    image2_mask = torch.ones((batch_size,), dtype=torch.bool, device=device)
    state = torch.zeros((batch_size, max_state_dim), dtype=dtype, device=device)
    language_tokens = torch.full((batch_size, token_length), pad_id, dtype=torch.long, device=device)
    language_attention_mask = torch.ones((batch_size, token_length), dtype=torch.bool, device=device)
    noise = torch.zeros((batch_size, chunk_size, max_action_dim), dtype=dtype, device=device)
    return (
        image,
        image2,
        image_mask,
        image2_mask,
        state,
        language_tokens,
        language_attention_mask,
        noise,
    )


def make_image_embed_inputs(
    policy: nn.Module, batch_size: int, token_length: int, device: str, dtype: torch.dtype
):
    image_inputs = make_inputs(policy, batch_size, token_length, device, dtype)
    image, image2, image_mask, image2_mask, state, language_tokens, language_attention_mask, noise = image_inputs
    with torch.no_grad():
        image_emb = policy.model.vlm_with_expert.embed_image(image)
        image2_emb = policy.model.vlm_with_expert.embed_image(image2)
    image_emb = image_emb.to(dtype)
    image2_emb = image2_emb.to(dtype)
    return (
        image_emb,
        image2_emb,
        image_mask,
        image2_mask,
        state,
        language_tokens,
        language_attention_mask,
        noise,
    )


def maybe_check_onnx(path: Path) -> str:
    try:
        import onnx
    except ImportError:
        return "skipped_no_onnx_package"
    model = onnx.load(path)
    onnx.checker.check_model(model)
    return "ok"


def maybe_build_engine(onnx_path: Path, engine_path: Path) -> dict[str, Any]:
    trtexec = subprocess.run(
        ["bash", "-lc", "command -v trtexec"],
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not trtexec:
        return {"status": "skipped_no_trtexec"}

    command = [
        trtexec,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    return {
        "status": "ok" if result.returncode == 0 else "failed",
        "returncode": result.returncode,
        "command": command,
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-4000:],
    }


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16

    policy, device = load_policy(args.policy_path, args.device, dtype)
    token_length = int(args.token_length or policy.config.tokenizer_max_length)
    if args.input_mode == "images":
        wrapper = SmolVLASampleActionsWrapper(policy).to(device=device).eval()
        inputs = make_inputs(policy, args.batch_size, token_length, device, dtype)
        input_names = [
            "image",
            "image2",
            "image_mask",
            "image2_mask",
            "state",
            "language_tokens",
            "language_attention_mask",
            "noise",
        ]
    else:
        wrapper = SmolVLASampleActionsFromEmbedsWrapper(policy).to(device=device).eval()
        inputs = make_image_embed_inputs(policy, args.batch_size, token_length, device, dtype)
        input_names = [
            "image_emb",
            "image2_emb",
            "image_mask",
            "image2_mask",
            "state",
            "language_tokens",
            "language_attention_mask",
            "noise",
        ]

    with torch.no_grad():
        reference = wrapper(*inputs)

    torch.onnx.export(
        wrapper,
        inputs,
        output,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
        input_names=input_names,
        output_names=["action_chunk"],
        dynamic_axes=None,
    )

    report: dict[str, Any] = {
        "onnx_path": str(output),
        "device": device,
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "token_length": token_length,
        "input_mode": args.input_mode,
        "input_shapes": {name: list(tensor.shape) for name, tensor in zip(input_names, inputs, strict=True)},
        "opset": args.opset,
        "output_shape": list(reference.shape),
        "onnx_check": maybe_check_onnx(output) if args.check else "skipped",
    }
    if args.build_engine:
        engine_output = Path(args.engine_output or output.with_suffix(".plan"))
        report["engine"] = maybe_build_engine(output, engine_output)

    report_path = output.with_suffix(".export_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
