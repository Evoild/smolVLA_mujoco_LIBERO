#!/usr/bin/env python3

"""Run SmolVLA eval/profile with TensorRT replacing only sample_actions core."""

from __future__ import annotations

import argparse
import ctypes
import csv
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn

from lerobot.configs import FeatureType, PreTrainedConfig
from lerobot.envs import close_envs, make_env, make_env_pre_post_processors
from lerobot.envs.factory import make_env_config
from lerobot.policies import make_pre_post_processors
from lerobot.policies.factory import get_policy_class
from lerobot.scripts.lerobot_eval import eval_policy_all
from lerobot.utils.constants import OBS_STATE
from lerobot.utils.random_utils import set_seed


try:
    import tensorrt as trt
except ImportError:  # pragma: no cover - depends on deployment machine
    trt = None


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--policy-path", required=True)
    common.add_argument("--engine-path", default=None)
    common.add_argument("--backend", choices=["pytorch", "trt", "trt-int8"], required=True)
    common.add_argument("--device", default="cuda")
    common.add_argument(
        "--trt-output-name",
        default="action_chunk",
        help="TensorRT output tensor to read when the engine exposes debug outputs.",
    )
    common.add_argument(
        "--use-cuda-graph",
        action="store_true",
        help="Capture/replay the TensorRT engine execution with CUDA Graph. Requires static shapes.",
    )
    common.add_argument(
        "--trt-plugin-library",
        action="append",
        default=[],
        help="TensorRT plugin .so to load before deserializing the engine. May be repeated.",
    )

    profile = subparsers.add_parser("profile", parents=[common])
    profile.add_argument("--output-dir", required=True)
    profile.add_argument("--task", default="libero_goal TensorRT INT8 sample_actions core profile")
    profile.add_argument("--warmup", type=int, default=5)
    profile.add_argument("--iters", type=int, default=30)

    eval_parser = subparsers.add_parser("eval", parents=[common])
    eval_parser.add_argument("--output-dir", required=True)
    eval_parser.add_argument("--tasks", default="libero_goal")
    eval_parser.add_argument("--seed", type=int, default=1000)
    eval_parser.add_argument("--episodes", type=int, default=10)
    eval_parser.add_argument("--batch-size", type=int, default=1)
    eval_parser.add_argument("--max-parallel-tasks", type=int, default=1)

    return parser.parse_args()


def torch_dtype_from_trt(dtype: Any) -> torch.dtype:
    if trt is not None and dtype == trt.float32:
        return torch.float32
    if trt is not None and dtype == trt.float16:
        return torch.float16
    if trt is not None and hasattr(trt, "bfloat16") and dtype == trt.bfloat16:
        return torch.bfloat16
    if trt is not None and dtype == trt.int64:
        return torch.int64
    if trt is not None and dtype == trt.int32:
        return torch.int32
    if trt is not None and dtype == trt.int8:
        return torch.int8
    if trt is not None and dtype == trt.bool:
        return torch.bool
    raise TypeError(f"unsupported TensorRT dtype: {dtype}")


class TensorRTCoreRunner:
    def __init__(
        self,
        engine_path: str,
        output_name: str = "action_chunk",
        use_cuda_graph: bool = False,
        plugin_libraries: list[str] | None = None,
    ):
        if trt is None:
            raise ImportError("tensorrt Python package is required for --backend trt-int8")
        logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(logger, "")
        self.plugin_libraries = []
        for library in plugin_libraries or []:
            handle = ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)
            self.plugin_libraries.append(handle)
        runtime = trt.Runtime(logger)
        with open(engine_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        if engine is None:
            raise RuntimeError(f"failed to deserialize TensorRT engine: {engine_path}")

        self.engine = engine
        self.context = engine.create_execution_context()
        self.tensor_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
        self.input_names = [
            name for name in self.tensor_names if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
        ]
        self.output_names = [
            name for name in self.tensor_names if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT
        ]
        if output_name in self.output_names:
            self.output_name = output_name
        elif len(self.output_names) == 1:
            self.output_name = self.output_names[0]
        else:
            raise RuntimeError(f"output {output_name!r} not found in TensorRT outputs: {self.output_names}")
        self.input_shapes = {
            name: tuple(int(dim) for dim in engine.get_tensor_shape(name)) for name in self.input_names
        }
        self.use_cuda_graph = bool(use_cuda_graph)
        self.cuda_graph: torch.cuda.CUDAGraph | None = None
        self.cuda_graph_failed: str | None = None
        self.static_inputs: dict[str, torch.Tensor] = {}
        self.static_outputs: dict[str, torch.Tensor] = {}
        self._static_io_ready = False

    def _prepare_input(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        expected_dtype = torch_dtype_from_trt(self.engine.get_tensor_dtype(name))
        tensor = tensor.to(device="cuda", dtype=expected_dtype)
        return tensor.contiguous()

    def _allocate_static_io(self, prepared: dict[str, torch.Tensor]) -> None:
        for name, tensor in prepared.items():
            self.context.set_input_shape(name, tuple(tensor.shape))
            self.static_inputs[name] = torch.empty_like(tensor)
            self.context.set_tensor_address(name, self.static_inputs[name].data_ptr())

        for output_name in self.output_names:
            output_shape = tuple(int(dim) for dim in self.context.get_tensor_shape(output_name))
            output_dtype = torch_dtype_from_trt(self.engine.get_tensor_dtype(output_name))
            output = torch.empty(output_shape, device="cuda", dtype=output_dtype)
            self.context.set_tensor_address(output_name, output.data_ptr())
            self.static_outputs[output_name] = output
        self._static_io_ready = True

    def _copy_to_static_inputs(self, prepared: dict[str, torch.Tensor]) -> None:
        if not self._static_io_ready:
            self._allocate_static_io(prepared)
        for name, tensor in prepared.items():
            static_tensor = self.static_inputs[name]
            if tuple(tensor.shape) != tuple(static_tensor.shape):
                raise ValueError(
                    f"CUDA Graph TensorRT runner requires static shape for {name}: "
                    f"got {tuple(tensor.shape)}, expected {tuple(static_tensor.shape)}"
                )
            if tensor.dtype != static_tensor.dtype:
                raise TypeError(f"CUDA Graph TensorRT runner dtype changed for {name}: {tensor.dtype} vs {static_tensor.dtype}")
            static_tensor.copy_(tensor, non_blocking=True)

    def _execute_direct(self, outputs: dict[str, torch.Tensor]) -> torch.Tensor:
        ok = self.context.execute_async_v3(stream_handle=torch.cuda.current_stream().cuda_stream)
        if not ok:
            raise RuntimeError("TensorRT execute_async_v3 returned false")
        return outputs[self.output_name]

    def _capture_cuda_graph(self) -> None:
        # TensorRT may allocate internal resources on first enqueue. Do one direct run before graph capture.
        ok = self.context.execute_async_v3(stream_handle=torch.cuda.current_stream().cuda_stream)
        if not ok:
            raise RuntimeError("TensorRT execute_async_v3 returned false before CUDA Graph capture")
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            ok = self.context.execute_async_v3(stream_handle=torch.cuda.current_stream().cuda_stream)
            if not ok:
                raise RuntimeError("TensorRT execute_async_v3 returned false during CUDA Graph capture")
        self.cuda_graph = graph

    def _execute_cuda_graph(self) -> torch.Tensor:
        if self.cuda_graph is None:
            try:
                self._capture_cuda_graph()
            except Exception as exc:
                self.cuda_graph_failed = repr(exc)
                self.use_cuda_graph = False
                return self._execute_direct(self.static_outputs)
        assert self.cuda_graph is not None
        self.cuda_graph.replay()
        return self.static_outputs[self.output_name]

    def __call__(self, **inputs: torch.Tensor) -> torch.Tensor:
        prepared = {name: self._prepare_input(name, inputs[name]) for name in self.input_names}
        if self.use_cuda_graph:
            self._copy_to_static_inputs(prepared)
            return self._execute_cuda_graph()

        for name, tensor in prepared.items():
            self.context.set_input_shape(name, tuple(tensor.shape))
            self.context.set_tensor_address(name, tensor.data_ptr())

        outputs = {}
        for output_name in self.output_names:
            output_shape = tuple(int(dim) for dim in self.context.get_tensor_shape(output_name))
            output_dtype = torch_dtype_from_trt(self.engine.get_tensor_dtype(output_name))
            output = torch.empty(output_shape, device="cuda", dtype=output_dtype)
            self.context.set_tensor_address(output_name, output.data_ptr())
            outputs[output_name] = output
        return self._execute_direct(outputs)


def load_policy(policy_path: str, device: str):
    cfg = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=[f"--device={device}"])
    policy_cls = get_policy_class(cfg.type)
    policy = policy_cls.from_pretrained(policy_path, config=cfg, strict=False)
    policy.eval()
    effective_device = str(policy.config.device)
    policy.to(effective_device)
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=policy_path,
        preprocessor_overrides={"device_processor": {"device": effective_device}},
    )
    return policy, (preprocessor, postprocessor), effective_device


def module_parameter_memory_mb(module: nn.Module) -> float:
    return sum(param.numel() * param.element_size() for param in module.parameters()) / (1024**2)


def install_trt_sample_actions(
    policy: nn.Module,
    engine_path: str,
    output_name: str = "action_chunk",
    backend_label: str = "trt-int8",
    use_cuda_graph: bool = False,
    plugin_libraries: list[str] | None = None,
) -> dict[str, Any]:
    runner = TensorRTCoreRunner(
        engine_path,
        output_name=output_name,
        use_cuda_graph=use_cuda_graph,
        plugin_libraries=plugin_libraries,
    )
    model = policy.model
    action_dim = int(policy.config.action_feature.shape[0])

    def match_static_shape(tensor: torch.Tensor, name: str, pad_value: int | float | bool = 0) -> torch.Tensor:
        expected = runner.input_shapes[name]
        if tuple(tensor.shape) == expected:
            return tensor
        if len(tensor.shape) != len(expected):
            raise ValueError(f"{name} rank mismatch: got {tuple(tensor.shape)}, expected {expected}")

        slices = tuple(slice(0, min(int(got), int(want))) for got, want in zip(tensor.shape, expected, strict=True))
        output = torch.full(expected, pad_value, dtype=tensor.dtype, device=tensor.device)
        output[slices] = tensor[slices]
        return output

    def trt_sample_actions(images, img_masks, lang_tokens, lang_masks, state, noise=None, **_kwargs):
        bsize = state.shape[0]
        if bsize != 1:
            raise ValueError("static TensorRT engine currently expects batch_size=1")
        if noise is None:
            actions_shape = (bsize, model.config.chunk_size, model.config.max_action_dim)
            noise = model.sample_noise(actions_shape, state.device)

        image_emb = model.vlm_with_expert.embed_image(images[0])
        image2_emb = model.vlm_with_expert.embed_image(images[1])
        action_chunk = runner(
            image_emb=image_emb,
            image2_emb=image2_emb,
            image_mask=img_masks[0],
            image2_mask=img_masks[1],
            state=state,
            language_tokens=match_static_shape(lang_tokens, "language_tokens", 0),
            language_attention_mask=match_static_shape(lang_masks, "language_attention_mask", False),
            noise=noise,
        )
        return action_chunk[:, :, :action_dim].to(dtype=torch.float32)

    model.sample_actions = trt_sample_actions
    return {
        "backend": backend_label,
        "engine_path": engine_path,
        "trt_engine_file_size_mb": Path(engine_path).stat().st_size / (1024**2),
        "replacement": "policy.model.sample_actions core",
        "trt_inputs": runner.input_names,
        "trt_outputs": runner.output_names,
        "trt_output_used": runner.output_name,
        "use_cuda_graph": use_cuda_graph,
        "trt_plugin_libraries": plugin_libraries or [],
    }


def maybe_install_backend(
    policy: nn.Module,
    backend: str,
    engine_path: str | None,
    trt_output_name: str = "action_chunk",
    use_cuda_graph: bool = False,
    plugin_libraries: list[str] | None = None,
) -> dict[str, Any]:
    if backend == "pytorch":
        return {"backend": "pytorch", "replacement": "none"}
    if not engine_path:
        raise ValueError("--engine-path is required for TensorRT backends")
    return install_trt_sample_actions(
        policy,
        engine_path,
        output_name=trt_output_name,
        backend_label=backend,
        use_cuda_graph=use_cuda_graph,
        plugin_libraries=plugin_libraries,
    )


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


def clone_batch(batch: dict[str, Any]) -> dict[str, Any]:
    return {key: value.clone() if torch.is_tensor(value) else value for key, value in batch.items()}


def sync(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def peak_memory_gb(device: str) -> float | None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024**3)
    return None


def summarize(values: list[float], prefix: str) -> dict[str, float]:
    sorted_values = sorted(values)
    return {
        f"{prefix}_mean_ms": statistics.fmean(values),
        f"{prefix}_median_ms": statistics.median(values),
        f"{prefix}_p95_ms": sorted_values[max(0, int(len(sorted_values) * 0.95) - 1)],
    }


def run_profile(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    policy, (preprocessor, _), device = load_policy(args.policy_path, args.device)
    backend_report = maybe_install_backend(
        policy,
        args.backend,
        args.engine_path,
        args.trt_output_name,
        use_cuda_graph=args.use_cuda_graph,
        plugin_libraries=args.trt_plugin_library,
    )
    raw_observation = make_raw_observation(policy.config, args.task)
    batch = preprocessor(raw_observation)

    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        for _ in range(args.warmup):
            policy.predict_action_chunk(clone_batch(batch))
    sync(device)

    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for idx in range(args.iters):
            current = clone_batch(batch)
            start = time.perf_counter()
            policy.predict_action_chunk(current)
            sync(device)
            rows.append({"iteration": idx, "policy_inference_e2e_ms": (time.perf_counter() - start) * 1000.0})

    latencies = [float(row["policy_inference_e2e_ms"]) for row in rows]
    summary: dict[str, Any] = {
        **backend_report,
        "device": device,
        "profile_entrypoint": "policy.predict_action_chunk",
        "profile_reason": "training policy.forward does not call sample_actions; deployment inference does",
        "pytorch_parameter_memory_mb": module_parameter_memory_mb(policy),
        "peak_gpu_memory_gb": peak_memory_gb(device),
        "policy_inference_fps": 1000.0 / statistics.fmean(latencies),
    }
    if summary["peak_gpu_memory_gb"] is not None:
        summary["peak_gpu_memory_mb"] = float(summary["peak_gpu_memory_gb"]) * 1024.0
    summary.update(summarize(latencies, "policy_inference_e2e"))

    with open(output_dir / "profile_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(output_dir / "profile_iterations.json", "w") as f:
        json.dump(rows, f, indent=2)
    with open(output_dir / "profile_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)
    with open(output_dir / "profile_iterations.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))


def run_eval(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    policy, (preprocessor, postprocessor), device = load_policy(args.policy_path, args.device)
    backend_report = maybe_install_backend(
        policy,
        args.backend,
        args.engine_path,
        args.trt_output_name,
        use_cuda_graph=args.use_cuda_graph,
        plugin_libraries=args.trt_plugin_library,
    )
    env_cfg = make_env_config("libero", task=args.tasks, max_parallel_tasks=args.max_parallel_tasks)
    envs = make_env(env_cfg, n_envs=args.batch_size, use_async_envs=False, trust_remote_code=False)
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy.config)

    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    try:
        with torch.no_grad():
            info = eval_policy_all(
                envs=envs,
                policy=policy,
                env_preprocessor=env_preprocessor,
                env_postprocessor=env_postprocessor,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                n_episodes=args.episodes,
                max_episodes_rendered=10,
                videos_dir=output_dir / "videos",
                start_seed=args.seed,
                max_parallel_tasks=args.max_parallel_tasks,
        )
        info["deployment"] = backend_report
        info["pytorch_parameter_memory_mb"] = module_parameter_memory_mb(policy)
        info["peak_gpu_memory_gb"] = peak_memory_gb(device)
        if info["peak_gpu_memory_gb"] is not None:
            info["peak_gpu_memory_mb"] = float(info["peak_gpu_memory_gb"]) * 1024.0
        with open(output_dir / "eval_info.json", "w") as f:
            json.dump(info, f, indent=2)
        print(json.dumps(info.get("overall", info), indent=2))
    finally:
        close_envs(envs)


def main() -> None:
    args = parse_args()
    if args.command == "profile":
        run_profile(args)
    elif args.command == "eval":
        run_eval(args)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
