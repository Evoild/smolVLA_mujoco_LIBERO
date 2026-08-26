# Quantization Runtime Issues

本文档只记录量化实验中脚本运行、TensorRT 构建、Q/DQ 覆盖校验和环境相关的问题。核心实验结果和结论保留在 `README.md`。

## B2 Dynamic Activation Scale TensorRT Build Failure

B2 的动态 runtime activation scale ONNX 可以通过 `onnx.checker`，但 TensorRT 10.16.1 在 build engine 时失败。

典型错误：

```text
MyelinCheckException: wrap_attention_op_in_kgen.cpp:1441: CHECK(graph->ssa_validation()) failed
Could not find any implementation for node {ForeignNode[/action_in_proj/MatMul_15273_act0_Abs.../Slice_1767]}
Created engine with size: 0 MiB
Assertion failure: false && "Attempting to access an empty engine!"
```

同时出现：

```text
DequantizeLinear [SCALE] has invalid precision Int8, ignored
```

结论：当前 TensorRT 版本对 `Abs/ReduceMax/Div` 形式的 runtime dynamic scale Q/DQ 子图不稳定。后续改用离线 calibration 后的静态 per-node/per-call activation scale。

## 3-5 BF16 Builder Path Mismatch

3-5A 的 `trtexec` 没有显式添加 `--bf16`，但 3-5B/C/D/E/F 第一轮脚本沿用了 3-4 默认 BF16 engine 的 `--bf16` 参数。

影响：第一轮 3-5B/C/D/E/F 结果属于“显式 `--bf16` 构建路径”的对照，不能直接解释 3-5A 为什么恢复。

修正：

```text
deploy/3-5Bprecision-obey-bf16.sh
deploy/3-5Cno-tf32-bf16.sh
deploy/3-5Dopt-level0-bf16.sh
deploy/3-5run-bf16-ablation.sh
```

修正后脚本不再显式传 `--bf16`，只改变实验指定参数。修正口径下，`--precisionConstraints=obey` 已足够让 TensorRT native-mixed engine 与 ONNX Runtime 基本等价。

## 3-7B Down Weight-Only Overmatch

3-7B 目标是：

```text
gate/up: W8A8
down: W8A16 / weight-only
```

首次 Q/DQ report 中：

```text
rewritten_linear_nodes: 64
rewritten_weight_only_linear_nodes: 64
```

预期 `rewritten_weight_only_linear_nodes` 应为 `32`。原因是 `DOWN_NODE_REGEX="^/mlp/down_proj(_[0-9]+)?/MatMul$"` 只依赖 ONNX node name，越过 VLM text_model 后继续命中了 `lm_expert/action expert` 的 `down_proj`。

修正方式：增加停止边界。

```text
STOP_BEFORE_ACTION_REGEX="^/action_in_proj/"
--stop-before-node-regex "$STOP_BEFORE_ACTION_REGEX"
```

并增加覆盖断言：

```text
rewritten_linear_nodes == 64
rewritten_weight_only_linear_nodes == 32
```

修正后 Q/DQ report：

```text
rewritten_linear_nodes: 64
rewritten_weight_only_linear_nodes: 32
```

## E4 Action-Only Q/DQ Parse Failure

E4 action-only 首次构建时出现 TensorRT parse 失败：

```text
QuantizeLinear /debug_core/q_proj_31/MatMul_14333_act0_QuantizeLinear
Assertion failed: K == scaleSize
Number of output channels = 480, number of scales = 960
```

原因：action-only wrapper 让 ONNX 节点名前面多了 `/debug_core/`。旧的：

```text
^/action_in_proj/
```

没有匹配到：

```text
/debug_core/action_in_proj/
```

导致 Q/DQ 插入越过 `action_in_proj`，误量化 ODE 迭代中的 expert `q/k/v/o`。这些 expert MatMul 和 VLM text_model 的 channel 维度不同，因此出现 `K != scaleSize`。

修正：

```text
^/(debug_core/)?action_in_proj/
```

## E4 Action-Only Expected Coverage

E4 后续出现覆盖断言：

```text
unexpected 4E Q/DQ coverage: rewritten_linear_nodes got 219, expected 224
```

这不是 Q/DQ 插入失败，而是 expected 值沿用了 debug graph 的 `224`。action-only graph 只保留最终 `action_chunk`，ONNX DCE 后在第一个 `action_in_proj` 之前实际只保留：

```text
k_proj: 32
v_proj: 32
q_proj: 31
o_proj: 31
mlp.gate_proj: 31
mlp.up_proj: 31
mlp.down_proj: 31
total: 219
```

因此 E4 action-only 的正确 expected 是 `219`。

## Step 5A Old BF16-Cast Directory

`runs/deploy/5/A-cast-bf16-obey` 曾经跑到旧 debug export/build 路径，只生成：

```text
smolvla_debug_core_native_mixed.onnx
smolvla_debug_core_vlm_mlp_full_attn_w8a8_qdq.onnx
smolvla_debug_core_*_precision_obey.plan
```

但没有：

```text
trtexec_int8.log
trtexec_int8_profile.json
deploy_eval/int8_action_only/profile/profile_summary.json
```

该目录只能说明 Q/DQ 插入和 engine build 成功，不能用于 FPS 判断。有效的 BF16-cast 性能结果来自 E4 action-only 目录：

```text
runs/deploy/4/E4-fps-diagnose-mlp-full-attn-w8a8-action-only
```

## Step 5C Attention-Only Coverage Failure

5C attention-only 首次运行出现：

```text
unexpected 4E Q/DQ coverage:
rewritten_linear_nodes: got 72, expected 126
output_cast_nodes: got 72, expected 126
```

原因：Q/DQ 插入脚本在 calibrated mode 下按“第 N 个 Linear call”取 scale。attention-only 子集下，action-only ONNX 图和 calibration JSON 的 Linear 顺序不完全同序，部分 `o_proj` 错拿 `mlp.gate_proj` 的 module 名，随后被 `include_module_regex` 排除。

修正：对 VLM text 的规则化 ONNX node name 优先反推出 module name，再按 module name 读取 calibration scale。

```text
/debug_core/q_proj_7/MatMul        -> layers.7.self_attn.q_proj
/debug_core/o_proj_7/MatMul        -> layers.7.self_attn.o_proj
/debug_core/mlp/down_proj_7/MatMul -> layers.7.mlp.down_proj
```

修复后验证：

```text
MLP-only:       rewritten_linear_nodes 93 / expected 93
Attention-only: rewritten_linear_nodes 126 / expected 126
```

## Step 5D Engine Inspector Requires CUDA

在无 CUDA 设备的 shell 中执行：

```bash
trtexec --loadEngine=... --dumpLayerInfo --exportLayerInfo=...
```

会失败：

```text
no CUDA-capable device is detected
```

因此 TensorRT layer precision 检查必须在能够正常运行 TensorRT engine 的同一 GPU 环境里执行。
