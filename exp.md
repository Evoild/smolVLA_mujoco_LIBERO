# TensorRT 量化部署标准闭环流程

## 1. 总体思路

TensorRT 量化部署并不是：

```text
模型 → INT8 → Engine → 完成
```

而应该是一个不断迭代的闭环：

```text
Baseline
    ↓
TensorRT FP16/BF16 Baseline
    ↓
量化方案设计
    ↓
Calibration
    ↓
INT8 Engine
    ↓
┌──────────────────────────────┐
│                              │
精度验证                     性能验证
│                              │
↓                              ↓
误差定位                     性能 Profiling
│                              │
↓                              ↓
精度优化                     性能优化
│                              │
└──────────────┬───────────────┘
               ↓
          重新构建 Engine
               ↓
          端到端任务评测
               ↓
          是否满足部署要求？
          ↓              ↓
         否              是
         ↓               ↓
       继续迭代          部署
```

因此整个量化部署可以理解为两个核心闭环：

### 精度闭环

```text
Calibration
    ↓
Quantization
    ↓
Error Analysis
    ↓
Precision Optimization
    ↓
Re-Quantization
```

### 性能闭环

```text
INT8 Engine
    ↓
Performance Profiling
    ↓
Kernel / Fusion Analysis
    ↓
Graph / Runtime Optimization
    ↓
Re-Benchmark
```

---

# 2. 阶段一：建立 Baseline

## 2.1 固定实验环境

在开始量化之前，需要固定：

- GPU
- CUDA
- TensorRT
- batch size
- input shape
- sequence length
- calibration dataset
- evaluation dataset
- 推理输入
- random seed
- benchmark warmup / iteration 数量

否则不同量化方案之间无法公平比较。

---

## 2.2 建立 PyTorch Baseline

记录原始模型：

- FP32 / BF16 / FP16
- Latency
- FPS / Throughput
- Peak GPU Memory
- Model Size
- Accuracy / Success Rate

对于 VLA，还建议记录：

- Action Chunk error
- First Action error
- Flow Matching 每一步误差
- Task Success Rate

---

## 2.3 建立 TensorRT FP16/BF16 Baseline

不要直接拿：

```text
PyTorch BF16
vs
TensorRT INT8
```

判断 INT8 加速效果。

应该首先建立：

```text
PyTorch BF16
      ↓
TensorRT BF16/FP16
      ↓
TensorRT INT8
```

其中真正用于判断量化收益的是：

```text
TensorRT BF16/FP16
        vs
TensorRT INT8
```

因为 TensorRT BF16/FP16 本身已经包含：

- Layer Fusion
- Constant Folding
- Kernel/Tactic Selection
- Memory Optimization
- Graph Optimization

这样才能把：

> TensorRT 本身的优化收益

和：

> INT8 量化本身的收益

区分开。

---

# 3. 阶段二：设计量化方案

## 3.1 确定量化对象

例如 Transformer/VLA 中：

### Attention

```text
q_proj
k_proj
v_proj
o_proj
```

### FFN

```text
gate_proj
up_proj
down_proj
```

### 其他

```text
Conv
Embedding
LM Head
Action Head
State Projection
Action Projection
```

不要默认所有层都必须 INT8。

---

## 3.2 确定 Weight 量化粒度

常见方案：

```text
per-tensor
per-channel
per-group
```

例如：

```text
Weight:
INT8 symmetric
per-output-channel
```

---

## 3.3 确定 Activation 量化粒度

常见：

```text
per-tensor
per-channel
per-group
```

例如 TensorRT GEMM 常见部署方案：

```text
Weight:
INT8 per-output-channel

Activation:
INT8 per-tensor
```

---

## 3.4 确定精度组合

例如：

```text
W8A8
W8A16
W4A8
W4A16
BF16
```

也可以进行 Mixed Precision：

```text
q_proj       W8A8
k_proj       W8A8
v_proj       W8A8
o_proj       W8A8

gate_proj    W8A8
up_proj      W8A8
down_proj    W8A16
```

---

# 4. 阶段三：Calibration

## 4.1 Calibration 的目标

Calibration 的核心任务是：

```text
真实输入
    ↓
模型 Forward
    ↓
收集 Activation
    ↓
统计 Activation Distribution
    ↓
确定 Dynamic Range
    ↓
确定 Quantization Scale
```

---

## 4.2 Dynamic Range

例如某层 activation：

```text
[-6.2, 6.2]
```

那么这个范围就是量化时希望 INT8 覆盖的动态范围。

对于对称 INT8：

\[
s = \frac{\max |x|}{127}
\]

例如：

\[
s=\frac{6.2}{127}
\]

量化：

\[
q=\mathrm{clip}
\left(
\mathrm{round}(x/s),
-128,
127
\right)
\]

反量化：

\[
\hat{x}=q\cdot s
\]

---

## 4.3 Dynamic Range 的确定方法

常见：

### MinMax

```text
T = max(abs(x))
```

### Percentile

```text
T = percentile(abs(x), 99.99)
```

例如：

```text
p99.9
p99.99
p99.999
```

### Entropy / KL Calibration

寻找使原始分布与量化分布之间差异较小的 threshold。

---

## 4.4 Calibration Dataset

Calibration Dataset 不一定越大越好。

更重要的是：

> 与真实 inference distribution 匹配。

例如 VLA 应该覆盖：

```text
不同任务
不同机器人状态
不同图像
不同 trajectory phase
不同语言指令
不同环境状态
```

而不是只随机抽一些图片。

---

# 5. 阶段四：生成 INT8 计算图

## 5.1 TensorRT Calibration 路线

传统流程：

```text
FP Model
    ↓
TensorRT INT8 Builder
    ↓
Calibrator
    ↓
Dynamic Range
    ↓
INT8 Engine
```

---

## 5.2 Explicit Q/DQ 路线

现代 ONNX / TensorRT 量化常使用：

```text
FP Activation
     ↓
QuantizeLinear
     ↓
INT8
     ↓
DequantizeLinear
     ↓
FP Representation
     ↓
Linear / GEMM
```

也就是：

```text
Q → DQ → Linear
```

Q/DQ 节点实际上向 TensorRT 描述：

```text
这个 Tensor 应该使用什么 scale
以什么粒度进行量化
在哪里进入/退出量化区域
```

TensorRT 再根据这些信息选择低精度 kernel。

---

# 6. 阶段五：精度验证

不要生成 INT8 Engine 后直接只跑最终 Success Rate。

首先应该做数值对齐。

---

## 6.1 固定输入比较

保证：

```text
相同 Input
相同 Noise
相同 Seed
相同 Model State
```

比较：

```text
BF16 Output
vs
INT8 Output
```

---

## 6.2 常用误差指标

### Cosine Similarity

衡量方向：

\[
\cos(x,\hat{x})
=
\frac{x\cdot\hat{x}}
{\|x\|\|\hat{x}\|}
\]

### Relative L2 Error

\[
E_{rel}
=
\frac{\|\hat{x}-x\|_2}
{\|x\|_2}
\]

### MAE

\[
MAE
=
\frac{1}{N}
\sum_i |\hat{x}_i-x_i|
\]

### Norm Ratio

\[
R=
\frac{\|\hat{x}\|_2}
{\|x\|_2}
\]

还可以记录：

```text
MSE
RMSE
Max Absolute Error
SQNR
```

---

# 7. 阶段六：精度损失定位

如果 INT8 精度下降，不要直接：

```text
换 percentile
→ 再跑
→ 不行
→ 再换 percentile
```

应该首先定位：

\[
\boxed{\text{误差到底从哪里产生}}
\]

---

## 7.1 逐层 Error Profiling

例如：

```text
Layer 0
 ↓
Layer 1
 ↓
Layer 2
 ↓
...
 ↓
Layer 31
```

统计：

```text
Input Relative L2
Output Relative L2
Cosine
Norm Ratio
SQNR
```

寻找：

```text
First Large Error Jump
```

---

## 7.2 区分 Weight Error 与 Activation Error

例如：

```text
BF16
W8A16
W8A8
```

如果：

```text
BF16 ≈ W8A16
W8A8 明显恶化
```

说明：

```text
Weight W8 不是主要问题
Activation A8 才是主要问题
```

---

# 8. Activation A8 常见问题

## 8.1 Clipping Dominated

Scale 太小：

```text
真实 activation：

[-100, 100]

量化范围：

[-10, 10]
```

于是：

```text
80  → 10
100 → 10
```

大量重要 activation 被 clipping。

解决方向：

```text
增大 Dynamic Range
提高 percentile
更好的 Calibration Dataset
```

---

## 8.2 Resolution Dominated

如果：

```text
普通 channel:
[-2, 2]

outlier channel:
[-100, 100]
```

per-tensor INT8 为了容纳 100：

\[
scale=\frac{100}{127}\approx0.787
\]

普通值：

```text
0.1
0.2
0.3
```

就很难精确表示。

这就是：

```text
Quantization Resolution Loss
```

---

# 9. Channel Dynamic-Range Imbalance

如果不同 channel：

```text
channel 0: [-2, 2]
channel 1: [-1, 2]
channel 2: [-3, 3]
channel 3: [-100, 100]
```

那么存在：

```text
Channel Dynamic-Range Imbalance
```

可以统计：

\[
R=
\frac{
\max_j(\text{channel max abs}_j)
}{
\operatorname{median}_j(\text{channel max abs}_j)
}
\]

如果：

```text
R = 100
R = 200
```

说明 channel 之间动态范围严重失衡。

这对：

```text
Activation INT8 per-tensor
```

尤其不友好，因为所有 channel 必须共享一个 scale。

---

# 10. Outlier / Channel Imbalance 的解决方法

## 10.1 Percentile / Clipping

优点：

```text
简单
不改变模型结构
```

缺点：

```text
只能在 clipping 与 resolution 之间折中
不能真正解决 channel imbalance
```

---

## 10.2 Per-Channel Activation

每个 channel：

```text
channel 0 → scale 0
channel 1 → scale 1
channel 2 → scale 2
...
```

优点：

```text
显著缓解 channel imbalance
```

缺点：

```text
实际 TensorRT INT8 GEMM/kernel 不一定支持所需粒度
```

---

## 10.3 Per-Group / Block-wise

多个 channel 共用一个 scale：

```text
channel 0~31    → scale 0
channel 32~63   → scale 1
...
```

属于：

```text
per-tensor
    ↓
per-group
    ↓
per-channel
```

之间的折中。

---

## 10.4 SmoothQuant

核心思想：

> 把 Activation 的部分动态范围转移给 Weight。

原始：

\[
Y=XW^T
\]

加入 channel-wise scale \(s\)：

\[
X'_j=\frac{X_j}{s_j}
\]

\[
W'_{:,j}=W_{:,j}s_j
\]

则：

\[
X'W'^T=XW^T
\]

数学结果保持不变，但：

```text
Activation channel range
```

变得更加均衡。

因此最终仍有机会使用：

```text
Activation:
INT8 per-tensor

Weight:
INT8 per-channel
```

这对于 TensorRT 部署非常有价值。

---

## 10.5 Mixed Precision

对于少量敏感层：

```text
普通层 → W8A8
敏感层 → W8A16 / BF16
```

例如：

```text
gate_proj    W8A8
up_proj      W8A8
down_proj    W8A16
```

目标不是：

```text
INT8覆盖率最大
```

而是：

\[
\boxed{
\text{在 Accuracy 与 Performance 之间寻找 Pareto 最优点}
}
\]

---

## 10.6 Critical Channel High Precision

如果误差主要集中在少数 channel：

```text
Top-K critical channels → BF16
其他 channels          → INT8
```

但这种方案需要额外 kernel/mixed-precision 支持，工程实现难度较高。

---

## 10.7 Rotation / Hadamard

通过正交变换把少数维度集中的 outlier 能量分散到多个 channel。

适用于严重 channel imbalance。

属于更高级的量化优化。

---

## 10.8 QAT

如果 PTQ 无法满足精度：

```text
PTQ
 ↓
SmoothQuant / Mixed Precision
 ↓
仍无法满足
 ↓
QAT
```

训练阶段加入 fake quantization，让模型主动适应量化误差。

---

# 11. 阶段七：端到端精度验证

数值误差通过后，最终还必须测试真实任务指标。

例如：

```text
Classification → Accuracy

Detection → mAP

VLA → Task Success Rate
```

对于 VLA，还应该检查：

```text
Action Chunk
First Action
Trajectory
Control Stability
Task Success Rate
```

因为：

```text
Cosine = 0.99
```

并不一定代表任务成功率不会下降。

---

# 12. 阶段八：性能 Profiling

精度通过后，再进入性能优化闭环。

不能假设：

```text
INT8 Engine
=
一定更快
```

实际性能：

\[
T_{total}
=
T_{GEMM}
+
T_{Attention}
+
T_{QDQ}
+
T_{Cast}
+
T_{Reformat}
+
T_{Memory}
+
T_{Launch}
+\cdots
\]

---

# 13. INT8 为什么可能没有明显加速？

理论：

```text
BF16 GEMM
   ↓
INT8 GEMM
   ↓
Tensor Core吞吐更高
```

实际可能变成：

```text
Q
↓
DQ
↓
INT8 GEMM
↓
Cast
↓
Reformat
```

因此最终收益：

\[
\boxed{
INT8\ GEMM\ Gain
-
Q/DQ\ Overhead
-
Reformat\ Overhead
-
Fusion\ Loss
}
\]

才是真正的端到端收益。

---

# 14. 性能 Profiling 需要检查什么？

重点统计：

```text
GEMM / MatMul
Q/DQ
Cast
Reformat
Reshape
Transpose
Softmax
LayerNorm/RMSNorm
Elementwise
H2D
D2H
Enqueue Time
GPU Compute Time
```

重点回答：

```text
INT8 Linear 是否真的使用 INT8 Kernel？

单个 INT8 GEMM 比 BF16 GEMM 快多少？

Q/DQ 占多少时间？

是否出现额外 Cast/Reformat？

量化后是否破坏原来的 Fusion？
```

---

# 15. Layer Fusion 优化

例如原 BF16：

```text
gate_proj ──┐
            ├── Fused Kernel
up_proj ────┘
```

量化后可能变成：

```text
Q → DQ → gate_proj
Q → DQ → up_proj
```

导致：

```text
Fusion Lost
```

于是：

\[
INT8\ Gain < Fusion\ Loss
\]

最终 INT8 甚至可能没有明显加速。

因此需要检查：

```text
量化前 kernel 数量
vs
量化后 kernel 数量
```

---

# 16. 调整 Quantization Mask

不要认为：

\[
\boxed{\text{量化越多 = 越快}}
\]

应该做模块级 speed ablation：

```text
QKV only

O only

QKVO

gate/up only

down only

FFN only

Attention + FFN
```

分别测：

```text
Δ Latency
Δ Throughput
Δ Accuracy
```

最终寻找：

\[
\boxed{
\text{最大性能收益}
+
\text{最小精度损失}
}
\]

---

# 17. CUDA Graph

如果 Engine 中有大量小 kernel：

```text
Q
DQ
GEMM
Cast
SiLU
Reshape
Transpose
...
```

正常情况下 CPU 需要不断 launch：

```text
CPU → kernel 1
CPU → kernel 2
CPU → kernel 3
CPU → kernel 4
...
```

CUDA Graph 将整个执行序列 capture：

```text
Q → DQ → GEMM → Cast → SiLU → ...
```

之后：

```text
CPU
 ↓
cudaGraphLaunch()
 ↓
GPU执行整个Graph
```

因此主要优化：

\[
\boxed{
CPU\ Enqueue / Kernel\ Launch\ Overhead
}
\]

而不是让 GEMM 数学计算本身变快。

---

# 18. Static Shape / Dynamic Shape

## Dynamic Shape

通过 Optimization Profile：

```text
MIN
OPT
MAX
```

告诉 TensorRT：

```text
模型可能运行在哪些 shape
最常见的 shape 是什么
```

---

## Static Shape

如果实际部署始终：

```text
batch = 1
sequence length = fixed
image size = fixed
```

则应该测试：

```text
MIN = OPT = MAX = Real Runtime Shape
```

固定 shape 有机会让 TensorRT：

```text
选择更合适的 tactic
进行更积极的优化
减少动态 shape overhead
```

---

# 19. Multi-Stream

Multi-Stream 主要用于：

\[
\boxed{Throughput}
\]

例如：

```text
Stream 0 → Request A
Stream 1 → Request B
Stream 2 → Request C
Stream 3 → Request D
```

多个独立 inference 可以并发执行。

但是：

```text
单 Request Latency
```

不一定下降，甚至可能因为资源竞争上升。

因此：

### Latency-sensitive

例如机器人 VLA：

```text
1~2 streams
small batch
```

### Throughput-sensitive

例如服务器：

```text
multiple streams
larger batch
dynamic batching
```

---

# 20. Batch Size

Batch 增大通常提高：

```text
GPU Utilization
Tensor Core Utilization
Throughput
```

但通常也增加单 batch latency。

因此必须区分：

\[
\boxed{Latency}
\]

和：

\[
\boxed{Throughput}
\]

不能因为：

```text
batch=16 FPS提高
```

就认为：

```text
单次模型推理变快
```

---

# 21. Plugin / Fused Kernel

当 Profiling 明确发现：

```text
多个小 kernel
大量 memory traffic
Q/DQ boundary
Fusion Loss
```

成为瓶颈时，可以进一步使用：

```text
TensorRT Plugin
Custom CUDA Kernel
Fused Kernel
```

例如 FFN：

```text
gate_proj → SiLU ──┐
                   × → down_proj
up_proj ───────────┘
```

可以考虑：

```text
Fused Gated-MLP Kernel
```

减少：

```text
Kernel Launch
Intermediate Memory Write
Intermediate Memory Read
```

这是较高级的部署优化阶段。

---

# 22. 最终 Benchmark

最终每个方案统一比较：

| 指标 | BF16 | INT8-A | INT8-B | Final |
|---|---:|---:|---:|---:|
| Latency | | | | |
| Speedup | 1.0× | | | |
| FPS | | | | |
| Throughput | | | | |
| Peak GPU Memory | | | | |
| Engine Size | | | | |
| Accuracy / Success Rate | | | | |
| Relative L2 | 0 | | | |
| Cosine | 1 | | | |

---

# 23. 完整 TensorRT 量化闭环

```text
┌──────────────────────────┐
│       FP/BF16 Model      │
└────────────┬─────────────┘
             ↓
┌──────────────────────────┐
│ Establish Baseline       │
│ Accuracy / Latency       │
│ Memory / Throughput      │
└────────────┬─────────────┘
             ↓
┌──────────────────────────┐
│ TensorRT BF16/FP16       │
│ Baseline                 │
└────────────┬─────────────┘
             ↓
┌──────────────────────────┐
│ Quantization Strategy    │
│ W8A8 / W8A16 / Mixed    │
└────────────┬─────────────┘
             ↓
┌──────────────────────────┐
│ Calibration              │
│ Activation Distribution  │
│ Dynamic Range / Scale    │
└────────────┬─────────────┘
             ↓
┌──────────────────────────┐
│ Q/DQ / INT8 Engine       │
└────────────┬─────────────┘
             ↓
      ┌──────┴──────┐
      ↓             ↓
 Precision        Performance
 Validation       Profiling
      ↓             ↓
 Layer Error      Kernel Time
 Analysis         Q/DQ
      ↓            Reformat
 Outlier          Fusion
 Clipping         Enqueue
 Resolution         ↓
 Channel          Performance
 Imbalance        Bottleneck
      ↓             ↓
┌──────────────┐ ┌───────────────┐
│Precision Opt │ │Performance Opt│
├──────────────┤ ├───────────────┤
│Calibration   │ │Quant Mask     │
│Percentile    │ │Fusion         │
│Per-channel   │ │CUDA Graph     │
│SmoothQuant   │ │Static Shape   │
│Mixed Prec.   │ │Plugin         │
│QAT           │ │Multi-stream   │
└──────┬───────┘ └───────┬───────┘
       └─────────┬────────┘
                 ↓
        Rebuild INT8 Engine
                 ↓
        End-to-End Benchmark
                 ↓
      Accuracy + Performance
                 ↓
          Deployment Target?
             ↙       ↘
           No         Yes
           ↓           ↓
        Iterate      Deploy
```

---

# 24. 最终需要形成的能力

完整的“量化部署能力”不只是：

```text
会把 FP16/BF16 模型转成 INT8
```

而是能够完成：

### 量化

```text
FP/BF16
→ Calibration
→ Q/DQ
→ INT8
→ TensorRT Engine
```

### 精度诊断

```text
Accuracy Drop
→ Layer-wise Error
→ Weight / Activation Attribution
→ Outlier / Clipping / Resolution
→ Channel Imbalance
```

### 精度优化

```text
Calibration
Percentile
Per-channel / Per-group
SmoothQuant
Mixed Precision
QAT
```

### 性能诊断

```text
Latency
→ Kernel Profiling
→ INT8 GEMM
→ Q/DQ
→ Cast/Reformat
→ Fusion
→ Enqueue
```

### 性能优化

```text
Quantization Mask
Layer Fusion
CUDA Graph
Static Shape
Tactic Selection
Plugin / Fused Kernel
Multi-stream / Batch
```

最终目标不是追求：

```text
INT8 覆盖率最高
```

而是寻找：

\[
\boxed{
\text{Accuracy}
+
\text{Latency}
+
\text{Memory}
+
\text{Throughput}
}
\]

之间最合适的 Pareto 最优方案。

---

# 25. 一句话总结

TensorRT 量化部署的标准闭环可以概括为：

> **先建立 BF16/FP16 TensorRT Baseline，再通过 Calibration 与 Q/DQ 构建 INT8 Engine；精度下降时通过逐层误差分析定位 clipping、resolution loss、channel imbalance 和敏感层，并通过 SmoothQuant、Mixed Precision 或 QAT 修复；性能不足时通过 Kernel Profiling 定位 INT8 GEMM、Q/DQ、Reformat、Fusion 和 Enqueue 开销，再通过量化 Mask、CUDA Graph、Static Shape 和 Fused Kernel 优化，最终以真实任务 Accuracy/Success Rate 与端到端 Latency 为标准不断迭代。**


| 精度下降原因/场景              | 文章提到的解决办法                                        | 核心作用                               |
| ---------------------- | ------------------------------------------------ | ---------------------------------- |
| Calibration 数据不代表真实分布  | **换更有代表性的校准集、增加校准样本**                            | 让 activation dynamic range 更接近真实推理 |
| INT8 dynamic range 不合适 | **调整每层动态范围** `ITensor::setDynamicRange(min,max)` | 改善 scale，减少 clipping / 量化误差        |
| 少数层对 INT8 很敏感          | **Mixed Precision，关键层保留 FP16**                   | 只量化稳定层，保护敏感层                       |
| PTQ 本身精度不够             | **QAT**                                          | 训练时模拟量化误差，让模型主动适应 INT8             |
| MinMax 对异常值敏感          | **Entropy Calibration**                          | 通过分布/KL 思路寻找更合适的量化范围               |
| FP16 某些算子数值敏感          | **敏感操作保持更高精度**                                   | 避免 Softmax、Norm 等数值问题              |
