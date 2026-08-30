# SmolVLA + RL Token 在线强化学习后训练面试报告

## 一、针对解决的什么问题

项目首先在 LIBERO Benchmark 上对 SmolVLA 进行闭环评测和 Failure Analysis。分析发现，模型并不是完全无法理解任务或生成操作轨迹，而是**失败高度集中在决定任务最终成败的关键操作阶段**。

主要表现为两类。

第一类是**抓取以及多阶段长时序任务中的误差累积**。例如：

> `open the top drawer and put the bowl inside`

模型可能已经能够完成打开抽屉、接近目标等前序动作，但任务需要连续经历：

$$\text{Open} \rightarrow \text{Approach} \rightarrow \text{Grasp} \rightarrow \text{Transport} \rightarrow \text{Place} \rightarrow \text{Release}$$

前序动作产生的小误差会不断改变后续 observation，使策略逐渐进入 demonstration 中覆盖不足的状态，最终在重新抓取、目标切换或者放置阶段失败。

第二类是**姿态和位置敏感的精细放置失败**。例如：

> `put the wine bottle on the rack`

SmolVLA 通常能够完成目标识别、接近、抓取和大范围搬运，但最终是否成功高度依赖 rack 附近的末端位置、瓶子姿态和 release timing。此时整体轨迹可能已经基本正确，真正需要修复的是最后几个关键动作。

因此，我对问题的判断不是：

> **VLA 不会完成这些任务。**

而是：

> **VLA 已经具有较好的 manipulation prior，但在闭环执行产生的困难状态以及抓取、姿态调整、精细放置等关键阶段仍存在局部 action error。**

所以项目目标也不是重新训练整个 SmolVLA，而是：

$$\boxed{ a_{RL}=a_{VLA}+\Delta a }$$

在尽量保持原始 VLA 能力的情况下，通过在线 RL 学习一个小的 residual correction，重点修复这些关键失败状态。

* * *

# 二、用什么方法

## 1. 理论分析：为什么 RL Token 可以解决这个问题

### 1.1 为什么单纯 SFT 难以继续解决

SmolVLA 的监督训练本质上学习：

$$\pi(a|o,l)$$

也就是根据视觉 Observation 和 Language Instruction 模仿 demonstration 中的 expert action。

它非常适合学习“应该抓哪个物体”“大致应该怎么移动”等通用 manipulation behavior。

但闭环执行时存在一个重要问题：

$$s_t^{policy}\neq s_t^{expert}$$

例如专家数据里的 wine bottle 始终以比较标准的姿态到达 rack，但真实 rollout 中可能已经发生轻微倾斜。

一旦进入这种 demonstration 覆盖不足的状态，后续误差可能继续累积。

更重要的是，SFT 优化的是：

$$\min_\theta L(a_{\theta},a_{expert})$$

而 Benchmark 真正关心的是：

$$\max_\theta P(\mathrm{Success})$$

因此对于**模型已经基本会做，但局部动作误差导致最终失败**的问题，环境提供的 closed-loop success/failure signal 有额外价值。

* * *

## 1.2 为什么不是直接用 RL 从头学习

LIBERO 的任务奖励主要围绕任务是否成功。如果直接使用随机初始化的连续控制 Actor：

$$a_{\mathrm{random}} \rightarrow \text{几乎无法完成复杂 manipulation}$$

就会产生严重的 cold-start 和 exploration difficulty。

而 SmolVLA 已经可以产生：

$$a_{\mathrm{ref}}$$

作为质量较高的 Action Prior。

因此采用 Residual RL：

$$a_{\mathrm{RL}} = a_{\mathrm{ref}} + \Delta a_{\mathrm{RL}}$$

让 SmolVLA 负责“**大体怎么完成任务**”，RL 负责“**这个状态下应该怎么修正**”。

这就把原本巨大的连续动作搜索空间压缩到 pretrained VLA policy 附近。

* * *

## 1.3 为什么需要 RL Token

如果直接把 VLM 大量视觉 hidden states 用于在线 RL，Actor/Critic 的输入和 Replay 数据都会比较大。

因此整个训练分成两个 Stage。

### Stage 1：Representation Learning

首先冻结 SmolVLA，从 VLM 得到：

$$z_{1:M}$$

也就是 prefix hidden states。

通过 RL Token Encoder：

$$z_{RL}=E(z_{1:M})$$

将大量视觉 token 压缩成紧凑状态表示。

训练时使用 Encoder-Decoder 做 Self-Reconstruction：

$$z_{1:M} \rightarrow z_{RL} \rightarrow \hat z_{1:M}$$

优化：

$$L_{recon} = \| \hat z-z \|^2$$

Stage 1 的作用可以简单理解成：

> **不是学习机器人怎么行动，而是先把 SmolVLA 已有的视觉语义表征压缩成一个适合后续 RL 使用的 state representation。**

* * *

### Stage 2：Online Residual RL

Stage 2 冻结 SmolVLA 和训练好的 RL Token Encoder。

完整控制流程：

```
LIBERO Observation
        ↓
Frozen SmolVLA
        ↓
Prefix Hidden States
        ↓
RL Token Encoder
        ↓
z_rl
        ↓
SmolVLA Reference Action Chunk
        ↓
Residual Actor
        ↓
a = a_ref + Δa
        ↓
LIBERO Env
        ↓
Reward / Done / Next State
        ↓
Replay Buffer
        ↓
Twin-Q Actor-Critic Update
```

Actor 不从头预测动作，而是学习：

$$\Delta a$$

并通过 BC Regularization：

$$L_{Actor} = -Q(s,a) + \beta \|a-a_{ref}\|^2$$

在“获得更高环境回报”和“不要偏离 VLA 太远”之间做 trade-off。

所以整个方法实际上是在求：

$$\boxed{ \text{提高任务成功率} \quad \text{s.t.} \quad a_{RL}\approx a_{VLA} }$$

这与“修复关键阶段局部动作误差”的问题非常匹配。

* * *

# 2. 实际分析：训练过程中遇到了什么问题，如何解决

真正的项目重点并不是把 Stage 1/2 跑通，而是 Stage 2 第一版训练效果并不好，之后通过 **Evaluation → Failure Analysis → 参数调整 → Fixed Evaluation** 建立了一个闭环。

* * *

## 问题一：重点 Task 3/9 训练后反而下降

第一轮 Stage 2 后，重点优化的 Task 3 和 Task 9 Success Rate 不升反降。

我首先没有继续盲目增加训练步数，而是保存 rollout，对失败轨迹进行分析。

发现两个任务分别代表两种典型困难：

**Task 3：**

$$\text{长时序 + 多阶段操作}$$

需要完成开抽屉、重新定位、抓取、搬运和放置。

**Task 9：**

$$\text{姿态敏感 + 精细放置}$$

主要困难集中在 rack 附近的最终 alignment 和 placement。

第一版：

$$Chunk\ Length=10$$

通过 Action Chunk Length 消融发现，较短 Chunk 虽然具有更高的 closed-loop replanning frequency，但在 Task 3 这类长时序多阶段任务上，连续动作的 temporal coherence 不够稳定。

因此增大 Action Chunk Size。

同时没有直接只用 Task 3/9 持续训练，而是调整为：

$$\boxed{ \text{Goal 全任务训练} \rightarrow \text{降低 LR} \rightarrow \text{Task 3/9 Targeted Fine-tuning} }$$

第一阶段先让 Actor 在完整任务分布上学习稳定的通用 correction。

第二阶段再用较低学习率重点优化 3/9，减少在窄任务分布上直接把 Actor 带偏的问题。

* * *

## 问题二：Actor 偏离 Reference Action 过大

随后进一步分析：

$$\|a_{Actor}-a_{Ref}\|$$

发现第一版 RL 中 Actor 已经明显偏离 SmolVLA Reference Action。

但：

$$\text{Action Drift}\uparrow$$

并没有对应：

$$\text{Success Rate}\uparrow$$

说明 Actor 不是“没学到东西”，而是**学得太激进**。

因此调整：

$$L_{Actor} = -Q+\beta L_{BC}$$

中的：

$$\beta$$

提高 BC Strength。

实验发现更强的 BC Constraint 效果更稳定。

它本质上限制：

$$\Delta a$$

的幅度，让 RL 不去重新生成整条 manipulation trajectory，而是围绕 pretrained VLA 做局部修正。

所以这里解决的是：

> **Policy Drift 问题。**

* * *

## 问题三：Exploration Noise 干扰精细操作

TD3 需要 Exploration Noise 来产生不同动作、探索高价值状态。

但是这个项目和普通的从零 RL 不同：

$$a_{ref}$$

本身已经是比较好的动作。

如果在：

* grasp；
* alignment；
* placement；
* release；

这些关键状态加入较大的 Gaussian Noise，就可能直接破坏原本合理的末端位置和姿态。

尤其是 Task 9，本身要求的就是精细 placement。

因此降低：

$$Action\ Exploration\ Noise$$

把探索范围进一步限制在：

$$a_{ref}$$

附近。

这个实验让我进一步确定：

> **对于 pretrained VLA 后训练，exploration 并不是越强越好，更重要的是在已有策略先验附近进行有效的 local exploration。**

* * *

## 问题四：UTD 过高导致窄分布 Replay 数据被反复拟合

进一步检查 Stage 2 代码时发现了一个比较关键的实现问题。

最开始：

$$UTD=5$$

但这里并不等于：

> 一个 Action Chunk 更新五次。

实际更新次数还和：

$$\frac{chunk\_len}{stride}$$

相关。

当：

$$chunk\_len=10,\quad stride=2$$

时：

$$N_{update} = 5\times\frac{10}{2} = 25$$

也就是说：

> **环境每产生一个 Action Chunk，Actor-Critic 实际进行了约 25 次梯度更新。**

而 Targeted Training 阶段 Replay Buffer 又主要由 Task 3/9 构成。

于是形成：

$$\text{少量困难任务数据} \rightarrow \text{Replay 重复采样} \rightarrow \text{大量 Gradient Update}$$

Critic Loss 即使下降得很好，也可能只是：

$$Q_{\phi}$$

很好地拟合了当前 Replay Distribution，并不代表 Actor 的真实闭环 Success Rate 提高。

因此降低 UTD，减少每条在线 transition 的重复利用强度，使：

$$\text{Environment Interaction} : \text{Gradient Update}$$

更加平衡。

* * *

## 最终形成的评测闭环

整个调参过程最终形成：

$$\boxed{ Rollout \rightarrow Success\ Rate \rightarrow Failure\ Analysis \rightarrow Training\ Metric \rightarrow Hyperparameter \rightarrow Fixed\ Evaluation }$$

而不是根据：

* Actor Loss；
* Critic Loss；
* Online Episode Reward；

单独判断模型有没有变好。

最终重点监控三类指标：

$$SR_{3,9}$$

衡量困难任务是否真正改善；

$$SR_{Guardrail}$$

衡量其它任务是否退化；

以及：

$$\|a_{Actor}-a_{Ref}\|$$

衡量 RL correction 是否已经过度偏离 VLA Prior。

所以四次主要调整分别对应：

| 发现的问题 | 诊断 | 调整 |
| --- | --- | --- |
| Task 3/9 下降 | 时序/训练分布 | Chunk + 全任务训练 → Target FT |
| Actor Drift | Actor 与 Reference 距离增大 | ↑ BC Strength |
| 精细阶段不稳定 | Exploration 扰动过大 | ↓ Action Noise |
| Replay 过拟合 | 真实 Updates/Chunk 过高 | ↓ UTD |

最后整个 Stage 2 从：

$$\text{Aggressive Policy Optimization}$$

逐渐调整为：

$$\boxed{\text{Conservative Residual Policy Optimization}}$$

* * *

# 3. 还有什么方法？为什么选择 RL Token

## 方法一：Action Expert SFT / LoRA

最直接的方法是收集 Task 3/9 更多 demonstration，然后继续微调 Action Expert。

优点是：

* 稳定；
* 实现简单；
* 不需要在线探索；
* 训练成本低。

问题是其优化目标仍然是：

$$a_{pred}\approx a_{expert}$$

不能直接利用：

$$success/failure$$

作为优化信号。

因此它更适合**任务适配和 demonstration distribution 内能力增强**。

RL Token 更适合本项目这种：

> VLA 已经基本会做，但是闭环执行中的局部动作误差决定最终任务成功。

* * *

## 方法二：针对 Failure State 补充 Demonstration

也可以人工收集：

* re-grasp；
* drawer insertion；
* rack alignment；
* failed placement recovery；

等数据，然后重新 SFT。

相比 RL，它的优势是 supervision 更强、更稳定。

缺点是需要提前知道：

> **什么状态会失败，以及这些状态下正确动作是什么。**

而很多困难状态实际上只有模型自己 rollout 后才会产生。

* * *

## 方法三：DAgger / Human Intervention

更进一步可以：

$$\text{Policy Rollout} \rightarrow \text{Failure State} \rightarrow \text{Expert Correction} \rightarrow \text{Dataset}$$

这实际上非常适合关键阶段失败问题。

它的优势是 correction signal 比 sparse reward 更直接，数据效率可能更高。

但缺点也很明显：

> 需要人工持续参与在线数据采集。

对于 LIBERO 仿真可以实现，但扩展到大规模任务时人工成本较高。

* * *

## 方法四：Reward Shaping

还可以给 Task 3 增加：

$$r_{open}, r_{grasp}, r_{insert}, r_{success}$$

给 Task 9 增加：

$$r_{distance}, r_{orientation}, r_{placement}$$

缓解 sparse reward 下的 credit assignment。

理论上可以提高 RL sample efficiency。

但是人为设计 Dense Reward 存在：

$$\text{Reward Hacking / Proxy Optimization}$$

风险。

所以项目优先使用 Benchmark 的 Success Signal，使训练目标和最终评价目标尽可能一致。

* * *

# 4. 后面还可以做什么

### 第一，Prioritized Replay

现在 Replay Buffer 的 transition 基本统一采样。

后续可以提高：

* failed grasp；
* near-success；
* terminal placement；
* recovery state；

的 sampling probability。

让 RL 把更多 update budget 用在真正决定 Success Rate 的状态上。

* * *

### 第二，Phase-Aware Residual Correction

现在：

$$\Delta a$$

整个任务都可以产生。

但实际失败主要集中在：

$$\text{Grasp / Alignment / Placement}$$

未来可以根据 manipulation phase 调节 residual magnitude。

例如 transport 阶段：

$$\Delta a\approx0$$

而 placement 阶段允许：

$$|\Delta a|\uparrow$$

进一步减少无意义 Policy Drift。

* * *

### 第三，Adaptive Action Chunk

当前一个任务统一使用固定 Chunk。

但：

$$\text{Long-Horizon Motion}$$

更需要 temporal coherence；

而：

$$\text{Precision Placement}$$

更需要 closed-loop replanning。

未来可以：

$$C_{transport}>C_{placement}$$

实现动态 Action Chunk。

* * *

### 第四，Human Intervention

对于真机 VLA，我认为这是比较自然的下一步。

当 Actor 进入：

$$\text{failed / unsafe / low-value state}$$

时由人工接管，获得 corrective demonstration，再结合 Online RL 训练。

最终形成：

$$\boxed{ VLA Prior + Online RL + Corrective Demonstration }$$

相比单纯扩大 RL rollout 数量，可能具有更高的数据效率。

* * *

# 三、取得了什么效果

最终通过：

$$\text{RL Token Representation} + \text{Residual Actor-Critic} + \text{BC Constraint} + \text{Targeted Online Training}$$

形成针对 SmolVLA 关键操作失败的在线后训练方案。

在最终统一 Fixed-Policy Evaluation 下，重点优化的 Task 3/9 平均 Success Rate：

$$\boxed{ 50\%\rightarrow82\% }$$

提升约：

$$\boxed{+32pp}$$

与此同时，其余 Guardrail Tasks 成功率保持在约：

$$\boxed{87\%}$$

没有因为重点优化两个困难任务而出现明显能力退化。

LIBERO-Goal Overall Success Rate 最终：

$$\boxed{ 79.5\%\rightarrow86.0\% }$$

提升：

$$\boxed{+6.5pp}$$

因此最终得到的结论并不是简单的：

> **“RL 比 SFT 好。”**

而是：

> **对于已经具备较强 manipulation prior 的 VLA，如果失败主要集中在抓取、姿态调整和精细放置等关键阶段，与其重新学习完整策略，更有效的思路是冻结 VLA 主干，以原始 Action Chunk 为策略先验，通过受 BC 约束的 Residual RL 利用闭环 Success/Failure Signal 学习局部动作纠偏。**

* * *

## 面试时最后可以压缩成 30 秒

如果面试官最后说“**所以你这个项目最核心做了什么？**”，建议你收束成：

> 我的核心工作是针对 SmolVLA 在抓取和精细放置关键阶段的集中失败做在线 RL 后训练。Stage 1 冻结 VLA，通过 Self-Reconstruction 把 VLM Prefix Hidden States 压缩成 RL Token；Stage 2 再冻结 VLA，以 SmolVLA Action Chunk 作为 Reference，用 TD3-style Actor-Critic 学 Residual Action Correction。第一版训练出现困难任务成功率下降，我通过 Fixed Evaluation 和失败轨迹分析，依次定位了 Action Chunk、Policy Drift、Exploration Noise 和 UTD 过高导致 Replay 过度拟合四个问题，最后把训练从 aggressive RL 调整成 BC 约束下的 conservative residual RL，使 Task 3/9 平均成功率从 50% 提升到 82%，同时其它任务保持在约 87%，Overall 从 79.5% 提升到 86%。