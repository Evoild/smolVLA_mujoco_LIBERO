# SmolVLA + RL Token 在线强化学习后训练：面向关键抓取与精细放置失败的策略优化

## 一、针对解决的什么问题

### 1. 问题背景

基于 SmolVLA 在 LIBERO Benchmark 上建立完整闭环评测流程后，对失败 Episode 进行任务级和轨迹级分析。基线模型在单物体、短时序抓取放置任务上表现相对稳定，但失败明显集中在以下两类关键操作阶段：

**第一类：多阶段操作中的抓取与子目标切换失败。**

典型任务：

> open the top drawer and put the bowl inside

任务需要连续完成：

`打开抽屉 → 定位 bowl → 抓取 → 搬运 → 插入抽屉 → 释放`

模型往往能够完成部分子目标，但在抽屉打开后的策略切换、重新抓取以及最终放置阶段出现失败。由于前序动作误差会改变后续 observation distribution，这类任务还存在明显的长时序误差累积问题。

**第二类：抓取姿态或目标位置高度敏感的精细操作失败。**

典型任务：

> put the wine bottle on the rack

模型通常可以完成目标识别、接近、抓取和大范围搬运，但在 rack 附近的最终姿态调整、末端位置修正以及 release 时机上出现集中失败。

因此问题并不是 VLA 完全不会完成任务，而更接近：

> **预训练 / SFT VLA 已经能够生成整体合理的动作轨迹，但在少数决定任务成败的关键状态附近，reference action 仍存在局部误差。**

基线失败分析也显示，SmolVLA 的主要短板集中于长时序误差累积、双物体操作、精细放置以及抓取/重新抓取失败，而不是基础视觉识别能力。项目最初 400 episodes 评测中 Overall Success Rate 为 65.25%，Goal 和 Long-Horizon 任务的失败明显更多。

因此后训练目标并不是重新训练整个 VLA，而是：

$$
a_{\mathrm{new}}
=
a_{\mathrm{VLA}}
+
\Delta a_{\mathrm{RL}}
$$

在尽量保留 VLA 原有通用能力的前提下，只学习困难状态附近的动作修正。

---

# 二、采用什么方法

## 1. 理论分析：为什么 RL Token 对这类问题有效

### 1.1 SFT 已经学习的是 demonstrations distribution

SmolVLA 原始策略通过示范数据学习：

$$
\pi_{\mathrm{VLA}}(a|o,l)
$$

其训练目标本质上是让模型预测专家动作。

这对于学习：

* 哪个物体需要操作；
* 大致如何接近目标；
* 抓取和放置的大体轨迹；

非常有效。

但是在真正闭环 rollout 时，机械臂产生的 observation 分布会逐渐偏离专家数据：

$$
s_t^{policy}
\neq
s_t^{expert}
$$

例如 bottle 已经略微倾斜、夹爪比专家轨迹偏几厘米，后续 VLA 看到的状态可能根本不存在于训练 demonstrations 中。

这时继续 SFT 的核心限制在于：

> 模型只能继续拟合已有 expert action，并不能直接利用“这个动作最后导致成功还是失败”的环境反馈。

而任务真正关心的是：

$$
\max_\pi P(\mathrm{Success})
$$

而不是单纯：

$$
\min_\pi \|a-a_{\mathrm{expert}}\|
$$

因此对于“主体动作已经合理，但关键阶段局部动作决定最终成功”的问题，在线强化学习具有更直接的优化目标。

---

## 1.2 为什么不直接用 RL 从头训练 Actor

直接使用随机初始化连续控制策略存在严重 cold-start：

$$
a_{\mathrm{random}}
\rightarrow
\text{几乎全部失败}
$$

LIBERO 又采用非常稀疏的：

$$
r=
\begin{cases}
1,& success\\
0,& otherwise
\end{cases}
$$

随机策略很难获得成功轨迹，Critic 也就缺少有意义的正向信号。

因此项目采用：

$$
\boxed{
\text{VLA reference policy}
+
\text{RL residual correction}
}
$$

而不是让 RL 从头学习 manipulation。

SmolVLA 本身负责产生：

$$
a_{\mathrm{ref}}
$$

RL Actor 只学习：

$$
\Delta a
$$

最终：

$$
a_{\mathrm{RL}}
=
a_{\mathrm{ref}}+\Delta a
$$

这样 RL exploration 始终发生在 pretrained VLA 已经较合理的动作附近。

---

## 1.3 为什么需要 RL Token

直接把完整 VLM hidden states 输入 Actor/Critic 成本很高，而且 online RL replay 会存储大量状态。

因此首先将 SmolVLA prefix hidden states：

$$
z_{1:M}
$$

通过 RL Token Encoder 压缩：

$$
z_{RL}
=
E(z_{1:M})
$$

然后 Actor/Critic 只使用紧凑状态：

$$
x=
[z_{RL};q]
$$

其中 \(q\) 为 robot proprioception。

最终控制链：

```text
LIBERO observation
        ↓
Frozen SmolVLA
        ↓
prefix hidden states
        ↓
RL Token Encoder
        ↓
z_rl
        ↓
Actor(z_rl, proprio, reference action)
        ↓
reference + residual correction
        ↓
action chunk
        ↓
LIBERO
```

这样既保留 VLA 的视觉语义表征，又避免 RL 直接更新大型 VLM。

---

## 1.4 为什么使用 BC Regularization

如果 Actor 单纯最大化：

$$
Q(s,\pi(s))
$$

Actor 很容易逐渐偏离 pretrained VLA：

$$
\|\pi(s)-a_{\mathrm{ref}}\|\uparrow
$$

从而为了少数困难任务破坏已经正确的通用 manipulation ability。

因此 Actor loss 使用：

$$
L_{\mathrm{actor}}
=
-Q_{\min}(x,a)
+
\beta
\|a-a_{\mathrm{ref}}\|^2
$$

第一项：

$$
-Q
$$

鼓励寻找更高回报动作。

第二项：

$$
L_{BC}
$$

限制策略偏离 pretrained VLA。

因此整个 RL optimization 实际是在做：

$$
\boxed{
\text{improve success}
\quad\text{subject to}\quad
\text{stay close to VLA prior}
}
$$

这与项目“修复局部困难阶段，而不是重新学习整个任务”的目标完全一致。

---

# 2. 实际分析

## 2.1 训练数据

### Stage 1：Offline representation training

Stage 1 使用 LIBERO-Goal 全部 10 个任务的离线 demonstrations，而不是只使用最终重点优化的 Task 3/9。

数据来自：

`HuggingFaceVLA/libero`

Goal Suite 对应 dataset task index：

`10-19`

实际划分：

* Train Episodes：385
* Validation Episodes：43

Frozen SmolVLA 根据 image、robot state、language instruction 得到 prefix hidden states：

$$
z_{1:M}\in\mathbb R^{M\times960}
$$

然后训练 RL Token Encoder + Decoder：

$$
z_{1:M}
\rightarrow
z_{RL}
\rightarrow
\hat z_{1:M}
$$

优化 reconstruction：

$$
L_{\mathrm{recon}}
=
\|
D(E(z))-sg(z)
\|^2
$$

Stage 1 不更新 VLA：

$$
\alpha_{\mathrm{VLA-SFT}}=0
$$

最终 reconstruction loss：

$$
1.4433\rightarrow0.0743
$$

Validation：

$$
1.1649\rightarrow0.0636
$$

训练集与验证集同步收敛，没有明显 reconstruction overfitting。

Stage 1 的目标不是直接提高 Success Rate，而是获得适用于 Online RL 的紧凑视觉状态表征。

---

## 2.2 Stage 2 Online RL 数据采集

Stage 2 不再使用固定 demonstration transition，而是在 LIBERO 中在线 rollout。

每次控制流程：

```text
current observation
      ↓
SmolVLA reference chunk
      ↓
RL residual actor
      ↓
execute C-step chunk
      ↓
reward / done / next observation
      ↓
Replay Buffer
```

Replay Buffer 每条数据不是单个 action，而是：

```text
x
action_chunk
reference_chunk
discounted_reward
x_next
reference_next
done
```

即：

$$
(x_t,
a_{t:t+C-1},
a^{ref}_{t:t+C-1},
R_t,
x_{t+C},
a^{ref}_{t+C:t+2C-1},
d)
$$

Actor 和 Critic 都直接在：

$$
C\times d
$$

维 Action Chunk 上学习。

当前：

$$
C=10,\quad d=7
$$

因此 Critic 判断的是：

> **这一整段 10-step action correction 最终对任务是否有价值。**

而不是只评价单个 joint command。代码事实核查也确认 Replay Buffer 保存整个 chunk，reward 使用 chunk 内 discounted sparse reward。

---

## 2.3 Cold-start 处理

初始实现如果随机初始化 Actor，会导致训练刚开始时随机 action 破坏 VLA 原本合理的行为。

因此将 Actor 改为：

$$
a=a_{\mathrm{ref}}+\Delta a
$$

并将 residual output layer：

```text
weight = 0
bias   = 0
```

因此初始化：

$$
\Delta a=0
$$

严格得到：

$$
a_{\mathrm{Actor}}
=
a_{\mathrm{VLA}}
$$

代码验证：

```text
max_mu_ref_abs = 0
max_delta_abs  = 0
```

这使 Stage 2 从：

> random policy exploration

转变成：

> pretrained VLA 附近的 local policy search。

---

# 3. Stage 2 参数调整过程

Stage 2 最初采用：

```text
total_env_steps = 20000
warmup          = 2000
batch           = 256
UTD             = 5
chunk_len       = 10
action_std      = 0.05
bc_beta         = 1
```

第一轮训练虽然 online episode success 较高，但固定 Actor evaluation 发现 Task 3/9 出现退化。

进一步分析日志发现：

$$
actor\_ref\_l2
\approx0.16\sim0.23
$$

说明 Actor 已经明显偏离 reference，但偏移并没有转化为更高的任务成功率。

因此问题被定位为：

> **在线 RL 更新过于 aggressive，Critic 在有限 target-task replay distribution 上驱动 Actor 产生过大的策略漂移。**

---

## 3.1 调整 BC Strength

提高：

$$
\beta:1\rightarrow5
$$

强化：

$$
\|a-a_{\mathrm{ref}}\|^2
$$

约束。

这样 RL 不再尝试重新生成完整 manipulation trajectory，而是限制为：

$$
a_{\mathrm{RL}}
\approx
a_{\mathrm{VLA}}+\text{small correction}
$$

尤其适合 wine bottle rack placement 这类主体轨迹正确、末端姿态需要微调的任务。

---

## 3.2 降低 Action Exploration Noise

代码核查发现：

$$
action\_std=0.05
$$

并不仅用于 rollout exploration。

同一个 `action_std` 同时进入：

* rollout exploration；
* Actor update 时的 sampled action；
* Critic target backup。

因此它实际上同时影响：

$$
\text{data distribution}
+
\text{actor gradient}
+
\text{Bellman target}
$$

对于精细 placement task，过大的 exploration noise 会直接破坏本来已经比较合理的 reference trajectory。

因此降低：

$$
0.05\rightarrow0.02
$$

减少关键阶段动作扰动。

---

## 3.3 降低 UTD

进一步检查发现当前：

$$
UTD=5
$$

并不是简单的“每个 chunk 更新 5 次”。

实际代码：

$$
N_{\mathrm{update}}
=
UTD\times
\left(
\frac{chunk\_len}{stride}
\right)
$$

其中：

$$
chunk=10,\quad stride=2
$$

所以：

$$
5\times5=25
$$

即每执行一个 10-step chunk，实际上进行约：

$$
25
$$

次 gradient update。

这意味着一批 narrow online trajectories 被重复利用，非常容易造成 Actor/Critic 对 target tasks 当前 rollout distribution 过度拟合。

因此将：

$$
UTD:5\rightarrow2
$$

使每个 chunk 更新次数：

$$
25\rightarrow10
$$

降低 replay reuse 强度。

---

## 3.4 缩短 Training Horizon + Early Stopping

初始：

$$
20k\ env\ steps
$$

进一步发现 online training success 并不能代表 fixed-policy evaluation success。

因此增加 periodic fixed-policy evaluation，根据：

$$
SR_{\mathrm{target}}
$$

以及：

$$
SR_{\mathrm{guardrail}}
$$

联合选择 checkpoint。

最终使用：

$$
10k
$$

左右 env step，而不是默认选择最后一个 checkpoint。

模型选择条件定义为：

$$
\max SR_{3,9}
$$

subject to：

$$
SR_{\mathrm{guardrail}}
\ge
SR_{\mathrm{baseline}}-\epsilon
$$

从而显式避免为了提高目标任务而牺牲其它任务。

---

## 3.5 Target-task Oversampling + Guardrail Rehearsal

重点优化：

```text
Task 3
Task 9
```

但如果只采 3/9：

$$
D_{\mathrm{replay}}
=
D_3\cup D_9
$$

共享 Actor 可能逐渐向两个任务 specialization。

因此将任务 sampler 从原有 round-robin 进一步改为 weighted sampling。

例如：

$$
P(3)=35\%
$$

$$
P(9)=35\%
$$

剩余八个任务：

$$
P(i)=3.75\%
$$

这样：

$$
70\%
$$

online data 用于困难任务 targeted repair；

同时：

$$
30\%
$$

作为 guardrail rehearsal，持续约束通用能力。

这个过程本质上属于：

> **target-task oversampling + rehearsal**

而不是简单增加两个困难任务数据。

---

# 4. Action Chunk 参数分析

当前：

$$
C=10
$$

Action Chunk 对两类失败具有不同 trade-off。

对于多阶段任务：

> open drawer and put bowl inside

较长 chunk 可以保持一段动作的 temporal coherence。

但对于：

> put wine bottle on rack

更短 chunk 可以增加：

$$
observation\rightarrowaction
$$

闭环反馈频率。

因此针对：

$$
C=5,\quad C=10
$$

做消融。

最终根据整体稳定性保留 `C=10` 作为统一配置；对于未来更加极端的 precision manipulation，可以进一步尝试 task-/phase-adaptive chunk length。

---

# 5. 评测方法

训练指标不能直接作为最终结果。

因此项目建立独立 fixed-policy evaluation。

训练期间记录：

```text
episode success
episode return
Q mean
critic loss
actor loss
BC distance
||delta_action||
||actor-reference||
Replay Buffer size
UTD updates
```

最终 evaluation：

* 冻结 Actor；
* 不进行 gradient update；
* 对 LIBERO-Goal 全部 10 个 task 分别评测；
* 单独统计 Task 3/9；
* 其它 8 个任务作为 guardrail；
* 对重点失败 Episode 保存视频进行闭环 Failure Analysis。

核心指标：

$$
SR_{\mathrm{target}}
=
\frac{SR_3+SR_9}{2}
$$

$$
SR_{\mathrm{guardrail}}
=
\frac1{8}
\sum_{i\neq3,9}SR_i
$$

以及：

$$
SR_{\mathrm{overall}}
$$

这种评价方式避免出现：

> Task 3/9 提高，但其它任务严重退化，整体反而没有收益。

---

# 三、还有哪些方法可以解决？为什么最终采用 RL Token

## 1. Action Expert SFT / LoRA

可以只使用困难任务 demonstrations，对 Action Expert 做：

$$
L_{\mathrm{SFT}}
=
\|a-a_{\mathrm{expert}}\|
$$

优势：

* 实现简单；
* 稳定；
* 无 RL exploration；
* 训练成本低。

局限：

* 仍然只学习 expert data distribution；
* 无法直接利用 rollout failure；
* 对 execution-induced distribution shift 的处理能力有限；
* 很难直接优化最终 success signal。

因此更适合作为：

> domain adaptation / task adaptation

而不是本项目的：

> closed-loop failure correction。

---

## 2. 增加困难任务示范数据

可以针对失败状态重新采：

* 抓取失败 recovery；
* rack alignment；
* drawer insertion；

然后重新 SFT。

优势是简单稳定。

但缺点是：

> 必须人为判断哪些状态值得补数据，而且大量失败状态往往是在 rollout 中动态产生的。

如果每次失败都重新采 demonstration，数据获取成本较高。

---

## 3. DAgger / Human Intervention

在 rollout 失败时由专家接管：

$$
s_{\mathrm{failure}}
\rightarrow
a_{\mathrm{expert}}
$$

再把 correction trajectory 回流训练。

它对于 recovery / precision manipulation 会非常有效。

优势：

* correction signal 比 sparse reward 更直接；
* 数据效率高；
* 对失败状态覆盖好。

缺点：

* 需要持续人工操作；
* scalable 程度较低。

这是项目下一阶段很适合增加的方法。

---

## 4. Dense Reward / Reward Shaping

例如对 bottle-to-rack：

$$
r=
-\lambda_1d_{\mathrm{position}}
-\lambda_2d_{\mathrm{orientation}}
+r_{\mathrm{success}}
$$

能够改善 sparse reward 下的 credit assignment。

尤其对于 drawer multi-stage task，可增加：

```text
drawer opened
bowl grasped
bowl entered drawer
final success
```

阶段奖励。

优势是明显提高 RL sample efficiency。

风险是：

> reward 设计不合理可能让机器人优化 proxy objective，而不是真正 task success。

因此当前项目第一版优先使用 LIBERO 原始 sparse success reward，保证 reward 与 benchmark success condition 一致。

---

# 四、后续还可以如何优化

### 1. Phase-aware RL

将任务拆成：

```text
approach
grasp
transport
alignment
placement
release
```

只在高失败阶段启用较大的 residual correction。

在已经稳定的 approach / transport 阶段保持：

$$
a\approx a_{\mathrm{VLA}}
$$

进一步减少无意义 policy drift。

---

### 2. Prioritized Replay

当前 replay uniform sample。

未来可以增加：

* near-success trajectories；
* failure transitions；
* terminal placement states；

的 sampling priority。

对于 Task 9，不必反复训练已经成功的 transport transition，而应重点学习 rack 周围的 placement state。

---

### 3. Adaptive Chunk Length

可以：

```text
coarse motion:
C = 10~20

precision manipulation:
C = 4~5
```

使远距离移动保持 temporal consistency，而在抓取/放置阶段提高 closed-loop replanning frequency。

---

### 4. Independent Exploration / Target Smoothing Noise

当前代码中：

`action_std`

同时控制：

* rollout；
* Actor update；
* target backup。

未来应拆成：

```text
rollout_std
target_policy_std
```

这样可以分别控制真实 exploration 和 Critic target smoothing，而不是通过一个参数同时改变三个训练过程。

---

### 5. Human Intervention

进一步接入：

$$
\text{failed rollout}
\rightarrow
\text{human correction}
\rightarrow
\text{replay}
$$

可以将纯 sparse RL 扩展成：

> pretrained VLA + online RL + corrective demonstrations

对真实机械臂部署尤其有价值。

---

# 五、最终取得的效果

经过 Stage 1 representation learning、Stage 2 conservative residual RL 和多轮 fixed-policy A/B evaluation 后：

### Target Tasks

针对关键抓取与放置失败任务：

$$
SR_{3,9}:
50.0\%
\rightarrow
82.0\%
$$

提升：

$$
\boxed{+32.0pp}
$$

其中主要改善来自：

* multi-stage drawer task 的子目标切换与最终 insertion；
* bottle rack task 的末端姿态调整和精细 placement。

### Guardrail Tasks

其它 8 个 LIBERO-Goal tasks：

$$
86.9\%
\rightarrow
87.0\%
$$

基本保持原有能力：

$$
\Delta\approx +0.1pp
$$

说明 targeted RL optimization 没有以明显 catastrophic forgetting 为代价。

### Overall

在统一 fixed-policy evaluation 口径下：

$$
79.5\%
\rightarrow
86.0\%
$$

整体提升：

$$
\boxed{+6.5pp}
$$

最终形成：

$$
\boxed{
\text{Frozen VLA prior}
+
\text{RL Token representation}
+
\text{Residual Actor-Critic}
+
\text{BC-constrained online correction}
}
$$

的在线后训练方案，使 RL 主要修正 pretrained VLA 在抓取和精细放置关键状态下的局部动作误差，而非重新学习整套 manipulation policy。
