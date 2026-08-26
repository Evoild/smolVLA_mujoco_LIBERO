# RLT 真机训练与推理流程（以插拔类任务为例）

> 目标任务示例：网线/充电线插入（对应论文 Ethernet / charger 任务）。
> 基于本仓库 `rlt/` 复现代码；方法细节见 `RLTOKEN_NOTES.md`，代码结构见 `rlt/README.md`。
>
> 环境：`PY=/home/feng/pi05_reproduce/.venv/bin/python`，跑 GPU 前 `nvidia-smi` 挑空闲卡。

## 总览

```
① 100 条示教训练 RL-token（可同时/先做 PI0.5 SFT），训完 PI0.5 与 RL-token 全部冻结
     示教从任务自然起点录全程（SFT 需要接近段），插入段录厚（含失败重试）
② PI0.5 warmup rollout + 人类必要接管，填 replay（成功/失败标注 reward）
     每个 episode 从 reset() 起点开始：已抓插头、距端口 3–5 cm、位姿加随机扰动
③ offline warm-start：用 warmup replay 先预训练 critic/actor 若干步   ← 上真机的关键安全步骤
④ online RL：actor 控制关键阶段，边跑边训
     每个 episode 结束（成功/超时）→ reset() 回关键阶段起点 → 开下一个 episode
⑤ 每隔固定 episode 做无接管 deterministic 评估（评估 episode 同样从 reset() 起点开始）
⑥ 部署：基础 PI0.5 走接近段 → 切换 RLT 策略完成插入
     部署无 reset()：PI0.5 交接给 RLT 的切换点扮演训练时 reset 起点的角色
```

**reset() 的角色**：`env.reset()` 只在训练/评估阶段（②–⑤）的每个 episode
边界被调用（`train_online.py` 主循环初始化一次、每个 episode 结束后一次），
真机实现为一段脚本化例程——机械臂沿安全路径退回预插入位姿（插头掉了则
人工递回或跑抓取脚本），再加随机偏移。它是论文 targeted practice 的载体：
把全部机器人练习时间集中在插入段，接近段不归 RL 管。

**三个分布必须对齐**，链条才闭合：

```
示教插入段的起始状态分布 ⊇ RL 训练 reset() 的起点分布 ≈ 部署时 PI0.5 → RLT 的切换点分布
```

示教覆盖要包住 reset 分布（RL token 与参考动作在那里才可靠）；reset 分布
要对准部署切换点（否则切换瞬间 actor 处于分布外）。

其中 ③ 是对论文串行复现版的工程加固：论文 rollout 与 learner 异步，learner 从
buffer 有数据起就在训练，warmup 结束时 actor 已经学过 warmup 数据；本仓库单进程
串行实现把更新 gate 在 warmup 之后（`rlt/train_online.py` 中
`if not warmup and len(buffer) >= cfg.batch_size:`），**warmup 结束后第一个 actor
chunk 来自随机初始化网络**，真机上不可裸跑。③ 实际上是把串行实现拉回异步版的等价行为。

---

## 阶段 0：任务定义与数据采集（1–2 天）

- **动作空间**：单臂建议末端增量位姿 6 维 + 夹爪 1 维 → `action_dim=7`；
  本体感知用关节位置 + 速度（7+7）→ `proprio_dim=14`。控制频率对齐 PI0.5
  训练数据（50 Hz 或数据集 fps）。
- **相机**：至少一路**腕部相机**（插拔的"最后一毫米"全靠它），加 1–2 路外部
  视角凑齐 PI0.5 的 3 路输入，224×224。
- **遥操作采集约 100 条演示**，存 LeRobot 数据集格式
  （`observation.images.*`、`observation.state`、`action`、固定 prompt 如
  `"insert the ethernet cable into the port"`）。
- **示教从任务自然起点录全程**（不要从 reset 起点开始录——SFT 需要接近段
  数据，部署时接近段归 PI0.5 管）。建议配比：
  - 主体（~70 条）：自然起点 → 接近 → 插入成功的全程演示；
  - 补充（~30 条，可选）：从类似 reset 起点的位姿开始的短演示，只录插入段，
    密集覆盖各种偏移/角度下的对准-插入-重试。
- **关键**：演示要覆盖插入失败后的重试/调整动作（故意先插偏再纠正、
  最终成功），而不是只录完美轨迹——后续 RL 在这个分布附近探索。
  纯失败（最终没成功）的 episode 不要：SFT 会把失败学进去；负样本由
  warmup/online 阶段的 rollout 自动提供。

## 阶段 1：训练 RL-token（+ 可选 PI0.5 SFT）

`checkpoint/pi05_base` 是 base 模型，对新机器人/任务通常要先 SFT
（用 `~/pi05_reproduce` 的训练脚本），得到 `checkpoint/pi05_plug_sft`。
验收标准：SFT 后的 PI0.5 裸跑能**接近插座并偶尔成功**（论文场景约 20%
成功率即可）——它不需要好，只需要够到关键阶段。

然后训练 RL token（约 1 小时 GPU）：

```bash
cd /mnt/sdb/feng/rltoken
CUDA_VISIBLE_DEVICES=1 $PY -m rlt.train_rl_token \
  --checkpoint checkpoint/pi05_plug_sft \
  --dataset your/plug_task --dataset-root /path/to/plug_dataset \
  --steps 5000 --batch-size 8 --out outputs/rl_token_plug
```

- 看 `loss_ro` 收敛到平台即可（参考：LIBERO-10 上 2000 步从 1.74 → ~0.3）。
- 也可合并两步：在 base 上直接 `--vla-sft-alpha 1.0` 联合训练，
  一次产出 SFT 模型（`pi05_sft.pt`）+ RL token。
- **训练完成后 PI0.5 与 RL-token 全部冻结**，后续阶段不再更新。

## 阶段 2：实现真机 `ChunkEnv`（唯一的对接代码）

实现 `rlt/envs.py` 中 `ChunkEnv` 协议的四个方法，替换 `train_online.py`
里的 `MockManipEnv`：

| 方法 | 实现要点 |
|---|---|
| `reset()` | **直接摆到关键阶段起点**：夹爪已抓插头、距端口 3–5 cm、位姿加随机扰动（论文 targeted practice——接近段不归 RL 管） |
| `step(action)` | 动作在 PI05 归一化空间，先过 postprocessor 反归一化再下发；返回 `(obs, reward, success)`。**稀疏 +1 奖励**：Ethernet 可自动判（交换机 link 灯 / 电气导通 / 力传感到位），否则操作员脚踏板确认 |
| `obs_to_batch()` | 相机图 + 关节状态 + 固定 prompt 打包成 preprocessor 输入 |
| `get_intervention()` | 检测遥操作设备（space mouse / 手柄）是否接管；接管时返回该 chunk 的 10 步人类动作。代码会自动把执行动作与 buffer 中的参考都替换为干预动作（论文 Sec. V 语义） |

**安全兜底写在 env 内部**（不依赖上层策略）：工作空间盒限位、力/扭矩
超阈值急停、单步动作幅度 clip。

## 阶段 3：warmup 填 replay

- 每个 episode 从 `reset()` 起点（关键阶段起点）开始；episode 结束
  （成功/超时截断）后再次 `reset()` 开下一个。
- 约 **2000 env 步 ≈ 30–50 个 episode**，全部执行 PI0.5 参考 chunk
  （`use_actor=False` 路径）。
- 人类只在危险时接管；干预动作自动入 buffer 当参考，本身是高质量监督信号。
- 每个 episode 的成功/失败要标注（reward 进 replay）；成功以 done=1 入库，
  时间截断以 done=0 入库（保留 bootstrap）——buffer 已按此实现。
- 每个 chunk 经 stride-2 子采样产生约 5 条转移，2000 步 warmup ≈ 1000 条转移。

## 阶段 4：offline warm-start（上真机前必须）

**目的**：critic 先从 warmup 数据学"哪些状态-动作更可能成功"；actor 被
BC 正则拉到接近 PI0.5 / 人类干预动作，避免第一个 actor chunk 输出乱动作。

**流程**：

```
warmup 填 replay
  → replay 数量 ≥ batch_size
  → 先做若干 gradient updates（建议按 UTD 记账：utd × N_warm/stride，
     即 5 × 2000/2 = 5000 步，与"异步版一直在训"等价）
  → 再允许 actor 控制真机
```

**注意**：

- 当前代码**没有**这一步（warmup 期间 `agent.update` 不执行），需要在
  `train_online.py` 主循环里加：warmup 结束的瞬间、第一次
  `run_chunk(use_actor=True)` 之前，插入一段离线更新循环。
- warm-start 期间 buffer 小（~1000 条），别把更新步数拉得过多导致 critic
  过拟合；5000 步（UTD 记账）是合理上限。
- 更根本的保险（可选叠加）：把 actor 改成**残差参数化**
  `μ = ref + Δ_θ(x, ref)`、末层零初始化——未训练的 actor 严格等于执行
  PI0.5 动作。这是对论文 Eq.4（直接输出 μ）的偏离，但真机上最稳。

## 阶段 5：online RL（论文经验：400–1000 episodes ≈ 15 分钟–5 小时机器人时间）

```bash
CUDA_VISIBLE_DEVICES=1 $PY -m rlt.train_online \
  --checkpoint checkpoint/pi05_plug_sft \
  --rl-token outputs/rl_token_plug/rl_token.pt \
  --action-dim 7 --proprio-dim 14 \
  --chunk-len 10 --action-std 0.05 --bc-beta 1.0 --utd 5 \
  --warmup-env-steps 2000 --max-episode-steps 300 \
  --num-inference-steps 5 \
  --out outputs/rlt_plug
```

（真机 env 接好后去掉 `--mock-env` 限制，换成你的 ChunkEnv。）

- **循环行为**：每个 chunk 边界一次 VLA prefill（同时产出 z_rl 和参考
  chunk ã_{1:50}）→ actor 输出 a_{1:10} 执行 → stride-2 子采样入 replay →
  每 chunk 做 `utd × C/stride = 25` 步更新（2 critic : 1 actor）。
- **人的角色**：只做安全接管 + 终端成功/失败标注。
- **实时性**：C=10 @ 50 Hz → 每 0.2 s 完成一次 prefill + 去噪 + actor 前向。
  5090 上 `--num-inference-steps` 降到 3–5 基本够；不够就异步化
  （边执行当前 chunk 边规划下一个）。
- **监控与调参**：
  - `q_mean` 应随成功样本增多稳步上升；`bc_dist` 先降后稳；
  - actor 动作离参考太远/乱动 → 升 `--bc-beta`；
    成功率卡在 PI0.5 水平不涨 → 降 `--bc-beta`；
  - 精度要求高、2×256 网络不够时按论文螺丝任务升
    `--n-layers 3 --hidden-dim 512`。

## 阶段 6：固定间隔无接管评估

每 **25 或 50 个训练 episode**，跑 **10–20 个无接管评估 episode**：

- actor 用 `deterministic=True`（`plan_chunk` 已支持），关闭探索噪声；
- 无人类干预；
- 记录：成功率、平均插入耗时、（训练段的）接管率、失败类型
  （没对准 / 插一半 / 撞限位 …）。

**注意**：当前代码没有独立评估循环（日志里的 success20 混着探索噪声和
干预），需要在 `train_online.py` 里加一个纯评估分支——评估 episode
不入 replay、不更新。

## 阶段 7：部署推理

完整任务的执行方式（论文做法）：

```
基础 PI0.5 走接近段（抓插头、移动到插座附近）
  → 切换点（先用人工触发或距离阈值；收尾可再微调 VLA 让它自己预测切换）
  → RLT 策略完成插入：每 10 步一次
     plan_chunk(use_actor=True, deterministic=True)
```

- 单次重规划成本 ≈ 原 PI0.5 一次推理（prefill + 去噪），RL token
  encoder（58M）与 actor MLP（<1M）开销可忽略；区别只是重规划频率
  从每 50 步提高到每 10 步。
- 动作全程在 PI05 归一化空间，出环境边界前必须过 postprocessor 反归一化。
- `plan_chunk(use_actor=False)` = 高频重规划的原始 PI0.5，可作对照基线。

## 预期收益校准（论文数据）

- Ethernet 插入：成功率持平或更高，速度约 2×，插入段涌现"贴住端口
  加压微调"的非演示行为；
- 螺丝任务：成功率 20% → 65%；
- 关键阶段最高 3× 提速，部分任务超过人类遥操作速度。

## 待补的代码清单

真正要写的新代码只有两块，其余是 `train_online.py` 内的小改动：

1. **真机 `ChunkEnv`**（阶段 2）——唯一的硬对接工作；
2. **offline warm-start**（阶段 4）——warmup 结束、actor 接管前插入
   `utd × N_warm/stride` 步离线更新；
3. 独立无接管评估循环（阶段 6）；
4. （可选）actor 残差参数化 + 末层零初始化（阶段 4 的加强保险）。
