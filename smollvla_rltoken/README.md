# RLT (RL Token) 复现 — 基于 LeRobot SmolVLA

复现论文 **RL Token: Bootstrapping Online RL with Vision-Language-Action Models**
(arXiv:2604.23073, Physical Intelligence)。方法调研见
[`/mnt/sdb/feng/rltoken/RLTOKEN_NOTES.md`](../rltoken/RLTOKEN_NOTES.md);
本仓库是 [`/mnt/sdb/feng/rltoken`](../rltoken)(PI0.5 版复现)向 **SmolVLA** 的移植。

论文基于闭源的 π0.6;本复现把同样的两阶段流程搭在
**SmolVLA**(lerobot 0.5.1 自带:SmolVLM2-500M VLM(取前 16 层,hidden 960)
+ 0.75× 宽度 action expert(hidden 720,自注意力/交叉注意力交替),
flow matching,H=chunk_size=50)之上,checkpoint 为 `./smolvla_base`
(SO-100 任务,3 相机 256×256、6 维状态/动作)。

## 运行环境

复用 `~/pi05_reproduce` 的虚拟环境(torch 2.10 + lerobot 0.5.1 + transformers 5.3,
另装了 `num2words` 供 SmolVLM processor 使用):

```bash
PY=/home/feng/pi05_reproduce/.venv/bin/python
```

SmolVLA 是 lerobot 0.5.1 的内置 policy,**无需**像 PI05 版那样 vendor 代码 +
兼容层;`rlt/smolvla_compat.py` 只做 checkpoint 加载。注意
`smolvla_base/config.json` 里 `load_vlm_weights: true`,首次加载会从 HF Hub
下载 `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` 底座(~1.2G,已入缓存),
然后再覆盖上 checkpoint 的微调权重。

## 与 PI05 版的对应关系

| 论文 | PI05 版复现 | 本 SmolVLA 版 |
|---|---|---|
| π0.6 最终层嵌入 z_{1:M} | PaliGemma prefix final hidden(宽 2048,3×256 图像 token) | SmolVLM2 prefix final hidden(宽 **960**;每相机 **64** 个 connector token,3 相机共 192;prefix 末尾还有 48 语言 token + **1 个 state token**,默认 image-only 丢弃) |
| ã_{1:H} ~ π_vla | `sample_actions` 复用 prefix KV cache | 同(`VLAFlowMatching.denoise_step`,`config.num_steps=10`) |
| RL token g_φ/d_φ | d_model=1024(≈vla_width/2) | d_model=**512**(≈960/2,~15M 参数;`--d-model 960` 可对齐 1×vla_width) |
| RL chunk C / 动作维 | C=10,d=14(双臂) | C=10,d=**6**(SO-100) |
| actor/critic、replay、Alg.1 | `actor_critic.py` / `replay_buffer.py` / `train_online.py` | 逐字节相同(与 VLA 无关) |

SmolVLA 相对 PI05 的两个结构差异,对提取逻辑的影响:

1. **prefix 里带 state token**:`embed_prefix(images, ..., state=state)` 会把
   `state_proj(s)` 作为最后一个 prefix token。RL 状态 x 已单独拼接本体感知,
   所以默认 `use_image_tokens_only=True` 只取前 n_img 个 token(与论文脚注一致,
   也顺带丢掉语言与 state token)。
2. **注意力掩码是 3D bool**(不像 PI05 需要 `_prepare_attention_masks_4d` +
   强制 eager),`vlm_with_expert.forward` 的第一个返回值直接就是
   prefix 最终层隐状态,提取代码反而更简单。

## 代码结构

| 文件 | 对应论文 | 内容 |
|---|---|---|
| `smolvla_compat.py` | — | checkpoint 加载(可选整体 cast dtype) |
| `configs.py` | App. B | `RLTokenConfig` / `ActorCriticConfig` / `OnlineRLConfig`(默认值按 smolvla_base:vla_width 960、action/proprio 6 维) |
| `rl_token.py` | Sec. IV-A, Eq.1-2 | `SmolVLAPrefixExtractor`(一次 prefix 前向同时产出 z_{1:M} 与 KV cache,cache 复用于参考 chunk 采样);`RLTokenModule`(encoder + e_rl 读出 z_rl;causal decoder teacher-forcing 自回归重建 sg(z)) |
| `actor_critic.py` | Sec. IV-B, Eq.3-5 | 固定 σ 高斯 chunk actor(参考动作 pass-through + 50% 输入 dropout)、双 Q ensemble critic(TD3 target、min backup)、`RLTAgent`(2 次 critic : 1 次 actor) |
| `replay_buffer.py` | Sec. V | chunk 级转移 + stride-2 子采样组装(o>0 的窗口跨两个相邻 chunk;H=50>C=10 保证平移参考可得) |
| `rlt_policy.py` | Sec. V Rollout | `RLTController`:chunk 边界一次 VLA 前向 → z_rl + ã_{1:H} → actor 输出 a_{1:C} |
| `envs.py` | — | `ChunkEnv` 协议(真机/仿真接入点,含干预接口)+ `MockManipEnv` 冒烟环境(6-DoF、相机键与 smolvla_base 一致、SmolVLM2 tokenizer + 尾部换行) |
| `train_rl_token.py` | Alg.1 行 1-3 | Stage 1:演示数据上训练 L_ro(可选 α·L_vla 联合 SFT;processor 用 `make_smolvla_pre_post_processors`) |
| `train_online.py` | Alg.1 行 4-19 | Stage 2:warmup → rollout → 子采样入 buffer → UTD 更新 |
| `scripts/smoke_test.py` | — | GPU 端到端自检(10 项断言) |

## 使用

### Stage 1 — 训练 RL token(离线,演示数据)

```bash
cd /mnt/sdb/feng/smollvla_rltoken
CUDA_VISIBLE_DEVICES=0 $PY -m rlt.train_rl_token \
  --checkpoint smolvla_base \
  --dataset lerobot/libero_10 \
  --dataset-root /home/feng/pi05_reproduce/data/lerobot/libero_10 \
  --steps 5000 --batch-size 16 --out outputs/rl_token_libero10
```

论文建议 2000–10000 步。`--vla-sft-alpha > 0` 可联合微调 VLA(默认冻结)。
注意:smolvla_base 是 SO-100 checkpoint,拿 LIBERO 演示只是验证管线;
正式使用时 `--dataset` 应指向与 checkpoint 任务/本体一致的演示数据
(数据集的相机键会覆盖 policy config 的 `input_features`)。

### Stage 2 — 在线 RL(Algorithm 1)

```bash
CUDA_VISIBLE_DEVICES=0 $PY -m rlt.train_online \
  --checkpoint smolvla_base \
  --rl-token outputs/rl_token_libero10/rl_token.pt \
  --mock-env --total-env-steps 100000 --warmup-env-steps 2000
```

真实机器人 / 仿真接入:实现 `envs.ChunkEnv` 协议(`reset` / `step` /
`obs_to_batch` / `get_intervention`),替换 `train_online.py` 中的
`MockManipEnv`。`obs_to_batch` 须给出与 SmolVLA 预处理一致的 batch
(图像 [0,1] float、`observation.state` 原始维度即可——`prepare_state`
自动 pad 到 32、SmolVLM2 tokenizer 的语言 token,状态/动作在归一化空间;
参考 `MockManipEnv`)。`get_intervention` 返回人类干预 chunk 时,执行动作与
buffer 中的参考都会按论文替换为干预动作。

### Smoke test

```bash
CUDA_VISIBLE_DEVICES=0 $PY scripts/smoke_test.py
```

## 实测(单张 RTX 5090,2026-07-07)

- smoke test 10 项断言全过:prefix z=(B,192,960)、M_total=241(192 img + 48
  lang + 1 state)、ref chunk (B,50,32)、L_ro 反传、actor/critic 更新、
  mini Algorithm-1 rollout→buffer→update。
- Stage 1(LIBERO-10、batch 8、混合精度默认布局):~3.8 it/s,
  L_ro 30 步内 1.46 → 0.45(PI05 版 2000 步 1.74 → ~0.3,收敛更快因为
  嵌入维度小)。
- Stage 2(mock env、batch 64):~33 env steps/s(含每 chunk 25 步梯度更新),
  1200 步内 success20 0 → 0.10、Q 值随稀疏奖励回传上升。

## 与论文的差异(已知取舍)

1. **基座模型**:π0.6(闭源)→ SmolVLA(450M);z 取 SmolVLM2 前 16 层
   final hidden(宽 960),默认只取图像 token(每相机 64 个,信息密度比
   PaliGemma 的 256/相机更高地压缩)。
2. **RL token 宽度**:论文图示 1×2048(=VLA 宽);默认 `d_model=512`
   (~15M 参数),`--d-model 960` 可对齐 1×vla_width。
3. **BC 正则**:Eq.5 的 ‖·‖² 实现为按元素 mean(与 sum 只差常数因子,
   被 β 吸收)。
4. **异步性**:论文 rollout 与更新异步;本实现单进程串行
   (每个 chunk 后做 `utd × C/stride` 步更新)。
5. **无真机**:附 `MockManipEnv` 验证管线;奖励/干预接口保留为论文语义。
6. **截断处理**:时间截断的 episode 以 done=0 入库(不清零 bootstrap),
   论文未说明该细节。
7. **warmup 冷启动**(与 PI05 版相同的注意点):串行实现 warmup 期间不做
   梯度更新,warmup 后第一个 actor chunk 来自未训练网络;上真机前建议
   warmup 结束先在 warmup replay 上补更新,或把 actor 改成
   `μ = ref + Δ_θ`(末层零初始化)的残差参数化。
