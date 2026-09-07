# CS2 预训练

> **模块**: `scripts/pretrain.py`（训练入口）+ `scripts/pretrain_model.py`（模型）
> **配置**: `config/pretrain-a100.yaml`（单卡 A100 80GB）
> **环境**: `conda activate cs2demo`

---

## 1. 概述

预训练任务：给定 `n_ticks` 个 tick（默认 16 = 4 秒 @ 0.25s/tick）的游戏场面信息，
对**每个玩家独立**预测后续 `n_ticks` 个 tick 的相机运动（离散 token 预测）。

目标：训练 player embedding，融合场上空间信息、历史时序信息和 per-tick 深度/位置/朝向
环境信息，为下游任务提供高质量的特征表示。

### 核心架构

```
Token Embedder (27 tokens/tick)
  → Spatial Transformer (per-tick, 27 token self-attention)
  → Temporal Transformer (per-player causal attention over time)
  → Token Decoder (GPT-style causal, per-tick 注入 depth/xyz/angle 条件)
  → Cross-Entropy Loss（纯离散 token 预测）
```

**设计要点：**
- **纯离散 token 预测**：每个 tick 的相机运动用 7 个离散 token 表示，统一 CE loss
- **3 路 per-tick 条件注入**：decoder 每个 tick group 注入 depth（64 射线编码）、
  absolute xyz、absolute angle（cos/sin yaw/pitch），让模型决策时"看到"未来环境的真实信息
- **残差修正**：tokenizer 编码时记录累积量化误差，防止离散化误差跨 tick 传播
- **玩家洗牌（shuffle）**：每个窗口独立随机 permutation 玩家索引，避免模型依赖固定编号

---

## 2. Camera Tokenizer

### 2.1 Token 布局（7 tokens/tick）

每个 tick 的相机运动编码为 7 个 token，**每个连续值用一个有符号全范围 token**：

| 位置 | 名称 | 编码方式 |
|------|------|---------|
| 0 | continue | binary（1=继续，0=停止/死亡） |
| 1 | d_forward | signed 单 token：`MOVE_OFFSET + bin` |
| 2 | d_right | signed 单 token |
| 3 | d_up | signed 单 token |
| 4 | d_pitch | signed 单 token，`PITCH_OFFSET + bin` |
| 5 | d_yaw | signed 单 token，`YAW_OFFSET + bin` |
| 6 | fire | binary（0/1） |

> 旧的 13-token 布局（sign token + magnitude bin）已废弃，当前为单 token 覆盖完整有符号范围。

### 2.2 词汇表（按 `pretrain-a100.yaml`：move_grid=4.0, angle_grid=5.0）

```
n_move_values  = 2×128/4   + 1 = 65    → MOVE_OFFSET=3,   tokens  3–67
n_pitch_values = 2×90/5    + 1 = 37    → PITCH_OFFSET=68, tokens 68–104
n_yaw_values   = 2×180/5   + 1 = 73    → YAW_OFFSET=105,  tokens 105–177
FIRE_0 = 178, FIRE_1 = 179
vocab_size = 180
```

| 范围 | Token ID | 说明 |
|------|----------|------|
| PAD | 0 | 填充 token |
| 1–2 | 1–2 | continue 0/1 |
| 3–67 | 3–67 | move bins（-128 ~ +128，分辨率 4.0） |
| 68–104 | 68–104 | pitch bins（-90 ~ +90，分辨率 5.0°） |
| 105–177 | 105–177 | yaw bins（-180 ~ +180，分辨率 5.0°） |
| 178–179 | 178–179 | fire 0/1 |

> 分辨率由 `move_grid_size` / `angle_grid_size` 配置，`pretrain.yaml`（小模型）用 1.0 分辨率时 vocab 会更大。

### 2.3 残差修正

编码时（逐 tick）：

```
residual = raw_value + accumulated_error
token    = quantize(residual)
decoded  = dequantize(token)
accumulated_error += raw_value - decoded
```

保证长时间序列的离散化误差不会累积导致轨迹漂移。

### 2.4 10D 连续标签 ↔ token

- `encode_sequence(labels, n_future_ticks)`：`[N, n_ticks, 10]` → `[N, n_ticks×7]`，GPU 向量化
- `decode_sequence(token_ids, n_future_ticks)`：`[N, n_ticks×7]` → `[N, n_ticks, 10]`
- PAD 位置解码为全零；`continue=0` 表示停止

---

## 3. 模型架构

### 3.1 Token Embedder（27 tokens/tick）

10 个玩家 + 1 个炸弹 + 16 个投掷物，每个实体编码为 d 维 token。

- **Player (0–9)**：`MLP1(pos+map_emb)` + `MLP2(state)` + `Σ Emb(weapon slots)`
  + `Σ MLP5(relation + target id)` + `MLP_sound(shots/footsteps)` + `DepthRayEncoder(64 rays)` + `id_emb`
  - 死亡玩家：`dead_emb + id_emb`
  - 共享组件之间通过 **Adapter**（bottleneck d→d/2→d）解耦梯度（`mlp1_player/bomb/proj_adapter`、`pid_self/rel_adapter`、`depth_enc_adapter`）
- **Bomb (10)**：`MLP1(pos+map_emb)` + `MLP4(bomb_state + round_time/160)`
  - `round_time` 经 bomb token 注入空间 attention
- **Projectile (11–26)**：`MLP1(pos+map_emb)` + `MLP3(type_emb + dur + is_active)`，空槽置零

输入特征（collated batch）：

| 张量 | 形状 | 说明 |
|------|------|------|
| `player_pos` | `[B,T,10,3]` | 归一化坐标（地图中心） |
| `player_state` | `[B,T,10,14]` | HP/护甲/闪光/角度 cos·sin/队伍/速度 |
| `player_inv` + `player_inv_mask` | `[B,T,10,9]` | 背包武器索引 |
| `player_rel_f` + `player_rel_i` + `player_rel_mask` | `[B,T,10,9,14]` | 与其他 9 名玩家的关系 |
| `player_sound` | `[B,T,10,2]` | [开火, 脚步] |
| `player_depth` | `[B,T,10,64,5]` | 64 射线方向深度（增强后含角度编码） |
| `player_alive_mask` | `[B,T,10]` | 存活 |
| `bomb_pos` / `bomb_state` | `[B,T,3]` / `[B,T,4]` | 炸弹 |
| `map_idx` | `[B,T]` | 地图索引 |
| `proj_pos` / `proj_type` / `proj_dur` / `proj_is_active` / `proj_mask` | `[B,T,16,...]` | 投掷物 |
| `tick_times_input` | `[B,T]` | round_seconds |

### 3.2 DepthRayEncoder

每条射线 5 维 `[log_dist, cos(yaw), sin(yaw), cos(pitch), sin(pitch)]`，
64 条射线过 `n_depth_ray_layers` 层 Transformer，mean-pool 得到 `[N, d]`。

64 射线 = 5 层同心圈：+60°(8) / +30°(12) / 0°(24) / −30°(12) / −60°(8)。

### 3.3 Spatial Transformer

27 个 token 之间 per-tick self-attention（`n_spatial_layers` 层，Pre-LN），
只 mask 空的投掷物槽位，输出前 10 个 player token。

### 3.4 Temporal Transformer

每玩家独立的 causal self-attention over time（`n_temporal_layers` 层），
叠加连续 sinusoidal time encoding（原始 round 秒数，不归一化）。

### 3.5 Token Decoder（GPT-style + per-tick 条件注入）

**训练（teacher forcing）**：输入 `[cond, tok_0, ..., tok_{seq-2}]`，预测全部 token 位置。

每个 tick group 的 **10 个 decoder 位置**布局（`TOKENS_PER_GROUP=10`，7 个 camera token + 3 个条件槽）：

```
group = tick × 10
target 侧（flat_targets，PAD 不计 loss）:
  pos group+0..2: PAD（3 个条件槽位，不预测）
  pos group+3..9: 7 个 camera token —— continue, d_forward, d_right, d_up, d_pitch, d_yaw, fire

输入侧（decoder_input = [cond, token_emb(gt[:-1])]，位置 i 的输入 = gt[i-1]）:
  pos group+0: cond（仅 tick 0）/ 上一 tick 的 fire token
  pos group+1: depth 条件注入（真实未来 depth 经 DepthRayEncoder 编码）
  pos group+2: xyz 条件注入（真实未来 absolute xyz 经共享 mlp1 编码）
  pos group+3: angle 条件注入（真实未来 cos/sin yaw/pitch 经 mlp_angle 编码）
               —— 该位置的输出预测 continue token
  pos group+4..9: 当前 tick 的前 6 个 camera tokens（teacher forcing）
```

- 条件槽位置的 target 为 PAD，不参与 loss；camera token 位置用 `gt_tokens` 作为 target
- 三个条件经各自的 **Adapter**（`depth_dec_adapter` / `xyz_dec_adapter` / `angle_dec_adapter`）投影到 decoder 空间
- `pos_encoding` 形状 `[1, seq_len+1, d]`（seq_len = 10×n_ticks，+1 为 cond）

**推理（autoregressive）**：`init_generate(conditions)` 预分配全长度状态 →
逐 tick `generate_group`（写 depth/xyz/angle 条件 → 生成 7 个 camera token）。
支持 **KV cache** 增量解码（`DecoderKVCache` + `forward_cached`，结果与全量 forward 一致）。

### 3.6 超参（`pretrain-a100.yaml`）

| 参数 | 值 | 说明 |
|------|-----|------|
| `d_model` | 512 | 隐藏维度 |
| `n_spatial_layers` | 5 | 空间 transformer 层数 |
| `n_temporal_layers` | 3 | 时间 transformer 层数 |
| `n_decoder_layers` | 6 | 解码器层数 |
| `n_depth_ray_layers` | 3 | DepthRayEncoder 层数 |
| `n_heads` | 8 | 注意力头数 |
| `d_ff` | 2048 | FFN 隐藏维度 |
| `dropout` | 0.1 | Dropout |
| `n_ticks` | 16 | 输入/输出窗口 tick 数 |
| `move_range` | 128.0 | 最大移动距离 |
| `move_grid_size` | 4.0 | 移动 token 分辨率 |
| `angle_grid_size` | 5.0 | 角度 token 分辨率 |

**参数量：约 61.8M**（d_model=512 时）。

---

## 4. 数据管线

### 4.1 窗口提取（在线，不落盘）

`CS2PretrainDataset` 读取 **round 级 WDS shard**，在 `__iter__` 中实时切窗口：

```
Round T ticks → 滑动窗口：n_ticks 输入 + n_ticks 输出 = 2×n_ticks−1 总 tick
stride + 随机抖动（jitter）滑动；start_min = 0（不 left-pad，不做假数据）
```

每个窗口包含：
- 输入特征：`player_pos/state/inv/rel/depth/sound` 等，T=n_ticks
- 标签：`label_camera [2×n_ticks−1, 10, 10]`（覆盖输入+输出全部 tick，训练时
  `unfold` 滑动切出每个输入 tick 对应的 n_ticks 输出窗口）
- per-tick 条件标签：`player_depth_labels`、`player_pos_labels`、`player_angle_labels`
  （覆盖输入+输出的 `2×n_ticks−1` 个 tick，供 decoder 条件注入）
- `tick_times_input` / `tick_times_output`（round_seconds，pad 位置外推）
- `output_mask`（输出窗口有效位）

窗口级 shuffle（`shuffle_buffer`）+ 每个窗口独立 **player shuffle**。

### 4.2 训练

```bash
conda activate cs2demo
python scripts/pretrain.py --config config/pretrain-a100.yaml
```

常用覆盖参数：

```bash
# resume（服务器上恢复训练）
python scripts/pretrain.py --config config/pretrain-a100.yaml --resume /path/to/latest_34k.pt

# 本地调试（CPU / 小 batch / 不编译 / 关闭 wandb）
python scripts/pretrain.py --config config/pretrain-a100.yaml \
    --device cpu --batch-size 2 --no-compile --no-wandb --max-samples 20
```

### 4.3 训练细节

- **优化**：AdamW(lr=3e-4, betas=(0.9,0.95))，linear warmup(500) → cosine decay → 0（total 300000 steps）
- **精度**：BF16 AMP（`use_amp`）、TF32 + cudnn.benchmark（`use_tf32`）、torch.compile（`use_compile`），均可 `--no-xxx` 关闭
- **梯度**：`max_grad_norm=1.0`，`grad_accum_steps` 支持梯度累积
- **保存**：`save_interval` 存 `step_{global_step:07d}.pt`（含 optimizer/scheduler），
  `val_interval` 时覆盖写 `latest.pt`；epoch 结束存 `epoch_{epoch:03d}.pt`；结束存 `final.pt`
- **验证**：数据目录下有 `test/` shard 时，`val_interval` 自动跑 teacher-forcing 验证

### 4.4 Resume

`--resume path.pt` 加载 checkpoint：

- 自动剥离 torch.compile 保存的 `_orig_mod.` 前缀，加载到底层原始模型（strict=False，兼容缺参数）
- 检查 checkpoint 的 lr 是否与当前 schedule 吻合：
  - 吻合 → 恢复 optimizer/scheduler 状态
  - 被用户改过 lr → 重置 optimizer（动量不适用新 lr），scheduler 步数置为 `global_step`
- checkpoint 加载到 CPU，避免占用 GPU 显存

**checkpoint 格式**：`{"model", "optimizer", "scheduler", "epoch", "global_step"}`。

---

## 5. Loss

纯 Cross-Entropy over all token positions：

```
loss = CE(token_logits.reshape(-1, vocab), gt_tokens.reshape(-1), ignore_index=PAD)
```

PAD token（id=0，含 3 个条件槽位位置）不计入 loss。metrics：`token_acc`。

---

## 6. Label 格式（`label_camera`）

10D 连续标签（世界对齐坐标系，v5）：

| 维度 | 名称 | 说明 |
|------|------|------|
| 0–2 | d_forward, d_right, d_up | 相机位移（世界坐标：forward/right 水平、up=纯世界 Z） |
| 3–4 | cos/sin(d_pitch) | 俯仰变化 |
| 5–6 | cos/sin(d_yaw) | 偏航变化（处理 359°→0° 回绕） |
| 7 | is_alive | tick t+1 存活 |
| 8 | is_firing | t→t+1 期间开火 |
| 9 | end | 有效=1, padding/末tick=0 |

`is_alive=1` 的位置参与运动 loss，`=0` 跳过。世界位移重建：
`wx = d_forward·cos(yaw) + d_right·sin(yaw)`，`wy = d_forward·sin(yaw) − d_right·cos(yaw)`，`wz = d_up`。

---

## 7. 推理

### 7.1 PredictionEngine

```python
from scripts.prediction_engine import PredictionEngine

engine = PredictionEngine(
    "config/pretrain-a100.yaml",
    "/Users/wanjungu/Downloads/latest_34k.pt",
    device="cuda",
    maps_dir="maps/optimized_obj_files",
)
result = engine.predict_at_tick(sample, query_tick=120)
```

推理流程（`predict_at_tick`）：
1. 截取输入窗口（最近 n_ticks 个 tick）→ Embedder → Spatial → Temporal
2. 提取最后一个 input tick 的 player embedding 作为 condition（`[10, d]`）
3. 起始状态：从 query_tick 的 pos/yaw/pitch/alive 初始化（游戏坐标）
4. 逐 tick 自回归：
   - 实时 raytrace 计算当前 pos/yaw/pitch 的 64 射线深度 → DepthRayEncoder → depth 条件
   - 当前 xyz / 角度 → `_compute_xyz_emb` / `_compute_angle_emb` 条件
   - `generate_group` 生成 7 个 camera token → `decode_sequence` → 10D 标签
   - `_apply_delta` 积分更新 pos/yaw/pitch/alive
5. 返回 GT 轨迹 vs 预测轨迹对比

参数：`temperature`（默认 0 = argmax）、`teacher_forcing_ticks`（前 N 个 tick 用 GT token 替代 AR 生成）。

### 7.2 评估脚本

```bash
# 单 tick 评估（test_pretrain.py，--key 为 sample key，可用 shard 内任意唯一前缀）
python scripts/test_pretrain.py --config config/pretrain-a100.yaml \
    --checkpoint /path/to/latest_34k.pt --data-dir examples/dataset \
    --key some_sample_key --tick 100

# 多 sample / 多 tick 汇总评估（evaluate_pretrain.py）
python scripts/evaluate_pretrain.py \
    --config config/pretrain-a100.yaml \
    --checkpoint /path/to/latest_34k.pt \
    --data-dir examples/dataset --split train \
    --max-samples 10 --tick-step 16 --mode both --device cpu
```

评估指标：
- **Teacher-Forcing**：loss、token_acc（按 token 类型拆分）
- **Auto-Regressive**：AR token_acc、ADE / FDE（预测轨迹 vs GT 轨迹的平均/最终位移误差）

### 7.3 下游 embedding

```python
emb = model.get_player_embeddings(batch)   # [B, T, 10, d]
```

返回每个 tick 每个玩家的 embedding（Embedder + Spatial + Temporal，不含 decoder）。
可视化脚本：`scripts/downstream/visualize_player_embeddings.py`（见 `scripts/downstream/README.md`）。

---

## 8. 文件结构

```
scripts/
  pretrain.py                    # 训练入口
  pretrain_model.py              # 模型定义（CameraTokenizer + CS2PretrainModel）
  prediction_engine.py           # 推理引擎（含 per-tick depth/xyz/angle 构建）
  evaluate_pretrain.py           # 多 sample/tick 汇总评估
  test_pretrain.py               # 单 tick 评估 + 注意力可视化
  create_pretrain_data.py        # Round WDS → Pretrain WDS（可选落盘）
  training_data/
    pretrain_processor.py        # 窗口提取（PretrainWindowExtractor + shuffle_players）
    torch_dataset.py             # CS2PretrainDataset + pretrain_collate_fn
    feature_builder.py           # 特征构建
    label_builder.py             # 标签构建（含 label_camera）
    depth_map.py                 # 深度图（raytrace）
    config.py                    # 常量与坐标变换
  downstream/
    visualize_player_embeddings.py   # 下游 embedding 可视化

config/
  pretrain.yaml                  # 小模型配置
  pretrain-a100.yaml             # A100 配置（当前训练用）

docs/
  pretrain.md                    # 本文档
  torch-dataset.md               # PyTorch Dataset 文档
  training-data-format.md        # 数据格式文档
```
