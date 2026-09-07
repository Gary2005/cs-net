# CS2 训练数据格式 V2

> **Version**: `cs2.training.v5`
> **CLI**: `python scripts/create_training_data.py --input-dir data/demos/json --output-dir data/dataset`
> **依赖**: `cs2demo` conda 环境

---

## 1. 概述

训练数据管线将 demo_parser 的 V2 JSON 输出（`cs2.demo.v2`）转换为 **WebDataset shard**，
每个 sample 为 **一个完整回合** 的时序数据，可直接被 PyTorch DataLoader 读取。

### 与旧版本的主要区别

| 特性 | 旧版 (V1) | 新版 (V2) |
|------|----------|----------|
| 时间维度 | 单 tick 窗口 (`T=1`) | 全回合时序 (`T=30~600+`) |
| 深度图 | 无 | 16 方向简化深度 (Open3D 射线检测) |
| 声音 | 无 | 开火/脚步声计数 |
| 玩家身份 | steamid 编号 | steamid + 职业选手名 |
| 回合组织 | tick 级滑动窗口 | 回合级打包 |
| Token 数 | 31 (10P+1B+20Proj) | 27 (10P+1B+16Proj) |
| 压缩 | zstd level 3 | zstd level 3 |

---

## 2. 快速开始

```bash
# 处理所有 JSON 文件（无深度图，最快）
python scripts/create_training_data.py \
    --input-dir data/demos/json \
    --maps-dir maps/optimized_obj_files \
    --output-dir data/dataset \
    --no-depth \
    --workers 4 \
    --verbose

# 包含深度图（需要地图 OBJ 文件）
python scripts/create_training_data.py \
    --input-dir data/demos/json \
    --maps-dir maps/optimized_obj_files \
    --output-dir data/dataset \
    --test-matches-per-map 4 \
    --workers 4 \
    --verbose

# 断点续传（跳过已处理的文件）
python scripts/create_training_data.py \
    --input-dir data/demos/json \
    --maps-dir maps/optimized_obj_files \
    --output-dir data/dataset \
    --done-file data/dataset/processed.txt
```

---

## 3. 输出结构

```
data/dataset/
  train/
    shards-00000.tar      # 训练集 shard（最多 5GB/个）
    shards-00001.tar
    ...
  test/
    shards-00000.tar      # 测试集 shard
    ...
  processed.txt           # 已处理文件列表（断点续传用）
```

### 训练/测试划分

每个回合以概率 `--test-split`（默认 0.02 = 2%）随机分配到测试集，其余进入训练集。
随机种子通过 `--seed`（默认 42）固定，确保可复现。

---

## 4. Sample 结构

每个 WebDataset sample 包含以下字段：

### 4.1 元数据 (`meta.json.zst`)

```json
{
    "format": "cs2.training.v5",
    "source_file": "navi-vs-faze-m1-dust2.json.gz",
    "match_teams": ["Navi", "Faze"],
    "map_name": "de_dust2",
    "map_idx": 1,
    "round_id": 5,
    "teams": ["CT", "T", "CT", "T", "CT", "T", "T", "T", "CT", "CT"],
    "winner": "CT",
    "end_reason": "bomb_exploded",
    "players": [
        {"steamid": "76561198000000001", "name": "s1mple"},
        ...
    ],
    "T": 82,
    "tick_interval": 0.25,
    "depth_config": {
        "n_directions": 64,
        "max_dist": 5000.0
    }
}
```

### 4.2 特征张量 (`*.npy.zst`)

所有张量使用 zstandard (level=3) 压缩的 numpy 数组。各字段的符号约定：

| 符号 | 含义 | 典型值 |
|------|------|--------|
| `T` | 本回合 tick 数 | 30 ~ 200 |
| `N` | 玩家人数 | 10 |
| `P` | 每 tick 最多投掷物 | 16 |

#### Token 0–9：玩家特征

| 字段 | 形状 | dtype | 说明 |
|------|------|-------|------|
| `player_pos` | `[T, 10, 3]` | float32 | x,y,z（地图中心归一化，范围 [-1,1]） |
| `player_alive_mask` | `[T, 10]` | bool | True = 玩家存活（统一替代原 pos/state/sound 三个 mask） |
| `player_state` | `[T, 10, 14]` | float32 | 见下方状态特征表 |
| `player_inv` | `[T, 10, 9]` | int32 | 背包武器索引（尾补 0） |
| `player_inv_mask` | `[T, 10, 9]` | bool | True = 该武器槽位有效 |
| `player_rel_f` | `[T, 10, 9, 14]` | float32 | 对其他玩家的关系特征 |
| `player_rel_i` | `[T, 10, 9]` | int32 | 关系目标玩家索引 |
| `player_rel_mask` | `[T, 10, 9]` | bool | True = 关系有效 |
| `player_depth` | `[T, 10, 64]` | float32 | 64 方向深度距离（log 归一化 [0,1]） |
| `player_depth_mask` | `[T, 10]` | bool | True = 深度图有效 |
| `player_sound` | `[T, 10, 2]` | float32 | `[是否开火, 是否有脚步]`（0/1） |

**状态特征 (14 dims)**：

| 索引 | 特征 | 归一化 |
|------|------|--------|
| 0 | HP | `/ 100` |
| 1 | 护甲 | `/ 100` |
| 2 | 头盔 | 0/1 |
| 3 | 拆弹器 | 0/1 |
| 4 | 闪光持续时间 | `/ 5` |
| 5 | 闪光透明度 | `/ 255` |
| 6 | cos(pitch) | — |
| 7 | sin(pitch) | — |
| 8 | cos(yaw) | — |
| 9 | sin(yaw) | — |
| 10 | 是否 CT | 0/1 |
| 11 | 前进速度（摄像机方向） | log-norm 有符号，max_dist=500，输入已乘 VELOCITY_SCALE(1/32) |
| 12 | 横移速度（摄像机右侧） | log-norm 有符号，max_dist=500，输入已乘 VELOCITY_SCALE(1/32) |
| 13 | 垂直速度（Z 轴） | log-norm 有符号，max_dist=500，输入已乘 VELOCITY_SCALE(1/32) |

**关系特征 (14 dims)**：

| 索引 | 特征 | 说明 |
|------|------|------|
| 0 | 前向距离（视线方向） | log-norm 有符号，正=沿视线前方 |
| 1 | 横向距离（视线右侧） | log-norm 有符号，正=视线右侧 |
| 2 | 垂直距离（视线上方） | log-norm 有符号，正=视线上方 |
| 3 | log(dist+1) / log(5001) | 对数距离 [0,1] |
| 4 | 是否队友 | 0/1 |
| 5 | 是否敌方 | 0/1 |
| 6 | 被我看见 | 0/1 |
| 7 | 看见我 | 0/1 |
| 8 | cos(视线角差) | 目标偏离视线的水平分量 |
| 9 | sin(视线角差) | 同上 |
| 10 | cos(视线仰角) | 目标相对视线的仰角 |
| 11 | sin(视线仰角) | 同上 |
| 12 | 目标是否存活 | 0/1 |
| 13 | 目标 HP | `/ 100` |

> 索引 0-3 均使用对数归一化：`sign(val) × log(|val|+1) / log(5001)`，
> 近处分辨力高于远处，全范围 [-1, 1]（dist_log 为 [0, 1]）。
> 索引 0-2 基于**眼睛位置**（Z+64）做 3D 旋转（含 yaw+pitch），
> 参考系为摄像机基向量（前=视线方向，右=垂直视线的水平向量，上=摄像机上方）。
> 索引 8-11 描述目标方向与视线的关系，与 0-2 互补。

#### Token 10：全局 / Bomb

| 字段 | 形状 | dtype | 说明 |
|------|------|-------|------|
| `bomb_pos` | `[T, 3]` | float32 | 炸弹位置（归一化） |
| `bomb_state` | `[T, 4]` | float32 | `[回合时间/160, 已安装, 已掉落, 安装时长/40]` |
| `map_idx` | `[T]` | int32 | 地图索引（常量） |

#### Token 11–26：投掷物

| 字段 | 形状 | dtype | 说明 |
|------|------|-------|------|
| `proj_pos` | `[T, 16, 3]` | float32 | 投掷物位置（归一化） |
| `proj_type` | `[T, 16]` | int32 | 投掷物类型索引，-1 = 空 |
| `proj_dur` | `[T, 16]` | float32 | 剩余持续时间 `/ 25`（飞行道具为 0，仅活跃烟雾/火焰有意义） |
| `proj_mask` | `[T, 16]` | bool | True = 该槽位有效 |
| `proj_is_active` | `[T, 16]` | int32 | 0 = 飞行道具，1 = 落地持续效果（烟雾/火焰） |

> `proj_is_active=0`（飞行）时 `proj_dur=0` 无意义；`proj_is_active=1`（落地）时 `proj_dur` = 剩余持续时间 / 25。
> 同一颗烟雾弹不会同时出现在 flying 和 active 两个状态——飞行时在 `grenades` 列表，落地后转入 `smokes` 列表。

投掷物类型：`smoke=0, inferno=1, he=2, flashbang=3, decoy=4, molotov=5`

#### Token 掩码

| 字段 | 形状 | dtype | 说明 |
|------|------|-------|------|
| `token_pad_mask` | `[T, 27]` | bool | 由 collate_fn 动态生成，True = pad token |

#### 标签

| 字段 | 形状 | dtype | 说明 |
|------|------|-------|------|
| `label_winrate` | `[T]` | float32 | 0 = CT获胜，1 = T获胜 |
| `label_nxt_kill` | `[T]` | int32 | 下一个获得击杀的玩家（0-9），10 = 无 |
| `label_nxt_death` | `[T]` | int32 | 下一个死亡的玩家（0-9），10 = 无 |
| `label_alive_end` | `[T, 10]` | float32 | 1.0 = 回合结束时存活 |
| `label_bombsite` | `[T]` | int32 | 炸弹安放包点：0 = A, 1 = B, 2 = 未知/未安放 |
| `label_win_reason` | `[T]` | int32 | 获胜原因：0 = CT全灭, 1 = T全灭, 2 = 炸弹爆炸, 3 = 炸弹拆除, 4 = 时间耗尽, 5 = 其他 |
| `label_camera` | `[T, 10, 10]` | float32 | **预训练标签**：每个 tick 相对上一 tick 的相机运动 + 辅助信号（见下方） |
> 标签有效性由 `is_alive`（dim 7）隐式表示：`is_alive=1` 的位置计算 loss，`=0` 跳过。

**相机运动标签 (`label_camera`) 10 维说明：**

| 索引 | 特征 | 说明 |
|------|------|------|
| 0 | `d_forward` | 水平面上沿视线方向（仅 yaw）的位移（游戏单位/tick） |
| 1 | `d_right` | 水平面上垂直于视线方向的位移（游戏单位/tick） |
| 2 | `d_up` | 世界 Z 轴（垂直）位移，不受视角 pitch 影响（游戏单位/tick） |
| 3 | `cos(d_pitch)` | 俯仰角变化的余弦 |
| 4 | `sin(d_pitch)` | 俯仰角变化的正弦 |
| 5 | `cos(d_yaw)` | 偏航角变化的余弦（已处理 359→0 回绕） |
| 6 | `sin(d_yaw)` | 偏航角变化的正弦（已处理 359→0 回绕） |
| 7 | `is_alive` | tick t+1 时刻是否存活（1=存活，0=死亡） |
| 8 | `is_firing` | t → t+1 期间是否开火（1=开枪，0=未开枪） |
| 9 | `end` | label 是否有效（1=有效，0=padding/末tick） |

> Loss 顺序：先 BCE(end)，end=1 则再 BCE(is_alive)，is_alive=1 则再 MSE(相机运动+is_firing)

> **v5 坐标系约定：位移分量使用世界对齐坐标系。**
> `d_forward` 和 `d_right` 在世界水平面上（仅由 yaw 决定，不受 pitch 影响），
> `d_up` = 纯世界 Z 位移（dz）。角度差 `Δpitch, Δyaw` 用 cos/sin 编码避免 359°→0° 回绕。
> 世界位移重建：`wx = d_forward*cos(yaw) + d_right*sin(yaw)`, `wy = d_forward*sin(yaw) - d_right*cos(yaw)`, `wz = d_up`。
> **`is_alive`（dim 7）即为有效性标记**：`is_alive=1` 处算 loss，`=0` 跳过。
> 末 tick（T-1）无 t+1，全零 is_alive=0。

> `label_nxt_kill` 和 `label_nxt_death` 是回合级别的 11 分类标签（0-9 = 玩家索引，10 = 无事件），每个 tick 指示下一对击杀/死亡发生的玩家。
> `label_bombsite` 是**回合级常量**（与 `label_winrate` 语义一致），
> 即使安放前的 tick 也填入最终安放的包点，让模型学会预测"这回合炸弹会安在哪个点"。
> 通过炸弹安放事件 + 玩家 `place` 字段（V2 JSON 顶层 `places` 字典）推断。
> `label_win_reason` 同样是回合级常量标签，由 `end_reason` 字段映射而来。

---

## 5. 深度图

### 5.1 简化方向深度（已存入数据集）

每个存活玩家，每个 tick，向 64 个方向发射射线（方向均相对于玩家当前朝向）。
采用 5 层同心圈结构，层间竖直间隔 30°：

```
+60° 层   8 条  每 45° 一条  → 头顶窗口/高台
+30° 层  12 条  每 30° 一条  → 楼梯/窗口上方
  0° 层  24 条  每 15° 一条  → 中心层（眼睛高度），门洞/掩体可分辨
-30° 层  12 条  每 30° 一条  → 低地/坑道
-60° 层   8 条  每 45° 一条  → 脚底坑道

总计: 8+12+24+12+8 = 64 条
```

起点：玩家眼睛高度 (Z + 64 游戏单位)
最远：5000 游戏单位

射线检测使用 **Open3D RaycastingScene**（BVH 加速），在地图 OBJ 几何体上进行。

- 首次加载地图 BVH：~0.22s
- 后续批量查询（一整队 5 人 × 64 条 = 320 条射线/tick）：<1ms（实测 ~0.6ms）

### 5.2 坐标系转换

```
游戏空间 (CS2): X=东, Y=北, Z=上。Yaw 0=+Y(北), Yaw 90=+X(东)
OBJ 空间: obj_x = game_y × 0.0254, obj_y = game_z × 0.0254, obj_z = game_x × 0.0254
```

玩家前方向量：`(cos(yaw)·cos(pitch), sin(yaw)·cos(pitch), sin(pitch))`

---

## 6. PyTorch DataLoader

> 预训练训练使用 `CS2PretrainDataset`（round WDS → 在线切窗口），见
> [`docs/torch-dataset.md`](torch-dataset.md) 和 `docs/pretrain.md` §4。
> 下面示例为直接读取 round 级样本（变长 T）的方式：

```python
import webdataset as wds
import zstandard as zstd
import numpy as np
import io

cctx = zstd.ZstdDecompressor()

def decode_sample(sample):
    """解码一个 WebDataset sample → numpy 张量字典."""
    result = {}
    for key, value in sample.items():
        if key == "__key__" or key.startswith("__"):
            continue
        if key.endswith(".npy.zst"):
            name = key[:-8]  # 去掉 .npy.zst
            decompressed = cctx.decompress(value)
            result[name] = np.load(io.BytesIO(decompressed))
        elif key.endswith(".json.zst"):
            import json
            result["meta"] = json.loads(cctx.decompress(value))
    return result

dataset = (
    wds.WebDataset("data/dataset/train/shards-*.tar")
    .shuffle(1000)
    .map(decode_sample)
)

for sample in dataset:
    T = sample["player_pos"].shape[0]
    print(f"Round: T={T}, map={sample['meta']['map_name']}")
    break
```

### collate_fn 示例

```python
def collate_fn(batch):
    """Padding 到 batch 内最大 T."""
    max_T = max(s["player_pos"].shape[0] for s in batch)
    # ... 对每个字段做 padding，生成 attention_mask
    return batched
```

---

## 7. 存储规模

假设 tick_interval=0.25s，每场 ~22 回合，每回合 ~80 tick：

| 组件 | 每回合 (未压缩) | 20 万回合 (zstd 后) |
|------|----------------|--------------------|
| 玩家特征 | ~0.6 MB | ~50 GB |
| 简化深度 | ~0.05 MB | ~5 GB |
| 声音 + 标签 + 元数据 | ~0.05 MB | ~5 GB |
| **合计** | **~0.7 MB** | **50–80 GB** |

一万多场比赛 ≈ 20–30 万回合，分布在 ~50–80 个 5GB shard 中。

---

## 8. 预训练数据格式（`cs2.training.pretrain.v5`）

预训练任务：给定前 `N_TICKS` 个 tick（默认 16，见 `config/pretrain-a100.yaml`）的场面信息，
对每个玩家独立预测后续 `N_TICKS` 个 tick 的相机运动（离散 token）。
位移分量使用世界对齐坐标系（v5）：`d_forward`/`d_right` 在水平面上，`d_up` = 纯世界 Z。

### 8.1 生成流程

预训练数据集有两条路径：

- **在线切窗口（当前训练默认）**：`CS2PretrainDataset` 直接读 round WDS shard，
  在 `__iter__` 中实时切窗口，**不落盘**（避免 300GB+ 数据翻倍）。训练入口
  `scripts/pretrain.py` 即用此方式。
- **离线落盘（可选）**：`create_pretrain_data.py` 预先生成窗口级 shard。

```
V2 JSON → create_training_data.py (含 camera label) → Round WDS shards
                                                              │
Round WDS shards → create_pretrain_data.py ─────────────────→ Pretrain WDS shards（可选）
                              │
                              └── CS2PretrainDataset 在线切窗口（训练默认）
```

```bash
# 第一步：生成 round 级样本（含相机标签）
python scripts/create_training_data.py \
    --input-dir data/demos/json \
    --output-dir data/dataset \
    --no-depth --workers 4

# 第二步（可选）：离线提取预训练窗口
python scripts/create_pretrain_data.py \
    --input-dir data/dataset \
    --output-dir data/pretrain_dataset \
    --n-ticks 16 --stride 8 --workers 4
```

### 8.2 Sample 结构

每个 pretrain sample 复用 round 样本的全部特征 key，但 T 固定为 `N_TICKS`（输入窗口）。
另外包含相机运动输出标签、per-tick 条件标签和窗口掩码：

| 字段 | 形状 | dtype | 说明 |
|------|------|-------|------|
| `label_camera` | `[2×N_TICKS−1, 10, 10]` | float32 | 覆盖输入+输出全部 tick 的相机运动标签（v5 世界对齐）；训练时 unfold 切出输出窗口 |
| `player_depth_labels` | `[2×N_TICKS−1, 10, 64, 5]` | float32 | 输入+输出全部 tick 的深度（增强后），decoder per-tick depth 条件 |
| `player_alive_mask_labels` | `[2×N_TICKS−1, 10]` | bool | 对应 tick 的存活标记（判断 depth 有效性） |
| `player_pos_labels` | `[2×N_TICKS−1, 10, 3]` | float32 | 未来 absolute xyz（decoder xyz 条件） |
| `player_angle_labels` | `[2×N_TICKS−1, 10, 4]` | float32 | 未来 cos/sin yaw + cos/sin pitch（decoder angle 条件） |
| `output_mask` | `[N_TICKS]` | bool | True = 非 pad 输出 tick（左对齐，真实数据在开头） |
| `tick_times_input` | `[N_TICKS]` | float32 | 输入窗口各 tick 的 round_seconds（pad 位置外推） |
| `tick_times_output` | `[N_TICKS]` | float32 | 输出窗口各 tick 的 round_seconds（pad 位置外推） |

现有 6 个下游任务标签（winrate 等）均保留，从输入窗口切片。

### 8.3 窗口提取策略（`PretrainWindowExtractor`）

- **滑动参数**：`N_TICKS=16`，`STRIDE=8`（50% 重叠），每步 `s += stride + jitter`
  （jitter ∈ [−stride/2, stride/2]，`PretrainWindowExtractor(jitter=...)` 构造参数，
  训练侧可用 `--no-jitter` 关闭）
- **最少要求**：`N_TICKS` 个真实输入 tick + 1 个真实输出 tick
- **起点**：`s ≥ 0`（不 left-pad，不做假数据）；窗口无左侧填充
- **短回合**：T < 2×N_TICKS−1 的回合可产生窗口，只要有足够真实 tick
  （输入不足 N_TICKS 或输出不足 1 个则跳过该窗口）
- **玩家洗牌**：每个窗口独立随机 permutation 玩家索引（并重映射 `player_rel_i`），
  防止模型依赖固定玩家编号

### 8.4 元数据

```json
{
    "format": "cs2.training.pretrain.v5",
    "source_sample_key": "navi-vs-faze-m1-dust2__round5_xxx",
    "window_start": 0,
    "n_valid_input": 16,
    "n_valid_output": 16,
    "n_ticks_config": 16,
    "stride": 8,
    "round_T": 82,
    "... 原始 round 元数据 ...": "..."
}
```

---

## 9. CLI 参考

```
python scripts/create_training_data.py \
    --input-dir DIR              # V2 JSON 文件目录 (.json / .json.gz)
    --maps-dir DIR               # 优化后的地图 OBJ 目录 (默认: maps/optimized_obj_files)
    --output-dir DIR             # WebDataset 输出目录 (默认: data/dataset)
    --test-matches-per-map N     # 每个地图用于测试的比赛数 (默认: 4)
    --no-depth                   # 禁用深度图生成
    --tick-interval F            # Tick 间隔秒数 (默认: 0.25)
    --max-shard-size N           # 每个 shard 最大 GB (默认: 5)
    --workers N                  # 并行 worker 数 (默认: 0=自动检测CPU核心数)
    --done-file PATH             # 断点续传记录文件
    --dry-run                    # 只统计不写入
    --verbose, -v                # 详细输出
```

### 预训练数据 CLI

```
python scripts/create_pretrain_data.py \
    --input-dir DIR              # Round 级 WebDataset 目录（含 train/ test/）
    --output-dir DIR             # Pretrain WebDataset 输出目录
    --n-ticks N                  # 输入/输出窗口 tick 数（默认: 64）
    --stride N                   # 滑动步长（默认: 16）
    --min-input-ticks N          # 最少有效输入 tick 数（默认: 32）
    --min-output-ticks N         # 最少有效输出 tick 数（默认: 1）
    --max-shard-size N           # 每个 shard 最大 GB（默认: 5）
    --workers N                  # 并行 worker 数（默认: 0=自动一半核心数）
    --require-camera             # 缺少 camera label 时报错而非跳过
    --verbose, -v                # 详细输出
```

---

## 10. 文件结构

```
scripts/
  create_training_data.py           # CLI 入口：V2 JSON → Round WDS
  create_pretrain_data.py           # CLI 入口：Round WDS → Pretrain WDS（新增）
  training_data/
    __init__.py
    config.py                       # 地图配置、坐标转换、常量
    map_loader.py                   # OBJ 加载 + Open3D BVH 缓存
    depth_map.py                    # 简化方向深度 + 完整深度图
    feature_builder.py              # 玩家/全局/投掷物特征构建
    label_builder.py                # 7 任务标签生成（含相机标签）
    round_processor.py              # 回合 → sample
    pretrain_processor.py           # 预训练窗口提取（新增）
    wds_writer.py                   # WebDataset + zstd 写入
    wds_reader.py                   # WebDataset 解码 + 验证 + 统计
    torch_dataset.py                # PyTorch IterableDataset + collate_fn
```

## 11. 支持的地图

| 地图 | OBJ 文件 | 中心坐标 |
|------|---------|----------|
| `de_dust2` | ✅ | (-199.0, 977.0, 32.22) |
| `de_mirage` | ✅ | (-605.89, -866.89, -171.62) |
| `de_inferno` | ✅ | (481.07, 1396.48, 137.91) |
| `de_nuke` | ✅ | (265.96, -772.5, -381.90) |
| `de_overpass` | ✅ | (-2027.39, -812.90, 324.95) |
| `de_ancient` | ✅ | (-435.5, -348.0, 43.65) |
| `de_anubis` | ✅ | (-77.39, 618.90, -6.80) |
| `de_cache` | ✅ | (724.16, 394.75, 1757.49) |
