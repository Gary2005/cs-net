# CS2 PyTorch Dataset 文档

> **模块**: `scripts/training_data/torch_dataset.py`
> **依赖**: `torch>=2.0`, `webdataset>=0.2`, `zstandard>=0.21`, `numpy>=1.24`

---

## 1. 预训练数据集（当前训练用）

`CS2PretrainDataset` 读取 **round 级 WDS shard**（`shards-*.tar`），在迭代时**实时切
固定长度窗口**（不落盘，避免 300GB+ 数据翻倍）。预训练训练入口
（`scripts/pretrain.py`）即使用此数据集。

```python
from scripts.training_data.torch_dataset import CS2PretrainDataset, pretrain_collate_fn
from torch.utils.data import DataLoader

ds = CS2PretrainDataset(
    "data/dataset", split="train",
    n_ticks=16, stride=8,          # 窗口 tick 数 + 滑动步长
    shuffle_buffer=2000,
    augment_depth=True,            # 深度图角度增强
    jitter=True,                   # 窗口起始位置随机抖动
)
loader = DataLoader(
    ds, batch_size=14,
    collate_fn=pretrain_collate_fn,
    num_workers=8, pin_memory=True,
)
```

### 1.1 窗口结构

每个窗口 = `n_ticks` 输入 tick + `n_ticks` 输出 tick（共 `2×n_ticks−1` 个 tick 的
数据被切片，输出窗口左对齐）。窗口包含：

- **输入特征**：`player_pos` / `player_state` / `player_inv` / `player_rel_*` /
  `player_sound` / `player_depth` / `bomb_*` / `proj_*` / `map_idx` /
  `player_alive_mask` / `tick_times_input`，T = n_ticks
- **输出标签**：`label_camera [2×n_ticks−1, 10, 10]`（覆盖输入+输出全部 tick 的相机
  运动 10D 标签，训练时由模型 `unfold` 滑动切出输出窗口；见 docs/pretrain.md §6）
- **per-tick 条件标签**（decoder 条件注入用）：
  - `player_depth_labels [2×n_ticks−1, 10, 64, 5]`
  - `player_pos_labels [2×n_ticks−1, 10, 3]`
  - `player_angle_labels [2×n_ticks−1, 10, 4]`（cos/sin yaw + cos/sin pitch）
  - `player_alive_mask_labels [2×n_ticks−1, 10]`
- **辅助**：`output_mask [n_ticks]`（输出窗口有效位）、`tick_times_output [n_ticks]`、
  `meta`（含 `window_start`、`n_valid_input/output`、`round_T` 等）

### 1.2 关键参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `data_dir` | — | Round WDS 根目录（含 `train/` `test/`） |
| `split` | `"train"` | `"train"` / `"test"` / `"both"` |
| `n_ticks` | 64 | 输入/输出窗口 tick 数 |
| `stride` | 16 | 滑动步长（重叠率 = 1 − stride/n_ticks） |
| `shuffle_buffer` | 10000 | 窗口级 shuffle buffer（0=不 shuffle） |
| `augment_depth` | True | 深度图 `[T,10,64]` → `[T,10,64,5]` 角度增强 |
| `max_samples` | None | 最大窗口数（调试用） |
| `jitter` | True | 窗口起始位置随机抖动 + 每步抖动 |

**多 worker 支持**：自动将 shard 均匀分给各 worker，避免重复读取。

### 1.3 窗口提取细节（`PretrainWindowExtractor`）

- 滑动起点 `s ≥ 0`（不 left-pad，不做假数据），最少 `n_ticks` 个有效输入 tick + 1 个有效输出 tick
- 每步 `s += stride + jitter`（jitter ∈ [−stride/2, stride/2]）
- 每个窗口独立 **player shuffle**（随机 permutation 玩家索引，并重映射 `player_rel_i`）
- `pretrain_collate_fn`：T 固定为 n_ticks，直接 `torch.stack`，无 padding

### 1.4 便捷函数

```python
from scripts.training_data.torch_dataset import create_pretrain_dataloader

loader = create_pretrain_dataloader(
    "data/dataset", split="train",
    batch_size=14, n_ticks=16, stride=8,
    shuffle_buffer=2000, num_workers=8,
)
```

---

## 2. 通用数据集（CS2Dataset，变长回合）

`CS2Dataset` 直接产出 **round 级样本**（变长 T），配合 `collate_fn` 按 batch 内最大
T padding。适合按回合组织数据的任务（如 winrate / nxt_kill 等回合级分类）。

```python
from scripts.training_data.torch_dataset import CS2Dataset, collate_fn
import torch

ds = CS2Dataset("data/dataset", split="train", shuffle_buffer=5000)
loader = torch.utils.data.DataLoader(
    ds, batch_size=8, collate_fn=collate_fn,
    num_workers=4, pin_memory=True,
)

for batch in loader:
    B, max_T = batch["T_mask"].shape   # T_mask: [B, max_T] True=有效 tick
    ...
```

### 2.1 参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `data_dir` | — | 数据集根目录（含 train/ test/） |
| `split` | `"train"` | `"train"` / `"test"` / `"both"` |
| `shuffle_buffer` | 1000 | WebDataset shuffle buffer（0=不 shuffle） |
| `augment_depth` | True | 深度图角度增强 |
| `max_samples` | None | 最大样本数（调试用） |

### 2.2 collate_fn 行为

变长 T 的 sample padding 到 batch 内最大 `max_T`，所有张量沿 dim=0 (T) padding：

| 类型 | 填充值 |
|------|--------|
| float 张量 | `0.0` |
| bool mask（普通） | `False` |
| int 张量（普通） | `0` |
| `label_nxt_kill/death` | `10`（无事件） |
| `label_bombsite` | `2`（未知） |
| `label_win_reason` | `5`（其他） |
| `proj_type` | `-1`（空槽） |

额外输出：`T_mask [B, max_T]`（True=有效 tick）、`token_pad_mask [B, max_T, 27]`
（True=pad token，由 T_mask 动态生成）、`meta`。

### 2.3 便捷函数

```python
from scripts.training_data.torch_dataset import create_dataloader

loader = create_dataloader(
    "data/dataset", split="train",
    batch_size=8, num_workers=4, shuffle_buffer=5000,
)
```

---

## 3. Batch 特征结构（27 tokens）

两个数据集产出的特征字段一致（形状以 `T` 或 `max_T` 表示时间维度）：

### 3.1 27 个 Token 结构

```
tokens  0– 9: 玩家 (10 tokens)
token     10: 炸弹/全局 (1 token)
tokens 11–26: 投掷物 (16 tokens)
```

### 3.2 玩家特征（Token 0–9）

| Key | 形状 | dtype | 说明 |
|-----|------|-------|------|
| `player_pos` | `[T,10,3]` | float32 | x,y,z 地图中心归一化坐标 |
| `player_alive_mask` | `[T,10]` | bool | True=存活 |
| `player_state` | `[T,10,14]` | float32 | HP/护甲/闪光/角度/队伍/速度 |
| `player_inv` | `[T,10,9]` | int64 | 背包武器索引 |
| `player_inv_mask` | `[T,10,9]` | bool | True=武器槽有效 |
| `player_rel_f` | `[T,10,9,14]` | float32 | 对其他玩家的关系特征 |
| `player_rel_i` | `[T,10,9]` | int64 | 关系目标玩家索引 |
| `player_rel_mask` | `[T,10,9]` | bool | True=关系有效 |
| `player_depth` | `[T,10,64]` → 增强后 `[T,10,64,5]` | float32 | 64 方向深度图 |
| `player_depth_mask` | `[T,10]` | bool | True=深度图有效 |
| `player_sound` | `[T,10,2]` | float32 | `[开火, 脚步]` 0/1 |

### 3.3 炸弹/全局（Token 10）

| Key | 形状 | dtype | 说明 |
|-----|------|-------|------|
| `bomb_pos` | `[T,3]` | float32 | 炸弹位置（归一化） |
| `bomb_state` | `[T,4]` | float32 | `[回合时间/160, 已安装, 已掉落, 安装时长/40]` |
| `map_idx` | `[T]` | int64 | 地图索引（0-7） |

### 3.4 投掷物（Token 11–26）

| Key | 形状 | dtype | 说明 |
|-----|------|-------|------|
| `proj_pos` | `[T,16,3]` | float32 | 投掷物位置（归一化） |
| `proj_type` | `[T,16]` | int64 | 类型：smoke=0, inferno=1, he=2, flashbang=3, decoy=4, molotov=5 |
| `proj_dur` | `[T,16]` | float32 | 剩余持续时间 `/25` |
| `proj_is_active` | `[T,16]` | int64 | 0=飞行，1=落地持续效果 |
| `proj_mask` | `[T,16]` | bool | True=槽位有效 |

### 3.5 标签

| Key | 形状 | dtype | 说明 |
|-----|------|-------|------|
| `label_winrate` | `[T]` | float32 | 0=CT赢, 1=T赢 |
| `label_nxt_kill` | `[T]` | int64 | 下一个击杀者索引（0-9），10=无 |
| `label_nxt_death` | `[T]` | int64 | 下一个死亡者索引（0-9），10=无 |
| `label_alive_end` | `[T,10]` | float32 | 1.0=回合结束时存活 |
| `label_bombsite` | `[T]` | int64 | 炸弹包点：0=A, 1=B, 2=未知 |
| `label_win_reason` | `[T]` | int64 | 0=CT全灭, 1=T全灭, 2=爆炸, 3=拆除, 4=超时, 5=其他 |
| `label_camera` | `[T,10,10]` | float32 | **预训练标签**：相机运动 10D（见 docs/pretrain.md §6） |

> **Meta**：`batch["meta"]` 是 `list[dict]`，包含 `map_name`, `T`, `winner`,
> `round_id`, `players` 等元数据，不参与 tensor 运算。

---

## 4. 深度图增强（`augment_depth`）

原始数据 `player_depth` 为 `[T,10,64]`（每条射线一个标量 `log_dist`）。
开启 `augment_depth=True`（默认）后自动扩展为 `[T,10,64,5]`：

```
[log_dist, cos(yaw_offset), sin(yaw_offset), cos(pitch_offset), sin(pitch_offset)]
```

`yaw_offset` / `pitch_offset` 是射线相对玩家当前朝向的偏移角，编码为 cos/sin 对让模型
区分不同方向的深度值。同样的增强也作用于 `player_depth_labels`（预训练条件标签）。

64 条射线 = 5 层同心圈：

```
+60° 层   8 条  每 45° 一条  → 头顶窗口/高台
+30° 层  12 条  每 30° 一条  → 楼梯/窗口上方
  0° 层  24 条  每 15° 一条  → 中心层（眼睛高度），门洞/掩体
-30° 层  12 条  每 30° 一条  → 低地/坑道
-60° 层   8 条  每 45° 一条  → 脚底坑道

总计: 8+12+24+12+8 = 64 条
```

角度编码是确定性常量，在 numpy 域 broadcast 拼接，开销 <0.1ms/sample。
关闭：`augment_depth=False`（`player_depth` 保持 `[T,10,64]`）。

---

## 5. CLI 测试

```bash
# 预训练数据集（窗口）
python -c "
from scripts.training_data.torch_dataset import CS2PretrainDataset, pretrain_collate_fn
from torch.utils.data import DataLoader
ds = CS2PretrainDataset('examples/dataset', split='train', n_ticks=16, stride=8, max_samples=20)
loader = DataLoader(ds, batch_size=2, collate_fn=pretrain_collate_fn)
for batch in loader:
    print({k: tuple(v.shape) for k, v in batch.items() if hasattr(v, 'shape')})
    break
"

# 通用数据集（变长回合）
python scripts/training_data/torch_dataset.py \
    --data-dir examples/dataset --split train \
    --batch-size 4 --max-samples 20
```
