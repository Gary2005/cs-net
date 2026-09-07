# spatial-only 下游任务微调

基于预训练模型的**最简下游任务**：不看路径、不看历史窗口，
只给**单个 tick 的完整局面**（10 玩家 + 炸弹 + 投掷物），为每个玩家预测**一个**概率。

**一个模型只训练一个任务**，用 `--task` 切换：

| `--task` | 预测目标 | label 来源 |
|----------|---------|-----------|
| `winrate` | 该玩家队伍的胜率 [0,1] | `label_winrate`（回合级，0=CT 1=T） |
| `alive_end` | 该玩家最终存活概率 [0,1] | `label_alive_end`（回合级） |
| `future_kill` | 该玩家在**本 tick 之后任意时刻**（死前）获得击杀的概率 | 从 `label_nxt_kill/death` 重建最后击杀 tick，`last_kill > t` |

**方法**：预训练 `embedder + SpatialTransformer` 全量微调（temporal / decoder 不参与），
每个玩家的 spatial embedding 过一个全连接 → 1 个 logit（BCEWithLogits）。

**数据**：不需要窗口。`SingleTickDataset` 从 round 级 WDS 逐 tick 切片
（每个 sample = 单个 tick 的完整局面），**player shuffle 增强也是每个 tick 独立**。
玩家在该 tick 已死亡 → mask 掉，不参与 loss。

**防过拟合的 tick 抽样**：同 round 相邻 tick 高度相关（输入几乎不变、
winner 标签恒定），全量采样会让训练样本大量近重复 → 训练后期过拟合
（val loss 回升、AUC 回落）。用 `keep_ratio` 控制训练保留比例：
- 每个 tick 以 `keep_ratio` 概率保留（伯努利），`1.0` = 全部保留；
  每 epoch 重新掷硬币（实测 keep_ratio=0.36 时两个 epoch 采样重合率
  ~4%），多 epoch 覆盖 round 全程更多时刻，随机性本身是隐式增强；
  每 round 至少保留 1 个 tick。
- val **不受影响**（全量 + `val_max_samples` 控制规模），确定性、指标可比。
- 训练循环支持**多 epoch**：`max_steps > 0` 时数据跑完一遍会自动重启
  迭代器继续下一 epoch，直到累计步数达标（`max_steps: 0` 保持只跑一遍）。

## 用法

```bash
conda activate cs2demo
# 推荐：config 管理全部参数（服务器上改 config 即可）
python scripts/downstream-spatial-only/finetune_spatial_only.py \
    --config config/finetune-spatial-only-a100.yaml

# 本地快速实验（CLI 覆盖 config）
python scripts/downstream-spatial-only/finetune_spatial_only.py \
    --config config/finetune-spatial-only-a100.yaml \
    --checkpoint /Users/wanjungu/Downloads/cs-net-v4-pro/cs-net-v4-pro.pt \
    --data-dir examples/dataset --task winrate \
    --batch-size 256 --lr 1e-4 --max-steps 2000 --device cpu
```

参数优先级 = 命令行 > `--config` yaml > 内置默认值。
`config/finetune-spatial-only-a100.yaml` 内置模型架构与
`config/pretrain-a100-pro.yaml` 一致（d_model=768），**必须与 checkpoint 匹配**
（base checkpoint 请用 512 的架构字段）。

常用参数：

| 参数 | 默认 | 说明 |
|------|------|------|
| `--checkpoint` | 必填 | 预训练 checkpoint 路径 |
| `--task` | `winrate` | winrate / alive_end / future_kill |
| `--data-dir` | `examples/dataset` | 回合级 WebDataset 目录（含 train/ test/） |
| `--split` | `train` | train / test / both |
| `--batch-size` | 256 | batch = 单个 tick 数（每 tick 10 玩家） |
| `--tick-stride` | 1 | tick 采样步长（2 = 隔一个取一个，省数据） |
| `--keep-ratio` | 1.0 | 训练保留比例：每个 tick 以该概率保留（1.0 = 全部；防近重复过拟合） |
| `--lr` | 1e-4 | 全量微调建议 1e-4 量级 |
| `--max-steps` | 2000 | 训练步数（0 = 跑完数据） |
| `--max-samples` | 0 | 最多用多少个 tick（调试，0 = 全部） |
| `--num-workers` | 0 | DataLoader worker 数 |
| `--device` | cpu | cpu / mps / cuda |
| `--save-dir` | `outputs/finetune_spatial_only` | 保存目录（不同 task 自动放子目录） |
| `--save-interval` | 1000 | 每 N 步保存 `*_step_{global_step}.pt`（0=关闭） |
| `--val-interval` | 500 | 每 N 步在 test 集验证（0=关闭） |
| `--val-max-samples` | 2000 | 每次 val 最多跑多少个 test tick |
| `--no-amp` / `--no-tf32` / `--no-compile` | 关 | 禁用加速开关 |

**训练日志**：每 `log_interval` 步打印该任务的 loss/acc/pos_rate；
`winrate` 任务在 val 时附 `ct_winrate_acc`（用存活玩家按阵营聚合出 CT 胜率并判对）。

**保存格式**：`{save_dir}/{task}/spatial_only_{ckpt_name}.pt`，含
`model_state`（embedder+spatial 全量微调后的权重，已剥离 `_orig_mod.` 前缀）、
`head_state`（d_model → 1 的全连接头）、`task`、`global_step`、`config`。
