<p align="center">
  <img src="visualizer/static/logo.svg" width="120" height="120" alt="cs-net-logo">
</p>

<h1 align="center">CS-NET v4</h1>

<p align="center">
  <strong>基于 Transformer 的 CS2 路径预测与局面预测框架</strong>
</p>

<p align="center">
  <a href="README.md">English</a>
  ·
  <a href="#-模型权重">模型权重</a>
  ·
  <a href="#-快速开始">快速开始</a>
  ·
  <a href="#-训练">训练</a>
  ·
  <a href="#-文档">文档</a>
</p>

---

## 本仓库包含什么

CS-NET v4 是 CS2（反恐精英 2）深度学习框架的开源版本，包含三部分：

| 组件 | 说明 | 入口 |
|---|---|---|
| **路径预测模型** | 预训练 Transformer：输入 16 tick 完整局面窗口（10 玩家 + 炸弹 + 投掷物 + 射线深度图），**自回归预测每个玩家未来的移动路径**（世界坐标），以离散移动/角度 token 表示。 | `scripts/pretrain_model.py`、`scripts/prediction_engine.py`、`scripts/pretrain.py` |
| **spatial-only 模型** | 最简下游任务族：仅凭**单个 tick** 的完整局面，预测每玩家**胜率 / 回合末存活 / 未来击杀**概率。每个模型 = 预训练 embedder + spatial transformer + 线性头（无路径、无历史窗口），一个任务一个 checkpoint。 | `scripts/spatial_only_predictor.py`、`scripts/downstream-spatial-only/finetune_spatial_only.py` |
| **3D 可视化工具** | Flask Web 应用：上传 `.dem` / `.json` / `.json.gz` 录像，3D 平滑回放（Three.js + 地图 OBJ），上传 checkpoint 可视化 **AI 预测路径 vs 真实路径**，加载 spatial-only 模型后整回合实时查看每玩家概率曲线。 | `visualizer/server.py` |

```
demo (.dem) ──demo_parser──▶ 回合 JSON ──create_training_data──▶ round WDS
round WDS ──create_pretrain_data──▶ 窗口 WDS ──pretrain.py──▶ 路径预测 ckpt
round WDS + 路径 ckpt ──finetune_spatial_only.py──▶ winrate / alive_end / future_kill
任意录像 ──visualizer/server.py──▶ 3D 回放 + AI 路径预测 + spatial-only 曲线
```

## 模型权重

Pro 架构（d_model=768，138.7M 参数）的已训练模型可在
[Releases](../../releases) 页面下载：

| 文件 | 任务 | 说明 |
|---|---|---|
| `cs-net-v4-pro.pt` | 路径预测（预训练） | 600k 步；完整模型（`model` + `global_step` 键） |
| `pretrain-v4-pro-win_rate.pt` | spatial-only `winrate` | `{task, model_state, head_state, config}` |
| `pretrain-v4-pro-alive_end.pt` | spatial-only `alive_end` | 同格式 |
| `pretrain-v4-pro-future_kill.pt` | spatial-only `future_kill` | 同格式 |

> spatial-only checkpoint 只含 embedder + spatial transformer 权重（外加线性头），
> 因此远小于完整路径预测模型。模型架构必须与 `config/pretrain-a100-pro.yaml` 一致。

### 下载

```bash
pip install huggingface_hub
python scripts/download_checkpoints.py                    # 全部 4 个 ckpt → checkpoints/
python scripts/download_checkpoints.py --pretrain-only    # 只下路径预测模型
```

或直接用 `hf` CLI：

```bash
hf download Gary2005/cs-net-v4 --local-dir checkpoints
```

### 验证模型能正确加载

```bash
python scripts/test_checkpoints.py --models-dir checkpoints
```

该测试会：按 config 构建模型并加载每个 checkpoint，检查架构完全匹配
（无 missing / unexpected 键）、所有权重有限值，并对 spatial-only 模型
跑一遍合成回合的逐 tick 推理。加 `--forward` 可额外对路径预测模型跑一次
合成局面的 16 tick 自回归推理（CPU 约 1-3 分钟）。

## 快速开始

### 1. 环境

```bash
conda create -n cs2demo python=3.10
conda activate cs2demo
pip install -r requirements.txt
```

用 `python scripts/check_env.py` 验证环境。推理支持 CPU / MPS / CUDA。

### 2. 3D 可视化工具 + AI 路径预测

```bash
# 启动并预加载本地 checkpoint（路径预测 + spatial-only 模型目录）
python visualizer/server.py --port 5000 \
    --checkpoint /path/to/cs-net-v4-pro.pt \
    --spatial-model-dir /path/to/spatial-ckpts \
    --device cpu            # 或 mps / cuda
```

打开 `http://127.0.0.1:5000/`，把 `.dem`/`.json`/`.json.gz` 录像拖进页面即可；
也可把录像放到 `examples/demo` / `examples/json` 下使用内置示例列表。
模型 checkpoint 也可以在页面内上传，不必走命令行参数。

完整功能与 API 列表见 [visualizer/README.md](visualizer/README.md)。

### 3. 路径预测推理（库 / CLI）

```python
from scripts.prediction_engine import PredictionEngine

engine = PredictionEngine(
    "config/pretrain-a100-pro.yaml",
    "cs-net-v4-pro.pt",
    device="cpu",
    maps_dir="maps/optimized_obj_files",
)
result = engine.predict_at_tick(sample, query_tick=120)
# result["trajectories"][p] = {"pred_traj": [...], "gt_traj": [...], ...}（世界坐标）
```

`sample` 是 `scripts/training_data/round_processor.py`（`process_round`）产出的
回合级 dict，与训练数据格式一致。

另有独立测试 CLI（读取 round 级 WDS shard）：

```bash
python scripts/prediction_engine.py \
    --config config/pretrain-a100-pro.yaml \
    --checkpoint /path/to/cs-net-v4-pro.pt \
    --data-dir /path/to/round_wds \
    --tick 200 --device cpu
```

### 4. spatial-only 推理

```python
from scripts.spatial_only_predictor import SpatialOnlyPredictor

predictor = SpatialOnlyPredictor("/path/to/spatial-ckpts", device="cpu")
out = predictor.predict_round_full(sample)   # 逐 tick 每玩家概率
```

模型目录按 ckpt 内 `task` 字段（`winrate` / `alive_end` / `future_kill`）自动发现，
每个任务加载一个模型。

## 训练

两个训练脚本均为 config 驱动（命令行参数可覆盖 yaml）：

```bash
# 预训练（路径预测）— A100 80GB 配置，600k 步
python scripts/pretrain.py --config config/pretrain-a100-pro.yaml

# spatial-only 下游微调（一次一个任务）
python scripts/downstream-spatial-only/finetune_spatial_only.py \
    --config config/finetune-spatial-only-a100.yaml \
    --checkpoint /path/to/cs-net-v4-pro.pt \
    --task winrate
```

详细文档：
- [docs/pretrain.md](docs/pretrain.md) — 预训练数据管线与训练
- [docs/training-data-format.md](docs/training-data-format.md) — round 级 WDS 格式
- [docs/torch-dataset.md](docs/torch-dataset.md) — 窗口数据集 / collate 内部实现
- [docs/demo-json-format.md](docs/demo-json-format.md) — 回合 JSON 格式（可视化工具输入）
- [scripts/downstream-spatial-only/README.md](scripts/downstream-spatial-only/README.md) — spatial-only 微调指南

数据管线入口：
- `demo_parser/` — `.dem` → 回合 JSON（`python -m demo_parser` 或 `scripts/demo_to_json.py`）
- `scripts/create_training_data.py` — 回合 JSON → round 级 WebDataset shards
- `scripts/create_pretrain_data.py` — round shards → 定长窗口 shards

## 仓库结构

```
config/                          # 训练配置（Pro 架构）
demo_parser/                     # .dem → 回合 JSON
maps/optimized_obj_files/        # 优化后的地图 OBJ（可视化 + 深度图）
replay_tool/filter.py            # JSON 后处理（可视化工具共用）
scripts/
  pretrain_model.py              # CS2PretrainModel / PretrainConfig
  prediction_engine.py           # 路径预测推理引擎
  pretrain.py                    # 预训练入口
  create_pretrain_data.py        # 窗口 shard 生成
  test_pretrain.py               # 单 sample teacher-forcing / AR 评估
  evaluate_pretrain.py           # 多样本评估
  spatial_only_predictor.py      # spatial-only 推理（可视化工具共用）
  create_training_data.py        # round shard 生成
  evaluate_demos.py              # demo 级评估（可视化工具扫描功能使用）
  training_data/                 # config / 深度图 / 特征 / 标签 / WDS IO / 数据集
  downstream-spatial-only/       # spatial-only 微调
visualizer/                      # Flask 3D 回放 + 预测 Web 应用
docs/                            # 详细中文文档
```

## 说明

- 所有输入窗口为 16 tick = 4 秒（0.25s/tick）；demo 解析固定 `interval=0.25` 与训练一致。
- 数据管线使用世界对齐（v5）坐标系；预测引擎会自动转换旧 v4 标签。
- `maps/optimized_obj_files/` 下的地图 OBJ 已针对可视化（Three.js）与射线深度
  （Open3D）做过优化。

## License

[MIT](LICENSE)
