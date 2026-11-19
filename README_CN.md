<p align="center">
  <img src="visualizer/static/logo.svg" width="120" height="120" alt="cs-net logo">
</p>

<h1 align="center">CS-NET <img src="https://img.shields.io/badge/version-v5-38bdf8?style=flat-square" alt="v5"></h1>

<p align="center">
  <strong>CS2 3D AI 分析回放工具</strong>
</p>

<p align="center">
  <a href="README.md">English</a>
  ·
  <a href="#快速开始">快速开始</a>
  ·
  <a href="#下载-ai-模型">下载模型</a>
  ·
  <a href="#怎么用">怎么用</a>
  ·
  <a href="#目录结构">目录结构</a>
  ·
  <a href="#文档">文档</a>
  ·
  <a href="#许可">许可</a>
</p>

<p align="center">
  <img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10%2B-blue">
  <img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-green">
</p>

---

## 这是什么

cs-net v5 是一个本地运行的AI分析工具，同时具有3D回放功能。

## 快速开始

需要 Python ≥ 3.10，以及 Go 工具链（只用来编译烟雾/火焰辅助程序，一次就够）。

```bash
git clone https://github.com/Gary2005/cs-net.git
cd cs-net

conda create -n cs-net-v5 python=3.10 -y
conda activate cs-net-v5
pip install -r requirements.txt

# 1) 编译 Go 辅助程序（烟雾体素 + 火焰格子，每台机器跑一次）
bash tools/demo_extract/build_sidechannel.sh      # PowerShell 里用 Git Bash

# 2) 解压地图几何（maps/*.obj.zip -> *.obj，解压后约 1 GB）
python scripts/extract_maps.py

# 3) 起工作台
python -m app.server --port 5000                  # → http://127.0.0.1:5000/library
```

第 2 步也可以在网页里做：`maps/` 还没解压时，库页面顶上会出现一条 **解压地图**
的横幅，点一下排进后台任务（和上面的命令是同一个脚本）。和权重一样，每个检出
只需要做一次。

### 下载 AI 模型

两份权重合计 665 MB，**不在 git 里**，放在 Hugging Face 上。**下过一次就行**，
两种方式等价：

* 在网页里：打开 Demo 库页面，`models/` 缺东西时顶上会出现横幅，点
  **下载模型**；
* 在命令行：`python scripts/download_models.py`。

两边跑的是同一个脚本（底层是 `huggingface_hub`，断点续传和校验都交给它）。
它先探 `huggingface.co`，**连不上就把 `HF_ENDPOINT` 切到 `hf-mirror.com`** ——
和你自己 `$env:HF_ENDPOINT = "https://hf-mirror.com"` 是同一件事
（`--mirror` 强制走镜像，`--endpoint URL` 指定站点，`--check` 只报状态不下载）。

环境有哪儿不对，跑 `python scripts/check_env.py`（Python、依赖、辅助程序、浏览器、unrar）。

## 怎么用

1. **加监听目录** —— 顶栏 **设置** → 粘贴放 demo 的目录（本机探测到的候选目录会列出来）→ **保存** → **扫描**。库默认是空的，这是有意的：目录必须由你指定。
2. **等提取** —— 后台把 `.dem` 提成 `.csz`（110 MB → 约 7 MB），进度在列表和顶栏 **任务** 抽屉里。
3. **看回放** —— 勾选一场（可多选）→ **载入分析** → 进 3D 回放页：选手、投掷物、烟雾体、火焰格子、击杀信息、Minimap、可见性。
4. **看玩家** —— **玩家** 页按人聚合库里的所有比赛；玩家档案里有传统统计，也有 **AI 数据** 一栏。
5. **开 AI** —— 回放页底部打开 **AI 预测面板**（回合胜率 + 每人五项指标）和 **AI 对枪**（下一次对枪谁赢，画成弧线）。

## 目录结构

```
app/                              Web app 本体：Demo 库、玩家页、AI 面板（Flask + SQLite）
visualizer/                       3D 回放页（Flask + Three.js），站点 logo 也在这
tools/demo_extract/               .dem -> .csz 提取器：逐 tick、zstd、烟雾/火焰解码
training/                         PyTorch 模型与训练脚本（AI 面板加载的就是它训出来的权重）
scripts/training_data/            .csz -> 训练用数据集
scripts/                          辅助脚本：download_models.py、extract_maps.py、check_env.py、export_weights.py
maps/optimized_obj_files/         减面后的地图几何（.obj.zip，用 extract_maps.py 解压）
models/                           两份写死的权重（不进 git，下载得到）
```

`.csz` 的编解码是**严格无损**的：`codec.save` → `codec.load` 逐位相同
（`codec.equal`），不是近似、不是四舍五入。提取器每个 tick 都采（≈64 Hz），
烟雾体素和火焰格子通过一个小 Go 辅助程序取到 `demoparser2` 拿不到的字段。

## 文档

| 文档 | 内容 |
|---|---|
| [app/README.md](app/README.md) | Demo 库、缩略图、玩家页、AI 面板、HTTP 接口 |
| [visualizer/README.md](visualizer/README.md) | 回放页的行为与内部实现 |
| [tools/demo_extract/README.md](tools/demo_extract/README.md) | `.csz` 字段树、采样方式、烟雾体素、火焰格子 |
| [tools/demo_extract/API.md](tools/demo_extract/API.md) | 字段级手册、实测量级、坑 |
| [scripts/training_data/README.md](scripts/training_data/README.md) | 怎么写任务、怎么切样本、数据集上的坑 |

## 许可

本仓库的**代码**是 MIT —— 见 [LICENSE](LICENSE)。

**游戏内容不是。** `visualizer/static/assets/`（武器模型、贴图、烟/火精灵、HUD
图标、雷达图）和 `maps/optimized_obj_files/` 的几何都派生自 Valve 的
Counter-Strike 2 内容，版权归 Valve Corporation 所有。
