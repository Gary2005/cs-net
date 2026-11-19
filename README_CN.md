<p align="center">
  <img src="visualizer/static/logo.svg" width="120" height="120" alt="cs-net logo">
</p>

<h1 align="center">CS-NET</h1>

<p align="center">
  <strong>逐 tick 的 CS2 demo 提取 · 3D 回放工作台 · 用真实比赛数据训练的 PyTorch 模型</strong>
</p>

<p align="center">
  <a href="README.md">English</a>
  ·
  <a href="#快速开始">快速开始</a>
  ·
  <a href="#数据流">数据流</a>
  ·
  <a href="#训练">训练</a>
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

CS-NET 把一份 `.dem` 变成本地可查的数据，再变成浏览器里的回放和训练集。四块：

| 模块 | 做什么 | 入口 |
|---|---|---|
| **`tools/demo_extract`** | 一份 `.dem` → 一份 `.csz`。**每个 tick 都采**（≈64 Hz），10 个玩家 + 投掷物 + 道具；另外通过一个小的 Go 辅助程序取到 `demoparser2` 拿不到的 **真实烟雾体素云**（32³，Morton 解码）和 **真实火焰格子**（三维坐标 + 表面法向）。列式编码 + zstd：110 MB 的 demo 变成 ~7 MB，同一份数据存 JSON 是 ~183 MB。 | `python -m demo_extract` |
| **`app/`** | Demo 库：监听目录、入库去重、按地图几何实时生成雷达缩略图、玩家档案、AI 概率面板。 | `python -m app.server` |
| **`visualizer/`** | 3D 回放工作台（Flask + Three.js）：在减面后的真实地图几何上平滑回放，烟雾体、火焰格子、投掷物轨迹、击杀信息、Minimap、可见性。 | `python visualizer/server.py` |
| **`scripts/training_data` + `training/`** | `.csz` → 任务数据集（`aim` 对枪窗口、`spatial_prediction` 单帧标签）→ PyTorch 训练。 | `python -m scripts.training_data.prepare`、`python -m training.aim.train` |

`.csz` 的编解码是**严格无损**的：`codec.save` → `codec.load` 逐位相同
（`codec.equal`），不是近似、不是四舍五入。

## 数据流

```
本机 .dem ──┐
            │  scraper/（HLTV，Playwright）──▶ .dem
            ▼
       .dem ──demo_extract──▶ .csz    逐 tick，zstd，比 JSON 小 ~26 倍
             （+ Go 辅助程序：火焰格子 + 烟雾体素）
                                │
       ┌────────────────────────┼─────────────────────────┐
       ▼                        ▼                         ▼
  app/ Demo 库            visualizer/ 3D 回放      scripts/training_data
  缩略图 · 玩家          浏览器里看比赛           prepared/<task>/ ──▶ training/
  AI 面板
```

## 快速开始

需要 Python ≥ 3.10，以及 Go 工具链（只用来编译火焰/烟雾辅助程序）。

```bash
git clone https://github.com/Gary2005/cs-net.git
cd cs-net

conda create -n cs-net python=3.10 -y
conda activate cs-net
pip install -r requirements.txt

# 1) 编译 Go 辅助程序（火焰格子 + 烟雾体素）
bash tools/demo_extract/build_sidechannel.sh

# 2) 解压地图几何（maps/*.obj.zip -> *.obj，约 1 GB）
python scripts/extract_maps.py

# 3) 检查环境（Python、依赖、辅助程序、浏览器、unrar）
python scripts/check_env.py

# 4) 起工作台
python -m app.server --port 5000        # → http://127.0.0.1:5000/library
```

库默认是空的，这是有意的 —— **监听目录必须由你指定**（顶栏 **设置** → 添加目录 → **保存**，
然后 **扫描**）。demo 会在后台提取成 `.csz`，从列表里一点就进 3D 回放。

### 命令行提取 / 检查一份 demo

```bash
export PYTHONPATH=tools                # PowerShell: $env:PYTHONPATH = "tools"

python -m demo_extract match.dem -o match.csz                    # 每个 tick
python -m demo_extract match.dem --interval 0.25 -o match.csz    # 4 Hz
python -m demo_extract match.dem --stride 8 -o match.csz         # 每 8 个 tick
python -m demo_extract --inspect match.csz                       # 看看里面有什么
python -m demo_extract match.dem --selftest                      # 提取 → 保存 → 读回 → 逐位比对
```

```python
from demo_extract import extract, codec

data = extract("match.dem")          # 每个 tick
codec.save(data, "match.csz")
assert codec.equal(data, codec.load("match.csz"))[0]
```

## 训练

```bash
# .csz -> prepared/<task>/{train,valid,test}
python -m scripts.training_data.prepare --task aim
python -m scripts.training_data.prepare --task spatial_prediction

# 训练（配置里默认是 /data/... 路径，用 --set data.root=... 覆盖）
python -m training.aim.train -c training/aim/config.yaml
python -m training.spatial_prediction.train -c training/spatial_prediction/config.yaml
```

| 任务 | 一条样本 | 标签 |
|---|---|---|
| `aim` | A 对 B 的 257 帧对枪窗口 | 逐帧 `P(A 击杀 B)` |
| `spatial_prediction` | **一帧** | 5 个头：CT 赢 / 终局存活 / 还能击杀 / 下一个死的 / 下一个杀的 |

不用显卡也能看样本：`python -m scripts.training_data_visualizer.aim`（5117 端口）和
`python -m scripts.training_data_visualizer.spatial_prediction`（5118 端口）。

**仓库里不带权重。** `models/` 已被 gitignore —— 光是导出的两份权重就有 665 MB，
留着优化器状态则单次训练要 ~2 GB。`app/` 的 AI 面板读的是 `app/ai/models.py`
里写死的路径，所以先导出一次：

```bash
python scripts/export_weights.py          # <run>/last.pt + config.yaml -> <run>.pt + <run>.yaml
```

## 目录结构

```
app/                              Demo 库 + 玩家页 + AI 面板（Flask、SQLite）
visualizer/                       3D 回放工作台（Flask + Three.js）
tools/demo_extract/               .dem → .csz：逐 tick 提取、编解码、烟雾/火焰解码
tools/cs2_assets/                 CS2 资产管线（GLB / 贴图、地图材质色烘焙）
scraper/                          HLTV → .dem → .csz（Playwright）
scripts/training_data/            任务框架：.csz → prepared 数据集
scripts/training_data_visualizer/ 独立样本检查器（只依赖 numpy）
training/                         PyTorch 模型与训练脚本
maps/optimized_obj_files/         减面地图几何（.obj.zip，用前先解压）
docs/                             PLAN.md（方案与调研）、TOOLS.md（提取工具）
notes/                            调研笔记（资产、demoparser2、保真度……）
```

## 文档

| 文档 | 内容 |
|---|---|
| [tools/demo_extract/README.md](tools/demo_extract/README.md) | `.csz` 字段树、采样方式、烟雾体素、火焰格子、已知取舍 |
| [tools/demo_extract/API.md](tools/demo_extract/API.md) | 字段级手册、实测量级、坑 |
| [visualizer/README.md](visualizer/README.md) | 回放工作台的行为与内部实现 |
| [app/README.md](app/README.md) | Demo 库、缩略图、玩家页、AI 面板 |
| [scripts/training_data/README.md](scripts/training_data/README.md) | 怎么写任务、怎么切样本、数据集上的坑 |
| [scraper/README.md](scraper/README.md) | HLTV 抓取与 Linux 上重编辅助程序 |
| [docs/PLAN.md](docs/PLAN.md) | v5 方案：这里的「真实游戏内容」指什么，逐条带出处 |
| [docs/TOOLS.md](docs/TOOLS.md) | 提取工具调研与实测 |
| [NOTES.md](NOTES.md) | AnimGraph2 模型移植笔记（资产侧） |

## 注意事项

- **辅助程序是分平台的。** `tools/demo_extract/sidechannel_go/` 里只有源码，没有可执行文件；
  请在本机跑 `build_sidechannel.sh`。用别处编译的二进制会直接 `Exec format error`，
  `check_env.py` 会检出并给出提示。
- **在意下蹲就别降采样。** 下蹲过渡只有约 200 ms（13 个 tick），`--interval 0.25`
  会吃掉其中 94%。
- **回合完整性会不齐。** 没有 `round_freeze_end` 的回合没有时间原点，会被跳过；
  原因写在 `skipped_rounds` 里，不是数据损坏。
- **体积不小。** `maps/` 里的地图几何是压缩存放的（约 207 MB）。

## 许可

本仓库的**代码**是 MIT —— 见 [LICENSE](LICENSE)。

**游戏内容不是。** `visualizer/static/assets/`（武器模型、贴图、烟/火精灵、HUD 图标、
雷达图）和 `maps/optimized_obj_files/` 的几何都派生自 Valve 的 Counter-Strike 2 内容，
版权归 Valve Corporation 所有，按
[Steam 用户协议](https://store.steampowered.com/subscriber_agreement/) 仅供个人、
非商业使用，**不在上面那个 MIT 许可的覆盖范围内**。转发这些文件、或发布捆绑它们的构建之前，
请先想清楚；`tools/cs2_assets/` 里的脚本可以从你自己的 CS2 安装里在本地重建这一套。
