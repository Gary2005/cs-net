# CS2 Vision Studio — 面向客户的 3D 回放 & AI 路径预测

一个面向客户（非开发者）的 CS2 可视化工具：上传 demo / json / json.gz，
3D 平滑回放比赛过程，并可上传预训练 checkpoint 可视化 AI 路径预测。

## 快速开始

```bash
conda activate cs2demo

# 启动（可在页面内上传模型，或用 --checkpoint 直接预加载）
python visualizer/server.py --port 5000

# 或直接加载本地 checkpoint / spatial-only 模型目录（路径预测与 spatial 均用 MPS）
python visualizer/server.py --port 5000 \
    --checkpoint /path/to/cs-net-v4-pro.pt \
    --spatial-model-dir /path/to/spatial-ckpts \
    --device cpu     # mps / cuda 亦可
```

> 设备默认**自动探测**（mps > cuda > cpu，页面内上传时下拉框默认选中探测结果）：
> Mac 自动用 mps，Windows 无 GPU 自动用 cpu，有 N 卡自动用 cuda。
> 路径预测与 spatial-only **共用同一个 `--device`**，也可显式指定。

浏览器打开 `http://127.0.0.1:5000/`。

## 功能

### 1. 回放数据加载

支持三种输入（拖拽或点击上传）：

| 格式 | 处理方式 |
|------|---------|
| `.dem` | 用 `demo_parser` 以 **0.25s 采样**转换为 JSON（与预训练数据一致） |
| `.json` | 直接加载（`cs2.demo.v2` 格式） |
| `.json.gz` | 解压后加载（也支持 `.tar.gz` 内单个 json） |

### 2. 3D 平滑回放

- Three.js 渲染，地图 OBJ 模型 + 科技感光照/粒子背景
- **平滑插值播放**：`currentTime` 连续推进，玩家位置/角度在 0.25s 采样点之间
  lerp 插值，任意倍速（0.25×~4×）下都流畅，不是"每秒 4 帧"跳变
- 玩家血条 / 名称 / 轨迹、炸弹、烟雾/火焰、投掷物轨迹、击杀信息、Minimap
- 空格播放/暂停、←→ 逐秒、R 重置视角
- 回合切换、时间线点击 seek、炸弹阶段高亮

### 3. JSON 打包下载

顶栏「⬇ JSON」按钮：把**最近加载的数据**打包为 `.json.gz` 下载。

- 上传 demo → 下载转换后的 0.25s 采样 json.gz（方便共享）
- 上传 json / json.gz → 下载整理后的 json.gz

### 4. AI 路径预测

1. 上传预训练 checkpoint（`.pt`，即 `pretrain.py` 保存的 checkpoint），
   可选推理设备：**MPS（Mac 加速）/ CPU / CUDA**
2. 选择预测起始 tick 和采样温度（0 = 确定性 argmax）
3. 点击「运行路径预测」：

- **紫色线**：AI 预测轨迹
- **绿色线**：真实轨迹（GT，来自数据标签）
- 起点/终点光点 + 玩家标签，轨迹上有流动光点动画
- 右侧面板显示每个玩家预测/GT 步数与终点位移误差

预测输入构建与预训练数据管线**完全对齐**：
- `round_processor.process_round`（v5 世界对齐坐标系）
- 深度图 raycast、label_camera 等与 `create_training_data.py` 一致
- 预测引擎自动处理 v4 → v5 坐标转换（旧数据兼容）

### 5. spatial-only 单局面预测（自动）

加载 **spatial-only 下游任务模型**（winrate / alive_end / future_kill，
embedder+spatial+head，`finetune_spatial_only.py` 保存的 checkpoint），
对**单个 tick 的完整局面**直接推理，无路径/decoder/历史：

1. 上传三个任务 checkpoint（`.pt`，可多选，任务由 ckpt 内部 `task` 字段识别），
   或启动时用 `--spatial-model-dir` 指定目录
2. **切换回合自动预测**：模型加载后，每次切换回合自动对该回合**全部 tick**
   逐个推理（单 tick 独立、速度很快）；结果按 (文件, 模型目录, device, 回合)
   在服务端缓存，来回切换回合不重算（前端也缓存，命中零请求）
3. **拖动时间线实时查看**：左侧玩家卡片每张都有一行概率 chip——
   队伍胜率 / 回合末存活 / 未来击杀（随播放位置实时更新，死亡玩家置灰划线）
4. **点击数值看曲线**：点击任一概率 chip 弹出该玩家该指标的**整回合曲线**
   （存活区间着色、阵亡时刻虚线标注、当前播放位置竖线跟随）
5. 右侧面板始终显示**聚合 CT 胜率曲线**：CT 玩家 1−P(T胜)、T 玩家 P(T胜)，
   ct = CT/(CT+T)，逐 tick 绘制，标注实际 Winner

推理器公共模块：`scripts/spatial_only_predictor.py`（visualizer 与其它工具共用）。

> **MPS 说明**：torch **2.9.x** 的 MPS 后端在本机跑 spatial-only 存在内存
> 损坏（间歇性 NaN + 有限但错误的值，非数据问题；逐 tick `torch.cat` 多个
> MPS 张量最严重，`torch.stack` 拼装后 NaN 消失但有限错误值仍偶发），
> 而路径预测（T=16 窗口 + decoder 自回归）在 MPS 上与 CPU 逐位一致——
> 差异在输入布局/批拼接路径，不在模型结构。该现象在 **torch 2.13** 上实测
> 已消失（5/5 独立 MPS 实例整回合与 CPU 逐位干净，max diff 1e-4，
> cat/stack 路径均无 NaN）。不过 PyTorch 侧仍有已知 MPS 问题未根治
> （[pytorch/pytorch#193487](https://github.com/pytorch/pytorch/issues/193487)：
> matmul 偶发静默错误结果，覆盖 2.7–2.13.0，仅被 #187441 掩盖）——
> 对数值敏感时可显式 `--device cpu`（整回合约 20s，服务端缓存后切回合秒回）。

## 文件结构

```
visualizer/
  server.py                 # Flask 后端（上传/下载/预测 API）
  templates/index.html      # 前端页面
  static/
    css/style.css           # 深空暗色主题
    js/main.js              # 入口：上传/回放/预测交互
    js/prediction.js        # 预测轨迹渲染（pred vs GT）
    js/scene.js             # Three.js 场景/相机/光照（复用自 replay_tool）
    js/map-loader.js        # OBJ 地图加载（复用）
    js/replay-core.js       # 数据加载/插值/回放引擎（复用）
    js/visuals.js           # 玩家/炸弹/烟雾等实体（复用）
    three/                  # Three.js 库（本地，无 CDN 依赖）
```

## API 一览

| 路由 | 方法 | 说明 |
|------|------|------|
| `/api/load` | POST | 上传 demo/json/json.gz 或指定路径加载 |
| `/api/download` | GET | 下载最近数据的 json.gz 打包 |
| `/api/maps` `/api/map/<name>` | GET | 地图列表 / OBJ 文件 |
| `/api/examples` | GET | 示例文件列表 |
| `/api/model/status` | GET | 预测引擎 + spatial-only 状态 |
| `/api/model/upload` | POST | 上传预训练 checkpoint（路径预测，multipart，可带 `device` 字段） |
| `/api/spatial/upload` | POST | 上传 spatial-only 任务 checkpoint（多文件，任务由 ckpt 识别） |
| `/api/predict` | POST | 对指定 round/tick 运行路径预测，返回轨迹 |
| `/api/predict/player-sampled` | POST | 单玩家并行采样多条路径 |
| `/api/predict/spatial` | POST | 单局面预测：每玩家 winrate/alive_end/future_kill 概率 |
| `/api/predict/spatial/curve` | POST | 整回合聚合 CT 胜率曲线 |
| `/api/predict/spatial/round` | POST | **整回合全任务逐 tick 自动预测**（切回合触发，服务端缓存，`cached` 字段标识命中） |
| `/api/scan/round` `/api/scan/all` | POST | 回合低概率移动扫描（路径模型） |

## 与预训练数据对齐的说明

当前训练用 `config/pretrain-a100.yaml`，数据为 **v5 世界对齐坐标系**
（`d_forward/d_right` 水平、`d_up` = 纯世界 Z）。旧数据（v4）是相机相对坐标系。

- 本工具解析 demo 时固定 `interval=0.25`（与预训练 `tick_interval=0.25` 一致）
- 预测输入经 `process_round` 构建，输出 v5 世界坐标轨迹，前端直接叠加在 3D 场景上
- 若上传的是旧 v4 标签数据，`prediction_engine` 会自动转换坐标系
