<p align="center">
  <img src="assets/logo.svg" width="120" height="120" alt="cs-net-logo">
</p>

<h1 align="center">CS-NET</h1>

<p align="center">
  <strong>基于深度学习的 Counter-Strike 比赛数据分析框架</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/Framework-PyTorch-ee4c2c.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
</p>

---

## 目录

- [项目简介](#项目简介)
- [项目介绍视频](#项目介绍视频)
- [它能做什么](#它能做什么)
- [模型架构](#模型架构)
- [环境配置（手把手教程）](#环境配置手把手教程)
  - [前提条件](#前提条件)
  - [第一步：安装 Conda](#第一步安装-conda)
  - [第二步：创建 Python 虚拟环境](#第二步创建-python-虚拟环境)
  - [第三步：安装项目依赖](#第三步安装项目依赖)
  - [第四步：下载预训练模型](#第四步下载预训练模型)
- [快速上手](#快速上手)
  - [获取 Demo 文件](#获取-demo-文件)
  - [将 Demo 转换为 JSON](#将-demo-转换为-json)
  - [运行案例分析与可视化](#运行案例分析与可视化)
- [输出结果解读](#输出结果解读)
- [项目结构](#项目结构)
- [训练你自己的模型（进阶）](#训练你自己的模型进阶)
- [常见问题 FAQ](#常见问题-faq)
- [贡献者](#贡献者)

---

## 项目简介

CS-NET 是一个基于 **Transformer** 架构的深度学习框架，用于分析 Counter-Strike（CS2）的比赛回放数据（`.dem` demo 文件）。它可以解析比赛录像，将游戏状态转化为 token 序列，然后通过预训练的 Transformer 模型进行多种实时预测。

简单来说：**给模型一段比赛回放，它能告诉你接下来谁会赢、谁会死、谁最可能拿到击杀。**

## 项目介绍视频

- Bilibili: https://www.bilibili.com/video/BV1rnQFB8Epu

## 它能做什么

CS-NET 支持以下四种预测任务：

| 预测任务 | 说明 | 输出形式 |
|---------|------|---------|
| **CT 胜率预测** | 当前时刻 CT 阵营赢下本局的概率 | 0~1 之间的概率值 |
| **存活预测** | 每个玩家在本局结束时还活着的概率 | 10 个玩家各一个概率值 |
| **下一次击杀预测** | 谁最可能拿到下一个击杀 / 谁最可能被杀 | 10+1 个玩家的概率分布 |
| **决斗预测** | 任意 CT-T 玩家对之间 1v1 的胜率 | 5x5 的概率矩阵 |

## 模型架构

CS-NET 使用三阶段 Transformer 架构：

```
Demo 文件 (.dem)
    │
    ▼
┌──────────────┐
│  状态提取器    │  解析 demo，提取每个采样时刻的游戏状态
│  (demoparser2) │  （玩家位置、血量、武器、投掷物等）
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Tick 分词器   │  将连续的游戏状态离散化为 token 序列
│  (TickTokenizer)│  词表大小：979
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Embedder     │  非因果 Transformer 编码器（6层, 10头）
│  (编码单帧)    │  将单个时刻的 token 序列编码为向量
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Processor    │  因果 Transformer 编码器（8层, 10头）
│  (时序建模)    │  建模连续时刻之间的时序关系（类似 GPT）
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  预测头       │  不同任务使用不同的 MLP 预测头
│  (Task Heads) │  存活 / 击杀 / 胜率 / 决斗
└──────────────┘
```

---

## 环境配置（手把手教程）

> 如果你是第一次接触 Python / 深度学习项目，请仔细阅读本节。

### 前提条件

你需要一台运行以下系统之一的电脑：
- **Windows 10/11**（推荐使用 WSL2，也可以直接在 Windows 上操作）
- **macOS**（Intel 或 Apple Silicon M1/M2/M3/M4 均可）
- **Linux**（Ubuntu、Debian 等）

硬件要求：
- **内存**：至少 8GB RAM（推荐 16GB）
- **存储空间**：至少 10GB 可用空间（模型 + 依赖包）
- **GPU（可选）**：NVIDIA 显卡可以加速推理，没有也没关系，CPU 也能跑

### 第一步：安装 Conda

Conda 是一个 Python 环境管理工具，可以帮你隔离不同项目的依赖，避免冲突。

#### Windows 用户

1. 访问 [Miniconda 官网](https://docs.conda.io/en/latest/miniconda.html)
2. 下载 **Miniconda3 Windows 64-bit** 安装包
3. 双击运行安装程序，一路点 "Next"（建议勾选 "Add Miniconda3 to my PATH environment variable"）
4. 安装完成后，打开 **Anaconda Prompt**（在开始菜单搜索）

#### macOS 用户

打开终端（Terminal），运行：

```bash
# Intel Mac
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
bash Miniconda3-latest-MacOSX-x86_64.sh

# Apple Silicon (M1/M2/M3/M4)
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
bash Miniconda3-latest-MacOSX-arm64.sh
```

按照提示完成安装，然后重启终端。

#### Linux 用户

```bash
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

按照提示完成安装，然后重启终端。

#### 验证安装

安装完成后，在终端输入：

```bash
conda --version
```

如果输出类似 `conda 24.x.x`，说明安装成功。

### 第二步：创建 Python 虚拟环境

打开终端（Windows 用户打开 Anaconda Prompt），依次运行：

```bash
# 创建一个名为 cs-net 的虚拟环境，Python 版本为 3.10
conda create -n cs-net python=3.10

# 等待安装完成后，激活环境
conda activate cs-net
```

激活成功后，终端提示符前面会出现 `(cs-net)` 字样，如：

```
(cs-net) your-username@your-computer:~$
```

> **注意**：以后每次打开新终端使用本项目时，都需要先运行 `conda activate cs-net` 来激活环境。

### 第三步：安装项目依赖

首先，确保你已经下载了本项目代码：

```bash
# 如果还没有克隆项目
git clone https://github.com/Gary2005/cs-net.git
cd cs-net
```

然后安装所有依赖包：

```bash
pip install -r requirements.txt
```

这一步会安装 PyTorch、demoparser2、HuggingFace Hub 等所有需要的 Python 库。安装过程可能需要几分钟，取决于你的网速。

> **如果安装 PyTorch 速度很慢或失败**：可以先单独安装 PyTorch。访问 [PyTorch 官网](https://pytorch.org/get-started/locally/)，选择你的系统和 CUDA 版本，复制对应的安装命令运行。然后再执行 `pip install -r requirements.txt` 安装其余依赖。

> **中国大陆用户加速提示**：可以使用清华镜像源加速下载：
> ```bash
> pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
> ```

### 第四步：下载预训练模型

运行以下命令，自动从 HuggingFace Hub 下载所有预训练模型和分词器配置文件：

模型权重文件也可以在这里获取：https://huggingface.co/gary2oos/cs-net

```bash
python -m examples.download_model
```

下载完成后，会在项目根目录创建 `cs-net-models/` 文件夹，结构如下：

```
cs-net-models/
├── alive/              # 存活预测模型
│   ├── alive_fine-tuning.pth
│   ├── tfm_alive_fine-tuning.yaml
│   └── tokenizer.yaml
├── duel/               # 决斗预测模型
│   ├── duel_fine-tuning.pth
│   ├── tfm_duel_fine-tuning.yaml
│   └── tokenizer.yaml
├── nxt_kill/           # 击杀预测模型
│   ├── nxt_kill_fine-tuning.pth
│   ├── tfm_nxt_kill_fine-tuning.yaml
│   └── tokenizer.yaml
└── win_rate/           # 胜率预测模型
    ├── win_rate_fine-tuning.pth
    ├── tfm_win_rate_fine-tuning.yaml
    └── tokenizer.yaml
```

> **如果 HuggingFace 下载很慢**：中国大陆用户可以设置 HuggingFace 镜像：
> ```bash
> export HF_ENDPOINT=https://hf-mirror.com
> python -m examples.download_model
> ```

---

## 快速上手

### 获取 Demo 文件

你需要一个 Counter-Strike 2 的比赛回放文件（`.dem` 格式）。可以从以下途径获取：

1. **HLTV**：访问 [hltv.org](https://www.hltv.org)，找到任意一场职业比赛，在比赛页面下载 demo
2. **CS2 游戏内**：在 CS2 游戏中，打开"观看"选项卡，下载你自己的比赛回放
3. **FACEIT / 完美世界**：从对应平台下载比赛 demo

下载后，将 `.dem` 文件放到项目目录下（比如 `examples/test.dem`）。

> **注意**：`.dem` 文件通常很大（几十到几百 MB），因此不包含在 Git 仓库中。

### 将 Demo 转换为 JSON

使用 `process_demo` 脚本将 demo 文件解析为结构化 JSON 数据：

```bash
python -m data.process_demo \
  -path examples/test.dem \
  -interval 0.25 \
  -out examples/test.json
```

参数说明：

| 参数 | 说明 | 默认值 |
|------|------|-------|
| `-path` | demo 文件路径 | （必填） |
| `-interval` | 采样间隔，单位：秒。每隔多少秒提取一帧游戏状态 | 0.5 |
| `-out` | 输出 JSON 文件路径 | （必填） |
| `-debug` | 是否输出调试信息 | 0（关闭） |
| `-compression` | 是否压缩 JSON 输出 | 0（关闭） |

> `-interval 0.25` 表示每 0.25 秒采样一次，精度更高但文件更大。推荐值为 0.25~0.5。

### 运行案例分析与可视化

使用以下命令对处理好的 JSON 数据进行模型推理：

```bash
python -m examples.case_study \
  --json_path examples/test.json \
  --alive_ckpt_dir cs-net-models/alive \
  --kill_ckpt_dir cs-net-models/nxt_kill \
  --winrate_ckpt_dir cs-net-models/win_rate \
  --duel_ckpt_dir cs-net-models/duel \
  --device cpu
```

参数说明：

| 参数 | 说明 |
|------|------|
| `--json_path` | 上一步生成的 JSON 文件路径 |
| `--alive_ckpt_dir` | 存活预测模型目录 |
| `--kill_ckpt_dir` | 击杀预测模型目录 |
| `--winrate_ckpt_dir` | 胜率预测模型目录 |
| `--duel_ckpt_dir` | 决斗预测模型目录 |
| `--device` | 推理设备（见下方说明） |
| `--remove_projectiles` | 可选，移除投掷物数据（减少噪声） |

`--device` 参数选择指南：

| 设备类型 | 值 | 适用场景 |
|---------|-----|---------|
| CPU | `cpu` | 没有独显 / 纯 CPU 推理 |
| NVIDIA GPU | `cuda` | 有 NVIDIA 显卡 + 已安装 CUDA |
| Apple Silicon | `mps` | Mac M1/M2/M3/M4 芯片 |

运行后，程序会提示你输入：
1. **回合编号**（Round number）：你想分析第几回合（从 0 开始计数）
2. **回合内时间**（Round seconds）：该回合开始后第几秒

然后模型会输出该时刻的所有预测结果。

---

## 输出结果解读

运行 case study 后，终端会显示一个 ASCII 雷达图和多个预测信息块。

### 雷达图对比

真实游戏内雷达截图：

![Ground Truth Radar](assets/radar.png)

case_study 输出的 ASCII 雷达图：

![Case Study Radar Output](assets/output.png)

- 蓝色背景 = CT 阵营玩家
- 黄色背景 = T 阵营玩家
- 灰色背景 `x` = 已阵亡玩家
- 箭头方向（`^` `v` `<` `>`）= 玩家面朝方向

### 完整输出示例

```text
======================================================================
Round: 4 | Time: 20.00s

CT Win Rate:
  0.1121

Alive Prediction:
  ztr             0.1372
  nota            0.5957
  xiELO           0.4869
  Matheos         0.1589
  zweih           0.5235
  volt            0.2621
  BELCHONOKK      0.4544
  Jame            0.5232
  Banjo           0.3307
  Jorko           0.3546

Next Killer Distribution:
  ztr             0.0808
  nota            0.1471
  xiELO           0.1298
  Matheos         0.1350
  zweih           0.1449
  volt            0.0737
  BELCHONOKK      0.0723
  Jame            0.0711
  Banjo           0.0802
  Jorko           0.0638
  <NO KILL>       0.0012

Next Death Distribution:
  ztr             0.1537
  nota            0.0676
  xiELO           0.1113
  Matheos         0.0912
  zweih           0.1023
  volt            0.1513
  BELCHONOKK      0.0910
  Jame            0.0704
  Banjo           0.0621
  Jorko           0.0981
  <NO DEATH>      0.0009

Duel Matrix (CT vs T)
P[CT beats T]

                     nota     xiELO     zweihBELCHONOKK      Jame
ztr                 0.330     0.414     0.352     0.435     0.310
Matheos             0.510     0.563     0.541     0.598     0.459
volt                0.337     0.402     0.395     0.499     0.359
Banjo               0.470     0.554     0.514     0.578     0.456
Jorko               0.361     0.480     0.439     0.520     0.369

======================================================================
```

### 各项输出含义

1. **`Round: X | Time: Ys`** — 确认实际匹配到的回合号和回合内时间
2. **`CT Win Rate`** — CT 阵营赢下本回合的概率。范围 `[0, 1]`，越接近 1 表示 CT 优势越大
3. **`Alive Prediction`** — 每位玩家在本回合结束时存活的概率。如果该玩家在当前时刻已死亡，会显示 `DEAD`
4. **`Next Killer Distribution`** — 谁最可能拿到下一个击杀的概率分布。`<NO KILL>` 表示接下来不会发生击杀
5. **`Next Death Distribution`** — 谁最可能是下一个被杀的概率分布。`<NO DEATH>` 表示接下来不会有人阵亡
6. **`Duel Matrix (CT vs T)`** — CT 对 T 的 1v1 决斗胜率矩阵。每个值表示 `P[CT 赢 T]`，大于 0.5 表示 CT 方占优

---

## 项目结构

```
cs-net/
├── README.md                   # 英文文档
├── README_CN.md                # 中文文档（本文件）
├── requirements.txt            # Python 依赖列表
├── assets/                     # 项目资源（logo、示例截图等）
│
├── config/                     # 训练配置文件（YAML 格式）
│   ├── tfm_pretrain_config.yaml          # 预训练配置
│   ├── tfm_alive_fine-tuning.yaml        # 存活预测微调配置
│   ├── tfm_duel_fine-tuning.yaml         # 决斗预测微调配置
│   ├── tfm_nxt_kill_fine-tuning.yaml     # 击杀预测微调配置
│   ├── tfm_win_rate_fine-tuning.yaml     # 胜率预测微调配置
│   └── demoparser_utils/
│       └── tokenizer.yaml                # 分词器配置
│
├── models/                     # 模型实现
│   ├── tfm_model.py            # 主模型（TickTransformerModel）
│   └── tfm_model_rope.py       # RoPE 位置编码变体
│
├── demoparser_utils/           # Demo 解析工具
│   ├── tick_tokenizer.py       # 游戏状态 → Token 序列
│   ├── state_extract.py        # 从 demo 提取游戏状态
│   └── tokenizer.yaml          # 分词器映射表（地图、武器等）
│
├── data/                       # 数据处理脚本
│   ├── process_demo.py         # .dem → .json 转换
│   └── create_training_data.py # .json → PyTorch 训练张量
│
├── dataset/                    # 数据集加载器
│   ├── streaming_dataset_all_label.py       # 单 GPU 流式数据集
│   └── ddp_streaming_dataset_all_label.py   # 多 GPU 分布式数据集
│
├── scripts/                    # 训练与推理脚本
│   ├── pretrain.py             # 预训练（Next Tick Prediction）
│   ├── alive_fine-tuning.py    # 存活预测微调
│   ├── duel_fine-tuning.py     # 决斗预测微调
│   ├── nxt_kill_fine-tuning.py # 击杀预测微调
│   ├── win_rate_fine-tuning.py # 胜率预测微调
│   └── inference.py            # 推理脚本
│
├── examples/                   # 示例代码
│   ├── case_study.py           # 案例分析（加载模型 + 可视化预测）
│   └── download_model.py       # 下载预训练模型
│
└── scraper/                    # 辅助工具
    └── seafile_client.py       # Seafile 云存储客户端
```

---

## 训练你自己的模型（进阶）

如果你有大量 demo 数据，可以自己训练模型。训练流程分为两个阶段：

### 阶段一：预训练（Next Tick Prediction）

模型通过预测下一个时刻的游戏状态来学习 CS 游戏的通用表征。

```bash
# 1. 准备训练数据：将 demo 转为 JSON，再将 JSON 转为训练张量
python -m data.process_demo -path your_demo.dem -interval 0.25 -out data.json
python -m data.create_training_data  # 根据配置文件中的数据路径

# 2. 开始预训练
python -m scripts.pretrain --config config/tfm_pretrain_config.yaml
```

### 阶段二：微调（Task-specific Fine-tuning）

在预训练模型基础上，针对特定预测任务进行微调：

```bash
# 存活预测
python -m scripts.alive_fine-tuning --config config/tfm_alive_fine-tuning.yaml

# 击杀预测
python -m scripts.nxt_kill_fine-tuning --config config/tfm_nxt_kill_fine-tuning.yaml

# 胜率预测
python -m scripts.win_rate_fine-tuning --config config/tfm_win_rate_fine-tuning.yaml

# 决斗预测
python -m scripts.duel_fine-tuning --config config/tfm_duel_fine-tuning.yaml
```

训练支持：
- **多 GPU 分布式训练**（DDP）
- **梯度累积**（默认 32 步，等效更大 batch size）
- **Cosine Annealing 学习率调度** + Warmup
- **Weights & Biases** 实验追踪（可选）
- **LoRA 微调**（参数高效微调，可选）

---

## 常见问题 FAQ

### Q: `pip install` 报错 / 超时怎么办？

**A:** 尝试使用国内镜像源：

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

如果 PyTorch 安装失败，先去 [PyTorch 官网](https://pytorch.org/get-started/locally/) 手动安装对应版本，再安装其余依赖。

### Q: `python -m examples.download_model` 下载很慢？

**A:** HuggingFace 在国内访问可能较慢，设置镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
python -m examples.download_model
```

### Q: 运行时报 `ModuleNotFoundError`？

**A:** 确保你已激活 conda 环境：

```bash
conda activate cs-net
```

并且是在项目根目录下运行命令。

### Q: 我的电脑没有 NVIDIA 显卡，能跑吗？

**A:** 完全可以。在运行 case_study 时将 `--device` 设为 `cpu` 即可。推理速度会稍慢，但功能完全一样。Mac 用户还可以使用 `--device mps` 利用 Apple Silicon 加速。

### Q: Demo 文件从哪里下载？

**A:** 主要来源：
- [HLTV](https://www.hltv.org) — 职业比赛 demo（免费）
- CS2 游戏内"观看"选项卡 — 你自己的比赛回放
- FACEIT / 完美世界 — 对应平台的比赛 demo

### Q: 支持哪些地图？

**A:** 模型的分词器配置中预定义了支持的地图列表。具体支持的地图可在 `demoparser_utils/tokenizer.yaml` 中查看 `maps` 字段。

### Q: `process_demo` 报错 / 某些 demo 解析失败？

**A:** 确保你的 demo 文件来自 CS2（而非 CS:GO），且文件完整没有损坏。不同版本的 CS2 更新可能导致 demo 格式变化，确保 `demoparser2` 是最新版本：

```bash
pip install --upgrade demoparser2
```

---

## 贡献者

- [Gary2005](https://github.com/Gary2005)
- [czdzx](https://github.com/czdzx)
