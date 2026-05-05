<p align="center">
  <img src="assets/logo.svg" width="120" height="120" alt="cs-net-logo">
</p>

<h1 align="center">CS-NET</h1>

<p align="center">
  <strong>面向 Counter-Strike 比赛数据分析的深度学习框架</strong>
</p>

<p align="center">
  <a href="README.md">English</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/Framework-PyTorch-ee4c2c.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
</p>

## 快速导航

- [项目概览](#项目概览)
- [预测任务](#预测任务)
- [快速开始](#快速开始)
- [Web App 使用方法](#web-app-使用方法)
- [Web App 功能](#web-app-功能)
- [致谢](#致谢)
- [贡献者](#贡献者)

---

## 项目概览

CS-NET 是一个基于 **Transformer** 的深度学习框架，用于分析 Counter-Strike 2 的比赛回放（`.dem` demo 文件）。它会解析比赛录像，把游戏状态转换成 token 序列，再交给预训练的 Transformer 模型做多种实时预测。

一句话概括：**给模型一段比赛回放，它能告诉你接下来谁会赢、谁会死，以及谁最可能拿到下一次击杀。**

## 预测任务

| 任务 | 说明 | 输出 |
|------|------|------|
| **胜率预测** | 当前回合 team1（按攻防映射）赢下本局的概率 | 0 到 1 之间的标量 |
| **存活预测** | 每个玩家在接下来 5 秒内仍然存活的概率 | 10 个玩家分别对应一个概率 |
| **下一次击杀预测** | 谁最可能拿到下一次击杀 | 10+1 类的概率分布 |
| **下一次阵亡预测** | 谁最可能成为下一次阵亡者 | 10+1 类的概率分布 |
| **决斗预测** | 任意 CT-T 玩家对之间的 1v1 胜率 | 5x5 概率矩阵 |

## 快速开始

### 1. 配置环境

创建 Python 环境并安装依赖：

```bash
conda create -n cs-net python=3.10
conda activate cs-net
pip install -r requirements.txt
```

### 2. 下载预训练模型

将所有预训练模型和分词器下载到 `./cs-net-models/`：

模型权重也可以在这里获取：https://huggingface.co/gary2oos/CS-Net-V3

```bash
python -m scripts.download_model
```

### 3. 将 Demo 转换为 JSON

使用 `process_demo` 脚本把 demo 文件解析为结构化 JSON：

`examples/test.dem` 故意没有包含在仓库中，因为 demo 文件通常非常大。
你需要自己下载一个 `.dem` 文件（例如来自 HLTV），并替换输入路径。

```bash
python -m data.process_demo \
  -path examples/test.dem \
  -interval 0.25 \
  -out examples/test.json
```

## Web App 使用方法

CS-NET 现在内置了一个交互式网页分析面板，可以直接上传 demo，并完成模型分析和基于 LLM 的赛后复盘。

> **署名说明**
> 内置的 2D 查看器改编自 [`sparkoo/csgo-2d-demo-viewer`](https://github.com/sparkoo/csgo-2d-demo-viewer)。
> 我们在上游 MIT 协议下使用该项目，并将其适配为 CS-NET 的 Flask 路由与模型预测叠加显示。

### 1. 启动 Web App

```bash
python -m demo_analysis.web_app
```

然后打开：

```text
http://127.0.0.1:7860
```

### 2. 在界面中分析 demo

1. 上传 .dem 文件。
2. 选择 **模型根目录**（通常是 `cs-net-models/`）。网页会一次性从根目录下加载 `alive / nxt_kill / nxt_death / win_rate / duel` 五个预测头，不需要再分别指定各自的子目录。
3. 选择推理设备（cpu / cuda / mps）。
4. 点击开始分析。

### 3. 生成 LLM 复盘

1. 填写 API Key、模型名和 Base URL（OpenAI 兼容）。
2. 选择界面语言（中文 / English）。
3. 点击生成 AI 复盘。

## Web App 功能

- 中英文双语界面与双语 LLM 输出。
- 回合胜率曲线 + 击杀事件标记。
- 鼠标悬停时间线即可查看该时刻的玩家贡献。
- **实时 2D 雷达**：鼠标在胜率曲线上移动时同步刷新，在真实地图 overview 上画出每个玩家的位置、阵营颜色、存活状态以及是否刚被闪。
- **逐 tick 指标面板**：四个预测头的输出完整展开，包括 5 秒内存活概率、下一击杀者分布、下一阵亡者分布，以及 CT vs T 的 5×5 对决胜率矩阵。
- **高级指标面板**：跨整场比赛聚合每个玩家的平均 kill / death / survive 概率、硬仗胜率（模型原本认为他会输的 1v1）、易仗胜率（模型原本看好他的 1v1）、highlight 率，以及按 |swing| 排序的关键击杀榜。
- **一键打开 2D 回放器**：在新标签页直接播放同一段 demo，包含烟雾 / 闪光 / 手雷弹道，并将 CS-NET 的胜率曲线叠加到 viewer 的时间线上。
- 当前回合最终贡献表 + 全场平均贡献表。
- MVP / SVP 标记。
- LLM 总结支持流式输出与 Markdown 渲染。
- 自动记住用户输入（浏览器本地存储）：API Key、模型名、Base URL、Temperature、设备、模型目录、Batch Size、语言。
- 为 LLM 提供攻防上下文，降低幻觉：每回合攻防归属、上下半场 CT/T 归属与分半比分。

## 致谢

Web App 中的 2D 回放器（`demo_analysis/static/viewer/` 下的全部文件）来自 **[sparkoo/csgo-2d-demo-viewer](https://github.com/sparkoo/csgo-2d-demo-viewer)**（作者 **Michal Vala**，MIT License，版权归 © 2023 Michal Vala 所有）。我们只是把它的静态资源路径接到 Flask 的 `/viewer/` 路由下，并把 CS-NET 模型输出的逐 tick 预测接入它的时间线。**回放器本身的 demo 解析、地图渲染和交互都由上游作者实现，相关技术信誉均归上游。**

该部分文件保留了原 MIT 许可证原文，见 [`demo_analysis/static/viewer/LICENSE`](demo_analysis/static/viewer/LICENSE)。如果你要进一步转发或再分发这部分代码，请一并保留该 LICENSE 文件与版权声明，以避免违反 MIT 协议。

## 贡献者

- [Gary2005](https://github.com/Gary2005)
- [czdzx](https://github.com/czdzx)
