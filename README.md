<p align="center">
  <img src="assets/logo.svg" width="120" height="120" alt="cs-net-logo">
</p>

<h1 align="center">CS-NET</h1>

<p align="center">
  <strong>A deep learning framework for Counter-Strike match data analysis</strong>
</p>

<p align="center">
  <a href="README_CN.md">中文文档</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/Framework-PyTorch-ee4c2c.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
</p>

## Quick Links

- [Project Overview](#-project-overview)
- [Prediction Tasks](#prediction-tasks)
- [Quick Start](#-quick-start)
- [Web App Usage](#-web-app-usage)
- [Web App Features](#-web-app-features)
- [Contributors](#-contributors)

---

## 📌 Project Overview

CS-NET is a **Transformer**-based deep learning framework for analyzing Counter-Strike 2 match replays (`.dem` demo files). It parses match recordings, converts game states into token sequences, and uses pre-trained Transformer models for multiple real-time predictions.

In short: **given a match replay, the model tells you who will win, who will die, and who is most likely to get the next kill.**

### Prediction Tasks

| Task | Description | Output |
|------|-------------|--------|
| **CT Win Rate** | Probability of the CT side winning the current round | Scalar in [0, 1] |
| **Alive Prediction** | Per-player probability of surviving to the end of the round | One probability per player |
| **Next Kill Prediction** | Who is most likely to get the next kill / who is most likely to die | Probability distribution over 10+1 classes |
| **Duel Prediction** | 1v1 win probability for any CT-T player pair | 5x5 probability matrix |

### Model Architecture

CS-NET uses a three-stage Transformer architecture:

```
Demo File (.dem)
    │
    ▼
┌──────────────────┐
│  State Extractor │  Parse demo, extract game state at each sampled tick
│  (demoparser2)   │  (player positions, HP, weapons, projectiles, etc.)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Tick Tokenizer  │  Discretize continuous game state into token sequences
│  (TickTokenizer) │  Vocabulary size: 979
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Embedder        │  Non-causal Transformer encoder (6 layers, 10 heads)
│  (Single Frame)  │  Encode tokens from one tick into a vector
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Processor       │  Causal Transformer encoder (8 layers, 10 heads)
│  (Temporal)      │  Model temporal dependencies across ticks (GPT-style)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Task Heads      │  Task-specific MLP prediction heads
│  (Predictions)   │  Alive / Kill / Win Rate / Duel
└──────────────────┘
```

## 🚀 Quick Start

### 1. Setup Environment

Create a Python environment and install dependencies:

```bash
conda create -n cs-net python=3.10
conda activate cs-net
pip install -r requirements.txt
```

### 2. Download Pre-trained Models

Download all pre-trained models and tokenizers to `./cs-net-models/`:

Model weights are also available here: https://huggingface.co/gary2oos/cs-net

```bash
python -m examples.download_model
```

### 3. Convert Demo to JSON

Process a Counter-Strike demo file (.dem) into structured JSON format:

`examples/test.dem` is intentionally NOT included in this repository because demo files are too large.
You must download a `.dem` file yourself (for example from HLTV) and replace the input path.

```bash
python -m data.process_demo \
  -path examples/test.dem \
  -interval 0.25 \
  -out examples/test.json
```

### 4. Run Case Study & Visualization

Generate predictions and visualizations using the processed data:

```bash
python -m examples.case_study \
  --json_path examples/test.json \
  --alive_ckpt_dir cs-net-models/alive \
  --kill_ckpt_dir cs-net-models/nxt_kill \
  --death_ckpt_dir cs-net-models/nxt_death \
  --winrate_ckpt_dir cs-net-models/win_rate \
  --duel_ckpt_dir cs-net-models/duel \
  --device cpu
```

**Note on `--device` flag:**
- Use `cuda` for NVIDIA GPUs
- Use `mps` for Apple Silicon (M1/M2/M3)
- Use `cpu` for CPU-only inference

Optional flag:
- `--remove_projectiles`: remove projectile and grenade entities from JSON before inference.

## 🌐 Web App Usage

CS-NET now includes an interactive web app for demo analysis and LLM-based post-game summary.

### 1. Start the web app

```bash
python -m demo_analysis.web_app
```

Then open:

```text
http://127.0.0.1:7860
```

### 2. Analyze a demo in UI

1. Upload a .dem file.
2. Select model directory (normally cs-net-models/win_rate).
3. Select device (cpu / cuda / mps).
4. Click Start Analysis.

### 3. Generate LLM summary

1. Fill API Key, model name, and Base URL (OpenAI-compatible).
2. Choose app language (Chinese / English).
3. Click Generate AI Review.

## ✨ Web App Features

- Bilingual UI and bilingual LLM output (Chinese / English).
- Round-by-round win-rate curve with kill markers.
- Hover-to-inspect player contribution at each timeline point.
- Current round final contribution table + full match average contribution table.
- MVP and SVP badges.
- LLM summary supports streaming output and Markdown rendering.
- Auto-save user settings in browser local storage:
  API Key, model name, base URL, temperature, device, model path, batch size, language.
- Team-side context for LLM:
  per-round attack/defense roles, first-half/second-half side assignment and half scores.

## 🤝 Contributors

- [Gary2005](https://github.com/Gary2005)
- [czdzx](https://github.com/czdzx)