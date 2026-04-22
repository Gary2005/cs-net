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
- [Acknowledgements](#-acknowledgements)
- [Contributors](#-contributors)

---

## 📌 Project Overview

CS-NET is a **Transformer**-based deep learning framework for analyzing Counter-Strike 2 match replays (`.dem` demo files). It parses match recordings, converts game states into token sequences, and uses pre-trained Transformer models for multiple real-time predictions.

In short: **given a match replay, the model tells you who will win, who will die, and who is most likely to get the next kill.**

### Prediction Tasks

| Task | Description | Output |
|------|-------------|--------|
| **Win Rate Prediction** | Probability of team1 (mapped from CT/T by side) winning the current round | Scalar in [0, 1] |
| **Alive Prediction** | Per-player probability of surviving the next 5 seconds | One probability per player |
| **Next Kill Prediction** | Which player is most likely to get the next kill | Probability distribution over 10+1 classes |
| **Next Death Prediction** | Which player is most likely to die next | Probability distribution over 10+1 classes |
| **Duel Prediction** | 1v1 win probability for any CT-T player pair | 5x5 probability matrix |

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

> **Attribution Notice**
> The built-in 2D viewer is a modified integration of
> [`sparkoo/csgo-2d-demo-viewer`](https://github.com/sparkoo/csgo-2d-demo-viewer).
> We use the upstream project under the MIT License and adapt it for CS-NET's
> Flask routes and model-prediction overlays.

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
2. Select the **model root directory** (normally `cs-net-models/`). The web app
   loads all five prediction heads (`alive`, `nxt_kill`, `nxt_death`,
   `win_rate`, `duel`) from their subdirectories in one go.
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
- **Live 2D radar** that syncs with the win-rate curve — player positions,
  team colour, alive/dead state, and "recently flashed" flag are drawn on the
  real minimap overview for every tick the cursor touches.
- **Per-tick metric panels** driven by all four prediction heads:
  5-second survival probability, next-kill distribution, next-death
  distribution, and the full 5×5 CT-vs-T duel matrix.
- **Advanced metrics table** aggregated across the whole match: per-player
  average kill/death/survival probability, hard-duel win rate (fights the
  model thought they would lose), easy-duel win rate (fights they were
  favoured in), highlight rate, plus a |swing|-sorted ranking of the most
  impactful kills.
- **One-click 2D replay viewer** — launches a bundled build of
  [`sparkoo/csgo-2d-demo-viewer`](https://github.com/sparkoo/csgo-2d-demo-viewer)
  in a new tab with the same demo, adding smoke/flash/grenade trajectories
  and an in-page timeline overlaid with CS-NET's predictions.
- Current round final contribution table + full match average contribution table.
- MVP and SVP badges.
- LLM summary supports streaming output and Markdown rendering.
- Auto-save user settings in browser local storage:
  API Key, model name, base URL, temperature, device, model path, batch size, language.
- Team-side context for LLM:
  per-round attack/defense roles, first-half/second-half side assignment and half scores.

## 🙏 Acknowledgements

The bundled 2D replay viewer under `demo_analysis/static/viewer/` is a lightly
modified build of the excellent open-source project
**[sparkoo/csgo-2d-demo-viewer](https://github.com/sparkoo/csgo-2d-demo-viewer)**
by **Michal Vala**, distributed under the MIT License (© 2023 Michal Vala).
All credit for the viewer's parsing, rendering, and UX belongs to the upstream
authors — CS-NET only rewires its asset paths and feeds in the
per-tick predictions from our models. Huge thanks to Michal and the upstream
contributors for making such a polished tool available to the community.

The original upstream license is reproduced verbatim at
[`demo_analysis/static/viewer/LICENSE`](demo_analysis/static/viewer/LICENSE)
and applies to every file in that directory. If you reuse or redistribute the
viewer portion of this repository, please preserve that notice.

## 🤝 Contributors

- [Gary2005](https://github.com/Gary2005)
- [czdzx](https://github.com/czdzx)