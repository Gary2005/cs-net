<p align="center">
  <img src="visualizer/static/logo.svg" width="120" height="120" alt="cs-net logo">
</p>

<h1 align="center">CS-NET <img src="https://img.shields.io/badge/version-v5-38bdf8?style=flat-square" alt="v5"></h1>

<p align="center">
  <strong>A CS2 3D AI analysis and replay tool</strong>
</p>

<p align="center">
  <a href="README_CN.md">中文</a>
  ·
  <a href="#quick-start">Quick start</a>
  ·
  <a href="#download-the-ai-models">Download models</a>
  ·
  <a href="#using-the-web-app">Using the web app</a>
  ·
  <a href="#repository-layout">Repository layout</a>
  ·
  <a href="#docs">Docs</a>
  ·
  <a href="#licensing">Licensing</a>
</p>

<p align="center">
  <img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10%2B-blue">
  <img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-green">
</p>

---

## What this is

cs-net v5 is an AI analysis tool that runs locally, with 3D replay built in.

## Quick start

Requirements: Python ≥ 3.10, and a Go toolchain (only to build the smoke/fire
side-channel, once).

```bash
git clone https://github.com/Gary2005/cs-net.git
cd cs-net

conda create -n cs-net-v5 python=3.10 -y
conda activate cs-net-v5
pip install -r requirements.txt

# 1) build the Go side-channel (smoke voxels + fire cells, once per machine)
bash tools/demo_extract/build_sidechannel.sh      # PowerShell: bash is Git Bash's

# 2) unpack the map geometry (maps/*.obj.zip -> *.obj, ~1 GB on disk)
python scripts/extract_maps.py

# 3) run the app
python -m app.server --port 5000                  # → http://127.0.0.1:5000/library
```

### Download the AI models

The two checkpoints behind the AI panels are 665 MB together and do not belong in
git, so they live on Hugging Face. Download them **once**, either way:

* in the app: open the library page and click **下载模型 / Download models** in the
  banner that appears when `models/` is missing something;
* from the CLI: `python scripts/download_models.py`.

Both run the same script, which is built on `huggingface_hub` (resume and
checksums included). It probes `huggingface.co` first and, when that is
unreachable, **switches `HF_ENDPOINT` to `hf-mirror.com`** automatically — the
same thing as exporting `HF_ENDPOINT=https://hf-mirror.com` yourself. `--mirror`
forces the mirror, `--endpoint URL` pins a host, and `--check` reports what is
missing without downloading anything.

`python scripts/check_env.py` verifies the whole environment (Python, modules,
side-channel binary, browser, unrar); run it if something looks off.

## Using the web app

1. **Add a watch folder** — top bar → **设置 / Settings** → paste the directory
   that holds your demos (candidates found on this machine are listed) → **Save**
   → **扫描 / Scan**. The library starts empty on purpose: the folder has to be
   yours.
2. **Wait for extraction** — each `.dem` becomes a `.csz` in the background
   (110 MB → ~7 MB). Progress shows in the list and in the **任务 / Tasks** drawer.
3. **Watch it back** — tick one or more demos → **载入分析 / Load analysis** →
   the 3D replay page: players, grenades, smoke volumes, fire cells, kill feed,
   minimap, visibility.
4. **Browse players** — the **玩家 / Players** page aggregates every demo in the
   library; each profile has aim/impact stats, and a **AI 数据 / AI stats** tab.
5. **Turn on AI** — bottom of the replay page: **AI 预测面板** (round win chance,
   per-player five heads) and **AI 对枪** (who wins the next duel, as an arc).

## Repository layout

```
app/                              The web app: demo library, player pages, AI panels (Flask + SQLite)
visualizer/                       3D replay studio (Flask + Three.js) — also the site logo
tools/demo_extract/               .dem -> .csz extractor: per-tick, zstd, smoke/fire decoders
training/                         PyTorch models and trainers (what the AI panels load)
scripts/training_data/            .csz -> task datasets for training
scripts/                          Helpers: download_models.py, extract_maps.py, check_env.py, export_weights.py
maps/optimized_obj_files/         Decimated map geometry (.obj.zip — unpack with extract_maps.py)
models/                           The two pinned checkpoints (git-ignored, downloaded)
```

The `.csz` codec is exact: `codec.save` → `codec.load` is bit-identical
(`codec.equal`), not approximate. The extractor samples every tick (≈64 Hz) and
reads smoke voxels and fire cells through a small Go side-channel for the fields
`demoparser2` cannot reach.

## Docs

| Doc | Contents |
|---|---|
| [app/README.md](app/README.md) | Demo library, thumbnails, player pages, AI panels, HTTP API |
| [visualizer/README.md](visualizer/README.md) | Replay studio behaviour and internals |
| [tools/demo_extract/README.md](tools/demo_extract/README.md) | `.csz` field tree, sampling modes, smoke voxels, fire cells |
| [tools/demo_extract/API.md](tools/demo_extract/API.md) | Field-level reference, measured magnitudes, pitfalls |
| [scripts/training_data/README.md](scripts/training_data/README.md) | Writing a task, slicing samples, dataset pitfalls |

## Licensing

The **code** in this repository is MIT — see [LICENSE](LICENSE).

The **game content** is not. `visualizer/static/assets/` (weapon models,
textures, smoke/fire sprites, HUD icons, radar images) and the geometry in
`maps/optimized_obj_files/` are derived from Valve's Counter-Strike 2 content,
© Valve Corporation.
