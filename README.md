<p align="center">
  <img src="visualizer/static/logo.svg" width="120" height="120" alt="cs-net logo">
</p>

<h1 align="center">CS-NET</h1>

<p align="center">
  <strong>Per-tick Counter-Strike 2 demo extraction, a 3D replay studio, and a PyTorch stack trained on real match data.</strong>
</p>

<p align="center">
  <a href="README_CN.md">中文</a>
  ·
  <a href="#quick-start">Quick start</a>
  ·
  <a href="#the-pipeline">Pipeline</a>
  ·
  <a href="#training">Training</a>
  ·
  <a href="#repository-layout">Layout</a>
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

CS-NET turns a `.dem` replay into a local, inspectable dataset — and then into a
browser replay studio and a training set. Four pieces:

| Piece | What it does | Entry point |
|---|---|---|
| **`tools/demo_extract`** | One `.dem` → one `.csz`. **Every tick** (≈64 Hz) for 10 players, projectiles and grenades — plus **real smoke voxel clouds** (32³, Morton-decoded) and **real fire cells** (3D position + surface normal) read through a small Go side-channel for the fields `demoparser2` cannot reach. Columnar + zstd: a 110 MB demo becomes ~7 MB, against ~183 MB as JSON. | `python -m demo_extract` |
| **`app/`** | Demo library: watch folders, dedupe on import, radar thumbnails generated from map geometry, per-player profiles, AI probability panels. | `python -m app.server` |
| **`visualizer/`** | 3D replay studio (Flask + Three.js): smooth playback on decimated real map geometry, smoke volumes, fire cells, projectile arcs, kill feed, minimap, visibility. | `python visualizer/server.py` |
| **`scripts/training_data` + `training/`** | `.csz` → task datasets (`aim` duel windows, `spatial_prediction` single-tick labels) → PyTorch training. | `python -m scripts.training_data.prepare`, `python -m training.aim.train` |

The `.csz` codec is exact: `codec.save` → `codec.load` is **bit-identical**
(`codec.equal`), not approximate.

## The pipeline

```
    .dem ──demo_extract──▶ .csz    per-tick, zstd, ~26× smaller than JSON
          (+ Go side-channel: fire cells + smoke voxels)
                                 │
        ┌────────────────────────┼─────────────────────────┐
        ▼                        ▼                         ▼
   app/ demo library      visualizer/ 3D studio     scripts/training_data
   thumbnails · players   replay in the browser    prepared/<task>/ ──▶ training/
   AI panels
```

## Quick start

Requirements: Python ≥ 3.10, and a Go toolchain (only to build the fire/smoke
side-channel).

```bash
git clone https://github.com/Gary2005/cs-net.git
cd cs-net

conda create -n cs-net python=3.10 -y
conda activate cs-net
pip install -r requirements.txt

# 1) build the Go side-channel (fire cells + smoke voxels)
bash tools/demo_extract/build_sidechannel.sh

# 2) unpack the map geometry (maps/*.obj.zip -> *.obj, ~1 GB)
python scripts/extract_maps.py

# 3) verify the environment (python, modules, side-channel, browser, unrar)
python scripts/check_env.py

# 4) run the workbench
python -m app.server --port 5000        # → http://127.0.0.1:5000/library
```

The library starts empty on purpose — the watch directory has to be yours
(top bar → **Settings** → add a folder → **Save**, then **Scan**). Demos are
extracted to `.csz` in the background and the round is one click away from the
3D replay.

### Extract or inspect a demo from the CLI

```bash
export PYTHONPATH=tools                # PowerShell: $env:PYTHONPATH = "tools"

python -m demo_extract match.dem -o match.csz                    # every tick
python -m demo_extract match.dem --interval 0.25 -o match.csz    # 4 Hz
python -m demo_extract match.dem --stride 8 -o match.csz         # every 8th tick
python -m demo_extract --inspect match.csz                       # what is inside
python -m demo_extract match.dem --selftest                      # extract → save → load → bit-compare
```

```python
from demo_extract import extract, codec

data = extract("match.dem")          # every tick
codec.save(data, "match.csz")
assert codec.equal(data, codec.load("match.csz"))[0]
```

## Training

```bash
# .csz -> prepared/<task>/{train,valid,test}
python -m scripts.training_data.prepare --task aim
python -m scripts.training_data.prepare --task spatial_prediction

# train (configs default to /data/... ; override with --set data.root=...)
python -m training.aim.train -c training/aim/config.yaml
python -m training.spatial_prediction.train -c training/spatial_prediction/config.yaml
```

| Task | Sample | Labels |
|---|---|---|
| `aim` | a 257-frame duel window for A vs B | `P(A kills B)`, per frame |
| `spatial_prediction` | **one** frame | 5 heads: CT win, alive at round end, can kill, next death, next kill |

Inspect a dataset without a GPU: `python -m scripts.training_data_visualizer.aim`
(port 5117) and `python -m scripts.training_data_visualizer.spatial_prediction`
(port 5118).

**Weights are not shipped.** `models/` is git-ignored — the two exported
checkpoints alone are ~665 MB, and keeping the optimizer state would push a
single run to ~2 GB. The AI panels in `app/` read the pinned paths in
`app/ai/models.py`, so export a trained run first:

```bash
python scripts/export_weights.py          # <run>/last.pt + config.yaml -> <run>.pt + <run>.yaml
```

## Repository layout

```
app/                              Demo library + player pages + AI panels (Flask, SQLite)
visualizer/                       3D replay studio (Flask + Three.js)
tools/demo_extract/               .dem → .csz: per-tick extractor, codec, smoke/fire decoders
scripts/training_data/            task framework: .csz → prepared datasets
scripts/training_data_visualizer/ standalone sample inspectors (numpy only)
training/                         PyTorch models and trainers
maps/optimized_obj_files/         decimated map geometry (.obj.zip — unzip first)
```

## Docs

| Doc | Contents |
|---|---|
| [tools/demo_extract/README.md](tools/demo_extract/README.md) | `.csz` field tree, sampling modes, smoke voxels, fire cells, known trade-offs |
| [tools/demo_extract/API.md](tools/demo_extract/API.md) | field-level reference, measured magnitudes, pitfalls |
| [visualizer/README.md](visualizer/README.md) | replay studio behaviour and internals |
| [app/README.md](app/README.md) | demo library, thumbnails, player pages, AI panels |
| [scripts/training_data/README.md](scripts/training_data/README.md) | writing a task, slicing samples, dataset pitfalls |

## Caveats

- **The side-channel binary is per-platform.** `tools/demo_extract/sidechannel_go/`
  ships sources, not an executable; run `build_sidechannel.sh` on your machine.
  A binary built elsewhere dies with `Exec format error` — `check_env.py` catches
  this and says so.
- **Do not downsample if you care about crouching.** The crouch transition is
  ~200 ms (13 ticks); `--interval 0.25` eats 94% of it.
- **Every-round completeness varies.** A round with no `round_freeze_end` has no
  time origin and is skipped; this is recorded in `skipped_rounds`, not data loss.
- **Assets are large.** The map geometry in `maps/` is stored zipped (~207 MB).

## Licensing

The **code** in this repository is MIT — see [LICENSE](LICENSE).

The **game content** is not. `visualizer/static/assets/` (weapon models,
textures, smoke/fire sprites, HUD icons, radar images) and the geometry in
`maps/optimized_obj_files/` are derived from Valve's Counter-Strike 2 content,
© Valve Corporation. They are licensed to you for personal, non-commercial use
under the [Steam Subscriber Agreement](https://store.steampowered.com/subscriber_agreement/),
**not** by the MIT license above. They are checked in because a clean checkout
cannot rebuild them (that needs a local CS2 install). Keep that in mind before
redistributing them or publishing builds that bundle them.
