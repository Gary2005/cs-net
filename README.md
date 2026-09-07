<p align="center">
  <img src="visualizer/static/logo.svg" width="120" height="120" alt="cs-net-logo">
</p>

<h1 align="center">CS-NET v4</h1>

<p align="center">
  <strong>Transformer-based path prediction &amp; game-state forecasting for Counter-Strike 2</strong>
</p>

<p align="center">
  <a href="README_CN.md">中文文档</a>
  ·
  <a href="#-model-checkpoints">Model Checkpoints</a>
  ·
  <a href="#-quick-start">Quick Start</a>
  ·
  <a href="#-training">Training</a>
  ·
  <a href="#-docs">Docs</a>
</p>

---

## What's in this repo

CS-NET v4 is an open-source release of the CS2 (Counter-Strike 2) deep learning
stack, containing three pieces:

| Component | Description | Entry point |
|---|---|---|
| **Path prediction model** | Pre-trained Transformer that takes a 16-tick window of the full game state (10 players + bomb + projectiles + raycast depth maps) and **auto-regressively predicts each player's future movement path** (world coordinates), tokenized as discrete move/angle tokens. | `scripts/pretrain_model.py`, `scripts/prediction_engine.py`, `scripts/pretrain.py` |
| **Spatial-only models** | The minimal downstream task family: from a **single tick** of the full game state, predict per-player **winrate / alive-at-round-end / future kill** probabilities. Each model = pre-trained embedder + spatial transformer + linear head (no path, no history window). One checkpoint per task. | `scripts/spatial_only_predictor.py`, `scripts/downstream-spatial-only/finetune_spatial_only.py` |
| **3D visualizer** | Flask web app: upload a `.dem` / `.json` / `.json.gz` replay, watch a smooth 3D replay (Three.js + map OBJ models), upload a checkpoint to visualize **AI-predicted paths vs. ground truth**, and load the spatial-only models to get live per-player probability curves across the whole round. | `visualizer/server.py` |

```
demo (.dem) ──demo_parser──▶ round JSON ──create_training_data──▶ round WDS shards
round WDS ──create_pretrain_data──▶ window WDS ──pretrain.py──▶ path prediction ckpt
round WDS + path ckpt ──finetune_spatial_only.py──▶ winrate / alive_end / future_kill ckpts
any replay ──visualizer/server.py──▶ 3D replay + AI path prediction + spatial-only curves
```

## Model checkpoints

Trained models for the Pro architecture (d_model=768, 138.7M params) can be
downloaded from the [releases](../../releases) page:

| File | Task | Notes |
|---|---|---|
| `cs-net-v4-pro.pt` | Path prediction (pre-training) | 600k steps; full model (`model` + `global_step` keys) |
| `pretrain-v4-pro-win_rate.pt` | spatial-only `winrate` | `{task, model_state, head_state, config}` |
| `pretrain-v4-pro-alive_end.pt` | spatial-only `alive_end` | same format |
| `pretrain-v4-pro-future_kill.pt` | spatial-only `future_kill` | same format |

> The spatial-only checkpoints only contain the embedder + spatial transformer
> weights (plus a linear head), so they are much smaller than the full path
> prediction checkpoint. The model architecture must match `config/pretrain-a100-pro.yaml`.

### Download

```bash
pip install huggingface_hub
python scripts/download_checkpoints.py          # all 4 checkpoints → checkpoints/
python scripts/download_checkpoints.py --pretrain-only   # path prediction only
```

Or directly with the `hf` CLI:

```bash
hf download gary2oos/cs-net-v4 --local-dir checkpoints
```

### Verify the checkpoints load

```bash
python scripts/test_checkpoints.py --models-dir checkpoints
```

The test builds the model from the config, loads each checkpoint, checks the
architecture matches exactly (no missing/unexpected keys), verifies all weights
are finite, runs a synthetic full-round inference for the spatial-only models,
and — using the bundled round data
[`examples/json/test.json.gz`](examples/json/test.json.gz) (de_mirage,
24 rounds) — runs the **full pipeline** end-to-end:
`json.gz → filter → inventory remap → process_round → model inference`, the
exact same preprocessing chain used in training and the visualizer. Add
`--forward` to also run autoregressive path prediction on a synthetic game
state and on the real round data (CPU: 1–3 min each).

## Quick Start

### 1. Environment

```bash
conda create -n cs2demo python=3.10
conda activate cs2demo
pip install -r requirements.txt
```

Verify with `python scripts/check_env.py`. The visualizer needs Flask and a
browser; inference needs a working PyTorch build (CPU / MPS / CUDA all work).

### 2. 3D visualizer + AI path prediction

```bash
# Start with local checkpoints preloaded (path prediction + spatial-only models)
python visualizer/server.py --port 5000 \
    --checkpoint /path/to/cs-net-v4-pro.pt \
    --spatial-model-dir /path/to/spatial-ckpts \
    --device cpu            # or mps / cuda
```

Open `http://127.0.0.1:5000/`, then either drag a `.dem`/`.json`/`.json.gz`
replay onto the page, or use the built-in example files (put replays under
`examples/demo` / `examples/json`). Model checkpoints can also be uploaded
through the UI instead of passing them on the command line.

See [visualizer/README.md](visualizer/README.md) for the full feature list and API.

### 3. Path prediction inference (library / CLI)

```python
from scripts.prediction_engine import PredictionEngine

engine = PredictionEngine(
    "config/pretrain-a100-pro.yaml",
    "cs-net-v4-pro.pt",
    device="cpu",
    maps_dir="maps/optimized_obj_files",
)
result = engine.predict_at_tick(sample, query_tick=120)
# result["trajectories"][p] = {"pred_traj": [...], "gt_traj": [...], ...} (world coords)
```

`sample` is a round-level dict produced by `scripts/training_data/round_processor.py`
(`process_round`) from parsed round JSON — the same format used for training.

A standalone test CLI is also available (reads a round-level WDS shard):

```bash
python scripts/prediction_engine.py \
    --config config/pretrain-a100-pro.yaml \
    --checkpoint /path/to/cs-net-v4-pro.pt \
    --data-dir /path/to/round_wds \
    --tick 200 --device cpu
```

### 4. Spatial-only inference

```python
from scripts.spatial_only_predictor import SpatialOnlyPredictor

predictor = SpatialOnlyPredictor("/path/to/spatial-ckpts", device="cpu")
out = predictor.predict_round_full(sample)   # per-tick per-player probabilities
```

The model directory is scanned for `.pt` files carrying a `task` field
(`winrate` / `alive_end` / `future_kill`); one model is loaded per task.

## Training

Both training scripts are config-driven (CLI overrides yaml):

```bash
# Pre-training (path prediction) — A100 80GB config, 600k steps
python scripts/pretrain.py --config config/pretrain-a100-pro.yaml

# spatial-only downstream fine-tuning (one task per run)
python scripts/downstream-spatial-only/finetune_spatial_only.py \
    --config config/finetune-spatial-only-a100.yaml \
    --checkpoint /path/to/cs-net-v4-pro.pt \
    --task winrate
```

Detailed docs:
- [docs/pretrain.md](docs/pretrain.md) — pre-training data pipeline & training
- [docs/training-data-format.md](docs/training-data-format.md) — round-level WDS format
- [docs/torch-dataset.md](docs/torch-dataset.md) — window dataset / collate internals
- [docs/demo-json-format.md](docs/demo-json-format.md) — the round JSON format (visualizer input)
- [scripts/downstream-spatial-only/README.md](scripts/downstream-spatial-only/README.md) — spatial-only fine-tuning guide

Data pipeline entry points:
- `demo_parser/` — parse `.dem` files to round JSON (`python -m demo_parser` or `scripts/demo_to_json.py`)
- `scripts/create_training_data.py` — round JSON → round-level WebDataset shards
- `scripts/create_pretrain_data.py` — round shards → fixed-length window shards

## Repository layout

```
config/                          # training configs (Pro architecture)
demo_parser/                     # .dem → round JSON
maps/optimized_obj_files/        # optimized OBJ map geometry (visualizer + depth maps)
replay_tool/filter.py            # JSON post-processing (shared by visualizer)
scripts/
  pretrain_model.py              # CS2PretrainModel / PretrainConfig
  prediction_engine.py           # path prediction inference engine
  pretrain.py                    # pre-training entry
  create_pretrain_data.py        # window shard creation
  test_pretrain.py               # single-sample teacher-forcing / AR evaluation
  evaluate_pretrain.py           # multi-sample evaluation
  spatial_only_predictor.py      # spatial-only inference (shared by visualizer)
  create_training_data.py        # round shard creation
  evaluate_demos.py              # demo-level evaluation (used by visualizer scanning)
  training_data/                 # config / depth maps / features / labels / WDS IO / datasets
  downstream-spatial-only/       # spatial-only fine-tuning
visualizer/                      # Flask 3D replay + prediction web app
docs/                            # detailed Chinese documentation
```

## Notes

- All input windows are 16 ticks = 4 seconds at 0.25 s/tick; demo parsing is
  fixed at `interval=0.25` to match training.
- The data pipeline uses world-aligned (v5) coordinates; the prediction engine
  transparently converts legacy v4 labels.
- Map OBJ files under `maps/optimized_obj_files/` are pre-optimized for both the
  visualizer (Three.js) and raycast depth computation (Open3D).

## License

[MIT](LICENSE)
