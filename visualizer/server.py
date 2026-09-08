#!/usr/bin/env python3
"""
cs-net — 面向客户的可视化工具后端。

功能:
  1. 上传 demo / json / json.gz 文件 → 3D 回放（demo 以 0.25s 采样转换，与预训练一致）
  2. 平滑插值播放（前端完成，0.25s 采样 + 插值）
  3. JSON 打包下载（无论上传 demo 还是 json，都返回转换/整理后的 json.gz）
  4. 预训练模型可视化：上传 checkpoint → 路径预测（device 支持 cpu/mps/cuda）
  5. spatial-only 单局面预测：加载 winrate / alive_end / future_kill 单局面
     模型（embedder+spatial+head），对单个 tick 的完整局面直接推理，输出
     每玩家概率 + 聚合胜率。前端切回合自动对整回合全部 tick 预测（服务端
     缓存，来回切换不重算），拖动时间线实时查看每玩家数值（无路径/decoder）
  6. 回合低概率移动扫描：/api/scan/round 扫描一个回合所有存活条件的未来路径
     teacher-forcing log p，返回分数最低的若干 (tick, player) 条件（分数越低 =
     越不寻常的移动）；点击条目 → 跳到该 tick 并运行预测（/api/predict 附带
     模型对自身预测路径的自评分 pred_logp）。扫描结果按 (文件, checkpoint, 回合)
     缓存，切回合不重复计算；/api/cache/clear 释放全部缓存（demo 解析 / 扫描）。

用法:
    conda activate cs2demo
    python visualizer/server.py --host 127.0.0.1 --port 5000

    或（命令行直接指定 checkpoint / spatial-only 模型目录，免上传）:
    python visualizer/server.py --checkpoint checkpoints/cs-net-v4-pro.pt --device mps \
        --spatial-model-dir checkpoints
    （先运行 scripts/download_checkpoints.py 下载模型，文件落在 checkpoints/ 目录）
"""

from __future__ import annotations

import gzip
import io
import json
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path
from typing import Optional

from flask import Flask, jsonify, render_template, request, send_file

import torch  # 模型推理 / checkpoint 读取（server 依赖 cs2demo 环境）
import numpy as np  # 回合扫描向量化（log p 聚合 / 位移）

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_PROJECT_ROOT), str(_PROJECT_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from demo_parser import parse_demo
from replay_tool.filter import filter_data
from training_data.config import N_PLAYERS
from scripts.spatial_only_predictor import SpatialOnlyPredictor

MAPS_DIR = _PROJECT_ROOT / "maps" / "optimized_obj_files"
CONFIG_PATH = _PROJECT_ROOT / "config" / "pretrain-a100-pro.yaml"
THREE_DIR = Path(__file__).resolve().parent / "static" / "three"


def _detect_device(prefer: Optional[str] = None) -> str:
    """推理设备：显式指定则用指定值（不支持时警告并回退 cpu），
    否则自动探测 mps > cuda > cpu。

    Windows 无 GPU / 无 MPS 的机器自动落到 cpu，Mac 自动用 mps。
    显式指定了当前环境不支持的设备（如 Windows 上 --device mps、
    无 CUDA 的 torch 上 --device cuda）时打印警告并回退 cpu，
    避免 .to('mps') / .to('cuda') 抛 RuntimeError。
    """
    if prefer == "mps":
        try:
            if getattr(torch.backends, "mps", None) is not None \
                    and torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        print("[Visualizer] ⚠ 指定了 mps 但当前 PyTorch 未链接 MPS 支持，"
              "已回退到 cpu（可用 --device cpu 显式指定）")
        return "cpu"
    if prefer == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        print("[Visualizer] ⚠ 指定了 cuda 但当前环境没有可用 CUDA，"
              "已回退到 cpu（可用 --device cpu 显式指定）")
        return "cpu"
    if prefer:
        return prefer
    try:
        if getattr(torch.backends, "mps", None) is not None \
                and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

# ── Demo 解析缓存（demo 解析很慢，30 分钟过期）──────────────────────────
_demo_cache: dict[str, tuple[float, dict]] = {}
CACHE_TTL = 1800

# 回合低概率移动扫描缓存：key = f"{source}|{ckpt}|{device}|{round_idx}"
# → {"round_idx", "map_name", "items": [...]}（模型/文件变化自然失效）
_scan_cache: dict[str, dict] = {}

# spatial-only 整回合自动预测缓存：key = f"{source}|{dir}|{device}|{round_idx}"
# → /api/predict/spatial/round 的完整 payload（切回合来回不重算）
_spatial_round_cache: dict[str, dict] = {}

# 最近加载的数据（用于 json.gz 下载 / 预测）
_last_data: Optional[dict] = None
_last_source: str = ""
_last_interval: float = 0.25

# 预测引擎状态
_prediction_engine = None
_prediction_device = "cpu"
_prediction_checkpoint_path: Optional[str] = None
_prediction_step: Optional[int] = None

# spatial-only 单局面预测器状态
_spatial_predictor: Optional[SpatialOnlyPredictor] = None
_spatial_model_dir: Optional[str] = None
_spatial_device = "cpu"


def _cleanup_orphan_ckpts() -> None:
    """清理 outputs/ 下未被引用的临时 checkpoint（垃圾回收）。

    上传的 checkpoint 以 NamedTemporaryFile(delete=False) 落在 outputs/tmp*.pt，
    只清理 tmp 前缀且不在当前引用中的文件：
      - 当前预训练 checkpoint（_prediction_checkpoint_path）
    其余（崩溃残留 / 删除失败 / 测试遗留）一律删除。
    每次启动服务与每次上传模型后调用。
    """
    referenced = set()
    if _prediction_checkpoint_path:
        referenced.add(str(Path(_prediction_checkpoint_path).resolve()))
    out_dir = _PROJECT_ROOT / "outputs"
    if not out_dir.is_dir():
        return
    removed = 0
    for f in sorted(out_dir.glob("tmp*.pt")):
        if str(f.resolve()) in referenced:
            continue
        try:
            f.unlink()
            removed += 1
        except OSError:
            pass
    if removed:
        print(f"[Visualizer] 已清理 {removed} 个临时 checkpoint（outputs/）")


def _read_ckpt_step(path: str) -> Optional[int]:
    """从 checkpoint 读取 global_step（尽力而为）。"""
    try:
        import torch
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        return ckpt.get("global_step")
    except Exception:
        return None


def create_app(checkpoint: Optional[str] = None, device: Optional[str] = None,
               spatial_model_dir: Optional[str] = None) -> Flask:
    """创建 Flask app。

    device: 统一推理设备（路径预测 + spatial-only 共用）。None = 自动探测
    （mps > cuda > cpu），例如 Windows 无 GPU 时自动落到 cpu，Mac 自动用 mps。
    注：torch < 2.13 的 MPS 后端存在内存损坏问题（间歇性 NaN + 有限但
    错误的值，非数据问题；已被 2.13 修复，实测 5/5 独立实例逐位干净）。
    为兼容旧 torch，_run_tasks_safe 仍保留 NaN 自动 CPU 兜底（有限错误
    值无法检测，但 2.13+ 上不会出现）。
    """
    global _prediction_device, _prediction_checkpoint_path, _prediction_step
    global _spatial_model_dir, _spatial_device

    _prediction_device = _detect_device(device)
    _prediction_checkpoint_path = checkpoint
    _prediction_step = _read_ckpt_step(checkpoint) if checkpoint else None
    _spatial_device = _prediction_device  # 统一设备：spatial-only 与路径预测共用
    _spatial_model_dir = spatial_model_dir

    # 确保 outputs/ 存在（页面内上传 checkpoint 的临时目录；
    # 不存在时 tempfile.NamedTemporaryFile(dir=outputs) 会 FileNotFoundError）
    (_PROJECT_ROOT / "outputs").mkdir(parents=True, exist_ok=True)

    # 启动时校验路径参数：路径不存在立即给出明确提示，而不是等浏览器
    # 轮询 /api/model/status 时才 500（避免误以为服务/模型坏了）
    if checkpoint and not Path(checkpoint).exists():
        print(f"[Visualizer] ⚠ checkpoint 路径不存在: {checkpoint}")
        print(f"   已忽略该参数（可在页面内上传）。若已运行 scripts/download_checkpoints.py，"
              f"模型在 {_PROJECT_ROOT / 'checkpoints'} 目录。")
        _prediction_checkpoint_path = None
        _prediction_step = None
    if spatial_model_dir and not Path(spatial_model_dir).is_dir():
        print(f"[Visualizer] ⚠ spatial-only 模型目录不存在: {spatial_model_dir}")
        print(f"   已忽略该参数（可在页面内上传）。若已运行 scripts/download_checkpoints.py，"
              f"spatial ckpt 在 {_PROJECT_ROOT / 'checkpoints'} 目录（--spatial-model-dir 指向该目录即可）。")
        _spatial_model_dir = None

    def _get_spatial_predictor():
        """懒加载 spatial-only 预测器。"""
        global _spatial_predictor
        if _spatial_predictor is None:
            if not _spatial_model_dir:
                return None
            _spatial_predictor = SpatialOnlyPredictor(
                _spatial_model_dir, device=_spatial_device)
        return _spatial_predictor

    if _spatial_model_dir:
        try:
            _get_spatial_predictor()
        except Exception as exc:
            print(f"[Visualizer] ⚠ spatial-only 模型加载失败: {exc}")
    _cleanup_orphan_ckpts()   # 启动时清理残留的临时 checkpoint

    app = Flask(
        __name__,
        template_folder=str(Path(__file__).resolve().parent / "templates"),
        static_folder=str(Path(__file__).resolve().parent / "static"),
    )
    app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024  # 1 GB

    # ── 页面 ──────────────────────────────────────────────────────────

    @app.route("/")
    def index():
        return render_template("index.html",
                               default_device=_prediction_device)

    @app.route("/three/<path:filename>")
    def serve_three(filename):
        from flask import send_from_directory
        return send_from_directory(str(THREE_DIR), filename)

    # ── 地图 API ──────────────────────────────────────────────────────

    @app.route("/api/maps")
    def list_maps():
        maps = []
        if MAPS_DIR.is_dir():
            for f in sorted(MAPS_DIR.glob("*.obj")):
                stat = f.stat()
                maps.append({
                    "name": f.stem,
                    "file": f.name,
                    "size_mb": round(stat.st_size / (1024 * 1024), 1),
                })
        return jsonify({"maps": maps})

    @app.route("/api/map/<name>")
    def serve_map(name: str):
        from flask import send_from_directory
        safe_name = Path(name).name
        map_path = MAPS_DIR / f"{safe_name}.obj"
        if not map_path.exists():
            map_path = MAPS_DIR / safe_name
        if not map_path.exists():
            return jsonify({"error": f"Map not found: {name}"}), 404
        return send_file(map_path, mimetype="text/plain")

    # ── 上传 / 加载 API ───────────────────────────────────────────────

    @app.route("/api/load", methods=["POST"])
    def load_data():
        """
        加载回放数据。

        Accepts:
          - Multipart file upload（.dem / .json / .gz / .json.gz / .tar.gz）
          - JSON body {"path": "/abs/path"}
        """
        global _last_data, _last_source, _last_interval

        if "file" in request.files:
            uploaded = request.files["file"]
            if not uploaded.filename:
                return jsonify({"error": "Empty filename"}), 400
            suffix = Path(uploaded.filename).suffix.lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                uploaded.save(tmp.name)
                tmp_path = tmp.name
            try:
                data, interval = _parse_upload(tmp_path, uploaded.filename)
                _validate_v2_format(data)
                filter_data(data)
                _last_data = data
                _last_source = uploaded.filename
                _last_interval = interval
                return jsonify({"status": "ok", "data": data,
                                "interval": interval, "source": uploaded.filename})
            except Exception as exc:
                traceback.print_exc()
                return jsonify({"error": str(exc)}), 500
            finally:
                Path(tmp_path).unlink(missing_ok=True)

        body = request.get_json(silent=True)
        if body and body.get("path"):
            file_path = Path(body["path"])
            if not file_path.exists():
                return jsonify({"error": f"File not found: {body['path']}"}), 404
            try:
                data, interval = _parse_upload(str(file_path), file_path.name)
                _validate_v2_format(data)
                filter_data(data)
                _last_data = data
                _last_source = file_path.name
                _last_interval = interval
                return jsonify({"status": "ok", "data": data,
                                "interval": interval, "source": file_path.name})
            except Exception as exc:
                traceback.print_exc()
                return jsonify({"error": str(exc)}), 500

        return jsonify({"error": "Provide 'file' upload or 'path' in body"}), 400

    # ── JSON 打包下载 ─────────────────────────────────────────────────

    @app.route("/api/download")
    def download_json():
        """
        下载最近加载数据的 JSON 压缩包（.json.gz）。

        - 上传 demo → 下载转换后的 0.25s 采样 json.gz
        - 上传 json / json.gz → 下载整理后的 json.gz
        """
        global _last_data, _last_source
        if _last_data is None:
            return jsonify({"error": "No data loaded yet"}), 404

        json_bytes = json.dumps(
            _last_data, ensure_ascii=False, indent=2, default=str
        ).encode("utf-8")
        compressed = gzip.compress(json_bytes, compresslevel=6)

        base = Path(_last_source).stem
        filename = f"{base}.json.gz"
        return send_file(
            io.BytesIO(compressed),
            mimetype="application/gzip",
            as_attachment=True,
            download_name=filename,
        )

    # ── 预训练模型 API ────────────────────────────────────────────────

    def _get_prediction_engine():
        """懒加载预测引擎。"""
        global _prediction_engine
        if _prediction_engine is None:
            if not _prediction_checkpoint_path or not CONFIG_PATH.exists():
                return None
            from scripts.prediction_engine import PredictionEngine
            print(f"[Visualizer] Loading PredictionEngine device={_prediction_device} "
                  f"ckpt={_prediction_checkpoint_path}")
            _prediction_engine = PredictionEngine(
                config_path=str(CONFIG_PATH),
                checkpoint_path=_prediction_checkpoint_path,
                device=_prediction_device,
                maps_dir=str(MAPS_DIR),
            )
        return _prediction_engine

    def _build_predict_inputs(round_idx: int, tick: int):
        """从最近加载的数据构建预训练 sample（/api/predict 与 player-sampled 共用）。

        Returns:
            (ts, query_tick, map_name) — torch sample + 规范化后的 query tick
        Raises:
            ValueError: 数据未加载 / round 越界
        """
        global _last_data
        if _last_data is None:
            raise ValueError("No replay data loaded. 请先上传 demo / json。")

        rounds = _last_data.get("rounds", [])
        if round_idx < 0 or round_idx >= len(rounds):
            raise ValueError(f"round_idx {round_idx} out of range")
        round_data = rounds[round_idx]

        # 注入 map_name（V2 中 map 在顶层）
        map_name = _last_data.get("map", _last_data.get("map_name", "unknown"))
        if "map_name" not in round_data:
            round_data["map_name"] = map_name
        if "map" not in round_data:
            round_data["map"] = map_name

        # 从 round 数据构建预训练 sample（与 create_training_data.py 一致，v5 世界对齐）
        from training_data.round_processor import process_round
        from training_data.torch_dataset import sample_to_torch
        from training_data.map_loader import get_map_geometry
        from create_training_data import _convert_inventory_indices

        # 与训练管线对齐：将 demo 动态 weapon 索引重映射为 config.py 的规范索引
        # （训练侧 create_training_data.py 也调用同一函数；否则预测输入与训练分布错位）
        import copy
        round_data = copy.deepcopy(round_data)
        _convert_inventory_indices({"weapons": _last_data.get("weapons", {}),
                                    "rounds": [round_data]})

        map_geom = None
        try:
            map_geom = get_map_geometry(map_name, MAPS_DIR)
        except Exception:
            map_geom = None

        sample = process_round(
            round_data,
            map_geom=map_geom,
            source_file=_last_source,
            match_teams=None,
            players_meta=_last_data.get("players"),
            tick_interval=_last_interval or 0.25,
            compute_depth=map_geom is not None,
            places=_last_data.get("places"),
        )
        ts = sample_to_torch(sample)

        round_T = sample["meta"]["T"]
        if tick < 0:
            tick = max(0, round_T // 2)
        query_tick = min(tick, max(0, round_T - 1))
        return ts, query_tick, map_name

    @app.route("/api/model/status")
    def model_status():
        """预测引擎 + spatial-only 状态。"""
        engine = _get_prediction_engine()
        params_m = 0
        if engine is not None:
            params_m = round(sum(p.numel() for p in engine.model.parameters()) / 1e6, 1)
        spatial = _get_spatial_predictor()
        return jsonify({
            "available": engine is not None,
            "checkpoint": Path(_prediction_checkpoint_path).name if _prediction_checkpoint_path else None,
            "device": _prediction_device,
            "params_m": params_m,
            "step": _prediction_step,
            "spatial": {
                "available": spatial is not None,
                "model_dir": _spatial_model_dir,
                "device": _spatial_device,
                "tasks": spatial.tasks if spatial else None,
            },
        })

    @app.route("/api/model/upload", methods=["POST"])
    def model_upload():
        """上传预训练 checkpoint（.pt）。"""
        global _prediction_engine, _prediction_checkpoint_path, _prediction_step, _prediction_device
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        uploaded = request.files["file"]
        if not uploaded.filename:
            return jsonify({"error": "Empty filename"}), 400

        # 解析可选 device 参数
        device = request.form.get("device", _prediction_device or "mps")
        if device not in ("cpu", "mps", "cuda"):
            return jsonify({"error": f"Unsupported device: {device}"}), 400

        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".pt", dir=str(_PROJECT_ROOT / "outputs")
        ) as tmp:
            uploaded.save(tmp.name)
            ckpt_path = tmp.name

        _prediction_checkpoint_path = ckpt_path
        _prediction_device = device
        _prediction_step = _read_ckpt_step(ckpt_path)
        _prediction_engine = None  # 强制重新加载
        _cleanup_orphan_ckpts()    # 新文件已在引用中，旧上传临时文件一并清理

        try:
            engine = _get_prediction_engine()
            if engine is None:
                return jsonify({"error": "Failed to init prediction engine"}), 500
            return jsonify({
                "status": "ok",
                "device": device,
                "checkpoint": uploaded.filename,
                "step": _prediction_step,
                "params_m": round(sum(p.numel() for p in engine.model.parameters()) / 1e6, 1),
            })
        except Exception as exc:
            traceback.print_exc()
            return jsonify({"error": f"Failed to load model: {exc}"}), 500

    @app.route("/api/predict", methods=["POST"])
    def predict():
        """
        对最近加载数据的指定 round/tick 运行路径预测。

        POST body (JSON):
            round_idx: int  — round 索引（默认 0）
            tick:      int  — query tick（默认窗口中间）
            temperature: float — 采样温度（默认 0 = argmax）
            teacher_forcing_ticks: int — 前 N tick 用 GT（默认 0）
            return_logp: bool — 附带模型对预测路径的自评分 pred_logp
                              （每 trajectory：per_tick/total/tokcount/ticks）

        Returns:
            {query_tick, input_T, output_T, map_name, trajectories: [...]}
        """
        engine = _get_prediction_engine()
        if engine is None:
            return jsonify({
                "error": "Prediction engine not loaded. 请先上传预训练 checkpoint (.pt)。"
            }), 400

        body = request.get_json(silent=True) or {}
        round_idx = int(body.get("round_idx", 0))
        tick = int(body.get("tick", -1))
        temperature = float(body.get("temperature", 0.0))
        tf_ticks = int(body.get("teacher_forcing_ticks", 0))
        return_logp = bool(body.get("return_logp", False))

        try:
            ts, query_tick, _ = _build_predict_inputs(round_idx, tick)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        try:
            result = engine.predict_at_tick(
                ts, query_tick,
                temperature=temperature,
                teacher_forcing_ticks=tf_ticks,
                return_logp=return_logp,
            )
        except Exception as exc:
            traceback.print_exc()
            return jsonify({"error": f"Prediction failed: {exc}"}), 500

        return jsonify(result)

    @app.route("/api/predict/player-sampled", methods=["POST"])
    def predict_player_sampled():
        """
        对单个玩家并行采样多条路径（decoder batch = num_samples，一次前向）。

        POST body (JSON):
            round_idx:   int  — round 索引（默认 0）
            tick:        int  — query tick（默认窗口中间）
            player_idx:  int  — 目标玩家索引 [0, 10)（默认 0）
            num_samples: int  — 采样条数（默认 8，范围 1-32）
            temperature: float — softmax 温度（默认 1.0；0=argmax 时所有采样相同）
            top_k:       int  — >0 时 top-k 采样（默认 0 = 关闭）
            top_p:       float — (0,1] nucleus 采样（默认 0.9）

        Returns:
            {query_tick, player_idx, num_samples, samples: [...], gt, ...}
        """
        engine = _get_prediction_engine()
        if engine is None:
            return jsonify({
                "error": "Prediction engine not loaded. 请先上传预训练 checkpoint (.pt)。"
            }), 400

        body = request.get_json(silent=True) or {}
        round_idx = int(body.get("round_idx", 0))
        tick = int(body.get("tick", -1))
        player_idx = int(body.get("player_idx", 0))
        num_samples = int(body.get("num_samples", 8))
        temperature = float(body.get("temperature", 1.0))
        top_k = int(body.get("top_k", 0))
        top_p = float(body.get("top_p", 0.9))

        if num_samples < 1 or num_samples > 32:
            return jsonify({"error": "num_samples must be in [1, 32]"}), 400

        try:
            ts, query_tick, _ = _build_predict_inputs(round_idx, tick)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        try:
            result = engine.predict_at_tick_player_sampled(
                ts, query_tick,
                player_idx=player_idx,
                num_samples=num_samples,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
        except Exception as exc:
            traceback.print_exc()
            return jsonify({"error": f"Prediction failed: {exc}"}), 500

        return jsonify(result)

    # ── spatial-only 模型上传 API ────────────────────────────────────────

    @app.route("/api/spatial/upload", methods=["POST"])
    def spatial_upload():
        """
        上传一个或多个 spatial-only 任务 checkpoint（.pt，多文件）。

        任务由 ckpt 内部 task 字段识别（winrate / alive_end / future_kill）。
        文件保存到 outputs/spatial_ckpts/ 目录（重复上传覆盖重建预测器）。

        POST multipart:
            file[]  — 一个或多个 spatial-only checkpoint
            device  — 推理设备（默认沿用当前）
        """
        global _spatial_model_dir, _spatial_predictor, _spatial_device
        files = request.files.getlist("file")
        if not files:
            return jsonify({"error": "No file uploaded"}), 400
        device = request.form.get("device", _spatial_device or "cpu")
        if device not in ("cpu", "mps", "cuda"):
            return jsonify({"error": f"Unsupported device: {device}"}), 400
        _spatial_device = device

        ckpt_dir = _PROJECT_ROOT / "outputs" / "spatial_ckpts"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        # 单次上传 = 整套模型；清掉旧上传避免混合
        for old in ckpt_dir.glob("*.pt"):
            old.unlink(missing_ok=True)

        results, ok_count = [], 0
        for uploaded in files:
            if not uploaded.filename:
                results.append({"name": "(unnamed)", "error": "Empty filename"})
                continue
            tmp_path = ckpt_dir / f"{uploaded.filename}"
            uploaded.save(str(tmp_path))
            try:
                ck = torch.load(tmp_path, map_location="cpu", weights_only=False)
            except Exception as exc:
                tmp_path.unlink(missing_ok=True)
                results.append({"name": uploaded.filename, "error": f"无法读取 checkpoint: {exc}"})
                continue
            task = ck.get("task")
            if task not in ("winrate", "alive_end", "future_kill")                     or "model_state" not in ck or "head_state" not in ck:
                tmp_path.unlink(missing_ok=True)
                results.append({
                    "name": uploaded.filename,
                    "error": "不是 spatial-only 下游任务 checkpoint（需要 task / model_state / head_state 字段）",
                })
                continue
            results.append({"name": uploaded.filename, "task": task,
                            "step": ck.get("global_step")})
            ok_count += 1
            del ck

        if ok_count == 0:
            return jsonify({"status": "error",
                            "error": results[0].get("error", "全部上传失败"),
                            "results": results}), 400

        _spatial_predictor = None
        _spatial_model_dir = str(ckpt_dir)
        _spatial_round_cache.clear()   # 模型变了 → 旧的整回合预测不再适用
        try:
            predictor = _get_spatial_predictor()
        except Exception as exc:
            traceback.print_exc()
            return jsonify({"status": "error",
                            "error": f"模型加载失败: {exc}", "results": results}), 500
        print(f"[Visualizer] spatial-only 模型已加载：{predictor.tasks}"
              f"（device={_spatial_device}）")
        return jsonify({"status": "ok", "results": results,
                        "failed": len(results) - ok_count,
                        "tasks": predictor.tasks,
                        "device": _spatial_device})

    # ── spatial-only 单局面预测 API（无路径/decoder）─────────────────────

    def _build_spatial_inputs(round_idx: int, tick: int):
        """从最近加载的数据构建 spatial-only 推理 sample（numpy dict，含 teams）。

        Returns:
            (sample, query_tick, map_name) — numpy sample + 规范化后的 query tick
        """
        global _last_data
        if _last_data is None:
            raise ValueError("No replay data loaded. 请先上传 demo / json。")
        rounds = _last_data.get("rounds", [])
        if round_idx < 0 or round_idx >= len(rounds):
            raise ValueError(f"round_idx {round_idx} out of range")
        round_data = rounds[round_idx]

        map_name = _last_data.get("map", _last_data.get("map_name", "unknown"))
        if "map_name" not in round_data:
            round_data["map_name"] = map_name
        if "map" not in round_data:
            round_data["map"] = map_name

        from training_data.round_processor import process_round
        from training_data.map_loader import get_map_geometry
        from create_training_data import _convert_inventory_indices

        import copy
        round_data = copy.deepcopy(round_data)
        _convert_inventory_indices({"weapons": _last_data.get("weapons", {}),
                                    "rounds": [round_data]})

        map_geom = None
        try:
            map_geom = get_map_geometry(map_name, MAPS_DIR)
        except Exception:
            map_geom = None

        sample = process_round(
            round_data,
            map_geom=map_geom,
            source_file=_last_source,
            match_teams=None,
            players_meta=_last_data.get("players"),
            tick_interval=_last_interval or 0.25,
            compute_depth=map_geom is not None,
            places=_last_data.get("places"),
        )
        # 确保 meta.teams 存在（spatial-only 聚合胜率需要 CT/T 划分）
        meta = sample.get("meta", {})
        if not meta.get("teams"):
            players_meta = _last_data.get("players") or []
            teams = []
            for i in range(N_PLAYERS):
                pm = players_meta[i] if i < len(players_meta) else None
                teams.append(pm.get("team") if isinstance(pm, dict) else None)
            meta["teams"] = teams or ["?"] * N_PLAYERS

        round_T = sample["meta"]["T"] if "T" in sample["meta"] else sample["player_pos"].shape[0]
        if tick < 0:
            tick = max(0, round_T // 2)
        query_tick = min(tick, max(0, round_T - 1))
        return sample, query_tick, map_name

    @app.route("/api/predict/spatial", methods=["POST"])
    def predict_spatial():
        """
        对指定 round/tick 运行 spatial-only 单局面预测（每玩家概率）。

        POST body (JSON):
            round_idx: int  — round 索引（默认 0）
            tick:      int  — query tick（默认回合中间）

        Returns:
            {tick, T, winrate/alive_end/future_kill: [10] 每玩家概率,
             winrate_agg: {ct, t, ct_winrate}, winrate_team: [10],
             player_teams, alive_mask, winner}
        """
        predictor = _get_spatial_predictor()
        if predictor is None:
            return jsonify({
                "error": "spatial-only 模型未加载。请上传任务 checkpoint 或启动时用 --spatial-model-dir 指定目录。"
            }), 400
        body = request.get_json(silent=True) or {}
        round_idx = int(body.get("round_idx", 0))
        tick = int(body.get("tick", -1))
        try:
            sample, query_tick, _ = _build_spatial_inputs(round_idx, tick)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        try:
            return jsonify(predictor.predict_tick(sample, query_tick))
        except Exception as exc:
            traceback.print_exc()
            return jsonify({"error": f"Prediction failed: {exc}"}), 500

    @app.route("/api/predict/spatial/curve", methods=["POST"])
    def predict_spatial_curve():
        """
        整回合 winrate 聚合曲线：每个 tick 一个聚合 CT 胜率（批量推理）。

        POST body (JSON):
            round_idx: int — round 索引（默认 0）

        Returns:
            {curve: [{ct, t, ct_winrate}], T, winner, player_teams}
        """
        predictor = _get_spatial_predictor()
        if predictor is None:
            return jsonify({"error": "spatial-only 模型未加载"}), 400
        if "winrate" not in predictor.tasks:
            return jsonify({"error": "未加载 winrate 任务，无法计算胜率曲线"}), 400
        body = request.get_json(silent=True) or {}
        round_idx = int(body.get("round_idx", 0))
        try:
            sample, _, _ = _build_spatial_inputs(round_idx, -1)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        try:
            return jsonify(predictor.predict_round_curve(sample))
        except Exception as exc:
            traceback.print_exc()
            return jsonify({"error": f"Prediction failed: {exc}"}), 500

    @app.route("/api/predict/spatial/round", methods=["POST"])
    def predict_spatial_round():
        """
        整回合全任务逐 tick 自动预测（前端切回合时触发，服务端缓存）。

        预测很快（单 tick 独立、无路径/decoder），一次请求算完整回合；
        结果按 (文件, 模型目录, device, 回合) 缓存，来回切换回合不重算。
        /api/cache/clear 或重新上传 spatial 模型时清空。

        POST body (JSON):
            round_idx: int — round 索引（默认 0）

        Returns:
            {cached, round_idx, T, tasks, ticks: [...]} — ticks 内每个元素:
              {tick, winrate/alive_end/future_kill: [10] 每玩家概率,
               winrate_agg: {ct, t, ct_winrate}, winrate_team: [10],
               alive_mask: [10]}
            另附 curve（winrate 任务已加载时）= 逐 tick 聚合 {ct, t, ct_winrate}，
            与 winner / player_teams，前端据此画聚合胜率曲线与玩家卡片数值。
        """
        predictor = _get_spatial_predictor()
        if predictor is None:
            return jsonify({
                "error": "spatial-only 模型未加载。请上传任务 checkpoint 或启动时用 --spatial-model-dir 指定目录。"
            }), 400
        if _last_data is None:
            return jsonify({"error": "No replay data loaded. 请先上传 demo / json。"}), 400
        body = request.get_json(silent=True) or {}
        round_idx = int(body.get("round_idx", 0))

        key = f"{_last_source}|{_spatial_model_dir}|{_spatial_device}|{round_idx}"
        cached = _spatial_round_cache.get(key)
        if cached is not None:
            return jsonify({"cached": True, **cached})

        try:
            sample, _, _ = _build_spatial_inputs(round_idx, -1)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        try:
            result = predictor.predict_round_full(sample)
        except Exception as exc:
            traceback.print_exc()
            return jsonify({"error": f"Prediction failed: {exc}"}), 500

        payload = {"round_idx": round_idx, **result}
        if "winrate" in predictor.tasks:
            payload["curve"] = [t["winrate_agg"] for t in payload["ticks"]]
        _spatial_round_cache[key] = payload
        # 简单容量上限：超过 40 个回合清最旧（dict 保持插入序）
        if len(_spatial_round_cache) > 40:
            for _k in list(_spatial_round_cache)[: len(_spatial_round_cache) - 40]:
                del _spatial_round_cache[_k]
        print(f"[Visualizer] 回合 {round_idx} spatial-only 整回合预测完成"
              f"（{payload['T']} ticks · 任务 {payload['tasks']} · "
              f"缓存 {len(_spatial_round_cache)} 回合）")
        return jsonify({"cached": False, **payload})

    # ── 回合低概率移动扫描 / 缓存管理 ─────────────────────────────────

    _SCAN_MIN_TICK_GAP = 8   # 列表内相邻条目的起点间隔下限：前后 2 秒（8 tick）内去重
    _SCAN_MIN_ALIVE_TICKS = 8   # 起点后至少存活 2 秒（8 tick）才参与扫描：
                                # 过滤"阵亡前 <2s"的短路径——那种低分多是被击杀导致，
                                # 不是玩家走位本身的问题（tokcount 太短，分数参考性弱）

    def _scan_round_cache_key(round_idx: int, player_idx: Optional[int],
                              top_n: int = 10) -> str:
        """扫描缓存键：文件 + checkpoint + device + 回合 + 玩家 + 选择算法版本 + top_n
        （任一变 → 自动失效；sel-v3 = 分数升序 + 起点间隔去重 + 阵亡前 2s 短路径过滤）。"""
        return (f"{_last_source}|{_prediction_checkpoint_path}|"
                f"{_prediction_device}|{round_idx}|{player_idx}|sel-v3|n{top_n}")

    @torch.no_grad()
    def _scan_round_logp(engine, ts: dict, map_name: str,
                         round_idx: int, top_n: int,
                         player_idx: Optional[int] = None) -> list:
        """扫描一个回合（可选指定玩家）所有存活条件的未来路径 teacher-forcing log p。

        player_idx=None → 全部玩家；指定 → 只扫描该玩家的条件（temporal /
        depth 编码 / decoder 都按单玩家裁剪，省 ~90% 算力）。

        与 collect_demos / evaluate_demos 的预训练部分完全同口径：
          - 条件 = 16-tick 因果窗口的 player embedding（_window_conditions）
          - 未来 ctx = 训练同款 depth/xyz/angle 逐 tick 编码（_round_ctx）
          - GT 相机 token 序列 teacher-forcing → 每 token log p
          - 分数 = per_tick（tick 等权：每个非 PAD tick 的 7 token 均值，
            再对有效 tick 取平均）——即"预训练走位评价"定稿口径；
            另附 total（有效 token 之和）与 disp（窗口总位移，游戏单位）。
        按 per_tick 升序（分数越低 = 模型越认为不可能）返回前 top_n 条；
        相邻条目的起点 tick 间隔 ≥ 8（前后 2 秒内去重：低概率时刻常扎堆出现，
        如死亡前几秒，去重后列表在时间上分散，覆盖更多不同局面）。
        只扫描起点后至少存活 _SCAN_MIN_ALIVE_TICKS tick（2 秒）的移动：
        阵亡前不足 2 秒的短路径（多为被击杀、而非走位问题）直接跳过。
        """
        from evaluate_demos import (  # noqa: E402
            _window_conditions,
            _round_ctx,
            _task_slices,
            _task_tick_displacement,
            N_PLAYERS,
        )
        from training_data.torch_dataset import augment_depth_with_angles  # noqa: E402
        # _build_predict_inputs 返回的 ts 未做深度角度增强（predict_at_tick 内部才做）；
        # embedder / _round_ctx 需要 [T,10,64,5]，这里与 evaluate_demos 调用方对齐
        if "player_depth" in ts and ts["player_depth"].ndim == 3:
            ts = augment_depth_with_angles(ts)
        n_ticks = engine.model_cfg.n_ticks
        device = engine.device
        T = ts["player_pos"].shape[0]
        t_max = T - n_ticks
        if t_max < 0:
            return []
        alive = ts["player_alive_mask"].numpy()          # [T, 10]
        # 每个玩家首个阵亡 tick（整场存活 = T）：过滤"阵亡前不足 2 秒"的移动
        death_tick = np.full(N_PLAYERS, T, dtype=np.int64)
        for p in range(N_PLAYERS):
            dead = np.flatnonzero(~alive[:, p])
            if dead.size:
                death_tick[p] = int(dead[0])
        if player_idx is not None:
            players = [player_idx]
            task_t = np.array(
                [t for t in range(t_max + 1)
                 if alive[t, player_idx]
                 and (death_tick[player_idx] - t) >= _SCAN_MIN_ALIVE_TICKS],
                dtype=np.int64)
            task_p_abs = np.full(task_t.shape, player_idx, dtype=np.int64)
            task_p_rel = np.zeros(task_t.shape, dtype=np.int64)   # 相对 ctx 索引
        else:
            players = None
            tasks = [(t, p) for t in range(t_max + 1) for p in range(N_PLAYERS)
                     if alive[t, p]
                     and (death_tick[p] - t) >= _SCAN_MIN_ALIVE_TICKS]
            task_t = np.array([t for t, _ in tasks], dtype=np.int64)
            task_p_abs = np.array([p for _, p in tasks], dtype=np.int64)
            task_p_rel = task_p_abs
        K = len(task_t)
        if K == 0:
            return []
        # ctx 索引用相对玩家（conds / _round_ctx 返回 [T, P, d]）
        tasks_ctx = [(int(t), int(p)) for t, p in zip(task_t, task_p_rel)]
        # label / 位移用绝对玩家
        tasks_abs = [(int(t), int(p)) for t, p in zip(task_t, task_p_abs)]

        conds = _window_conditions(engine.model, ts, range(t_max + 1), 16,
                                   device, seg_ticks=64, players=players)
        d_enc, x_enc, a_enc = _round_ctx(engine.model, ts, map_name, device,
                                         seg_ticks=64, players=players)
        cond_t = conds[task_t, task_p_rel]               # [K, d]

        labels = torch.stack(
            [ts["label_camera"][t:t + n_ticks, p] for t, p in tasks_abs]).to(device)
        tokens = engine.model.tokenizer.encode_sequence(labels, n_ticks)

        pad = engine.model.tokenizer.PAD
        tick_idx = np.arange(n_ticks)[:, None] * 10 + np.arange(3, 10)  # [16,7]
        per_tick_logp = np.full((K, n_ticks), np.nan)
        totals = np.zeros(K)
        tokcounts = np.zeros(K, dtype=np.int64)
        batch = 128
        for i0 in range(0, K, batch):
            idx = slice(i0, min(i0 + batch, K))
            k = idx.stop - idx.start
            logits, flat = engine.model.decoder(
                cond_t[idx], tokens[idx],
                depth_ctx=_task_slices(d_enc, tasks_ctx[i0:i0 + batch], n_ticks),
                xyz_ctx=_task_slices(x_enc, tasks_ctx[i0:i0 + batch], n_ticks),
                angle_ctx=_task_slices(a_enc, tasks_ctx[i0:i0 + batch], n_ticks),
            )
            logp = torch.log_softmax(logits, dim=-1)     # [k,160,vocab]
            mask = flat != pad                           # [k,160]
            gathered = logp.gather(-1, flat.clamp(min=0).unsqueeze(-1)).squeeze(-1)
            totals[i0:i0 + k] = (gathered * mask).sum(-1).cpu().numpy()
            tokcounts[i0:i0 + k] = mask.sum(-1).cpu().numpy()
            gm = (gathered * mask).cpu().numpy()
            cm = mask.cpu().numpy().astype(np.float32)
            with np.errstate(invalid="ignore", divide="ignore"):
                per_tick_logp[i0:i0 + k] = np.where(
                    cm[:, tick_idx].sum(-1) > 0,
                    gm[:, tick_idx].sum(-1) / np.maximum(cm[:, tick_idx].sum(-1), 1),
                    np.nan,
                )
        if device.type == "mps":
            torch.mps.empty_cache()

        # 分数 = tick 等权均值（每个非 PAD tick 的 7 token 均值再平均）
        scores = np.nanmean(per_tick_logp, axis=1)       # [K]
        disp = _task_tick_displacement(ts, task_t, task_p_abs, n_ticks, T)  # [K,16]
        disp_sum = disp.sum(axis=1)

        players_meta = _last_data.get("players", []) if _last_data else []
        teams = ts.get("meta", {}).get("teams") or ["?"] * N_PLAYERS
        order = np.argsort(scores, kind="stable")        # 升序：最不可能在前
        items = []
        selected_ticks: list = []                        # 已入选条目的起点 tick
        for i in order:
            if not np.isfinite(scores[i]):
                continue
            t = int(task_t[i])
            # 与已入选条目起点太近（前后 _SCAN_MIN_TICK_GAP 内）→ 跳过，
            # 分数更高的那个低概率时刻让位给更分散的其它局面
            if any(abs(t - s) <= _SCAN_MIN_TICK_GAP for s in selected_ticks):
                continue
            if len(items) >= top_n:
                break
            p = int(task_p_abs[i])
            name = (players_meta[p].get("name") if p < len(players_meta)
                    and isinstance(players_meta[p], dict) else None) or f"P{p}"
            items.append({
                "tick": int(task_t[i]),
                "player": p,
                "name": name,
                "team": teams[p] if p < len(teams) else None,
                "per_tick": float(scores[i]),
                "total": float(totals[i]),
                "tokcount": int(tokcounts[i]),
                "disp": float(disp_sum[i]),
            })
            selected_ticks.append(t)
        return items

    @app.route("/api/scan/round", methods=["POST"])
    def scan_round():
        """
        扫描一个回合的低概率移动（缓存命中时秒回）。

        POST body (JSON):
            round_idx:  int — 回合索引（默认 0）
            top_n:      int — 返回分数最低的条件数（默认 10，上限 50）
            player_idx: int|None — 只扫描该玩家（默认 None = 全部玩家；
                              选单个玩家可大幅节省计算时间）

        Returns:
            {cached, round_idx, player_idx, map_name, round_T, n_conds,
             min_tick_gap, min_alive_ticks, items: [...]}
            items 按 per_tick 升序（最不可能在前），且相邻条目起点 tick 间隔 ≥
            min_tick_gap（默认 8 = 2 秒，避免低概率时刻扎堆）；仅保留起点后仍
            存活 ≥ min_alive_ticks tick（默认 8 = 2 秒）的移动（过滤阵亡前短路径）：
              {tick, player, name, team, per_tick, total, tokcount, disp}
        """
        engine = _get_prediction_engine()
        if engine is None:
            return jsonify({
                "error": "Prediction engine not loaded. 请先上传预训练 checkpoint (.pt)。"
            }), 400
        if _last_data is None:
            return jsonify({"error": "No replay data loaded. 请先上传 demo / json。"}), 400

        body = request.get_json(silent=True) or {}
        round_idx = int(body.get("round_idx", 0))
        top_n = max(1, min(int(body.get("top_n", 10)), 50))
        player_idx = body.get("player_idx")
        if player_idx is not None:
            player_idx = int(player_idx)
            if not (0 <= player_idx < N_PLAYERS):
                return jsonify({"error": f"player_idx 越界: {player_idx}"}), 400

        key = _scan_round_cache_key(round_idx, player_idx, top_n)
        if key in _scan_cache:
            return jsonify({"cached": True, **_scan_cache[key]})

        try:
            ts, _, map_name = _build_predict_inputs(round_idx, -1)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        try:
            items = _scan_round_logp(engine, ts, map_name, round_idx, top_n,
                                     player_idx=player_idx)
        except Exception as exc:
            traceback.print_exc()
            return jsonify({"error": f"扫描失败: {exc}"}), 500

        payload = {
            "round_idx": round_idx,
            "player_idx": player_idx,
            "map_name": map_name,
            "round_T": int(ts["player_pos"].shape[0]),
            "n_conds": int(len(items)),   # 实际返回条数（≤ top_n）
            "min_tick_gap": _SCAN_MIN_TICK_GAP,
            "min_alive_ticks": _SCAN_MIN_ALIVE_TICKS,
            "items": items,
        }
        _scan_cache[key] = payload
        print(f"[Visualizer] 回合 {round_idx} 扫描完成"
              f"（{'全部玩家' if player_idx is None else f'玩家 {player_idx}'}）："
              f"{payload['n_conds']} 个低概率移动（缓存 {len(_scan_cache)} 回合）")
        return jsonify({"cached": False, **payload})

    @app.route("/api/scan/status")
    def scan_status():
        """查询某回合是否已扫描（前端切回合时显示缓存状态，不触发计算）。"""
        round_idx = int(request.args.get("round_idx", 0))
        player_idx = request.args.get("player_idx")
        if player_idx is not None and player_idx != "":
            player_idx = int(player_idx)
        else:
            player_idx = None
        top_n = max(1, min(int(request.args.get("top_n", 10)), 50))
        cached = _scan_cache.get(_scan_round_cache_key(round_idx, player_idx, top_n))
        return jsonify({
            "cached": cached is not None,
            "n_items": len(cached["items"]) if cached else 0,
        })

    # ── 全回合批量扫描（后台线程逐回合计算，每回合复用单回合缓存）──

    _scan_all_jobs: dict = {}   # player_idx -> {running, done, error, total, current, items}

    def _scan_all_worker(engine, player_idx: int, top_n: int):
        """后台线程：逐回合扫描该玩家低概率移动；结果增量累积到 job["items"]。
        每回合先查共享缓存（与单回合扫描同一 key），未命中才算并写回。"""
        job = _scan_all_jobs[player_idx]
        try:
            rounds = _last_data.get("rounds", []) if _last_data else []
            job["total"] = len(rounds)
            for r in range(len(rounds)):
                job["current"] = r
                key = _scan_round_cache_key(r, player_idx, top_n)
                cached = _scan_cache.get(key)
                if cached is not None:
                    items = cached.get("items", [])
                else:
                    ts, _, map_name = _build_predict_inputs(r, -1)
                    items = _scan_round_logp(engine, ts, map_name, r, top_n,
                                             player_idx=player_idx)
                    _scan_cache[key] = {
                        "round_idx": r,
                        "player_idx": player_idx,
                        "map_name": map_name,
                        "round_T": int(ts["player_pos"].shape[0]),
                        "n_conds": int(len(items)),
                        "min_tick_gap": _SCAN_MIN_TICK_GAP,
                        "min_alive_ticks": _SCAN_MIN_ALIVE_TICKS,
                        "items": items,
                    }
                for it in items:
                    it = dict(it)
                    it["round_idx"] = r
                    job["items"].append(it)
                if not job.get("running"):
                    break
            job["done"] = True
        except Exception as exc:
            traceback.print_exc()
            job["error"] = str(exc)
        finally:
            job["running"] = False

    @app.route("/api/scan/all", methods=["POST"])
    def scan_all():
        """
        后台批量扫描某玩家所有回合的低概率移动（每回合 top_n 条，
        与单回合扫描同一缓存：已算过的回合直接命中，秒过）。

        POST body (JSON):
            player_idx: int — 扫描哪个玩家（必选）
            top_n:      int — 每回合保留条数（默认 3，上限 50）

        Returns:
            {started, player_idx, total_rounds}
            进度用 GET /api/scan/all/status?player_idx=N 轮询。
        """
        engine = _get_prediction_engine()
        if engine is None:
            return jsonify({
                "error": "Prediction engine not loaded. 请先上传预训练 checkpoint (.pt)。"
            }), 400
        if _last_data is None:
            return jsonify({"error": "No replay data loaded. 请先上传 demo / json。"}), 400
        body = request.get_json(silent=True) or {}
        player_idx = int(body.get("player_idx", 0))
        if not (0 <= player_idx < N_PLAYERS):
            return jsonify({"error": f"player_idx 越界: {player_idx}"}), 400
        top_n = max(1, min(int(body.get("top_n", 3)), 50))
        rounds = _last_data.get("rounds", [])
        if not rounds:
            return jsonify({"error": "回放数据没有回合"}), 400
        if _scan_all_players_job.get("running"):
            return jsonify({"error": "全部玩家预热进行中，请等它结束再做单玩家全回合扫描。"}), 409
        job = _scan_all_jobs.get(player_idx)
        if job and job.get("running"):
            return jsonify({"started": False, "running": True, "player_idx": player_idx,
                            "total_rounds": len(rounds),
                            "current_round": job.get("current", 0)})
        _scan_all_jobs[player_idx] = {
            "running": True, "done": False, "error": None,
            "total": len(rounds), "current": 0, "items": [],
        }
        t = threading.Thread(target=_scan_all_worker,
                             args=(engine, player_idx, top_n), daemon=True)
        t.start()
        print(f"[Visualizer] 启动玩家 {player_idx} 全回合扫描"
              f"（{len(rounds)} 回合，top_n={top_n}）")
        return jsonify({"started": True, "player_idx": player_idx,
                        "total_rounds": len(rounds)})

    @app.route("/api/scan/all/status")
    def scan_all_status():
        """批量扫描进度（增量返回已完成回合的汇总 items，前端边算边显示）。"""
        player_idx = int(request.args.get("player_idx", 0))
        job = _scan_all_jobs.get(player_idx)
        if not job:
            return jsonify({"started": False, "running": False, "done": False,
                            "error": None, "total_rounds": 0, "current_round": 0,
                            "scan_all": True, "n_items": 0, "items": []})
        return jsonify({
            "started": True,
            "running": bool(job.get("running")),
            "done": bool(job.get("done")),
            "error": job.get("error"),
            "total_rounds": job.get("total", 0),
            "current_round": job.get("current", 0),
            "scan_all": True,
            "n_items": len(job["items"]),
            "items": job["items"],
        })

    # ── 全部玩家 × 全部回合扫描预热（后台线程：只填共享缓存，不做汇总显示）──

    _scan_all_players_job: dict = {}   # {running, done, error, top_n, total_rounds,
                                       #  total_jobs, done_jobs, current_round, current_player}

    def _scan_all_players_worker(engine, top_n: int):
        """后台线程：逐回合 × 逐玩家扫描低概率移动，结果只写入共享 _scan_cache
        （key 与单回合/单玩家扫描完全一致）。目的是预热缓存：之后点任一玩家
        “扫描全部回合”全部命中缓存、秒回。不累积 items（无汇总显示）。"""
        job = _scan_all_players_job
        try:
            rounds = _last_data.get("rounds", []) if _last_data else []
            job["total_rounds"] = len(rounds)
            job["total_jobs"] = len(rounds) * N_PLAYERS
            done = 0
            for r in range(len(rounds)):
                job["current_round"] = r
                ts, _, map_name = _build_predict_inputs(r, -1)
                for p in range(N_PLAYERS):
                    job["current_player"] = p
                    key = _scan_round_cache_key(r, p, top_n)
                    if key not in _scan_cache:
                        items = _scan_round_logp(engine, ts, map_name, r, top_n,
                                                 player_idx=p)
                        _scan_cache[key] = {
                            "round_idx": r,
                            "player_idx": p,
                            "map_name": map_name,
                            "round_T": int(ts["player_pos"].shape[0]),
                            "n_conds": int(len(items)),
                            "min_tick_gap": _SCAN_MIN_TICK_GAP,
                            "min_alive_ticks": _SCAN_MIN_ALIVE_TICKS,
                            "items": items,
                        }
                    done += 1
                    job["done_jobs"] = done
                if not job.get("running"):
                    break
            job["done"] = True
        except Exception as exc:
            traceback.print_exc()
            job["error"] = str(exc)
        finally:
            job["running"] = False

    @app.route("/api/scan/all/players", methods=["POST"])
    def scan_all_players():
        """
        一键预热：后台扫描【全部玩家 × 全部回合】的低概率移动，
        结果只写入共享缓存（key 与单回合/单玩家扫描一致），不做汇总显示。
        预热完成后，点任一玩家的“扫描全部回合”全部命中缓存、秒回。

        POST body (JSON):
            top_n: int — 每个（回合 × 玩家）保留条数（默认 3，上限 50）

        Returns:
            {started, top_n, total_rounds, total_jobs}
            进度用 GET /api/scan/all/players/status 轮询。
        """
        engine = _get_prediction_engine()
        if engine is None:
            return jsonify({
                "error": "Prediction engine not loaded. 请先上传预训练 checkpoint (.pt)。"
            }), 400
        if _last_data is None:
            return jsonify({"error": "No replay data loaded. 请先上传 demo / json。"}), 400
        body = request.get_json(silent=True) or {}
        top_n = max(1, min(int(body.get("top_n", 3)), 50))
        rounds = _last_data.get("rounds", [])
        if not rounds:
            return jsonify({"error": "回放数据没有回合"}), 400
        job = _scan_all_players_job
        if job.get("running"):
            return jsonify({"started": False, "running": True, "top_n": top_n,
                            "total_rounds": len(rounds),
                            "total_jobs": len(rounds) * N_PLAYERS,
                            "done_jobs": job.get("done_jobs", 0)})
        # 避免与单玩家批量扫描并发抢 MPS 算力
        for j in _scan_all_jobs.values():
            if j.get("running"):
                return jsonify({"error": "已有单玩家全回合扫描在跑，请等它结束再预热全部玩家。"}), 409
        job.clear()
        job.update({
            "running": True, "done": False, "error": None, "top_n": top_n,
            "total_rounds": len(rounds), "total_jobs": len(rounds) * N_PLAYERS,
            "done_jobs": 0, "current_round": 0, "current_player": 0,
        })
        t = threading.Thread(target=_scan_all_players_worker,
                             args=(engine, top_n), daemon=True)
        t.start()
        print(f"[Visualizer] 启动全部玩家 × {len(rounds)} 回合扫描预热"
              f"（共 {len(rounds) * N_PLAYERS} 个任务，top_n={top_n}）")
        return jsonify({"started": True, "top_n": top_n,
                        "total_rounds": len(rounds),
                        "total_jobs": len(rounds) * N_PLAYERS})

    @app.route("/api/scan/all/players/status")
    def scan_all_players_status():
        """全部玩家预热进度（只报进度，不返回汇总 items）。"""
        job = _scan_all_players_job
        if not job.get("running") and not job.get("done"):
            return jsonify({"started": False, "running": False, "done": False,
                            "error": None, "top_n": None,
                            "total_rounds": 0, "total_jobs": 0, "done_jobs": 0,
                            "current_round": 0, "current_player": 0})
        return jsonify({
            "started": True,
            "running": bool(job.get("running")),
            "done": bool(job.get("done")),
            "error": job.get("error"),
            "top_n": job.get("top_n"),
            "total_rounds": job.get("total_rounds", 0),
            "total_jobs": job.get("total_jobs", 0),
            "done_jobs": job.get("done_jobs", 0),
            "current_round": job.get("current_round", 0),
            "current_player": job.get("current_player", 0),
        })

    @app.route("/api/cache/clear", methods=["POST"])
    def cache_clear():
        """
        释放所有缓存：demo 解析缓存 + 回合扫描缓存 + spatial-only 整回合预测缓存。
        已加载的回放数据与模型本身保留（不影响当前页面）。
        """
        global _demo_cache, _scan_cache
        n_demo = len(_demo_cache)
        n_scan = len(_scan_cache)
        n_spatial = len(_spatial_round_cache)
        _demo_cache.clear()
        _scan_cache.clear()
        _spatial_round_cache.clear()
        _scan_all_jobs.clear()
        _scan_all_players_job.clear()
        print(f"[Visualizer] 已释放缓存：demo={n_demo} 扫描={n_scan} spatial={n_spatial}")
        return jsonify({
            "status": "ok",
            "cleared": {"demo": n_demo, "scan": n_scan, "spatial": n_spatial},
        })

    # ── 示例文件 ──────────────────────────────────────────────────────

    @app.route("/api/examples")
    def list_examples():
        examples = []
        for sub in ["demo", "json"]:
            d = _PROJECT_ROOT / "examples" / sub
            if d.exists():
                for f in sorted(d.iterdir()):
                    if f.is_file() and not f.name.startswith('.'):
                        examples.append({
                            "type": sub,
                            "name": f.name,
                            "path": str(f),
                            "size_mb": round(f.stat().st_size / (1024 * 1024), 1),
                        })
        return jsonify({"examples": examples})

    return app


# ── 工具函数 ──────────────────────────────────────────────────────────────

def _parse_upload(path: str, filename: str) -> tuple[dict, float]:
    """解析上传文件 → (V2 JSON dict, tick_interval)。"""
    lower = filename.lower()
    suffix = Path(lower).suffix

    if suffix == ".dem":
        return _cached_parse(path, interval=0.25), 0.25

    if suffix in (".json", ".gz"):
        # json / json.gz / .tar.gz 里包含 json（demo json 也可能被 tar 打包）
        data = _load_json_any(path)
        return data, 0.25

    raise ValueError(f"Unsupported file type: {suffix}（支持 .dem / .json / .json.gz）")


def _load_json_any(path: str) -> dict:
    """加载 json 或 json.gz（兼容 tar.gz 内的单个 json）。"""
    p = Path(path)
    raw = p.read_bytes()

    # gzip 解压（json.gz 或任意 gzip 压缩的 json）
    if p.suffix.lower() == ".gz" or raw[:2] == b"\x1f\x8b":
        try:
            import tarfile
            import io as _io
            tf = tarfile.open(fileobj=_io.BytesIO(raw), mode="r:*")
            members = [m for m in tf.getmembers() if m.isfile() and m.name.endswith((".json", ".json.gz"))]
            if members:
                mf = tf.extractfile(members[0])
                member_raw = mf.read()
                if member_raw[:2] == b"\x1f\x8b":
                    member_raw = gzip.decompress(member_raw)
                return json.loads(member_raw.decode("utf-8"))
            tf.close()
        except (tarfile.TarError, Exception):
            pass
        try:
            return json.loads(gzip.decompress(raw).decode("utf-8"))
        except Exception:
            pass

    # 直接 json
    try:
        return json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise ValueError(f"无法解析 JSON 文件: {exc}")


def _cached_parse(demo_path: str, interval: float = 0.25) -> dict:
    """解析 demo（带缓存）。"""
    cache_key = f"{demo_path}:{interval}"
    now = time.time()
    if cache_key in _demo_cache:
        ts, data = _demo_cache[cache_key]
        if now - ts < CACHE_TTL:
            return data
    print(f"[Visualizer] Parsing demo {demo_path} (interval={interval}s)...")
    data = parse_demo(demo_path, interval=interval, verbose=False)
    _demo_cache[cache_key] = (now, data)
    expired = [k for k, (ts, _) in _demo_cache.items() if now - ts > CACHE_TTL]
    for k in expired:
        del _demo_cache[k]
    return data


def _validate_v2_format(data: dict) -> None:
    """快速校验 V2 JSON 格式。"""
    if not isinstance(data, dict):
        raise ValueError("Data must be a JSON object")
    fmt = data.get("format", "")
    if fmt != "cs2.demo.v2":
        raise ValueError(
            f"Expected format 'cs2.demo.v2', got '{fmt}'. "
            "Use demo_parser to convert .dem files."
        )
    if "rounds" not in data or not data.get("rounds"):
        raise ValueError("Missing 'rounds' key in data")
    if "players" not in data:
        raise ValueError("Missing 'players' key in data")


def main():
    import argparse
    ap = argparse.ArgumentParser(description="cs-net Visualizer Server")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--checkpoint", default=None,
                    help="预训练 checkpoint 路径（路径预测；也可在页面内上传）")
    ap.add_argument("--spatial-model-dir", default=None,
                    help="spatial-only 下游任务 checkpoint 目录（winrate/alive_end/"
                         "future_kill 单局面模型；也可在页面内上传）")
    ap.add_argument("--device", default=None, choices=["cpu", "mps", "cuda"],
                    help="推理设备（路径预测与 spatial-only 共用；默认自动探测: "
                         "mps > cuda > cpu，Mac 自动用 mps，Windows 无 GPU 自动 cpu）")
    args = ap.parse_args()

    device = _detect_device(args.device)

    app = create_app(checkpoint=args.checkpoint, device=device,
                     spatial_model_dir=args.spatial_model_dir)

    print(f"""
╔════════════════════════════════════════════════════════════╗
║                  cs-net — 3D Replay Studio                 ║
║                                                            ║
║  Open in browser: http://{args.host}:{args.port}/
║  Maps: {MAPS_DIR}
║  Checkpoint: {args.checkpoint or '(页面内上传)'}
║  Spatial-only: {args.spatial_model_dir or '(页面内上传)'}
║  Device:     {device}（路径预测 + spatial-only）
╚════════════════════════════════════════════════════════════╝
""")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
