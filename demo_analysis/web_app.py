import json
import os
import re
import subprocess
import sys
import threading
import uuid
from pathlib import Path
from typing import Any

from flask import (
    Flask,
    Response,
    jsonify,
    render_template,
    request,
    send_from_directory,
    stream_with_context,
)

from demo_analysis import high_level_analysis
from demo_analysis import llm_summary as llm_summary_module


os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


ROOT_DIR = Path(__file__).resolve().parents[1]
ASSETS_DIR = ROOT_DIR / "assets"
UPLOAD_DIR = ROOT_DIR / "demo_analysis" / "uploads"
OUTPUT_DIR = ROOT_DIR / "demo_analysis" / "outputs"
MODEL_ROOT = ROOT_DIR / "cs-net-models"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, template_folder="templates", static_folder="static")
ANALYSIS_CACHE: dict[str, dict[str, Any]] = {}
ANALYSIS_JOBS: dict[str, dict[str, Any]] = {}
ANALYSIS_LOCK = threading.Lock()

VIEWER_DIR = Path(__file__).resolve().parent / "static" / "viewer"


def _has_running_jobs() -> bool:
    with ANALYSIS_LOCK:
        for job in ANALYSIS_JOBS.values():
            if job.get("status") in {"queued", "running"}:
                return True
    return False


def cleanup_runtime_artifacts(clear_state: bool = True) -> dict[str, int]:
    """Delete stale uploaded demos/results and optionally reset in-memory state."""
    removed_uploads = 0
    removed_outputs = 0

    for path in UPLOAD_DIR.glob("*"):
        if path.is_file() or path.is_symlink():
            path.unlink(missing_ok=True)
            removed_uploads += 1

    for path in OUTPUT_DIR.glob("*"):
        if path.is_file() or path.is_symlink():
            path.unlink(missing_ok=True)
            removed_outputs += 1

    if clear_state:
        with ANALYSIS_LOCK:
            ANALYSIS_JOBS.clear()
        ANALYSIS_CACHE.clear()

    return {
        "uploads": removed_uploads,
        "outputs": removed_outputs,
    }


def choose_default_device() -> str:
    return "cpu"


ALL_HEAD_SUBDIRS = ("alive", "nxt_kill", "nxt_death", "win_rate", "duel")
REQUIRED_HEAD_SUBDIRS = ("win_rate",)  # only win_rate is mandatory; other heads fall back


def _has_ckpt_and_yaml(path: Path) -> bool:
    """True if `path` directly contains a checkpoint (.pt/.pth) and a non-tokenizer yaml."""
    if not path.is_dir():
        return False
    has_yaml = any(
        p.suffix == ".yaml" and "tokenizer" not in p.name.lower()
        for p in path.iterdir()
    )
    has_ckpt = any(p.suffix in {".pt", ".pth"} for p in path.iterdir())
    return has_yaml and has_ckpt


def _looks_like_model_root(path: Path) -> bool:
    """A model root has (at minimum) a `win_rate/` subdir."""
    if not path.exists() or not path.is_dir():
        return False
    return all((path / sub).is_dir() for sub in REQUIRED_HEAD_SUBDIRS)


def normalize_model_root(path_str: str) -> Path | None:
    """
    Accept any of:
      - A model root containing `win_rate/` (and optionally other heads).
      - A bare legacy checkpoint dir (win_rate itself) — passed through as-is;
        the inference script auto-detects it as a single-head legacy layout.
      - A path *inside* a model root (e.g. `.../win_rate`) — normalize to its parent.
    """
    if not path_str:
        return None
    path = Path(path_str)
    if not path.is_absolute():
        path = (ROOT_DIR / path).resolve()
    if _looks_like_model_root(path):
        return path
    parent = path.parent
    if _looks_like_model_root(parent):
        return parent
    # Legacy: user points directly at a single-head checkpoint dir (e.g. win_rate/).
    if _has_ckpt_and_yaml(path):
        return path
    return None


def discover_model_paths() -> list[str]:
    options: list[str] = []
    if _looks_like_model_root(MODEL_ROOT):
        options.append(str(MODEL_ROOT))
    # Legacy fallback: surface MODEL_ROOT/win_rate if only that exists.
    legacy = MODEL_ROOT / "win_rate"
    if not options and _has_ckpt_and_yaml(legacy):
        options.append(str(legacy))
    return options


def choose_default_model_path(model_options: list[str]) -> str:
    if model_options:
        return model_options[0]
    if MODEL_ROOT.exists() and MODEL_ROOT.is_dir():
        return str(MODEL_ROOT)
    return ""


safe_float = high_level_analysis.safe_float


def safe_nonnegative_int(value, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(0, parsed)


def _append_job_log(job_id: str, line: str) -> None:
    clean = line.rstrip("\n")
    with ANALYSIS_LOCK:
        job = ANALYSIS_JOBS.get(job_id)
        if not job:
            return
        job["logs"].append(clean)
        if len(job["logs"]) > 300:
            job["logs"] = job["logs"][-300:]


def _update_job_progress_by_log(job_id: str, line: str) -> None:
    total_match = re.search(r"Total rounds with sampled ticks:\s*(\d+)", line)
    round_match = re.search(r"Processing round\s+(\d+)\s+with", line)

    with ANALYSIS_LOCK:
        job = ANALYSIS_JOBS.get(job_id)
        if not job:
            return

        if "Loading model from" in line:
            job["phase"] = "加载模型中"
            job["message"] = "正在加载胜率模型..."
        elif "Sampling ticks" in line or "Processing demo" in line:
            job["phase"] = "解析 Demo 中"
            job["message"] = "正在解析 demo (通常 30~60s)"
        elif "Extracted states for" in line and "Starting inference" in line:
            job["phase"] = "模型推理中"
            job["message"] = "开始逐回合推理 (通常不到 1 分钟)"
        elif "Results saved to" in line:
            job["phase"] = "结果收集中"
            job["message"] = "结果已生成，正在整理数据..."

        if total_match:
            job["total_rounds"] = int(total_match.group(1))

        if round_match:
            current_round = int(round_match.group(1))
            job["current_round"] = current_round
            total_rounds = int(job.get("total_rounds", 0))
            if total_rounds > 0:
                finished = int(job.get("processed_rounds", 0)) + 1
                finished = min(finished, total_rounds)
                job["processed_rounds"] = finished
                progress = int(100 * finished / total_rounds)
                job["progress"] = max(min(progress, 99), 1)
                job["message"] = f"正在处理第 {current_round} 局 ({finished}/{total_rounds})"


def _run_analysis_job(
    job_id: str,
    upload_path: Path,
    model_path: str,
    device: str,
    batch_size: str,
    output_path: Path,
) -> None:
    with ANALYSIS_LOCK:
        if job_id not in ANALYSIS_JOBS:
            return
        ANALYSIS_JOBS[job_id]["status"] = "running"
        ANALYSIS_JOBS[job_id]["phase"] = "准备中"
        ANALYSIS_JOBS[job_id]["message"] = "任务已启动，正在准备执行..."

    cmd = [
        sys.executable,
        "-m",
        "demo_analysis.get_round_win_rate",
        "--demo_path",
        str(upload_path),
        "--model_root",
        model_path,
        "--device",
        device,
        "--output",
        str(output_path),
        "--batch_size",
        batch_size,
    ]

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT_DIR),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            env=env,
        )

        stdout_lines: list[str] = []
        if proc.stdout is not None:
            for line in proc.stdout:
                stdout_lines.append(line)
                _append_job_log(job_id, line)
                _update_job_progress_by_log(job_id, line)

        return_code = proc.wait()

        if return_code != 0:
            with ANALYSIS_LOCK:
                job = ANALYSIS_JOBS.get(job_id)
                if job:
                    job["status"] = "failed"
                    job["phase"] = "失败"
                    job["message"] = "分析脚本运行失败"
                    job["error"] = "\n".join(job.get("logs", [])[-80:])[-4000:]
            return

        if not output_path.exists():
            with ANALYSIS_LOCK:
                job = ANALYSIS_JOBS.get(job_id)
                if job:
                    job["status"] = "failed"
                    job["phase"] = "失败"
                    job["message"] = "分析结果文件不存在"
                    job["error"] = "分析结果文件不存在"
            return

        with output_path.open("r", encoding="utf-8") as f:
            raw_results = json.load(f)

        dashboard = high_level_analysis.build_dashboard_payload(raw_results)
        analysis_id = uuid.uuid4().hex
        ANALYSIS_CACHE[analysis_id] = {
            "dashboard": dashboard,
            "raw": raw_results,
            "source_file": str(upload_path),
            "result_file": str(output_path),
            "stdout": "".join(stdout_lines),
        }

        with ANALYSIS_LOCK:
            job = ANALYSIS_JOBS.get(job_id)
            if job:
                job["status"] = "succeeded"
                job["phase"] = "完成"
                job["message"] = "分析完成"
                job["progress"] = 100
                job["analysis_id"] = analysis_id
                job["dashboard"] = dashboard
    except Exception as exc:
        with ANALYSIS_LOCK:
            job = ANALYSIS_JOBS.get(job_id)
            if job:
                job["status"] = "failed"
                job["phase"] = "失败"
                job["message"] = "后台任务异常"
                job["error"] = str(exc)


@app.route("/")
def index():
    # Keep disk usage bounded for local usage: clear stale artifacts on each open.
    if not _has_running_jobs():
        cleanup_runtime_artifacts(clear_state=True)

    model_options = discover_model_paths()
    return render_template(
        "index.html",
        default_device=choose_default_device(),
        model_options=model_options,
        default_model_path=choose_default_model_path(model_options),
    )


@app.get("/assets/<path:filename>")
def asset_file(filename: str):
    return send_from_directory(ASSETS_DIR, filename)


# ---- third_party 2D replay viewer (built SPA mounted under /viewer/) ----

@app.get("/viewer/")
@app.get("/viewer/player")
def viewer_index():
    """Serve the SPA shell for the viewer's home and /player routes."""
    return send_from_directory(VIEWER_DIR, "index.html")


@app.get("/viewer/<path:filename>")
def viewer_assets(filename: str):
    """Serve any built asset (JS/CSS/PNG/wasm/worker.js) under /viewer/."""
    target = (VIEWER_DIR / filename).resolve()
    try:
        target.relative_to(VIEWER_DIR.resolve())
    except ValueError:
        return jsonify({"error": "invalid path"}), 404
    if not target.is_file():
        # Unknown sub-route → fall back to SPA shell so client routing still works.
        return send_from_directory(VIEWER_DIR, "index.html")
    return send_from_directory(VIEWER_DIR, filename)


def build_viewer_timeline(dashboard: dict[str, Any]) -> dict[str, Any]:
    """Flatten per-tick predictions into the shape the 2D viewer's overlays expect.

    The bundled viewer (`static/viewer/`) reads a `winrateurl` JSON and drives its
    win-rate curve, next-kill / next-death bars, and duel matrix from the entry
    that matches the current round + round_seconds. Fields consumed by the viewer:
    round, seconds, ct_win_rate, alive_pred (10), next_kill (11), next_death (11),
    duel (10x10), players_info[{name}]. See `winRateData.js:normalizeTimelineEntry`.
    """
    timeline: list[dict[str, Any]] = []
    for rd in dashboard.get("rounds") or []:
        round_id = rd.get("round_id")
        for tick in rd.get("ticks") or []:
            seconds = tick.get("round_seconds")
            if seconds is None:
                continue
            players = [
                {"name": p.get("name")}
                for p in (tick.get("players_info") or [])
                if p.get("name")
            ]
            timeline.append(
                {
                    "round": round_id,
                    "seconds": seconds,
                    "ct_win_rate": tick.get("ct_win_rate"),
                    "alive_pred": tick.get("alive_pred"),
                    "next_kill": tick.get("next_kill"),
                    "next_death": tick.get("next_death"),
                    "duel": tick.get("duel"),
                    "players_info": players,
                }
            )
    return {
        "meta": {"roundBase": 1},
        "timeline": timeline,
    }


@app.get("/api/winrate_timeline/<analysis_id>")
def serve_winrate_timeline(analysis_id: str):
    """Serve the per-tick prediction timeline consumed by the 2D viewer overlays."""
    if not re.fullmatch(r"[0-9a-fA-F]{8,64}", analysis_id):
        return jsonify({"error": "bad analysis_id"}), 400
    entry = ANALYSIS_CACHE.get(analysis_id)
    if not entry:
        return jsonify({"error": "analysis not found"}), 404
    return jsonify(build_viewer_timeline(entry["dashboard"]))


@app.get("/api/demo_file/<run_id>")
@app.get("/api/demo_file/<run_id>.dem")
def serve_uploaded_demo(run_id: str):
    """Stream the originally uploaded .dem file so the in-browser parser can fetch it.

    Accepts both `<run_id>` and `<run_id>.dem` — the 2D viewer's WASM parser picks
    demo format from the URL suffix, so we surface `.dem` by default.
    """
    if not re.fullmatch(r"[0-9a-fA-F]{8,64}", run_id):
        return jsonify({"error": "bad run_id"}), 400
    matches = sorted(UPLOAD_DIR.glob(f"{run_id}_*"))
    if not matches:
        return jsonify({"error": "demo not found"}), 404
    target = matches[0]
    return send_from_directory(
        UPLOAD_DIR,
        target.name,
        as_attachment=False,
        download_name=target.name.split("_", 1)[-1],
        mimetype="application/octet-stream",
    )


@app.get("/api/download_result/<analysis_id>")
def download_result(analysis_id: str):
    """Download the raw parsed JSON for a completed analysis."""
    if not re.fullmatch(r"[0-9a-fA-F]{8,64}", analysis_id):
        return jsonify({"error": "bad analysis_id"}), 400
    entry = ANALYSIS_CACHE.get(analysis_id)
    if not entry:
        return jsonify({"error": "analysis not found"}), 404
    raw = entry.get("raw")
    if raw is None:
        return jsonify({"error": "raw results not available"}), 404
    return jsonify(raw)


@app.post("/api/load_json")
def load_json_result():
    """Load a pre-parsed JSON result file and build the dashboard from it."""
    if _has_running_jobs():
        return jsonify({"error": "已有任务正在运行，请等待完成后再操作"}), 409

    json_file = request.files.get("json_file")
    if json_file is None or json_file.filename == "":
        return jsonify({"error": "请上传 JSON 文件"}), 400

    try:
        data = json.load(json_file)
    except json.JSONDecodeError as exc:
        return jsonify({"error": f"JSON 解析失败: {exc}"}), 400

    if not isinstance(data, dict):
        return jsonify({"error": "JSON 根节点必须是对象"}), 400

    # Detect format: dashboard payload has "rounds" as a list
    if "rounds" in data and isinstance(data.get("rounds"), list):
        dashboard = data
        raw = data
    else:
        raw = data
        try:
            dashboard = high_level_analysis.build_dashboard_payload(raw)
        except Exception as exc:
            return jsonify({"error": f"无法解析此 JSON 为有效的分析结果: {exc}"}), 400

    analysis_id = uuid.uuid4().hex
    ANALYSIS_CACHE[analysis_id] = {
        "dashboard": dashboard,
        "raw": raw,
        "source_file": json_file.filename,
        "result_file": None,
        "stdout": "Loaded from uploaded JSON file",
    }

    return jsonify(
        {
            "analysis_id": analysis_id,
            "dashboard": dashboard,
            "run_id": None,
        }
    )


@app.post("/api/analyze")
def analyze_demo():
    dem_file = request.files.get("demo_file")
    if dem_file is None or dem_file.filename == "":
        return jsonify({"error": "请上传 .dem 文件"}), 400

    if _has_running_jobs():
        return jsonify({"error": "已有任务正在运行，请等待完成后再上传新 demo"}), 409

    cleanup_runtime_artifacts(clear_state=True)

    model_path = request.form.get("model_path", "").strip()
    device = request.form.get("device", choose_default_device()).strip()
    batch_size = request.form.get("batch_size", "32").strip()

    if not model_path:
        return jsonify({"error": "请先选择模型根目录"}), 400

    model_dir = normalize_model_root(model_path)
    if model_dir is None:
        return jsonify({
            "error": (
                f"模型根目录无效: {model_path}\n"
                f"至少需要 win_rate 子目录（或直接传入 win_rate 目录）；"
                f"alive / nxt_kill / nxt_death / duel 缺失时会自动走 fallback。"
            )
        }), 400

    model_path = str(model_dir)

    run_id = uuid.uuid4().hex
    upload_path = UPLOAD_DIR / f"{run_id}_{Path(dem_file.filename).name}"
    output_path = OUTPUT_DIR / f"{run_id}.json"
    dem_file.save(upload_path)

    job_id = uuid.uuid4().hex
    with ANALYSIS_LOCK:
        ANALYSIS_JOBS[job_id] = {
            "status": "queued",
            "phase": "排队中",
            "message": "任务已提交，等待执行",
            "progress": 0,
            "logs": [
                "已接收任务，准备开始分析...",
                "提示: 解析 demo 通常 30~60s，模型推理通常不到 10 分钟。",
            ],
            "error": "",
            "total_rounds": 0,
            "processed_rounds": 0,
            "current_round": None,
            "analysis_id": None,
            "dashboard": None,
            "run_id": run_id,
        }

    worker = threading.Thread(
        target=_run_analysis_job,
        args=(job_id, upload_path, model_path, device, batch_size, output_path),
        daemon=True,
    )
    worker.start()

    return jsonify({"job_id": job_id})


@app.get("/api/analyze_status/<job_id>")
def analyze_status(job_id: str):
    with ANALYSIS_LOCK:
        job = ANALYSIS_JOBS.get(job_id)
        if not job:
            return jsonify({"error": "无效的 job_id"}), 404

        logs = job.get("logs", [])
        payload = {
            "status": job.get("status", "unknown"),
            "phase": job.get("phase", ""),
            "message": job.get("message", ""),
            "progress": job.get("progress", 0),
            "total_rounds": job.get("total_rounds", 0),
            "processed_rounds": job.get("processed_rounds", 0),
            "current_round": job.get("current_round"),
            "logs": "\n".join(logs[-120:]),
            "error": job.get("error", ""),
            "analysis_id": job.get("analysis_id"),
            "run_id": job.get("run_id"),
        }

        if job.get("status") == "succeeded":
            payload["dashboard"] = job.get("dashboard")

    return jsonify(payload)


@app.post("/api/llm_summary")
def llm_summary():
    body = request.get_json(silent=True) or {}
    analysis_id = body.get("analysis_id", "")
    api_key = body.get("api_key", "").strip()
    model_name = body.get("model_name", "").strip()
    base_url = body.get("base_url", "https://api.openai.com/v1").strip()
    temperature = safe_float(body.get("temperature", 0.95), 0.95)
    language = body.get("language", "zh").strip().lower()
    max_detailed_rounds = safe_nonnegative_int(
        body.get("max_detailed_rounds"),
        llm_summary_module.MAX_DETAILED_TACTICAL_ROUNDS,
    )

    if not analysis_id or analysis_id not in ANALYSIS_CACHE:
        return jsonify({"error": "无效的 analysis_id，请先完成 DEM 分析"}), 400
    if not api_key:
        return jsonify({"error": "请提供 API Key"}), 400
    if not model_name:
        return jsonify({"error": "请提供模型名称"}), 400

    dashboard = ANALYSIS_CACHE[analysis_id]["dashboard"]
    config = llm_summary_module.LlmSummaryConfig(
        api_key=api_key,
        model_name=model_name,
        base_url=base_url,
        temperature=temperature,
        language=language,
    )
    try:
        content = llm_summary_module.llm_summary_sync(
            dashboard,
            config,
            max_detailed_rounds=max_detailed_rounds,
        )
        return jsonify({"summary": content})
    except Exception as exc:
        return jsonify({"error": f"请求 LLM 接口异常: {exc}"}), 500


@app.post("/api/llm_summary_stream")
def llm_summary_stream():
    body = request.get_json(silent=True) or {}
    analysis_id = body.get("analysis_id", "")
    api_key = body.get("api_key", "").strip()
    model_name = body.get("model_name", "").strip()
    base_url = body.get("base_url", "https://api.openai.com/v1").strip()
    temperature = safe_float(body.get("temperature", 0.95), 0.95)
    language = body.get("language", "zh").strip().lower()
    max_detailed_rounds = safe_nonnegative_int(
        body.get("max_detailed_rounds"),
        llm_summary_module.MAX_DETAILED_TACTICAL_ROUNDS,
    )

    if not analysis_id or analysis_id not in ANALYSIS_CACHE:
        return jsonify({"error": "无效的 analysis_id，请先完成 DEM 分析"}), 400
    if not api_key:
        return jsonify({"error": "请提供 API Key"}), 400
    if not model_name:
        return jsonify({"error": "请提供模型名称"}), 400

    dashboard = ANALYSIS_CACHE[analysis_id]["dashboard"]
    config = llm_summary_module.LlmSummaryConfig(
        api_key=api_key,
        model_name=model_name,
        base_url=base_url,
        temperature=temperature,
        language=language,
    )

    @stream_with_context
    def generate_text_stream():
        try:
            yield from llm_summary_module.llm_summary_stream_sync(
                dashboard,
                config,
                max_detailed_rounds=max_detailed_rounds,
            )
        except Exception as exc:
            yield f"\n\n[LLM error] {exc}"

    return Response(generate_text_stream(), content_type="text/plain; charset=utf-8")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=7860, debug=True)
