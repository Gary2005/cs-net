import math
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import yaml
from demoparser2 import DemoParser
from flask import Flask, jsonify, render_template, request, send_from_directory


TOOLS_DIR = Path(__file__).resolve().parent
ROOT_DIR = TOOLS_DIR.parent
CALLOUT_CONFIG_DIR = ROOT_DIR / "config" / "callouts"
DEFAULT_NEAREST_THRESHOLD = 300
OVERVIEW_DIR = ROOT_DIR / "demo_analysis" / "static" / "overviews"

app = Flask(
    __name__,
    template_folder=str(TOOLS_DIR / "templates"),
    static_folder=str(TOOLS_DIR / "static"),
    static_url_path="/tool_static",
)


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if hasattr(value, "item"):
        return jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def empty_config() -> dict[str, Any]:
    return {"defaults": {"nearest_threshold": DEFAULT_NEAREST_THRESHOLD}, "maps": {}}


def map_config_path(map_name: str) -> Path:
    name = str(map_name)
    if not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError(f"Invalid map name: {name!r}")
    return CALLOUT_CONFIG_DIR / f"{name}.yaml"


def load_config() -> dict[str, Any]:
    data = empty_config()
    if not CALLOUT_CONFIG_DIR.exists():
        return data
    for path in sorted(CALLOUT_CONFIG_DIR.glob("*.yaml")):
        with path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        if not isinstance(cfg, dict):
            cfg = {}
        cfg.setdefault("nearest_threshold", DEFAULT_NEAREST_THRESHOLD)
        cfg.setdefault("polygons_cn", [])
        cfg.setdefault("polygons_en", [])
        data["maps"][path.stem] = cfg
    return data


def save_config(data: dict[str, Any]) -> None:
    defaults = data.get("defaults") if isinstance(data.get("defaults"), dict) else {}
    threshold = defaults.get("nearest_threshold", DEFAULT_NEAREST_THRESHOLD)
    maps = data.get("maps")
    if not isinstance(maps, dict):
        maps = {}
    CALLOUT_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    for map_name, cfg in sorted(maps.items()):
        if not isinstance(cfg, dict):
            continue
        cfg.setdefault("nearest_threshold", threshold)
        cfg.setdefault("polygons_cn", [])
        cfg.setdefault("polygons_en", [])
        with map_config_path(map_name).open("w", encoding="utf-8", newline="\n") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False, width=120)


def sample_demo(path: Path, max_ticks: int = 650) -> dict[str, Any]:
    print(f"Parsing demo {path}...")
    parser = DemoParser(str(path))
    header = parser.parse_header()
    map_name = header.get("map_name", "unknown_map")

    meta = parser.parse_ticks(wanted_props=["game_time", "total_rounds_played"])
    ticks = sorted(int(x) for x in meta["tick"].drop_duplicates().tolist())
    if len(ticks) > max_ticks:
        stride = max(1, math.ceil(len(ticks) / max_ticks))
        ticks = ticks[::stride]

    df = parser.parse_ticks(
        wanted_props=["X", "Y", "Z", "last_place_name", "is_alive", "team_num"],
        ticks=ticks,
    )
    samples = []
    skipped = 0
    for row in df.itertuples():
        x = finite_float(getattr(row, "X", None))
        y = finite_float(getattr(row, "Y", None))
        z = finite_float(getattr(row, "Z", None))
        if x is None or y is None or z is None:
            skipped += 1
            continue
        samples.append(
            {
                "tick": int(row.tick),
                "name": str(getattr(row, "name", "")),
                "steamid": str(getattr(row, "steamid", "")),
                "team_num": "CT" if getattr(row, "team_num", None) == 3 else "T",
                "is_alive": bool(getattr(row, "is_alive", False)),
                "x": x,
                "y": y,
                "z": z,
                "last_place_name": str(getattr(row, "last_place_name", "") or ""),
            }
        )

    place_counts = Counter(
        s["last_place_name"] for s in samples if s["last_place_name"]
    )
    print(
        f"Parsed {len(samples)} samples from demo, "
        f"skipped {skipped} invalid player rows, found {len(place_counts)} unique last_place_name values"
    )
    return {
        "map_name": map_name,
        "sample_count": len(samples),
        "samples": samples,
        "last_place_counts": place_counts.most_common(80),
        "skipped_samples": skipped,
        "header": jsonable(header),
    }


@app.get("/")
def index():
    return render_template("callouts_annotator.html")


@app.get("/api/config")
def api_config():
    return jsonify(load_config())


@app.post("/api/config")
def api_save_config():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Expected JSON object"}), 400
    try:
        save_config(data)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"ok": True, "path": str(CALLOUT_CONFIG_DIR)})


@app.post("/api/demo")
def api_demo():
    demo = request.files.get("demo")
    if demo is None or not demo.filename:
        return jsonify({"error": "Upload a .dem file"}), 400
    with tempfile.NamedTemporaryFile(suffix=".dem", delete=False) as f:
        tmp_path = Path(f.name)
        demo.save(f)
    try:
        return jsonify(sample_demo(tmp_path))
    finally:
        tmp_path.unlink(missing_ok=True)


@app.get("/overviews/<path:filename>")
def overviews(filename: str):
    return send_from_directory(OVERVIEW_DIR, filename)


if __name__ == "__main__":
    print("Open http://127.0.0.1:7871")
    app.run(host="127.0.0.1", port=7871)
