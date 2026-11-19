"""
cs-net — the demo library and replay workbench, as one web app.

``create_app`` builds the existing visualizer (unchanged, so its tests and
static assets keep working) and layers the demo library on top:

* ``/``          -> the demo library (see :mod:`app.api`)
* ``/players``   -> the player index, aggregated from every demo in the library
* ``/players/<steamid>`` -> one player's profile
* ``/replay``    -> the 3D replay page, the visualizer's own ``index.html``
* ``/api/library/*`` -> library API
* ``/api/*``     -> the visualizer's API, untouched

Run it::

    conda activate cs-net-v5
    python -m app.server --port 5000
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from flask import Flask, render_template

from . import api, config, db, jobs, watcher

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def create_app(maps_dir: Optional[Path] = None, *, start_background: bool = True) -> Flask:
    """Assemble the full application."""
    from visualizer.server import create_app as create_visualizer_app

    app = create_visualizer_app(maps_dir=maps_dir, default_page="library")

    # Before init_db: it creates the data directory, and a freshly created
    # (empty) one would look like "already migrated" to the move below.
    moved = db.migrate_legacy_data_dir()
    if moved:
        print(f"[cs-net] 数据目录已从 {config.LEGACY_DATA_DIR} 移到 {moved}")

    db.init_db()
    # Nothing can be extracting at startup, so any row that says it is was
    # interrupted.  Left alone it reads "提取中" forever and nothing ever
    # finishes it, because the watcher only queues newly discovered files.
    stale = db.reset_stale_extracting()
    if stale:
        print(f"[cs-net] 上次中断了 {len(stale)} 个提取任务，已重新标记为待入库：{stale}")

    app.register_blueprint(api.bp)
    # AI 预测（aim / spatial_prediction）。单独的 blueprint：它要读回放会话，
    # 和库的 `/api/library/*` 是两件不相干的事。import 很轻 —— torch、numba
    # 都在路由函数里才 import，页面启动不受影响。
    from .ai.api import bp as ai_bp

    app.register_blueprint(ai_bp)

    # 玩家页的两份统计（传统 / AI，每场明细 + 跨场合计）。单独一个 blueprint：
    # 它们要解码 .csz、跑模型、写磁盘缓存，和库的元数据接口是两件事。
    # 同样只在路由函数里 import 重的东西 —— 页面启动不受影响。
    from .stats_api import bp as stats_bp

    app.register_blueprint(stats_bp)

    def _page(template: str, **context):
        """Render one of the library's own pages.

        Rendered HTML gets no Cache-Control from Flask, and a heuristically
        cached page paired with a fresh .js is a confusing bug to chase.  Same
        reasoning as the visualizer's index route.
        """
        resp = app.make_response(render_template(template, **context))
        resp.headers["Cache-Control"] = "no-cache"
        return resp

    @app.get("/library")
    def library_page():
        return _page(
            "library.html",
            sources=config.SOURCE_ORDER,
            source_labels=config.SOURCE_LABELS,
        )

    @app.get("/players")
    def players_page():
        return _page("players.html")

    @app.get("/players/<steamid>")
    def player_page(steamid: str):
        # The page is a shell; everything comes from /api/library/players/...
        # so an unknown SteamID renders the page and shows "not found" rather
        # than a bare 404 the user cannot navigate back from.
        return _page("player.html", steamid=steamid)

    @app.get("/healthz")
    def healthz():
        return {
            "ok": True,
            "demos": db.stats()["total"],
            "players": db.list_players(limit=1)["total"],
            "jobs_pending": jobs.manager.pending_count(),
        }

    if start_background:
        jobs.manager.start()
        if config.Config().get("watch_enabled", True):
            watcher.watcher.start()

    return app


def banner(host: str, port: int, *, maps: Optional[Path], data_dir: Path,
           sampling: str, columns: Optional[int] = None,
           color: Optional[bool] = None) -> str:
    """启动横幅：logo 字符画 + 信息框。

    对齐、版式（并排还是上下）、要不要打颜色都由 :mod:`visualizer.banner` 决定 ——
    这里只准备内容，不做任何空格算术。``columns`` / ``color`` 是给测试用的：
    正常启动时传 None，让它按真实终端自己判断。
    """
    from visualizer.banner import startup

    return startup(
        [
            ("Demo 库:", f"http://{host}:{port}/library"),
            ("玩家:", f"http://{host}:{port}/players"),
            ("回放页:", f"http://{host}:{port}/replay"),
            None,
            ("数据目录:", str(data_dir)),
            ("地图:", str(maps) if maps else "(未找到，回退到空地 + 网格)"),
            ("提取器:", f"demo_extract（{sampling}）"),
        ],
        title="cs-net v5 · Demo 库 + 3D 回放",
        columns=columns,
        color=color,
    )


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="cs-net v5 · Demo 库 + 3D 回放")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--maps-dir", default=None,
                    help="optimized_obj_files 目录；默认自动查找")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--no-watch", action="store_true",
                    help="不启动目录监听（仍可在界面里手动扫描）")
    args = ap.parse_args()

    if args.no_watch:
        config.Config().update({"watch_enabled": False})

    app = create_app(
        maps_dir=Path(args.maps_dir) if args.maps_dir else None,
        start_background=True,
    )
    from visualizer.server import _resolve_maps_dir

    maps = _resolve_maps_dir(args.maps_dir)
    print()
    print(banner(args.host, args.port, maps=maps, data_dir=config.DATA_DIR,
                 sampling=config.Config().sampling_label()))
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
