"""
cs-net — demo library and replay workbench.

The library layer: a SQLite-backed catalogue of local ``.dem`` files with
watch-directory auto-import, ``.csz`` extraction through ``tools/demo_extract``,
and a handoff into the existing 3D replay visualizer.

Module map::

    config.py    settings + platform path candidates + source inference
    db.py        SQLite schema and queries
    probe.py     cheap demo reads (header, real team names)
    metadata.py  scoreboard / roster / watched players from an extracted demo
    radar.py     map OBJ -> radar thumbnail
    ingest.py    discovery, dedup, extraction pipeline
    uploads.py   browser uploads: stream to data/imports, content dedup
    watcher.py   polling directory watcher
    jobs.py      background job queue
    api.py       /api/library/* HTTP surface
    server.py    create_app() / entry point
"""

from .config import Config

__all__ = ["Config"]
__version__ = "5.1.0"
