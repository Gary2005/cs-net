#!/usr/bin/env python3
"""`/api/maps`、`/api/map/<name>`、`/api/map/<name>/colors`：地图目录是**懒解析**的。

为什么值得单测
--------------
`maps/optimized_obj_files/*.obj` 不进 git，只带 `.obj.zip`。用户很可能在 app 已经
跑起来之后，才点库页面顶上那条「解压地图」横幅把几何解出来 —— 如果地图目录只在
`create_app()` 里解析一次，这次解压就白做了：`/api/map/<name>` 一直 404，回放里
永远是空地，非得重启一次 app 才看得见建筑。

所以 `create_app` 里的 `_maps_now()` 会在"启动时没找到"的情况下每次请求再找一遍，
找到之后记住（不会每个请求都 glob）。这个文件钉住的就是：

* 启动时没有 `.obj` → `/api/maps` 空、`/api/map/…` 404、回放页说"未找到地图"
* 文件出现之后（进程没重启）→ 立刻 200、页面那句提示也变了
* 找到之后不再重复解析（`_resolve_maps_dir` 只被多叫了找到它的那一次）
* 不存在的图仍然 404，不会因为"懒解析"就把 404 变成 200

Run:  python visualizer/tests/test_maps_api.py
"""

from __future__ import annotations

import shutil
import sys
import uuid
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "tools"))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass

import visualizer.server as vs  # noqa: E402

_passed = 0
_failed: list[str] = []

_OBJ = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"


def check(name: str, cond: bool, detail: str = "") -> None:
    global _passed
    if cond:
        _passed += 1
        print(f"  \u2713 {name}")
    else:
        _failed.append(name)
        print(f"  \u2717 {name}" + (f"\n      {detail}" if detail else ""))


def eq(name: str, got, want) -> None:
    check(name, got == want, f"got {got!r}, want {want!r}")


def scratch() -> Path:
    base = _ROOT / ".build" / "test-tmp"
    base.mkdir(parents=True, exist_ok=True)
    d = base / uuid.uuid4().hex[:12]
    d.mkdir()
    return d


def main() -> int:
    print("visualizer 地图接口（懒解析地图目录）")

    tmp = scratch()
    real_resolver = vs._resolve_maps_dir
    calls = {"n": 0}

    def late_resolver(explicit):  # noqa: ARG001 - 接口要的就是这个签名
        """进程启动那一次还没有 .obj，之后才有。"""
        calls["n"] += 1
        if calls["n"] == 1:
            return None
        return tmp.resolve() if list(tmp.glob("*.obj")) else None

    vs._resolve_maps_dir = late_resolver
    try:
        # 解压之前：目录里只有 zip
        (tmp / "de_dust2.obj.zip").write_bytes(b"PK\x03\x04 fake")

        app = vs.create_app(maps_dir=tmp)
        client = app.test_client()

        eq("启动时只解析一次地图目录", calls["n"], 1)
        eq("/api/maps 是空的", client.get("/api/maps").get_json(),
           {"maps": [], "dir": None})
        eq("/api/map/de_dust2 → 404", client.get("/api/map/de_dust2").status_code, 404)
        check("回放页说没找到地图",
              "未找到地图 OBJ" in client.get("/replay").get_data(as_text=True))

        # 解压完成（进程没重启）
        (tmp / "de_dust2.obj").write_text(_OBJ, encoding="utf-8")
        body = client.get("/api/maps").get_json()
        eq("/api/maps 立刻看得见", body.get("maps"), ["de_dust2"])
        eq("… dir 也填上了", body.get("dir"), str(tmp.resolve()))
        r = client.get("/api/map/de_dust2")
        eq("/api/map/de_dust2 → 200", r.status_code, 200)
        check("… 返回的就是那份 OBJ", b"f 1 2 3" in r.data, str(r.data[:40]))
        check("回放页不再说没找到",
              "未找到地图 OBJ" not in client.get("/replay").get_data(as_text=True))

        # 找到之后记住
        before = calls["n"]
        for _ in range(3):
            client.get("/api/maps")
        eq("找到之后不再重复解析", calls["n"], before)

        # 404 还是 404
        eq("/api/map/unknown → 404", client.get("/api/map/unknown").status_code, 404)
        eq("… colors 也是 404",
           client.get("/api/map/unknown/colors").status_code, 404)
        eq("没有 .colors.bin 时 colors → 404",
           client.get("/api/map/de_dust2/colors").status_code, 404)
    finally:
        vs._resolve_maps_dir = real_resolver
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n{_passed} passed, {len(_failed)} failed")
    for name in _failed:
        print(f"  FAILED: {name}")
    return 0 if not _failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
