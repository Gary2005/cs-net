#!/usr/bin/env python3
"""`scripts/extract_maps.py` 和两个 `/api/library/maps*` 路由的单元测试。

**不碰真的 943 MB 地图**：所有的 zip 都是这个测试自己捏的（几百字节），
`de_dust2.obj.zip`、`de_mirage.obj.zip` 这类名字只是为了让检查读起来像真的。
量的是那些"错了也不会报错、只会在用户那儿静默失败"的地方：

* 要解压什么 = 目录里的 `*.obj.zip` 推出来的；zip 里带目录要拍平成 `<map>.obj`，
  因为网页和 `radar.py` 认的是平铺布局
* `inspect()` 只 `stat()`，判"解压好了没"比的是 zip 头里写的大小 —— 比"文件
  存在"严：断电留下的半个 `.obj` 会让回放画出一半地图
* 解压本身：先写 `.part` 再原子改名（半截文件不会被当成完整）、进度是真实
  字节且只增不减、解压好了的跳过、`--force` 才重来
* 坏 zip（CRC 错 / 里面没有 .obj）：一句人话 + 不留 `.part`、不假装成功
* 路由：缺才排队、已经齐了不排队、连点两下不会把同一批文件解两遍

Run:  python app/tests/test_extract_maps.py
"""

from __future__ import annotations

import shutil
import sys
import uuid
import zipfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass

from scripts import extract_maps as em  # noqa: E402

_passed = 0
_failed: list[str] = []
_skipped: list[str] = []


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


#: 一份"几何"：几十 KB 的文本，压不压缩都行，但要够大能分几个 chunk。
_BODY = b"v 1.0 0.0 0.0\n" * 4000


def make_zip(base: Path, name: str, member: str, body: bytes = _BODY,
             compression: int = zipfile.ZIP_DEFLATED) -> Path:
    path = base / f"{name}.obj.zip"
    with zipfile.ZipFile(path, "w", compression) as zf:
        zf.writestr(member, body)
    return path


# ═══════════════════════════════════════════════════════════════════════════
# 要解压什么
# ═══════════════════════════════════════════════════════════════════════════


def test_targets() -> None:
    tmp = scratch()
    try:
        eq("空目录：没有目标", em.targets(tmp), [])

        make_zip(tmp, "de_dust2", "de_dust2.obj")
        make_zip(tmp, "de_mirage", "nested/de_mirage.obj")   # 带一层目录
        (tmp / "notes.txt").write_text("不是 zip", encoding="utf-8")
        (tmp / "de_stray.obj").write_text("已经解压过的", encoding="utf-8")

        got = em.targets(tmp)
        eq("只认 *.obj.zip", [t.map for t in got], ["de_dust2", "de_mirage"])
        eq("解压后的落点拍平到 <map>.obj",
           [t.obj_path.name for t in got], ["de_dust2.obj", "de_mirage.obj"])
        eq("路径都在同一个目录", {t.obj_path.parent for t in got}, {tmp})
        eq("大小来自 zip 头", [t.bytes for t in got], [len(_BODY), len(_BODY)])
        eq("成员名照抄 zip 里的（可能带目录）",
           [t.member for t in got], ["de_dust2.obj", "nested/de_mirage.obj"])

        # zip 里没有 .obj：不能静默跳过，要说出来
        bad = scratch()
        try:
            make_zip(bad, "de_weird", "readme.txt")
            try:
                em.targets(bad)
                check("没有 .obj 的 zip 抛 MapExtractError", False)
            except em.MapExtractError as exc:
                check("没有 .obj 的 zip 抛 MapExtractError", True)
                check("… 消息里点名是哪个包", "de_weird" in str(exc), str(exc))
        finally:
            shutil.rmtree(bad, ignore_errors=True)

        # 根本不是 zip 的文件
        broken = scratch()
        try:
            (broken / "de_half.obj.zip").write_bytes(b"PK\x03\x04 not a real zip")
            try:
                em.targets(broken)
                check("坏 zip 抛 MapExtractError", False)
            except em.MapExtractError as exc:
                check("坏 zip 抛 MapExtractError", True)
                check("… 消息里点名是哪个包", "de_half" in str(exc), str(exc))
        finally:
            shutil.rmtree(broken, ignore_errors=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ═══════════════════════════════════════════════════════════════════════════
# 本地状态
# ═══════════════════════════════════════════════════════════════════════════


def test_inspect() -> None:
    tmp = scratch()
    try:
        empty = em.inspect(tmp)
        eq("空目录：ok=true（没东西要解压）", empty["ok"], True)
        eq("空目录：zips=0", empty["zips"], 0)
        eq("空目录：没有地图条目", empty["maps"], [])
        eq("空目录：missing 为空", empty["missing"], [])

        make_zip(tmp, "de_dust2", "de_dust2.obj")
        make_zip(tmp, "de_mirage", "de_mirage.obj")
        zips = sum(p.stat().st_size for p in tmp.glob("*.obj.zip"))

        packed = em.inspect(tmp)
        eq("只有 zip：ok=false", packed["ok"], False)
        eq("只有 zip：两张都缺", packed["missing"], ["de_dust2", "de_mirage"])
        eq("只有 zip：unpacked_bytes 是解压后总量",
           packed["unpacked_bytes"], 2 * len(_BODY))
        eq("只有 zip：packed_bytes 是压缩包总量", packed["packed_bytes"], zips)
        eq("只有 zip：present_bytes=0", packed["present_bytes"], 0)
        check("只有 zip：条目里带 zip / path / bytes",
              all(m["zip"] and m["path"] and m["bytes"] for m in packed["maps"]),
              str(packed["maps"]))
        eq("只有 zip：没有 .part 标记", [m["partial"] for m in packed["maps"]],
           [False, False])

        # 解压到一半的 .obj：比 zip 里写的小，必须算"没解压好"
        half = tmp / "de_dust2.obj"
        half.write_bytes(_BODY[: len(_BODY) // 2])
        (tmp / "de_mirage.obj.part").write_bytes(_BODY[:4096])
        broken = em.inspect(tmp)
        eq("半个 .obj：ok=false", broken["ok"], False)
        item = next(m for m in broken["maps"] if m["map"] == "de_dust2")
        eq("半个 .obj：truncated=true", item["truncated"], True)
        eq("半个 .obj：exists=false", item["exists"], False)
        eq("半个 .obj：present_bytes 是实际大小",
           item["present_bytes"], len(_BODY) // 2)
        mirage = next(m for m in broken["maps"] if m["map"] == "de_mirage")
        eq("留着 .part：partial=true", mirage["partial"], True)
        eq("… 照样算缺", mirage["exists"], False)

        # 齐了
        for name in ("de_dust2", "de_mirage"):
            (tmp / f"{name}.obj").write_bytes(_BODY)
        full = em.inspect(tmp)
        eq("都解压好了：ok=true", full["ok"], True)
        eq("都解压好了：missing 为空", full["missing"], [])
        eq("都解压好了：present_bytes=unpacked_bytes",
           full["present_bytes"], full["unpacked_bytes"])

        # 齐了以后再坏一个 zip：不能 500，要当成"这张没解压好"
        destroyed = scratch()
        try:
            good = make_zip(destroyed, "de_ok", "de_ok.obj")
            (destroyed / "de_ok.obj").write_bytes(_BODY)
            (destroyed / "de_bad.obj.zip").write_bytes(b"PK\x03\x04 truncated")
            state = em.inspect(destroyed)
            eq("坏 zip 不会让 inspect 抛错，ok=false", state["ok"], False)
            eq("… 算进 missing", state["missing"], ["de_bad"])
            bad = next(m for m in state["maps"] if m["map"] == "de_bad")
            check("… 条目里带原因", bool(bad.get("error")), str(bad))
            eq("… 好包照样是解压好的",
               next(m for m in state["maps"] if m["map"] == "de_ok")["exists"], True)
            check("… 好包的 zip 还在", good.is_file())
        finally:
            shutil.rmtree(destroyed, ignore_errors=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ═══════════════════════════════════════════════════════════════════════════
# 解压
# ═══════════════════════════════════════════════════════════════════════════


def test_extract() -> None:
    tmp = scratch()
    try:
        make_zip(tmp, "de_dust2", "de_dust2.obj")
        make_zip(tmp, "de_mirage", "nested/de_mirage.obj")

        progress: list[tuple[float, str]] = []
        lines: list[str] = []
        summary = em.extract(
            root=tmp,
            progress=lambda f, m: progress.append((f, m)),
            log=lines.append,
        )

        eq("两张都解压了", sorted(summary["extracted"]), ["de_dust2.obj", "de_mirage.obj"])
        eq("没有跳过的", summary["skipped"], [])
        eq("字节数对得上", summary["bytes"], 2 * len(_BODY))
        check("ok=true", summary["ok"] is True)
        eq("摘要里带上解压后的本地状态", summary["state"]["ok"], True)

        eq("落盘内容一模一样", (tmp / "de_dust2.obj").read_bytes(), _BODY)
        eq("带目录的成员也拍平了", (tmp / "de_mirage.obj").read_bytes(), _BODY)
        check("没有留下 .part", not list(tmp.glob("*.part")), str(list(tmp.glob("*.part"))))

        check("进度从 0 走到 1（没倒退）",
              progress and progress[0][0] == 0.0 and progress[-1][0] == 1.0
              and all(b[0] >= a[0] for a, b in zip(progress, progress[1:])),
              f"{progress[:2]} … {progress[-2:]}")
        check("进度里带文件名", any(".obj" in msg for _f, msg in progress),
              str(progress[-1]))
        check("进度是真实字节（不是 0/1 两档）", len(progress) >= 4, f"{len(progress)}")
        eq("日志里按顺序点名两个文件", len(lines), 2)
        check("日志第一行是 de_dust2", "de_dust2.obj" in lines[0], str(lines))

        # 再来一次：都在，一个字节都不解
        again = em.extract(root=tmp, progress=lambda f, m: None)
        eq("第二次：没有解压", again["extracted"], [])
        eq("第二次：两张都跳过", sorted(again["skipped"]), ["de_dust2.obj", "de_mirage.obj"])
        eq("第二次：字节数 0", again["bytes"], 0)

        # 半个 .obj：要整张重解（不是接着写）。旁边再放一个上次被打断留下的
        # .part，重解必须从 0 开始写，不能拼出一个"前半截 + 后半截"的文件。
        (tmp / "de_dust2.obj").write_bytes(_BODY[:100])
        (tmp / "de_dust2.obj.part").write_bytes(b"stale half file\n" * 100)
        # 已经解压好的那张旁边也留一个 .part：那是上上次的残骸，该被清掉
        (tmp / "de_mirage.obj.part").write_bytes(b"stale half file\n" * 100)
        repaired = em.extract(root=tmp, progress=lambda f, m: None)
        eq("半个 .obj 会被重解", repaired["extracted"], ["de_dust2.obj"])
        eq("… 内容补全了", (tmp / "de_dust2.obj").read_bytes(), _BODY)
        eq("… 没把旧 .part 拼进去", (tmp / "de_dust2.obj").stat().st_size, len(_BODY))
        eq("已经解压好的那张没被重写", (tmp / "de_mirage.obj").read_bytes(), _BODY)
        check("残留的 .part 都被清掉了", not list(tmp.glob("*.part")),
              str(list(tmp.glob("*.part"))))

        # --force：跳过判断整个不算
        forced = em.extract(root=tmp, force=True, progress=lambda f, m: None)
        eq("force 时两张都重解", sorted(forced["extracted"]),
           ["de_dust2.obj", "de_mirage.obj"])
        eq("force 时没有跳过", forced["skipped"], [])
        eq("force 时字节数还是总量", forced["bytes"], 2 * len(_BODY))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_extract_errors() -> None:
    # 1) CRC 对不上（下载 / 复制坏的 zip）：读的时候才炸，报一句人话
    tmp = scratch()
    try:
        path = make_zip(tmp, "de_dust2", "de_dust2.obj", body=b"A" * 2000,
                        compression=zipfile.ZIP_STORED)
        raw = path.read_bytes()
        path.write_bytes(raw.replace(b"A" * 2000, b"B" * 2000))
        try:
            em.extract(root=tmp, progress=lambda f, m: None)
            check("CRC 坏掉时抛 MapExtractError", False)
        except em.MapExtractError as exc:
            check("CRC 坏掉时抛 MapExtractError", True)
            check("… 消息里点名是哪个文件", "de_dust2" in str(exc), str(exc))
        check("… 没有留下半个 .obj", not (tmp / "de_dust2.obj").exists())
        check("… 也没留下 .part", not list(tmp.glob("*.part")),
              str(list(tmp.glob("*.part"))))
        eq("… 状态还是没解压", em.inspect(tmp)["ok"], False)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # 2) zip 里没有 .obj：解压前就该拒掉
    tmp2 = scratch()
    try:
        make_zip(tmp2, "de_weird", "readme.txt")
        try:
            em.extract(root=tmp2, progress=lambda f, m: None)
            check("没有 .obj 的包解压时抛 MapExtractError", False)
        except em.MapExtractError as exc:
            check("没有 .obj 的包解压时抛 MapExtractError", True)
            check("… 点名是哪个包", "de_weird" in str(exc), str(exc))
    finally:
        shutil.rmtree(tmp2, ignore_errors=True)


# ═══════════════════════════════════════════════════════════════════════════
# 路由
# ═══════════════════════════════════════════════════════════════════════════


def test_routes() -> None:
    from flask import Flask

    from app import api as api_mod
    from app import config
    from app.api import bp as library_bp
    from app.jobs import manager

    app = Flask(__name__)
    app.register_blueprint(library_bp)
    client = app.test_client()

    tmp = scratch()
    other = scratch()
    saved_maps = config.MAPS_DIR
    try:
        config.MAPS_DIR = tmp
        make_zip(tmp, "de_dust2", "de_dust2.obj")
        make_zip(tmp, "de_mirage", "de_mirage.obj")

        # 1) 缺文件：横幅要的那份状态
        r = client.get("/api/library/maps")
        eq("状态 → 200", r.status_code, 200)
        body = r.get_json()
        eq("… ok=false", body.get("ok"), False)
        eq("… 带上缺哪些", body.get("missing"), ["de_dust2", "de_mirage"])
        eq("… unpacked_bytes 给出来了", body.get("unpacked_bytes"), 2 * len(_BODY))
        eq("… 还没有解压任务", body.get("extract"), None)
        eq("… 不缓存", r.headers.get("Cache-Control"), "no-store")

        # 2) 点按钮 → 排一个任务（没有 worker 就不会真去解 943 MB）
        r = client.post("/api/library/maps/extract")
        eq("解压 → 200", r.status_code, 200)
        job_id = r.get_json().get("job_id")
        check("… 返回 job_id", bool(job_id), str(r.get_json()))
        job = manager.get(job_id)
        check("… 任务进了队列", job is not None and job.state == "queued",
              job.state if job else "None")
        eq("… 任务类型是 maps", job.kind, api_mod.MAPS_JOB_KIND)
        eq("… 任务名给人看", job.label, "解压地图几何")

        # 3) 连点两下：返回同一个任务，不排第二个
        r = client.post("/api/library/maps/extract")
        eq("再点一次返回同一个任务", r.get_json().get("job_id"), job_id)
        eq("… 并说明已经有一个在跑", r.get_json().get("already"), True)

        # 4) 状态接口把那个任务一起给出来（横幅读它的 step / progress）
        body = client.get("/api/library/maps").get_json()
        check("状态里带解压任务", isinstance(body.get("extract"), dict),
              str(body.get("extract")))
        eq("… 就是刚才那个", body["extract"]["id"], job_id)

        # 5) 都解压好了：不排队
        config.MAPS_DIR = other
        make_zip(other, "de_ok", "de_ok.obj")
        (other / "de_ok.obj").write_bytes(_BODY)
        r = client.post("/api/library/maps/extract")
        eq("齐了 → 200", r.status_code, 200)
        eq("齐了 → 不排任务", r.get_json().get("job_id"), None)
        eq("齐了 → already=true", r.get_json().get("already"), True)

        # 6) 显式 force：齐了也重解（想覆盖一份坏掉的地图时用）
        #    先把队列里那个模拟任务结束掉，否则"别排第二个"的守卫会先命中
        manager.get(job_id).state = "cancelled"
        r = client.post("/api/library/maps/extract?force=1")
        forced = r.get_json().get("job_id")
        check("force=1 时排新任务", bool(forced) and forced != job_id, str(r.get_json()))
        eq("… 新任务也进了队列", manager.get(forced).state, "queued")
    finally:
        config.MAPS_DIR = saved_maps
        manager.clear_finished()
        for job_id in [j["id"] for j in manager.list() if j["kind"] == "maps"]:
            manager.get(job_id).state = "cancelled"
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(other, ignore_errors=True)


# ═══════════════════════════════════════════════════════════════════════════
# 页面接线：横幅的 id / 接口路径 / 文案
# ═══════════════════════════════════════════════════════════════════════════


def test_wiring() -> None:
    html = (_ROOT / "app" / "templates" / "library.html").read_text(encoding="utf-8")
    js = (_ROOT / "app" / "static" / "js" / "library.js").read_text(encoding="utf-8")
    css = (_ROOT / "app" / "static" / "css" / "library.css").read_text(encoding="utf-8")
    i18n = (_ROOT / "visualizer" / "static" / "js" / "i18n" / "library.js").read_text(
        encoding="utf-8"
    )

    for el_id in ("map-banner", "map-detail", "map-bar", "map-bar-fill", "btn-extract-maps"):
        check(f"library.html has #{el_id}", f'id="{el_id}"' in html)
    check("the banner reuses the shared setup-banner styling",
          'class="setup-banner map' in html)
    check("the model banner moved to the same styling", 'class="setup-banner ai' in html)

    check("library.js asks the server for the map state",
          "api('/api/library/maps')" in js)
    check("library.js posts to the extract endpoint",
          "post('/api/library/maps/extract')" in js)
    check("the button is wired", "MAP_BANNER.bind()" in js)
    check("the banner loads on start", "MAP_BANNER.load()" in js)
    check("the shared renderer is used by both banners",
          js.count("setupBanner({") == 2, f"{js.count('setupBanner({')}")

    check("the maps banner has its own accent colour", ".setup-banner.map" in css)

    for key in ("title", "btn", "btnTitle", "missing", "running", "queued",
                "preparing", "done", "failed"):
        full = f"'lib.maps.{key}'"
        eq(f"i18n has {full} in both languages", i18n.count(full), 2)


# ═══════════════════════════════════════════════════════════════════════════


def main() -> int:
    print("scripts/extract_maps + /api/library/maps 单元测试")
    for name, fn in [
        ("targets", test_targets),
        ("inspect", test_inspect),
        ("extract", test_extract),
        ("extract errors", test_extract_errors),
        ("routes", test_routes),
        ("wiring", test_wiring),
    ]:
        print(f"\n[{name}]")
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            _failed.append(f"{name} raised {type(exc).__name__}")
            print(f"  \u2717 {name} raised {type(exc).__name__}: {exc}")
            import traceback
            traceback.print_exc()

    print(f"\n{_passed} passed, {len(_failed)} failed"
          + (f", {len(_skipped)} skipped" if _skipped else ""))
    for name in _failed:
        print(f"  FAILED: {name}")
    return 0 if not _failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
