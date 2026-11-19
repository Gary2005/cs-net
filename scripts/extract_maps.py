#!/usr/bin/env python3
"""把 `maps/optimized_obj_files/*.obj.zip` 解压成 `*.obj`（3D 回放和雷达缩略图要它）。

为什么要单独一个脚本
--------------------
仓库里带的是**压缩过的**地图几何（8 个 zip，约 207 MB），解压后约 943 MB，
所以 `*.obj` 不进 git —— 干净检出之后 `maps/optimized_obj_files/` 里只有 zip，
3D 回放会退化成一块空地加网格，雷达缩略图也生成不出来。

这和 `scripts/download_models.py` 是同一类事：仓库里缺的东西，跑一次补上。
网页上「解压地图」那个按钮调的就是 :func:`extract`，命令行跑的是同一个函数，
所以两边不可能解出不一样的结果。

解压怎么做
----------
* **先写 `<map>.obj.part` 再原子改名**：943 MB 解压到一半被关掉，不会留下一个
  看起来正常的半个 `.obj`（`inspect()` 按字节数判断完整性，半截文件会被当成
  缺，下次重跑同一个任务）。
* 逐块读写（1 MB），所以进度条是**真实字节**而不是"第几个文件"。
* zip 里如果带一层目录（`sub/de_dust2.obj`），落盘时拍平成 `de_dust2.obj` ——
  网页和 `radar.py` 认的是 `<map>.obj` 这个平铺布局。

用法
----
    python scripts/extract_maps.py            # 解压缺的，已经有的跳过
    python scripts/extract_maps.py --check    # 只看状态，不解压
    python scripts/extract_maps.py --force    # 重新解压一遍
    python scripts/extract_maps.py --json     # 机器可读（脚本 / 自测用）
    python scripts/extract_maps.py --dir D:\\maps\\optimized_obj_files

`--check` 在"还有没解压的"时退出码是 1，方便串进别的脚本。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent

#: 压缩包和 `.obj` 都放在这个目录里（`app/config.py::MAPS_DIR` 指向同一处）。
MAPS_DIR = _REPO_ROOT / "maps" / "optimized_obj_files"

#: 解压中的临时后缀。只在解压过程中存在，正常结束会被 :func:`os.replace` 换掉。
PART_SUFFIX = ".part"

#: 每次读写多少字节。1 MB 让进度条有约 950 步（够细），又不至于把 syscall 变成瓶颈。
_CHUNK = 1 << 20

#: 进度回调最多按这个粒度发一次 —— 它最终是给浏览器 1.2 s 轮询看的。
_PROGRESS_STEPS = 200


class MapExtractError(RuntimeError):
    """解压失败。消息是一句人话 —— 网页横幅和任务抽屉直接拿去显示。"""


# ═══════════════════════════════════════════════════════════════════════════
# 有哪些地图 / 解压好了没
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class Target:
    """一个 `.obj.zip` 和它解压出来的 `.obj`。"""

    #: 地图名，`de_dust2`（也是 `.obj` 的文件名主干）
    map: str
    zip_path: Path
    obj_path: Path
    #: zip 里的成员名（可能带目录）
    member: str
    #: zip 头里写的解压后大小，用来判断 `.obj` 完整不完整
    bytes: int


def _base(root: Optional[Path]) -> Path:
    return Path(root) if root is not None else MAPS_DIR


def _human(n: Optional[int]) -> str:
    if n is None:
        return "?"
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"  # pragma: no cover - 走不到


def _short(exc: BaseException, limit: int = 160) -> str:
    text = str(exc).strip() or type(exc).__name__
    return text if len(text) <= limit else text[:limit].rstrip() + " …"


def _zip_member(zip_path: Path) -> tuple[str, int]:
    """`*.obj.zip` → (成员名, 解压后字节数)。

    只读中央目录（8 个 zip 合计 1.4 ms），不解压任何数据。
    """
    try:
        with zipfile.ZipFile(zip_path) as zf:
            infos = [
                i for i in zf.infolist()
                if not i.is_dir() and i.filename.lower().endswith(".obj")
            ]
    except (OSError, zipfile.BadZipFile) as exc:
        raise MapExtractError(f"{zip_path.name} 读不了：{_short(exc)}") from exc
    if not infos:
        raise MapExtractError(f"{zip_path.name} 里没有 .obj")
    return infos[0].filename, int(infos[0].file_size)


def _build(base: Path) -> tuple[list[Target], list[tuple[str, str]]]:
    """→ (能解压的目标, [(名字, 读不了的原因)])。

    坏掉的 zip 不抛出来：`inspect()` 要把"这个包坏了"当成一条状态显示给用户，
    而不是让整个接口 500。
    """
    targets: list[Target] = []
    broken: list[tuple[str, str]] = []
    for zip_path in sorted(base.glob("*.obj.zip")):
        try:
            member, size = _zip_member(zip_path)
        except MapExtractError as exc:
            broken.append((zip_path.name, str(exc)))
            continue
        # zip 里可能带目录，落盘一律拍平：网页认 `<map>.obj` 这个平铺布局
        name = Path(member).name
        targets.append(
            Target(
                map=Path(name).stem,
                zip_path=zip_path,
                obj_path=base / name,
                member=member,
                bytes=size,
            )
        )
    return targets, broken


def targets(root: Optional[Path] = None) -> list[Target]:
    """这个目录里有哪些地图要解压（坏的 zip 在这里抛错）。"""
    found, broken = _build(_base(root))
    if broken:
        raise MapExtractError("；".join(reason for _name, reason in broken))
    return found


def _complete(target: Target) -> bool:
    """`.obj` 在不在、够不够长。

    比的是 zip 头里写的大小：字节数够了就当解压好了。故意不用"文件存在" ——
    断电 / 关窗口留下的半个文件会让回放画出一半地图，那比缺文件更难查。
    """
    path = target.obj_path
    return path.is_file() and path.stat().st_size >= target.bytes


def inspect(root: Optional[Path] = None) -> dict[str, Any]:
    """本地状态（**纯磁盘**，不发任何请求）：哪些解压好了、缺哪些、多少字节。

    网页横幅每次打开都会问这个，所以要快：只 `stat()`，不解压、不读内容。
    """
    base = _base(root)
    found, broken = _build(base)
    maps: list[dict[str, Any]] = []
    missing: list[str] = []
    packed = 0
    unpacked = 0
    present = 0

    for target in found:
        path = target.obj_path
        actual = path.stat().st_size if path.is_file() else None
        ok = actual is not None and actual >= target.bytes
        try:
            packed += target.zip_path.stat().st_size
        except OSError:  # pragma: no cover - 列目录之后被删了
            pass
        unpacked += target.bytes
        if ok:
            present += actual or 0
        else:
            missing.append(target.map)
        maps.append(
            {
                "map": target.map,
                "zip": str(target.zip_path),
                "path": str(path),
                "bytes": target.bytes,
                "exists": ok,
                "present_bytes": actual,
                # 文件在、但比 zip 里写的小：上次解压没做完
                "truncated": actual is not None and not ok,
                # 还留着 `.part`：上次解压到一半被打断
                "partial": path.with_name(path.name + PART_SUFFIX).is_file(),
            }
        )

    for name, reason in broken:
        missing.append(name.removesuffix(".obj.zip"))
        maps.append(
            {
                "map": name.removesuffix(".obj.zip"),
                "zip": str(base / name),
                "path": None,
                "bytes": None,
                "exists": False,
                "present_bytes": None,
                "truncated": False,
                "partial": False,
                "error": reason,
            }
        )

    return {
        "ok": not missing,
        "root": str(base),
        # 目录里有几个 zip；0 表示这个仓库没带地图（网页横幅就不出现）
        "zips": len(found) + len(broken),
        "maps": maps,
        "missing": missing,
        "packed_bytes": packed,
        "unpacked_bytes": unpacked,
        "present_bytes": present,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 解压
# ═══════════════════════════════════════════════════════════════════════════


def _extract_one(target: Target, on_bytes: Callable[[int], None]) -> int:
    """解压一个文件：写 `.part` → 校验字节数 → 原子改名。返回写入的字节数。"""
    part = target.obj_path.with_name(target.obj_path.name + PART_SUFFIX)
    written = 0
    try:
        part.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(target.zip_path) as zf, \
                zf.open(target.member) as src, open(part, "wb") as dst:
            while True:
                chunk = src.read(_CHUNK)
                if not chunk:
                    break
                dst.write(chunk)
                written += len(chunk)
                on_bytes(written)
    except Exception as exc:  # noqa: BLE001 - 见下
        # zip 里的坏数据可能是 `zlib.error` / `BadZipFile`，也可能是磁盘满 /
        # 文件被占用的 `OSError`；一个个列会漏。`Exception` 收得住它们，又不会
        # 把 KeyboardInterrupt 当成普通失败。
        _unlink(part)
        raise MapExtractError(
            f"{target.obj_path.name} 解压失败：{_short(exc)}"
        ) from exc

    if written != target.bytes:
        _unlink(part)
        raise MapExtractError(
            f"{target.obj_path.name} 解出来 {_human(written)}，"
            f"zip 里写的是 {_human(target.bytes)}（这个 zip 可能坏了）"
        )
    os.replace(part, target.obj_path)
    return written


def _unlink(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass


def extract(
    *,
    root: Optional[Path] = None,
    force: bool = False,
    progress: Optional[Callable[[float, str], None]] = None,
    log: Optional[Callable[[str], None]] = None,
) -> dict[str, Any]:
    """解压 `root` 下所有没解压的 `.obj.zip`。

    `progress(0..1, 一句话)` 报真实字节进度（网页横幅的进度条读它），
    `log(一句话)` 报"第几个文件 / 跳过了什么"。已经解压好的跳过，`force`
    才会重来一遍。
    """
    base = _base(root)
    started = time.time()
    say = log or (lambda _msg: None)
    report = progress or (lambda _frac, _msg: None)

    found, broken = _build(base)
    if broken:
        raise MapExtractError("；".join(reason for _name, reason in broken))

    todo = [t for t in found if force or not _complete(t)]
    skipped = [] if force else [t.obj_path.name for t in found if _complete(t)]
    total = sum(t.bytes for t in todo)

    # 已经解压好的那张如果还留着 `.part`（上次断在中途、后来又补上了），
    # 顺手删掉：它最多 130 MB，留着只会让人以为还没解压完。
    for target in found:
        if target in todo:
            continue
        part = target.obj_path.with_name(target.obj_path.name + PART_SUFFIX)
        if part.is_file():
            _unlink(part)
            say(f"清掉了残留的 {part.name}")

    if not todo:
        report(1.0, "地图几何已经解压好了")
        say(f"{len(found)} 张地图都在，跳过。")
        return {
            "ok": True,
            "root": str(base),
            "extracted": [],
            "skipped": skipped,
            "bytes": 0,
            "elapsed": round(time.time() - started, 1),
            "state": inspect(base),
        }

    # 进度只许往前走：某个文件重试时不要把条子打回头。
    emitted = {"frac": 0.0}

    def report_frac(frac: float, message: str) -> None:
        emitted["frac"] = max(emitted["frac"], min(float(frac), 0.999))
        report(emitted["frac"], message)

    # 每涨这么多字节报一次，而不是每 1 MB 都报（950 次对 1.2 s 的轮询没意义）
    step = max(1, total // _PROGRESS_STEPS)
    report_frac(0.0, f"准备解压 {len(todo)} 张地图（共 {_human(total)}）")

    extracted: list[str] = []
    done = 0
    for index, target in enumerate(todo, start=1):
        say(f"[{index}/{len(todo)}] {target.obj_path.name}（{_human(target.bytes)}）")
        base_done = done
        # `seen` 是**这个文件**已经报过的字节数：跨文件累加会让第 2 张图只在
        # 结束时报一次（它的 read 从头开始数，永远追不上全局那份）。
        seen = [0]

        def on_bytes(read: int, *, _base=base_done, _seen=seen,
                     _name=target.obj_path.name, _size=target.bytes) -> None:
            if read - _seen[0] < step and read < _size:
                return
            _seen[0] = read
            report_frac(
                (_base + read) / total,
                f"{_name}  {_human(_base + read)} / {_human(total)}",
            )

        done += _extract_one(target, on_bytes)
        extracted.append(target.obj_path.name)

    report(1.0, "地图几何解压完成")
    return {
        "ok": True,
        "root": str(base),
        "extracted": extracted,
        "skipped": skipped,
        "bytes": done,
        "elapsed": round(time.time() - started, 1),
        "state": inspect(base),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 命令行
# ═══════════════════════════════════════════════════════════════════════════


def _console_progress(frac: float, message: str) -> None:
    line = f"\r  {frac * 100:5.1f}%  {message}"
    sys.stdout.write(line + " " * 4)
    sys.stdout.flush()


def _utf8_console() -> None:
    """让 ✓/✗ 在 GBK 控制台上也能打出来（否则第一行就是 UnicodeEncodeError）。"""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def _print_check(state: dict[str, Any]) -> None:
    print(f"目录：{state['root']}")
    print()
    if not state["maps"]:
        print(f"  这个目录里没有 *.obj.zip（{state['root']}）")
        return
    width = max(len(m["map"]) for m in state["maps"])
    for item in state["maps"]:
        if item.get("error"):
            mark, detail = "✗", f"读不了（{item['error']}）"
        elif item["exists"]:
            mark, detail = "✓", _human(item["present_bytes"])
        elif item["truncated"]:
            mark = "✗"
            detail = f"{_human(item['present_bytes'])} / {_human(item['bytes'])}，没解压完"
        else:
            mark, detail = "✗", f"没解压（{_human(item['bytes'])}）"
        print(f"  {mark}  {item['map']:<{width}}  {detail}")
    print()
    if state["ok"]:
        print(f"{len(state['maps'])} 张地图都解压好了（{_human(state['present_bytes'])}）。")
    else:
        print(
            f"还有 {len(state['missing'])} 张没解压："
            f"{_human(state['unpacked_bytes'] - state['present_bytes'])}。"
        )
        print("运行 `python scripts/extract_maps.py` 解压（网页顶上也会出现按钮）。")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="解压 cs-net 的地图几何（maps/optimized_obj_files/*.obj.zip）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "解压后约 943 MB；3D 回放的建筑和雷达缩略图都要它。\n"
            "已经解压好的不会重来，断在半截的那张会被整张重解。"
        ),
    )
    parser.add_argument("--check", action="store_true",
                        help="只看状态，不解压（还有没解压的时退出码 1）")
    parser.add_argument("--force", action="store_true", help="已经解压过也重新解")
    parser.add_argument("--dir", metavar="DIR", help="地图目录（默认 maps/optimized_obj_files）")
    parser.add_argument("--json", action="store_true", help="输出 JSON")
    parser.add_argument("--quiet", "-q", action="store_true", help="不打印进度")
    args = parser.parse_args(argv)

    root = Path(args.dir).expanduser() if args.dir else None
    _utf8_console()

    try:
        if args.check:
            state = inspect(root)
            if args.json:
                print(json.dumps(state, ensure_ascii=False, indent=2))
            else:
                _print_check(state)
            return 0 if state["ok"] else 1

        summary = extract(
            root=root,
            force=args.force,
            progress=None if args.quiet else _console_progress,
            log=None if args.quiet else (lambda msg: print(f"  {msg}")),
        )
        if not args.quiet:
            sys.stdout.write("\r" + " " * 78 + "\r")
            sys.stdout.flush()
        if args.json:
            print(json.dumps(summary, ensure_ascii=False, indent=2))
        else:
            print(
                f"完成：解压 {len(summary['extracted'])} 张（{_human(summary['bytes'])}），"
                f"跳过 {len(summary['skipped'])} 张，用时 {summary['elapsed']} s。"
            )
            print("地图几何已就绪：python -m app.server")
        return 0
    except MapExtractError as exc:
        if args.json:
            print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False, indent=2))
        else:
            print(f"失败：{exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:  # pragma: no cover - 手动打断
        print("\n已中断（重跑一次会把没解压完的那张重解）", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
