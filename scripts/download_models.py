#!/usr/bin/env python3
"""从 Hugging Face 下载网页用的那几份权重（v5）。

为什么要单独一个脚本
--------------------
`models/` 在 .gitignore 里（两份 `.pt` 合计 665 MB，不该进 git），而
`app/ai/models.py::PINNED_WEIGHTS` 又把路径写死了 —— 干净检出之后 AI 面板
一定报"权重不存在"。这个脚本补的就是这一步：把 HF 仓库 `gary2oos/cs-net-v5`
里的同名文件放到本地对应位置。

仓库里的目录结构和仓库一模一样（`models/aim/<run>.pt` + `<run>.yaml`），
所以**不另写一张文件表** —— 要下什么由 `PINNED_WEIGHTS` 推出来，换版本只改
`app/ai/models.py` 那一处。

用什么下
--------
`huggingface_hub.hf_hub_download`（断点续传、校验、限速、xet 都交给它）。
`local_dir` 给的是仓库根：仓库里的 `models/aim/<run>.pt` 就是落盘相对路径，
文件直接压在网页读的那几个位置上（hub 自己的元数据落在 `<repo>/.cache/`，
已 gitignore）。

镜像
----
国内直连 huggingface.co 时经常连不上（DNS 被投毒 / TLS 握手被重置）。脚本
**先探测再决定**：能连上官方站点就用官方的，连不上自动换成
https://hf-mirror.com ，并在输出里说明用的是哪个。换站点靠的是
`HF_ENDPOINT` —— 和用户自己 `export HF_ENDPOINT=...` 是同一件事，只是不用
重开进程。也可以手动指定：

    --mirror                 只用镜像，不试官方（官方确定被墙时快一点）
    --endpoint https://...   把某个站点排在最前面
    HF_ENDPOINT=https://...  同上，环境变量版（huggingface_hub 本来就认它）

用法
----
    python scripts/download_models.py            # 下载缺的，已有的跳过
    python scripts/download_models.py --check    # 只看状态，不下载（不需要 hub）
    python scripts/download_models.py --force    # 已有也重下
    python scripts/download_models.py --mirror   # 强制走镜像
    python scripts/download_models.py --json     # 机器可读（脚本 / 自测用）

网页上「下载模型」那个按钮调的是同一个 :func:`download`：`app/ai/api.py`
把活儿扔进 jobs 队列，横幅上的进度条读的就是这里传出去的
`progress(0..1, 一句话)`。所以命令行和网页下载的**是同一份代码、同一批文件**。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.ai.models import MODELS_DIR, PINNED_WEIGHTS  # noqa: E402  (要先加 sys.path)


def default_root() -> Path:
    """四个文件的下落根：``<root>/models/<task>/<file>``。

    开发态就是仓库根。打包后权重和 exe 并排（``<bundle>/models``），而本文件的
    ``__file__`` 在 ``_internal/`` 里 —— 直接用 ``_REPO_ROOT`` 会让 full 版的横幅
    永远说"缺 4 个文件、合计 665 MB"，明明权重就躺在 exe 旁边。所以从
    :data:`MODELS_DIR` 反推：它会按冻结态 / portable / %LOCALAPPDATA% 解析。
    """
    return Path(MODELS_DIR).parent

#: 权重所在的 HF 仓库。改了版本号这里也要跟着改，同时改 `PINNED_WEIGHTS`。
REPO_ID = "gary2oos/cs-net-v5"
HF_ENDPOINT = "https://huggingface.co"
MIRROR_ENDPOINT = "https://hf-mirror.com"

#: 探测站点能不能用：只发一个 HEAD，连不上要尽快失败，不能让用户等半分钟。
_PROBE_TIMEOUT = 8.0

#: `.pt` 小于这个就当它是坏的 —— HF 没开 LFS 时会发一个几百字节的指针文本，
#: 尺寸"对得上"但 torch 一读就炸，比缺文件更难查。
_MIN_PT_BYTES = 10 * 1024 * 1024


class DownloadError(RuntimeError):
    """下载失败。消息是一句人话 —— 网页横幅直接拿去显示。"""


# ═══════════════════════════════════════════════════════════════════════════
# 要下载什么
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class Target:
    """一个要下载的文件：仓库里叫什么、本地落在哪。"""

    task: str
    #: 仓库里的路径，`models/aim/<run>.pt`（也是 `hf_hub_download` 的 filename）
    remote: str
    #: 本地落点，默认 `<repo>/models/aim/<run>.pt`
    local: Path


def targets(root: Optional[Path] = None) -> list[Target]:
    """`PINNED_WEIGHTS` → 四个文件（两份 `.pt` + 各自同名的 `.yaml`）。

    `root` 只在测试/自测时用（把文件下到别处），默认是 :func:`default_root`。
    """
    base = Path(root) if root is not None else default_root()
    out: list[Target] = []
    for task, weights in PINNED_WEIGHTS.items():
        rel = weights.relative_to(MODELS_DIR)  # aim/20261119-013008.pt
        for path in (weights, weights.with_suffix(".yaml")):
            out.append(
                Target(
                    task=task,
                    remote=f"models/{rel.parent.as_posix()}/{path.name}",
                    local=base / "models" / rel.parent / path.name,
                )
            )
    return out


def _present(target: Target) -> bool:
    """本地这份文件在不在、像不像一份真的权重。"""
    path = target.local
    if not path.is_file():
        return False
    return path.stat().st_size >= _MIN_PT_BYTES if path.suffix == ".pt" else True


def inspect(root: Optional[Path] = None) -> dict[str, Any]:
    """本地状态（**不联网**）：每个文件在不在、多大、缺哪些。

    网页横幅每次打开都会问这个 —— 所以这里绝不能发请求。
    """
    files = []
    missing = []
    for target in targets(root):
        path = target.local
        size = path.stat().st_size if path.is_file() else None
        ok = _present(target)
        files.append(
            {
                "task": target.task,
                "remote": target.remote,
                "path": str(path),
                "exists": ok,
                "bytes": size,
                # 文件在、但不是一份完整的权重（多半是 HF 的 LFS 指针文本）
                "truncated": (not ok) and size is not None,
            }
        )
        if not ok:
            missing.append(target.remote)
    return {
        "ok": not missing,
        "repo": REPO_ID,
        "root": str(Path(root) if root is not None else default_root()),
        "files": files,
        "missing": missing,
        "present_bytes": sum(f["bytes"] or 0 for f in files if f["exists"]),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 站点：官方 / 镜像
# ═══════════════════════════════════════════════════════════════════════════


def endpoints(prefer: Optional[str] = None, mirror_only: bool = False) -> list[str]:
    """按优先级排好的候选站点（去重、去掉末尾斜杠）。

    顺序是"越靠前越优先"：`--endpoint` > `HF_ENDPOINT` > 官方 > 镜像。
    `--mirror` 时跳过官方站点 —— 官方被墙的情况下那 8 秒探测纯属白等。
    """
    candidates = [prefer, os.environ.get("HF_ENDPOINT")]
    if not mirror_only:
        candidates.append(HF_ENDPOINT)
    candidates.append(MIRROR_ENDPOINT)

    seen: set[str] = set()
    out: list[str] = []
    for base in candidates:
        if not base:
            continue
        clean = str(base).strip().rstrip("/")
        if clean and clean not in seen:
            seen.add(clean)
            out.append(clean)
    return out


def _short(exc: BaseException, limit: int = 160) -> str:
    text = str(exc).strip() or type(exc).__name__
    if not text or text == type(exc).__name__:
        return type(exc).__name__
    return text if len(text) <= limit else text[:limit].rstrip() + " …"


def _human(n: Optional[int]) -> str:
    if n is None:
        return "?"
    if n == 0:
        return "0 B"
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"  # pragma: no cover - 走不到


def _file_url(base: str, remote: str) -> str:
    """仓库里一个文件的下载地址（HF 会 302 到 CDN，urllib 自动跟）。"""
    return f"{base}/{REPO_ID}/resolve/main/{remote}"


def _head(base: str, remote: str, timeout: float = _PROBE_TIMEOUT) -> tuple[bool, Optional[int], str]:
    """HEAD 一个文件：`(能不能用, 字节数, 失败原因)`。

    只发 HEAD 不发 GET —— 探测一个 393 MB 的文件不该真的传 393 MB。这一条
    故意不用 huggingface_hub：它要么下整个文件、要么只查仓库元数据（两次
    往返），而这里只需要"这个站点通不通、这个文件多大"。所以 `--check`
    在没有装 huggingface_hub 的机器上也能跑。
    """
    request = urllib.request.Request(
        _file_url(base, remote), method="HEAD", headers={"User-Agent": "cs-net-model-downloader/5"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            length = resp.headers.get("Content-Length")
            size = int(length) if length and length.isdigit() else None
            return True, size, ""
    except urllib.error.HTTPError as exc:
        return False, None, f"HTTP {exc.code} {exc.reason}"
    except (urllib.error.URLError, OSError, ValueError) as exc:
        reason = getattr(exc, "reason", None) or exc
        return False, None, _short(reason)


def choose_endpoint(
    plan: list[Target],
    *,
    prefer: Optional[str] = None,
    mirror_only: bool = False,
    log: Optional[Callable[[str], None]] = None,
) -> str:
    """探测候选站点，返回第一个能用的。

    探测对象挑**最小的那个文件**（`.yaml`，几百字节）—— 连 HEAD 都省不了，
    但至少别拿 393 MB 的 `.pt` 当探针。
    """
    say = log or (lambda _msg: None)
    probe = next((t for t in plan if t.remote.endswith(".yaml")), plan[0])
    failures: list[str] = []

    for base in endpoints(prefer, mirror_only):
        ok, _size, reason = _head(base, probe.remote)
        if ok:
            if failures:
                say(f"站点：{base}（前面的站点不可用，已自动切换）")
            else:
                say(f"站点：{base}")
            return base
        failures.append(f"{base}（{reason}）")
        say(f"{base} 不可用（{reason}），换下一个")

    raise DownloadError(
        "连不上模型仓库，官方和镜像都试过了："
        + "；".join(failures)
        + "。检查网络 / 代理，或者手动指定 --endpoint <你信任的 HF 镜像>"
    )


# ═══════════════════════════════════════════════════════════════════════════
# 下载（huggingface_hub）
# ═══════════════════════════════════════════════════════════════════════════

_ENDPOINT_ARG: Optional[bool] = None  #: 这个 hub 版本认不认 `endpoint=`


def _quiet_hub() -> None:
    """别让 huggingface_hub 自己打日志进度 —— 我们已经在报同一件事了。"""
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)


def _hub_supports_endpoint() -> bool:
    """`hf_hub_download(endpoint=...)` 是 hub 2.x 才有的参数。"""
    global _ENDPOINT_ARG
    if _ENDPOINT_ARG is None:
        try:
            import inspect as _inspect

            from huggingface_hub import hf_hub_download

            _ENDPOINT_ARG = "endpoint" in _inspect.signature(hf_hub_download).parameters
        except (ImportError, TypeError, ValueError):
            _ENDPOINT_ARG = False
    return _ENDPOINT_ARG


def _bar_class(on_bytes: Callable[[int, Optional[int]], None]) -> Any:
    """把 hub 的下载进度条接到 `on_bytes(已下, 总大小)` 上，自己不画。

    `disable=True` 是有意的：命令行这边由脚本自己刷一行（和库里其它脚本一致），
    网页那边 stdout 根本不是终端 —— 两条进度条只会打架。tqdm 关掉之后 `.n`
    不再自增，所以这里自己数。
    """
    from tqdm.auto import tqdm

    class _Bar(tqdm):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            kwargs["disable"] = True
            super().__init__(*args, **kwargs)
            self._csnet_done = 0

        def update(self, n: int = 1) -> Any:
            out = super().update(n)
            self._csnet_done += n
            on_bytes(self._csnet_done, self.total)
            return out

    return _Bar


def _download_one(
    base: str,
    target: Target,
    *,
    local_dir: Path,
    force: bool,
    expected: Optional[int],
    on_bytes: Callable[[int, Optional[int]], None],
) -> int:
    """用 huggingface_hub 下**一个**文件，返回落盘字节数。

    - 断点续传、校验、重试都是 hub 自己的（先写 `.incomplete` 再改名），
      665 MB 在慢网上下到 99% 断掉，重跑一次会接着下；
    - 站点用 `endpoint=` 显式传，不靠全局状态 —— 下到一半换镜像不用重建进程，
      也不会踩到调用方自己设的 `HF_ENDPOINT`；
    - 老版本 hub 没有这个参数时退回改环境变量（和用户手动 export 等价）。
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - 环境问题
        raise DownloadError(
            "下载权重需要 huggingface_hub：pip install huggingface_hub"
        ) from exc

    kwargs: dict[str, Any] = {
        "repo_id": REPO_ID,
        "filename": target.remote,
        "local_dir": str(local_dir),
        "force_download": force,
        "tqdm_class": _bar_class(on_bytes),
    }
    if _hub_supports_endpoint():
        kwargs["endpoint"] = base
    else:
        os.environ["HF_ENDPOINT"] = base
        from huggingface_hub import constants

        constants.ENDPOINT = base

    path = Path(hf_hub_download(**kwargs))
    if not path.is_file():
        raise DownloadError(f"{target.remote} 下载后没有落盘：{path}")

    size = path.stat().st_size
    if expected is not None and size != expected:
        raise DownloadError(
            f"{target.remote} 只有 {_human(size)} / {_human(expected)}，重跑一次会接着下"
        )
    return size


def download(
    *,
    root: Optional[Path] = None,
    files: Optional[list[str]] = None,
    force: bool = False,
    prefer: Optional[str] = None,
    mirror_only: bool = False,
    progress: Optional[Callable[[float, str], None]] = None,
    log: Optional[Callable[[str], None]] = None,
) -> dict[str, Any]:
    """把缺的权重下到本地，返回一份摘要（网页会把它塞进任务结果里）。

    `progress(0..1, 一句话)` 是给网页进度条用的；`log(一句话)` 是给命令行
    看的叙事（"用哪个站点""第几个文件"）。两个都可以不给。
    """
    say_progress = progress or (lambda _frac, _msg: None)
    say = log or (lambda _msg: None)

    todo = targets(root)
    if files:
        wanted = {str(f).replace("\\", "/").lstrip("/") for f in files}
        known = {t.remote for t in todo}
        unknown = sorted(wanted - known)
        if unknown:
            raise DownloadError(
                f"仓库里没有这些文件：{', '.join(unknown)}（可选：{', '.join(sorted(known))}）"
            )
        todo = [t for t in todo if t.remote in wanted]

    plan = [t for t in todo if force or not _present(t)]
    skipped = [t.remote for t in todo if t not in plan]
    for rel in skipped:
        say(f"已有，跳过 {rel}")

    if not plan:
        say_progress(1.0, "模型已就绪")
        return {
            "ok": True,
            "endpoint": None,
            "downloaded": [],
            "skipped": skipped,
            "bytes": 0,
            "elapsed": 0.0,
        }

    root_dir = Path(root) if root is not None else default_root()
    # `local_dir` 给仓库根：仓库里的路径 `models/aim/<run>.pt` 就是落盘相对路径，
    # 文件直接压在网页读的那几个位置上。hub 自己的元数据在 `<repo>/.cache/`。
    local_dir = root_dir
    started = time.time()
    _quiet_hub()

    base = choose_endpoint(plan, prefer=prefer, mirror_only=mirror_only, log=say)
    fallbacks = [c for c in endpoints(prefer, mirror_only) if c != base]

    # 先问一遍每个文件多大：总进度要靠它算，顺便在动手前就发现"远端 404"。
    sizes: dict[str, Optional[int]] = {}
    for target in plan:
        _ok, size, reason = _head(base, target.remote)
        sizes[target.remote] = size
        if size is None:
            say(f"警告：{target.remote} 拿不到大小（{reason}），仍尝试下载")
    grand_total = sum(s for s in sizes.values() if s) or None

    # 进度只许往前走：hub 重试同一个文件时会把计数打回 0，别让条子倒着走。
    reported = {"frac": 0.0}

    def emit(frac: float, message: str) -> None:
        reported["frac"] = max(reported["frac"], min(float(frac), 0.999))
        say_progress(reported["frac"], message)

    downloaded: list[str] = []
    done_before = 0
    emit(0.0, f"准备下载 {len(plan)} 个文件（共 {_human(grand_total)}）")
    for index, target in enumerate(plan, start=1):
        rel = target.remote
        expected = sizes.get(rel)
        say(f"[{index}/{len(plan)}] {rel}"
            + (f"（{_human(expected)}）" if expected else ""))

        def on_bytes(done: int, total: Optional[int], *, _rel=rel, _index=index,
                     _before=done_before, _expected=expected) -> None:
            if grand_total:
                frac = (_before + min(done, total or done)) / grand_total
            else:  # 拿不到大小就按"第几个文件 + 本文件百分比"算
                inner = (done / total) if total else 0.0
                frac = (_index - 1 + inner) / len(plan)
            emit(frac, f"{_rel}  {_human(done)} / {_human(total or _expected)}")

        attempts = [base, *fallbacks]
        errors: list[str] = []
        size = 0
        for pos, attempt in enumerate(attempts):
            try:
                size = _download_one(
                    attempt, target, local_dir=local_dir, force=force,
                    expected=expected, on_bytes=on_bytes,
                )
            except Exception as exc:  # noqa: BLE001 - hub 抛的异常五花八门
                msg = _short(exc)
                errors.append(f"{attempt}（{msg}）")
                more = pos + 1 < len(attempts)
                say(f"{attempt} 下不了 {rel}（{msg}）" + ("，换下一个站点" if more else ""))
                continue
            if attempt != base:
                say(f"{rel} 改用 {attempt} 下载")
                fallbacks = [c for c in fallbacks if c != attempt]
                base = attempt
            break
        else:
            raise DownloadError(f"{rel} 下载失败：" + "；".join(errors))

        done_before += size
        downloaded.append(rel)

    say_progress(1.0, "下载完成")
    summary: dict[str, Any] = {
        "ok": True,
        "endpoint": base,
        "downloaded": downloaded,
        "skipped": skipped,
        "bytes": done_before,
        "elapsed": round(time.time() - started, 1),
    }
    summary["state"] = inspect(root)
    return summary


# ═══════════════════════════════════════════════════════════════════════════
# 命令行
# ═══════════════════════════════════════════════════════════════════════════


def _console_progress(frac: float, message: str) -> None:
    """一行原地刷新的进度（CR），和库里其它脚本的观感一致。"""
    line = f"\r  {frac * 100:5.1f}%  {message}"
    sys.stdout.write(line + " " * 4)
    sys.stdout.flush()


def _print_check(state: dict[str, Any], remote: Optional[dict[str, Any]]) -> None:
    print(f"仓库：{state['repo']}")
    print(f"本地：{state['root']}")
    if remote and remote.get("endpoint"):
        print(f"站点：{remote['endpoint']}")
    print()
    width = max(len(f["remote"]) for f in state["files"]) if state["files"] else 0
    for item in state["files"]:
        mark = "✓" if item["exists"] else "✗"
        if item["exists"]:
            detail = _human(item["bytes"])
        elif item["truncated"]:
            detail = f"{_human(item['bytes'])}，不是完整的权重"
        else:
            detail = "缺失"
        print(f"  {mark}  {item['remote']:<{width}}  {detail}")
    print()
    if state["ok"]:
        print(f"本地 {len(state['files'])} 个文件都在（{_human(state['present_bytes'])}）。")
    else:
        print(f"缺 {len(state['missing'])} 个文件。运行 `python scripts/download_models.py` 下载。")
        if remote and remote.get("error"):
            print(f"（远端探测失败：{remote['error']}）")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=f"下载 cs-net 网页用的权重（{REPO_ID}）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "镜像：默认先试 huggingface.co，连不上自动改用 hf-mirror.com"
            "（等价于把 HF_ENDPOINT 切过去）。\n"
            "已有文件不会重下；断掉的下载重跑一次会从断点接着下。"
        ),
    )
    parser.add_argument("--check", action="store_true",
                        help="只看状态，不下载（缺文件时退出码 1；不需要 huggingface_hub）")
    parser.add_argument("--force", action="store_true", help="已有文件也重新下载")
    parser.add_argument("--mirror", action="store_true", help="只用 hf-mirror.com，不试官方站点")
    parser.add_argument("--endpoint", metavar="URL", help="指定站点，排在最前面试")
    parser.add_argument("--file", action="append", metavar="REL",
                        help="只处理指定文件（仓库内路径，可重复）")
    parser.add_argument("--dest", metavar="DIR", help="下载到别的目录（默认仓库根）")
    parser.add_argument("--json", action="store_true", help="输出 JSON")
    parser.add_argument("--quiet", "-q", action="store_true", help="不打印进度")
    args = parser.parse_args(argv)

    root = Path(args.dest).expanduser().resolve() if args.dest else None

    try:
        if args.check:
            state = inspect(root)
            remote: Optional[dict[str, Any]] = None
            try:
                base = choose_endpoint(targets(root), prefer=args.endpoint,
                                       mirror_only=args.mirror)
                sizes = {}
                for target in targets(root):
                    _ok, size, _reason = _head(base, target.remote)
                    sizes[target.remote] = size
                for item in state["files"]:
                    item["remote_bytes"] = sizes.get(item["remote"])
                remote = {"endpoint": base}
            except DownloadError as exc:
                remote = {"endpoint": None, "error": str(exc)}

            if args.json:
                print(json.dumps({**state, "remote": remote}, ensure_ascii=False, indent=2))
            else:
                _print_check(state, remote)
            return 0 if state["ok"] else 1

        summary = download(
            root=root,
            files=args.file,
            force=args.force,
            prefer=args.endpoint,
            mirror_only=args.mirror,
            progress=None if args.quiet else _console_progress,
            log=None if args.quiet else (lambda msg: print(f"  {msg}")),
        )
        if not args.quiet:
            sys.stdout.write("\r" + " " * 78 + "\r")
            sys.stdout.flush()
        if args.json:
            print(json.dumps(summary, ensure_ascii=False, indent=2))
        else:
            done = len(summary["downloaded"])
            print(f"完成：下载 {done} 个文件（{_human(summary['bytes'])}），"
                  f"跳过 {len(summary['skipped'])} 个，用时 {summary['elapsed']} s。")
            print("AI 面板现在可以用了：python -m app.server")
        return 0
    except DownloadError as exc:
        if args.json:
            print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False, indent=2))
        else:
            print(f"失败：{exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:  # pragma: no cover - 手动打断
        print("\n已中断（再跑一次会从断点接着下）", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
