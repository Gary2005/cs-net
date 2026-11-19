#!/usr/bin/env python3
"""`scripts/download_models.py` 和两个 `/api/ai/models*` 路由的单元测试。

**不联网、不下载 665 MB**：`hf_hub_download` 被换成一个只写几百字节假文件的
桩，`_head`（探测站点 / 问文件大小）也换掉。量的是那些"错了也不会报错、只会在
用户那儿静默失败"的地方：

* 要下哪些文件 = `PINNED_WEIGHTS` 推出来的（换版本不用改两处），四个文件
  里 `.pt` 和同名 `.yaml` 必须成套，且落盘路径就是网页读的那几条
* `inspect()` **不联网**，而且一个几百字节的 `.pt`（HF 没开 LFS 时的指针
  文本）要算"坏的"，不能算"有了"
* 站点优先级：官方 → 镜像；`--mirror` 只走镜像；官方挂掉要**自动**换镜像，
  两个都挂要报出两家分别为什么不行
* 下载本身：`local_dir` 是仓库根、`force_download` 跟着 `--force` 走、进度
  只增不减、下完不用重下、换站点后一次成功
* 没装 huggingface_hub 时给一句能照做的话，而不是 ImportError 栈
* 路由：缺文件才排队、已经齐了不排队、连点两下不会排出两个 665 MB 的下载

Run:  python app/tests/test_download_models.py
"""

from __future__ import annotations

import shutil
import sys
import uuid
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass

from scripts import download_models as dm  # noqa: E402

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


def _hub():
    """装不装 huggingface_hub 是环境的事，不是这个脚本的错。"""
    try:
        import huggingface_hub

        return huggingface_hub
    except ImportError:  # pragma: no cover - 环境问题
        return None


# ═══════════════════════════════════════════════════════════════════════════
# 要下什么 / 本地状态
# ═══════════════════════════════════════════════════════════════════════════


def test_targets() -> None:
    from app.ai import models

    got = dm.targets()
    rels = [t.remote for t in got]

    eq("四个文件（两份 .pt + 各自同名的 .yaml）", len(got), 4)
    check("远端路径都在 models/ 下", all(r.startswith("models/") for r in rels), str(rels))
    for task, weights in models.PINNED_WEIGHTS.items():
        rel = weights.relative_to(models.MODELS_DIR).as_posix()
        check(f"{task}: 仓库里的路径就是本地相对路径", f"models/{rel}" in rels, str(rels))
        check(f"{task}: 配一份同名的 .yaml",
              f"models/{weights.with_suffix('.yaml').relative_to(models.MODELS_DIR).as_posix()}" in rels,
              str(rels))
        target = next(t for t in got if t.remote == f"models/{rel}")
        eq(f"{task}: 默认下到仓库根的 models/ 里", target.local, models.PINNED_WEIGHTS[task])
        eq(f"{task}: 任务名带上", target.task, task)
        check(f"{task}: 远端路径和 hf_hub_download 的 filename 是同一个",
              target.remote.endswith(target.local.name), target.remote)

    # 换个根目录（自测 / --dest）只影响本地路径
    other = dm.targets(_ROOT / ".build" / "elsewhere")
    eq("--dest 只改本地落点", [t.remote for t in other], rels)
    eq("… 落点跟着根走",
       other[0].local, _ROOT / ".build" / "elsewhere" / "models" / "aim" / other[0].local.name)


def test_inspect() -> None:
    tmp = scratch()
    try:
        empty = dm.inspect(tmp)
        eq("空目录：ok=false", empty["ok"], False)
        eq("空目录：四个都缺", len(empty["missing"]), 4)
        eq("空目录：present_bytes=0", empty["present_bytes"], 0)
        check("空目录：每个文件都带 remote/path", all("remote" in f and "path" in f
                                                      for f in empty["files"]), str(empty["files"]))

        # 真的齐了（.pt 用够大的假文件，否则会被当成 LFS 指针）
        for target in dm.targets(tmp):
            target.local.parent.mkdir(parents=True, exist_ok=True)
            if target.local.suffix == ".pt":
                target.local.write_bytes(b"w" * (dm._MIN_PT_BYTES + 1))
            else:
                target.local.write_text("model: {}\n", encoding="utf-8")
        full = dm.inspect(tmp)
        eq("文件齐了：ok=true", full["ok"], True)
        eq("文件齐了：missing 为空", full["missing"], [])
        check("文件齐了：present_bytes 加起来", full["present_bytes"] > 0, str(full["present_bytes"]))

        # HF 没开 LFS 时会发一个几百字节的指针文本 —— "存在"但不能用
        pointer = dm.targets(tmp)[0].local
        pointer.write_bytes(b"version https://git-lfs.github.com/spec/v1\n")
        broken = dm.inspect(tmp)
        eq("几百字节的 .pt 算缺", broken["ok"], False)
        item = next(f for f in broken["files"] if f["remote"] == dm.targets(tmp)[0].remote)
        eq("… 标成 truncated", item["truncated"], True)
        eq("… exists 是 false", item["exists"], False)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ═══════════════════════════════════════════════════════════════════════════
# 站点选择 / 自动换镜像
# ═══════════════════════════════════════════════════════════════════════════


def test_endpoints() -> None:
    saved_env = dm.os.environ.pop("HF_ENDPOINT", None)
    try:
        eq("默认：官方优先，镜像兜底",
           dm.endpoints(), [dm.HF_ENDPOINT, dm.MIRROR_ENDPOINT])
        eq("--mirror：只剩镜像", dm.endpoints(mirror_only=True), [dm.MIRROR_ENDPOINT])
        eq("--endpoint 排最前",
           dm.endpoints(prefer="https://example.test")[:2],
           ["https://example.test", dm.HF_ENDPOINT])
        eq("末尾斜杠和重复项都去掉",
           dm.endpoints(prefer="https://hf-mirror.com/"),
           [dm.MIRROR_ENDPOINT, dm.HF_ENDPOINT])

        dm.os.environ["HF_ENDPOINT"] = "https://env.test/"
        eq("用户自己设的 HF_ENDPOINT 也算候选（hub 认的就是它）",
           dm.endpoints()[:2], ["https://env.test", dm.HF_ENDPOINT])
    finally:
        dm.os.environ.pop("HF_ENDPOINT", None)
        if saved_env is not None:
            dm.os.environ["HF_ENDPOINT"] = saved_env


def test_fallback() -> None:
    plan = dm.targets()
    saved_head = dm._head
    try:
        # 官方挂、镜像通 → 自动用镜像，并且说明"切换过"
        lines: list[str] = []

        def head_official_down(base, remote, timeout=dm._PROBE_TIMEOUT):
            if base == dm.HF_ENDPOINT:
                return False, None, "握手被重置"
            return True, 692, ""

        dm._head = head_official_down
        eq("官方不可达 → 用镜像", dm.choose_endpoint(plan, log=lines.append), dm.MIRROR_ENDPOINT)
        check("… 明确说了在用镜像", any(dm.MIRROR_ENDPOINT in ln for ln in lines), str(lines))
        check("… 也说了官方为什么不行", any(dm.HF_ENDPOINT in ln for ln in lines), str(lines))

        # 两家都挂 → 报错里要有两家各自的原因，不能只说"失败了"
        def head_all_down(base, remote, timeout=dm._PROBE_TIMEOUT):
            return False, None, f"{base} 不通"

        dm._head = head_all_down
        try:
            dm.choose_endpoint(plan)
            check("两家都不可达时抛 DownloadError", False)
        except dm.DownloadError as exc:
            check("两家都不可达时抛 DownloadError", True)
            check("… 消息里带镜像地址", dm.MIRROR_ENDPOINT in str(exc), str(exc))
            check("… 也带官方的地址", dm.HF_ENDPOINT in str(exc), str(exc))
    finally:
        dm._head = saved_head


# ═══════════════════════════════════════════════════════════════════════════
# 真的下（hf_hub_download 换成桩）
# ═══════════════════════════════════════════════════════════════════════════

_PT_BODY = b"P" * 512
_YAML_BODY = b"model: {}\n"


class _FakeHub:
    """冒充 `hf_hub_download`：按需写假文件、记下每次调用的参数。

    `endpoint` 写成显式参数是为了让 `_hub_supports_endpoint()` 的签名探测
    看出"这个 hub 认 endpoint="——桩的行为要和真 hub 一致。
    """

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.fail_at: set[str] = set()

    def __call__(self, *, repo_id, filename, local_dir, force_download=False,
                 tqdm_class=None, endpoint=None, **kwargs):
        self.calls.append(
            {"repo_id": repo_id, "filename": filename, "local_dir": local_dir,
             "force_download": force_download, "endpoint": endpoint, **kwargs}
        )
        if endpoint in self.fail_at:
            raise ConnectionError(f"{endpoint} 连不上")
        dest = Path(local_dir) / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        body = _PT_BODY if dest.suffix == ".pt" else _YAML_BODY
        # 走一遍进度条：真 hub 是边下边 update 的
        bar = tqdm_class(total=len(body), unit="B")
        bar.update(len(body) // 2)
        bar.update(len(body) - len(body) // 2)
        dest.write_bytes(body)
        return str(dest)

    def force_flags(self) -> list[bool]:
        return [c["force_download"] for c in self.calls]


def _patch_head(monkey_targets: bool = True):
    """`_head` 换成"站点通、大小已知"，探测就不联网了。"""
    sizes = {t.remote: (512 if t.remote.endswith(".pt") else 10) for t in dm.targets(_ROOT)}

    def head(base, remote, timeout=dm._PROBE_TIMEOUT):
        return True, sizes.get(remote), ""

    dm._head = head


def test_download() -> None:
    hub = _hub()
    if hub is None:  # pragma: no cover - 环境问题
        _skipped.append("download")
        print("  - SKIP：这台机器没装 huggingface_hub")
        return

    tmp = scratch()
    saved_min, saved_head = dm._MIN_PT_BYTES, dm._head
    saved_hub_download = hub.hf_hub_download
    dm._MIN_PT_BYTES = 64          # 假权重只有几百字节
    fake = _FakeHub()
    hub.hf_hub_download = fake
    _patch_head()
    try:
        # 1) 全下：本地路径、参数、进度
        progress: list[tuple[float, str]] = []
        summary = dm.download(root=tmp, prefer=dm.HF_ENDPOINT,
                              progress=lambda f, m: progress.append((f, m)))

        eq("用的是探测到的站点", summary["endpoint"], dm.HF_ENDPOINT)
        eq("四个文件都下了", len(summary["downloaded"]), 4)
        eq("跳过 0 个", summary["skipped"], [])
        check("ok=true", summary["ok"] is True)
        eq("摘要里带上下的本地状态", summary["state"]["ok"], True)

        eq("hub 被调了四次", len(fake.calls), 4)
        eq("repo_id 对", {c["repo_id"] for c in fake.calls}, {dm.REPO_ID})
        eq("local_dir 是仓库根（仓库路径就是落盘路径）",
           {c["local_dir"] for c in fake.calls}, {str(tmp)})
        eq("filename 用的是仓库内路径",
           sorted(c["filename"] for c in fake.calls),
           sorted(t.remote for t in dm.targets(tmp)))
        eq("默认不 force", fake.force_flags(), [False] * 4)

        for target in dm.targets(tmp):
            want = _PT_BODY if target.local.suffix == ".pt" else _YAML_BODY
            eq(f"{target.remote} 落盘内容对得上", target.local.read_bytes(), want)

        check("进度从 0 走到 1（没倒退）",
              progress and progress[0][0] == 0.0 and progress[-1][0] == 1.0
              and all(b[0] >= a[0] for a, b in zip(progress, progress[1:])),
              f"{progress[:3]} … {progress[-3:]}")
        check("进度里带文件名", any("models/aim" in msg for _f, msg in progress), str(progress[-1]))

        # 2) 再跑一次：都在，一个字节都不下
        del fake.calls[:]
        again = dm.download(root=tmp, prefer=dm.HF_ENDPOINT, progress=lambda f, m: None)
        eq("第二次：没有下载", again["downloaded"], [])
        eq("第二次：四个都跳过", len(again["skipped"]), 4)
        eq("第二次：根本没问 hub", fake.calls, [])
        eq("第二次：字节数 0", again["bytes"], 0)

        # 3) --force：已有也重下，并且把 force_download 传下去
        dm.download(root=tmp, prefer=dm.HF_ENDPOINT, force=True, progress=lambda f, m: None)
        eq("force 时四个都重下", len(fake.calls), 4)
        eq("… force_download=True", fake.force_flags(), [True] * 4)
    finally:
        dm._MIN_PT_BYTES, dm._head = saved_min, saved_head
        hub.hf_hub_download = saved_hub_download
        shutil.rmtree(tmp, ignore_errors=True)


def test_download_failover() -> None:
    hub = _hub()
    if hub is None:  # pragma: no cover
        _skipped.append("failover")
        print("  - SKIP：这台机器没装 huggingface_hub")
        return

    tmp = scratch()
    saved_min, saved_head = dm._MIN_PT_BYTES, dm._head
    saved_hub_download = hub.hf_hub_download
    dm._MIN_PT_BYTES = 64
    fake = _FakeHub()
    hub.hf_hub_download = fake
    _patch_head()
    try:
        # 1) 探测时官方是通的，真下的时候官方挂了 → 自动换镜像，而不是整批失败
        fake.fail_at = {dm.HF_ENDPOINT}
        lines: list[str] = []
        summary = dm.download(root=tmp, progress=lambda f, m: None, log=lines.append)

        eq("官方下不动 → 改用镜像", summary["endpoint"], dm.MIRROR_ENDPOINT)
        eq("四个文件都下来了", len(summary["downloaded"]), 4)
        check("镜像站点也是 endpoint= 传下去的",
              all(c["endpoint"] == dm.MIRROR_ENDPOINT for c in fake.calls[1:]),
              str([c["endpoint"] for c in fake.calls]))
        check("说清楚换了站点", any(dm.MIRROR_ENDPOINT in ln for ln in lines), str(lines))
        check("文件真的落盘了",
              all(t.local.is_file() for t in dm.targets(tmp)), str(list(tmp.rglob("*.pt"))))

        # 2) 两家都下不动 → 一句人话 + 两家各自的原因
        tmp2 = scratch()
        fake2 = _FakeHub()
        fake2.fail_at = {dm.HF_ENDPOINT, dm.MIRROR_ENDPOINT}
        hub.hf_hub_download = fake2
        try:
            dm.download(root=tmp2, progress=lambda f, m: None)
            check("都下不动时抛 DownloadError", False)
        except dm.DownloadError as exc:
            check("都下不动时抛 DownloadError", True)
            check("… 消息里有官方站点", dm.HF_ENDPOINT in str(exc), str(exc))
            check("… 也有镜像站点", dm.MIRROR_ENDPOINT in str(exc), str(exc))
        finally:
            shutil.rmtree(tmp2, ignore_errors=True)
    finally:
        dm._MIN_PT_BYTES, dm._head = saved_min, saved_head
        hub.hf_hub_download = saved_hub_download
        shutil.rmtree(tmp, ignore_errors=True)


def test_download_errors() -> None:
    hub = _hub()
    if hub is None:  # pragma: no cover
        _skipped.append("errors")
        print("  - SKIP：这台机器没装 huggingface_hub")
        return

    tmp = scratch()
    saved_head = dm._head
    _patch_head()
    try:
        # 1) 点名一个仓库里没有的文件 → 说清楚有哪些可选，不是 KeyError
        try:
            dm.download(root=tmp, files=["models/nope.pt"])
            check("点名不存在的文件时抛 DownloadError", False)
        except dm.DownloadError as exc:
            check("点名不存在的文件时抛 DownloadError", "nope.pt" in str(exc), str(exc))

        # 2) 没装 huggingface_hub：提示 pip install，而不是甩一个 ImportError 栈
        saved_mod = sys.modules.get("huggingface_hub")
        saved_bool = dm._ENDPOINT_ARG
        dm._ENDPOINT_ARG = True      # 别让签名探测去碰真的 hub
        sys.modules["huggingface_hub"] = None  # import 它会抛 ImportError
        try:
            dm.download(root=tmp, progress=lambda f, m: None)
            check("没装 hub 时抛 DownloadError", False)
        except dm.DownloadError as exc:
            check("没装 hub 时抛 DownloadError", "huggingface_hub" in str(exc), str(exc))
            check("… 告诉用户怎么装", "pip install" in str(exc), str(exc))
        finally:
            if saved_mod is None:
                sys.modules.pop("huggingface_hub", None)
            else:
                sys.modules["huggingface_hub"] = saved_mod
            dm._ENDPOINT_ARG = saved_bool
    finally:
        dm._head = saved_head
        shutil.rmtree(tmp, ignore_errors=True)


def test_progress_bar() -> None:
    """进度条只是把数字转出去，自己不该往终端打任何东西。"""
    events: list[tuple[int, object]] = []
    bar_cls = dm._bar_class(lambda done, total: events.append((done, total)))
    bar = bar_cls(total=100, unit="B")
    bar.update(30)
    bar.update(70)
    eq("总大小透传", [e[1] for e in events], [100, 100])
    eq("已下字节累加", [e[0] for e in events], [30, 100])


# ═══════════════════════════════════════════════════════════════════════════
# 路由
# ═══════════════════════════════════════════════════════════════════════════


def test_routes() -> None:
    from flask import Flask

    from app.ai.api import bp as ai_bp
    from app.jobs import manager

    app = Flask(__name__)
    app.register_blueprint(ai_bp)
    client = app.test_client()

    saved_inspect = dm.inspect
    missing_state = {
        "ok": False, "repo": dm.REPO_ID, "root": str(_ROOT),
        "files": [], "missing": ["models/aim/x.pt"], "present_bytes": 0,
    }
    try:
        # 1) 缺文件：横幅要的那份状态
        dm.inspect = lambda root=None: dict(missing_state)
        r = client.get("/api/ai/models")
        eq("状态 → 200", r.status_code, 200)
        body = r.get_json()
        eq("… ok=false", body.get("ok"), False)
        eq("… 带上缺哪些", body.get("missing"), ["models/aim/x.pt"])
        eq("… 还没有下载任务", body.get("download"), None)
        eq("… 不缓存", r.headers.get("Cache-Control"), "no-store")

        # 2) 点按钮 → 排一个任务（没有 worker 就不会真的去下 665 MB）
        r = client.post("/api/ai/models/download")
        eq("下载 → 200", r.status_code, 200)
        job_id = r.get_json().get("job_id")
        check("… 返回 job_id", bool(job_id), str(r.get_json()))
        job = manager.get(job_id)
        check("… 任务进了队列", job is not None and job.state == "queued",
              job.state if job else "None")
        eq("… 任务类型是 models", job.kind, "models")
        eq("… 任务名给人看", job.label, "下载 AI 模型")

        # 3) 连点两下：返回同一个任务，不排第二个
        r = client.post("/api/ai/models/download")
        eq("再点一次返回同一个任务", r.get_json().get("job_id"), job_id)
        eq("… 并说明已经有一个在跑", r.get_json().get("already"), True)

        # 4) 状态接口把那个任务一起给出来（横幅读它的 step / progress）
        dm.inspect = lambda root=None: dict(missing_state)
        body = client.get("/api/ai/models").get_json()
        check("状态里带下载任务", isinstance(body.get("download"), dict), str(body.get("download")))
        eq("… 就是刚才那个", body["download"]["id"], job_id)

        # 5) 文件齐了：不排队
        dm.inspect = lambda root=None: {**missing_state, "ok": True, "missing": []}
        r = client.post("/api/ai/models/download")
        eq("齐了 → 200", r.status_code, 200)
        eq("齐了 → 不排任务", r.get_json().get("job_id"), None)
        eq("齐了 → already=true", r.get_json().get("already"), True)

        # 6) 显式 force：齐了也重下（换了版本想重新拉一遍）
        #    先把队列里那个模拟任务结束掉，否则"别排第二个"的守卫会先命中
        manager.get(job_id).state = "cancelled"
        r = client.post("/api/ai/models/download?force=1")
        forced = r.get_json().get("job_id")
        check("force=1 时排新任务", bool(forced) and forced != job_id, str(r.get_json()))
        eq("… 新任务也进了队列", manager.get(forced).state, "queued")
    finally:
        dm.inspect = saved_inspect
        manager.clear_finished()
        for job_id in [j["id"] for j in manager.list() if j["kind"] == "models"]:
            manager.get(job_id).state = "cancelled"


# ═══════════════════════════════════════════════════════════════════════════


def main() -> int:
    print("scripts/download_models + /api/ai/models 单元测试")
    for name, fn in [
        ("targets", test_targets),
        ("inspect", test_inspect),
        ("endpoints", test_endpoints),
        ("fallback to mirror", test_fallback),
        ("download", test_download),
        ("download failover", test_download_failover),
        ("download errors", test_download_errors),
        ("progress bar", test_progress_bar),
        ("routes", test_routes),
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
