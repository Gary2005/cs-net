#!/usr/bin/env python3
"""双击启动器（`app/desktop.py`）和打包自检（`app/selftest.py`）。

窗口本身没法在无头环境里测，所以这里量的是"窗口之前"和"窗口之后"的东西：

* 找空闲端口：拿到的端口必须真的能 bind（不能是 5000 那种被占的）
* 单实例记录：写进 `server.json` 的地址能被读回来；指向死端口的记录要能识别成
  "上次没退干净"并清掉 —— 否则玩家第二次双击会以为程序卡住
* `assets_summary()` 不能抛（窗口打开时就要用它，抛了窗口就打不开）
* **`--packaging-selftest` 真的能跑完**：子进程里起服务、打路由、import 重依赖，
  退出码 0。这一条是每次改动 `app/server.py` 之后最划算的回归测试。

Run:  python app/tests/test_desktop.py
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
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

_passed = 0
_failed: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    global _passed
    if cond:
        _passed += 1
        print(f"  \u2713 {name}")
    else:
        _failed.append(name)
        print(f"  \u2717 {name}" + (f"\n      {detail}" if detail else ""))


def eq(name: str, got: object, want: object) -> None:
    check(name, got == want, f"got {got!r}, want {want!r}")


def _scratch() -> Path:
    """见 `app/tests/test_paths.py::_scratch` 的说明（不用 mkdtemp 的原因）。"""
    base = _ROOT / ".build" / "test-tmp"
    base.mkdir(parents=True, exist_ok=True)
    path = base / f"desktop-{uuid.uuid4().hex[:12]}"
    path.mkdir()
    return path


def _load_desktop(data_dir: Path):
    """把 `app.desktop` 在一个指到临时目录的数据目录下导入。

    `app.config` 的模块级常量是 import 时算出来的，所以要先把环境变量设好、
    再清掉已经 import 的模块，才能让这一份"指到别处"。
    """
    os.environ["CSNET_DATA_DIR"] = str(data_dir)
    for name in [m for m in sys.modules if m == "app" or m.startswith("app.")]:
        del sys.modules[name]
    from app import desktop

    return desktop


def test_port_and_url() -> None:
    print("port / url")
    desktop = _load_desktop(_scratch())
    port = desktop.free_port("127.0.0.1")
    check("端口在合理区间", 1024 < port < 65536, str(port))
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", port))       # 真的能 bind 才算"空闲"
    eq("url", desktop.local_url("127.0.0.1", 5123), "http://127.0.0.1:5123/library")
    eq("0.0.0.0 显示成 127.0.0.1",
       desktop.local_url("0.0.0.0", 1), "http://127.0.0.1:1/library")
    # 记录里存的是库页面地址，健康检查必须回到 origin 的 /healthz，
    # 不能拼成 /library/healthz（那会把"在跑"误判成"上次没退干净"）。
    eq("healthz 换 origin",
       desktop._health_url("http://127.0.0.1:5/library"),
       "http://127.0.0.1:5/healthz")


def test_state_roundtrip() -> None:
    print("server.json / 单实例记录")
    data = _scratch()
    desktop = _load_desktop(data)

    eq("一开始没有记录", desktop._running_instance(), None)
    desktop._write_state("http://127.0.0.1:9/library")
    state = json.loads((data / "server.json").read_text(encoding="utf-8"))
    eq("写进去的地址", state["url"], "http://127.0.0.1:9/library")
    eq("pid 是当前进程", state["pid"], os.getpid())

    # 端口 9 上不会有服务 → 这条记录是"上次没退干净"，读的时候要清掉。
    eq("死记录 → None", desktop._running_instance(), None)
    check("死记录被删掉", not (data / "server.json").exists())

    desktop._write_state("http://127.0.0.1:9/library")
    desktop._clear_state()
    check("_clear_state 也能删", not (data / "server.json").exists())


def test_assets_summary() -> None:
    print("assets_summary / 窗口里那一行")
    desktop = _load_desktop(_scratch())
    text = desktop.assets_summary()
    check("返回的是字符串", isinstance(text, str), repr(text))
    check("提到了地图", "地图" in text, repr(text))


def test_ui_mode() -> None:
    print("界面形态 / 图标 + 托盘")
    desktop = _load_desktop(_scratch())

    # 图标：仓库里有 packaging/cs-net.png 就该读得出来；没有也该是 None 而不是抛异常
    # （图标缺失不该让程序起不来）。
    image = desktop._ui_icon()
    check("图标能读出来（或者干脆没有）", image is None or image.size[0] > 0, repr(image))
    check("show_tray 可调用", callable(desktop.show_tray))
    check("show_window 可调用", callable(desktop.show_window))

    # 三个界面开关要真的注册了：--help 的输出里能看到。
    env = dict(os.environ)
    env["CSNET_DATA_DIR"] = str(_scratch())
    proc = subprocess.run(
        [sys.executable, "-m", "app.desktop", "--help"],
        cwd=str(_ROOT), env=env, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=120,
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    eq("--help 退出码 0", proc.returncode, 0)
    for flag in ("--window", "--no-tray", "--no-window"):
        check(f"--help 里有 {flag}", flag in out, out[-400:])


def test_selftest_subprocess() -> None:
    print("--packaging-selftest / 端到端起一次")
    data = _scratch()
    env = dict(os.environ)
    env["CSNET_DATA_DIR"] = str(data)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(_ROOT), str(_ROOT / "tools"), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    proc = subprocess.run(
        [sys.executable, "-m", "app.desktop", "--packaging-selftest"],
        cwd=str(_ROOT), env=env, capture_output=True, text=True,
        # 中文 + GBK 控制台：子进程按 UTF-8 输出（PYTHONIOENCODING 管它），
        # 父进程不显式指定就会按 locale（GBK）解码，于是 UnicodeDecodeError。
        encoding="utf-8", errors="replace", timeout=600,
    )
    out = proc.stdout or ""
    check("退出码 0", proc.returncode == 0, (proc.stderr or "")[-1500:])
    check("打了 SELFTEST_JSON", "SELFTEST_JSON" in out, out[-800:])
    line = next((ln for ln in out.splitlines() if ln.startswith("SELFTEST_JSON")), "")
    if line:
        payload = json.loads(line[len("SELFTEST_JSON "):])
        eq("自检 ok", payload["ok"], True)
        check("至少 20 项检查", payload["total"] >= 20, str(payload["total"]))
        eq("没有失败项", payload["failed"], [])
    check("日志写进了临时数据目录", (data / "logs" / "desktop.log").is_file())


def main() -> int:
    for fn in (test_port_and_url, test_state_roundtrip, test_assets_summary,
               test_ui_mode, test_selftest_subprocess):
        print()
        fn()

    print()
    if _failed:
        print(f"FAILED {len(_failed)} / {_passed + len(_failed)}")
        for name in _failed:
            print(f"  - {name}")
        return 1
    print(f"all {_passed} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
