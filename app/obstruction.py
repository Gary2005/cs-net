"""app 侧的整回合遮挡缓存入口。

几何、射线、稀疏构建全在 ``scripts/training_data`` 里 —— 那是造数和推理**共用
的唯一实现**，app 不复制一份（复制一份就有两个"遮挡"的定义，迟早对不上）。
这里只做两件事：

* 把 ``sys.path`` 配好（``scripts`` / ``tools`` 不是安装好的包）；
* 在 demo 层面按回合遍历，把进度折算成整个 demo 的百分比。

缓存落在 ``<csz>.obs/r<回合>.npz``，见 ``training_data/obstruction_cache.py``。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

#: 进度回调：``progress(0..1, 一句话)``。
Progress = Optional[Callable[[float, str], None]]


def ensure_round(
    csz_path: str | Path | None,
    rnd: dict,
    map_name: str,
    *,
    progress: Progress = None,
) -> np.ndarray:
    """一个回合的 ``(T,10,10) int8`` 遮挡：命中缓存就读，否则整回合重算并写回。"""
    from scripts.training_data import obstruction_cache
    from scripts.training_data.constants import N_PLAYERS

    return obstruction_cache.ensure(
        csz_path, rnd, map_name, N_PLAYERS, progress=progress
    )


def build_demo(
    csz_path: str | Path,
    data: dict,
    *,
    progress: Progress = None,
) -> dict[str, Any]:
    """整份 demo 的每个回合建一遍缓存（已经命中的回合只是读一次 npz）。

    每条回合单独兜异常：一个回合的数据坏了不该让整份 demo 的缓存建不起来，
    更不该让"提取"这一步失败 —— 缺缓存最多是下次用的时候慢一点。

    返回 ``{"rounds", "built", "cached", "failed", "map"}``：
    ``built`` 是这次真算了的回合数，``cached`` 是读到现成缓存的回合数。
    """
    map_name = str((data.get("meta") or {}).get("map") or "")
    rounds = data.get("rounds") or []
    say = progress or (lambda _frac, _msg: None)
    built = cached = 0
    failed: list[dict[str, str]] = []
    total = max(1, len(rounds))

    for i, rnd in enumerate(rounds):
        base = i / total
        span = 1.0 / total
        rid = int(rnd.get("id", i))
        say(base, f"缓存逐帧遮挡（回合 {i + 1}/{total}）…")
        try:
            before = _cached_round(csz_path, rnd, map_name)
            ensure_round(
                csz_path,
                rnd,
                map_name,
                progress=lambda frac, msg, _b=base, _s=span: say(
                    _b + _s * max(0.0, min(1.0, frac)), f"{msg}（回合 {i + 1}/{total}）"
                ),
            )
            if before:
                cached += 1
            else:
                built += 1
        except Exception as exc:  # noqa: BLE001 - 缓存失败不该让提取失败
            # 不逐条打印：一张没有地图几何的地图会让每个回合都失败一次，20 行
            # 噪音不如一行汇总（调用方把 `failed` 的长度打出来）。
            failed.append({"round": rid, "error": f"{type(exc).__name__}: {exc}"})

    say(1.0, "逐帧遮挡缓存完成")
    return {
        "map": map_name,
        "rounds": len(rounds),
        "built": built,
        "cached": cached,
        "failed": failed,
    }


def _cached_round(csz_path: str | Path, rnd: dict, map_name: str) -> bool:
    """这个回合现在有没有可用的缓存（只用来统计"命中/新建"）。"""
    from scripts.training_data import obstruction_cache

    return obstruction_cache.has(
        csz_path,
        int(rnd.get("id", 0)),
        len(rnd.get("ticks") or ()),
        map_name=map_name,
    )


def discard(csz_path: str | Path | None) -> None:
    """``.csz`` 被删时一起清掉它的遮挡缓存。

    这里**不复用** ``scripts/training_data/obstruction_cache.discard``：那条 import
    会连带把 ``scripts.training_data`` 的包级导入（demoparser2，约 1.5 s）拉进一次
    删除请求。目录名规则（``.csz`` → ``.obs``）就一行，两处都写着；
    ``test_ai.py::test_obstruction_cache`` 会断言删完目录真的没了，规则漂了会红。
    """
    if not csz_path:
        return
    d = Path(csz_path).with_suffix(".obs")
    if not d.is_dir():
        return
    for p in list(d.glob("*.npz")) + list(d.glob("*.tmp")):
        try:
            p.unlink()
        except OSError:
            pass
    try:
        d.rmdir()
    except OSError:
        pass        # 还有别人的文件 / 非空：留着，不致命
