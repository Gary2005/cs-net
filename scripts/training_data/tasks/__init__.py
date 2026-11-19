"""任务注册表：**一个任务 = `tasks/` 下的一个文件**。

    tasks/example.py    ← `--task example`
    tasks/mytask.py     ← `--task mytask`（你自己加的）
    tasks/_template.py  不是任务（下划线开头），是给你复制的空白模板
    tasks/base.py       不是任务，是契约本身（`Task` 基类）

**加一个任务只需要一步**：复制 `example.py`（或 `_template.py`）成 `tasks/mytask.py`，
改里面的类。不用动 `prepare.py` / `dataset.py` / 这个文件里的任何一行 —— 它们都是
扫这个目录发现任务的。

任务名和 import 是**惰性**的：`available()` 只 glob 文件名，只有真的用到某个任务才
import 它。所以一个任务文件写坏了，不会连累别的任务，也不会连累 `--todos`。
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

from .base import Task

#: 造数据目录里的元数据文件名（`out_dir/task.json`）。它记着这个目录是**哪个任务**
#: 造出来的，用来挡住"拿 example 的样本按 mytask 去读"这类静默错配。
TASK_META = "task.json"


def task_files() -> list[Path]:
    """所有任务文件的路径，按名字排序。以 `_` 开头的和下划线文件不算。"""
    here = Path(__file__).parent
    return sorted(
        p for p in here.glob("*.py") if not p.stem.startswith("_") and p.stem != "base"
    )


def available() -> list[str]:
    """所有可用的任务名（= 文件名去掉 `.py`）。"""
    return [p.stem for p in task_files()]


def get_task(name: str) -> Task:
    """按名字取任务实例（`tasks/<name>.py` 里的 `TASK`）。取不到就报错，不返回 None。"""
    if name not in available():
        names = available()
        if names:
            body = "\n".join(
                f"    {n:<12} {t.desc if (t := _peek(n)) else '（这个文件 import 失败）'}"
                for n in names
            )
            head = f"  可用任务（就是 tasks/<名>.py 的文件名）：\n{body}"
        else:
            head = "  tasks/ 下还没有任务文件。"
        raise ValueError(
            f"没有任务 {name!r}。\n{head}\n"
            f"  新任务：复制 tasks/example.py 成 tasks/{name}.py 再改"
        )

    mod = importlib.import_module(f"{__name__}.{name}")
    task = getattr(mod, "TASK", None)
    if not isinstance(task, Task):
        raise TypeError(
            f"tasks/{name}.py 里没有 `TASK = <你的类>()`（模块末尾那一行），"
            f"现在读到的是 {task!r}"
        )
    return task


def _peek(name: str) -> Task | None:
    """拿 desc 用；import 失败就返回 None（列可用任务不该因为一个坏文件而炸）。"""
    try:
        task = getattr(importlib.import_module(f"{__name__}.{name}"), "TASK", None)
    except Exception:
        return None
    return task if isinstance(task, Task) else None


def read_meta(out_dir: str | Path) -> dict:
    """读 `out_dir/task.json`。没有这个文件就返回 `{}`（手工拼的目录没有它）。"""
    p = Path(out_dir) / TASK_META
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except ValueError as e:
        raise ValueError(f"{p} 坏了（{e}）—— 删掉它，或者重新跑一次 prepare.py") from e


def recorded_task(out_dir: str | Path) -> str | None:
    """`out_dir` 是哪个任务造出来的。没有 `task.json`（或里面没记）就是 `None`。"""
    name = read_meta(out_dir).get("task")
    return str(name) if name else None


__all__ = [
    "Task",
    "TASK_META",
    "available",
    "get_task",
    "read_meta",
    "recorded_task",
    "task_files",
]
