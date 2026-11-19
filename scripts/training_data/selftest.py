"""跑通检查：`python -m scripts.training_data --task example`

不依赖 torch 也能跑（`SampleDataset` 在没有 torch 时退化成普通类，只是不能进 DataLoader）。

体检三件事：

  1. csz 侧：文件能不能加载、格式对不对、每回合的 side-channel 有没有坏
  2. 任务侧：`tasks/<任务名>.py` 能不能取到、`n_samples()` / `read_sample()` 跑不跑得动
  3. 数据侧：`prepared/<task>/{train,valid,test}/` 三个 split 各有多少样本

最后把「`write_round` 里 `round_tensors()` 会给你什么」摆出来，那是你写任务时的原料。

`--todos` 只列出还没写的 TODO（跳过 csz 体检，秒回）：

    python -m scripts.training_data --todos
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

from ._paths import REPO_ROOT
from .constants import REQUIRED_FORMAT
from .dataset import HAS_TORCH, SampleDataset
from .features import round_events
from .prepare import SPLITS
from .store import CszStore
from .tasks import available, get_task, read_meta, task_files

# 输出用 ASCII 符号：Windows 控制台默认 GBK，⚠/✓ 会 UnicodeEncodeError。
_OK = "[OK]"
_WARN = "[!]"

#: 只匹配 TODO 的**标题行**（`# ── TODO 1 ── ...`），不匹配下面缩进的续行。
#: 续行里会出现"必须在 TODO 3 之前"这种引用，不能算一处待写。
_TODO_RE = re.compile(r"^\s*#\s*──\s*(TODO\s+\d+.*)$")


def _todo_sources() -> list[Path]:
    """要扫 TODO 的文件：框架两个 + 所有已注册的任务。

    `tasks/_template.py`（下划线开头）**不算** —— 它是给你复制的空白模板，
    里面的 TODO 不是"你还要写的"，复制成 `tasks/<名>.py` 之后才会被数到。
    """
    here = Path(__file__).parent
    return [here / "prepare.py", here / "dataset.py", *task_files()]


def _todo_report() -> int:
    """把还没写的 TODO 列出来，返回条数。"""
    print("\n-- 还没写的部分（TODO）--")
    here = Path(__file__).parent
    n = 0
    for path in _todo_sources():
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            m = _TODO_RE.match(line)
            if m:
                n += 1
                label = re.sub(r"\s*─+\s*", "  ", m.group(1)).strip()
                try:
                    shown = path.relative_to(here).as_posix()
                except ValueError:
                    shown = str(path)
                print(f"  {shown}:{lineno:<4} {label}")
    if n:
        print(f"  共 {n} 处。写完一处就删掉那处注释块和它下面的 raise。")
    else:
        print("  没有 TODO 了。")
    print(f"  新任务：复制 tasks/_template.py 成 tasks/<你的任务名>.py（名字就是 --task 的值）")
    return n


def _default_csz_dir() -> str:
    return str(REPO_ROOT / "data" / "demos" / "csz")


def _default_data_dir(task: str) -> str:
    return str(REPO_ROOT / "data" / "demos" / "prepared" / task)


def _selftest(csz_dir: str, data_dir: str, task: str) -> int:
    print(f"CSZ 目录 : {csz_dir}")
    print(f"任务     : {task}   (tasks/{task}.py；现有 {available()})")
    t0 = time.perf_counter()
    store = CszStore.from_dir(csz_dir)
    print(f"文件数   : {len(store.paths)}  (CszStore 构造 {time.perf_counter() - t0:.2f}s，尚未 load)")
    print(f"格式要求 : {REQUIRED_FORMAT}  (不是这个版本会在 load 时直接报错)")
    print(f"torch    : {('有 ' + __import__('torch').__version__) if HAS_TORCH else '缺失 -- 只测数据层'}")

    print("\n-- 逐文件体检 --")
    total_rounds = bad_rounds = 0
    for fi, p in enumerate(store.paths):
        t0 = time.perf_counter()
        d = store.data(fi)
        load_s = time.perf_counter() - t0
        print(f"\n[{fi}] {p.name}")
        print(f"    load={load_s:5.2f}s  map={store.map_name(fi)}  rounds={len(d['rounds'])}  "
              f"weapons={len(d['weapons'])}  places={len(d['places'])}")
        print(f"    format={d['format']}  players={len(d['players'])}")
        n_fire = sum(len(r.get("fire") or []) for r in d["rounds"])
        n_smoke = sum(len(r.get("smoke") or []) for r in d["rounds"])
        print(f"    fire/smoke 实例数（load 时已合并）: {n_fire} / {n_smoke}")
        for rid in store.round_ids(fi):
            total_rounds += 1
            why = store.round_health(fi, rid)
            if why:
                bad_rounds += 1
                print(f"    {_WARN} round {rid:>2}: {why}  -> 跳过")
    print(f"\n回合合计 : {total_rounds}，体检不合格 {bad_rounds}")
    st = store.merge_stats()
    print(f"fire/smoke : 合并前 {st['before']} 条 -> 合并后 {st['after']} 条"
          f"（改写 {st['fixed']} 条 change 的 mode，自检失败 {st['bad_rounds']} 个回合）")
    if bad_rounds:
        print(f"             造数据时这 {bad_rounds} 个必须跳过"
              f"（store.round_health(fi, rid) is not None）")
    else:
        print("             全部健康 —— 切碎的 fire/smoke 已在 load 时合并回一条，"
              "prepare.py 里那行 continue 不会滤掉任何回合（细节见 README §5.2）")

    print("\n-- 任务 --")
    try:
        t = get_task(task)
    except (ValueError, TypeError) as e:
        print(f"{_WARN} 取不到任务：{e}")
        t = None
    if t is not None:
        print(f"tasks/{task}.py  desc={t.desc!r}  ext={t.ext!r}  glob={t.pattern!r}")

    print("\n-- 读数据层 --")
    root = Path(data_dir)
    print(f"数据根目录: {root}")
    if not root.is_dir():
        print("跳过 —— 目录不存在。先造数据：")
        print(f"         python -m scripts.training_data.prepare --task {task}")
    else:
        try:
            meta = read_meta(root)
        except ValueError as e:
            meta = {}
            print(f"{_WARN} {e}")
        if meta:
            print(f"task.json : task={meta.get('task')!r}  seed={meta.get('seed')!r}  "
                  f"ratios={meta.get('ratios')!r}  created={meta.get('created')!r}")
        for split in SPLITS:
            d = root / split
            if not d.is_dir():
                print(f"  {split:6s}: {_WARN} 缺这个子文件夹 —— build() 没建？")
                continue
            try:
                ds = SampleDataset(d, task=task)
                n = len(ds)
            except Exception as e:
                print(f"  {split:6s}: {_WARN} {type(e).__name__}: {e}")
                continue
            print(f"  {split:6s}: {n:4d} 个样本（{len(ds.samples)} 个样本文件），"
                  f"例如 {ds.samples[0].name if ds.samples else '（空）'}")
            if n == 0:
                # 合法产物（整张图都分到别的 split 去了）。提醒一句，因为下一步
                # DataLoader(shuffle=True) 会报 num_samples=0，报错信息看不出原因。
                print(f"          {_WARN} 这个 split 是空的 —— 训练脚本里要跳过它，"
                      f"别直接丢给 DataLoader")
                continue
            if split == "train":  # 只试第一个 split
                try:
                    s = ds[0]
                except Exception as e:
                    print(f"          ds[0] 失败 {type(e).__name__}: {e}")
                    print(f"          -> 看 tasks/{task}.py 的 read_sample()")
                else:
                    if isinstance(s, dict):
                        for k, v in s.items():
                            shape = getattr(v, "shape", None)
                            dtype = getattr(v, "dtype", type(v).__name__)
                            print(f"          ds[0][{k!r}] {shape} {dtype}")
                    else:
                        print(f"          ds[0] -> {type(s).__name__}（不是 dict，"
                              f"collate 得自己管）")

    print("\n-- 任务里 round_tensors 能给你什么（示例：fi=0, rid=0）--")
    rnd = store.round(0, 0)
    print(f"round 键 : {sorted(rnd.keys())}")
    print(f"           ticks={len(rnd['ticks'])}  start_tick={rnd['start_tick']}  winner={rnd.get('winner')!r}")

    tensors = store.round_tensors(0, 0)
    print(f"round_tensors: {len(tensors)} 个 key")
    for k in ("players", "alive", "teams", "inventory", "spotted", "ticks"):
        print(f"    {k:10s} {str(list(tensors[k].shape)):16s} {tensors[k].dtype}")
    mb = sum(v.nbytes for v in tensors.values()) / 1e6
    print(f"    这一个回合合计 {mb:.1f} MB（4 个 demo 全部 89 回合实测共 817 MB）")

    print("    事件原始 vs 过滤后（round_events 会滤掉不属于本回合的，别直接遍历原始列表）")
    for kind in ("kills", "damage", "sound"):
        raw = len(rnd.get("events", {}).get(kind, []))
        kept = round_events(rnd, kind)
        flag = "" if raw == len(kept) else "   <- 有事件被滤掉，直接用原始列表会做错标签"
        print(f"    {kind:8s} {raw:4d} -> {len(kept):4d}{flag}")

    n_todo = _todo_report()
    if n_todo:
        print(f"\n{_OK} 数据层跑通 —— 还有 {n_todo} 处 TODO 等你写")
    else:
        print(f"\n{_OK} selftest 通过")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="scripts.training_data", description="CSZ 数据层 + 任务检查"
    )
    ap.add_argument(
        "--task", default="example", help=f"任务名 = tasks/<名>.py 的文件名。现有：{available()}"
    )
    ap.add_argument("--csz-dir", default=None, help="原始 csz 目录，默认 data/demos/csz")
    ap.add_argument(
        "--data-dir",
        default=None,
        help="造好的数据目录，默认 data/demos/prepared/<task>",
    )
    ap.add_argument("--todos", action="store_true", help="只列出还没写的 TODO，跳过 csz 体检")
    args = ap.parse_args(argv)
    if args.todos:  # 不碰 csz，秒回
        _todo_report()
        return 0
    return _selftest(
        args.csz_dir or _default_csz_dir(),
        args.data_dir or _default_data_dir(args.task),
        args.task,
    )


if __name__ == "__main__":
    raise SystemExit(main())
