"""造数据层：把 `.csz` 文件夹 dump 成 `SampleDataset` 能读的数据文件夹。

    python -m scripts.training_data.prepare --task example

**这个文件是框架，不用改。** "一个 .csz 要变成什么数据集"写在 `tasks/` 里 ——
一个任务一个文件，完整可跑的例子见 `tasks/example.py`，空白模板见 `tasks/_template.py`。

产出结构 —— `out_dir` 下按 split 分三个子文件夹，旁边放一个 `task.json` 记着是谁造的：

    data/demos/prepared/example/          ← out_dir（命令行默认就是 prepared/<task>）
    ├── task.json                           {"task": "example", "seed": 0, ...}
    ├── train/    de_mirage_f0_r3.npz  ...
    ├── valid/    ...
    └── test/     ...

读的时候参数是**具体的那个 split 子文件夹**（不是 `prepared`，也不是 `prepared/example`），
而且要传同一个任务名：

    ds = SampleDataset("data/demos/prepared/example/train", task="example")

分工
----
    tasks/<名>.py   ★ 你写：一个回合 dump 成什么（`write_round`）、怎么读回来
                    （`read_sample`）、一共几个样本（`n_samples`）、文件叫什么
    prepare.py      框架：取数、体检过滤、按 demo 划分 split、逐文件单遍、--overwrite 重跑
    dataset.py      框架：glob 样本文件，`__len__` / `__getitem__` 转发给任务

这是本目录里**唯一** import `store.py` / `demo_extract` 的地方（`tasks/*.py` 的
`write_round` 拿到的是框架传进去的 store，自己不 import）。训练侧只认磁盘上的样本。
**跑一次就行**，不要放进训练循环。

一个回合写几个样本文件，随你
----------------------------
`write_round` 每次只拿到一个 `path`，但**没有任何东西限制你写几个文件** ——
自己 `path.with_name(...)` 派生兄弟名字就行。这时：

* `n_samples(samples)` 用默认的 `len(samples)` 就自动对得上：读侧只认磁盘上的文件，
  根本不知道"回合"这回事；
* 下面 `[OK]` 报的数字是**样本文件数**（最后 glob 出来的），和 `len(SampleDataset(...))`
  永远一致 —— 它**不是** `write_round` 的调用次数。进度行里那个 `累计 N 回合` 才是。

**不要**用"调了几次 `write_round`"去推样本数：一个回合写 3 个文件时，回合数 20、
样本数 60。这条以前是错的（报 20 而磁盘上有 60），现在按磁盘数报。

为什么要分"造数据 / 读数据"两层：见 `dataset.py` 的 docstring。

为什么是"逐文件单遍 + 处理完就丢"（实测）
------------------------------------------
csz 是**一个 zstd 帧包住整个对象图**（`codec.loads()` 先 `read_bytes()` 整文件、
整帧解压、再解出整个 dict）。所以：

* **想知道一个文件里有哪些回合，就必须先把它整个 load 进内存。** 没有"只读清单"
  这种便宜路径 —— 所以"第一遍只数、第二遍再处理"等于把整个 csz 目录解析两遍。
* load 之后是 **Python 对象图**，膨胀得极凶：实测 4 个 csz 一共 44.1 MB，全部 load
  完常驻 **4233 MB** —— **×96**。
* 所以"先把所有回合列成清单再统一处理"（两段式）在 100 GB 上 = **9.6 TB 常驻**，
  必然 OOM。

`build()` 因此是**单遍**的：load 一个文件 → 处理它的所有回合 → `clear_cache()`
丢掉 → 下一个文件。实测峰值恒定 **1674 MB**（≈ 最大的那一个文件），和 csz 总量无关。

这条约束是**按进程**算的：`--workers N`（默认 4）就是 N 个进程各自单遍，互不通信
（划分只看文件名，谁算都一样）。所以峰值 ≈ **N × 1.7 GB**。本机实测（32 核、4 个 csz）：
`--workers 1` 152 s、`2` 107 s、`4` 72 s（≈2.1×），再加收益递减 —— 瓶颈是
`codec.load()` 造几百万个 Python float 的内存带宽。

⚠ 代价：**不能"先知道总数再按比例切"**。见下一条。

划分：一个 demo（= 一个 csz = 一张地图）整场进同一个 split
----------------------------------------------------------
划分落在**文件**上：一个 csz 里所有回合必然进同一个 split，不会被拆到两边。
哈希算在**文件名**上（不是 `file_idx`）：

    split = demo_split(store.paths[fi].name, seed=seed, train=train, valid=valid)

* **不能用 `file_idx`**：往 csz 目录里加文件会让下标整体平移，已经分好的 demo 会
  换 split，两次实验的 val loss 就没法比了。文件名是稳定的。
* **不能用回合**：同一个 demo 的相邻回合共享走位习惯、经济、对手打法。按回合随机
  划分会让 valid/test 的 loss **比真实泛化能力好看，而且不报任何错** —— 这种偏差
  最难查，因为它只让数字变好。按文件划分从根上没有这个问题。

三个好处：

1. **不需要知道总数**，所以能单遍流式跑（上面那个内存约束的直接后果）。
2. **可以多进程分片跑**：划分是纯函数，每个进程管一批文件、各算各的，不用通信。
   按比例随机切（`rng.permutation`）是全局排列，做不到这一点 —— 而单进程在
   100 GB 上要 ~28 小时。
   ⚠ 但**没有 resume**，所以每个进程必须写**自己的 `--out-dir`**（给一个只含这批
   文件的 `--csz-dir`，链接或复制都行），跑完把三个 split 子目录并起来。
   往同一个 `out_dir` 里并发写现在行不通：先到的会留下样本，后到的会被
   "目标非空"挡住；真加了 `--overwrite` 就会把别人的删掉。
3. **划分只跟文件名有关，跟内容无关**：可复现、可解释。

⚠ 已知取舍：粒度就是**一张地图**。实测这 4 个文件里有两张来自同一场比赛
（`2392136` 的 m2-nuke 和 m3-inferno），它们**可能被分到不同 split**。想按比赛
绑在一起，把哈希输入换成 `name.split("-", 1)[0]`（比赛 id）即可。

⚠ 现在只有 **4 个 demo**，0.9/0.05/0.05 切出来 **valid / test 大概率是空的**
（4 × 0.05 = 0.2 张图）。这是**预期行为，不是 bug** —— 现在的目标是跑通 pipeline；
等 csz 攒到几千个再在服务器上重跑，比例自然就准了。

空 split 是合法的：`out_dir/valid/` 建出来、里面一个文件都没有。读的时候
`SampleDataset(out_dir/'valid', task=...)` 不会报错、`len(ds) == 0`（见 `dataset.py`），
但训练脚本里要 `if len(ds) == 0: 跳过` —— `DataLoader(shuffle=True)` 不接受 0 个样本。

抽多少：`--count` / `--map-balance`
-----------------------------------
`--count N` = **只用 N 个 csz**。`N` 不小于目录里的文件数时这条不生效（等于全用）。
`N` 小于文件数时按 `seed` **随机抽** N 个 —— 同一个 `seed` + 同一个目录，抽中的是同一批，
可复现。它和 `--limit` 是两回事：`--limit` 是"排序后的**前** N 个"（不看内容、只给冒烟
测试用），`count` 是"**随机** N 个"。两个一起给 `build()` 会直接报错，不猜你想要哪个。

`--map-balance` = **按地图均分名额**，只在 `count < 文件数` 时生效。目录里有 K 张图
（按实际出现过的算，不写死 8），每张图先拿 `count // K` 个；某张图自己要是不够这个数
就**全拿**；这样通常拿不满 `count`，剩下的名额再从**还没被选中的文件**里随机补齐。
8 张图都在时，就是"每张图抽 count/8"。

⚠ **哪些地图参与**：默认只跑 `constants.MAP_TO_IDX` 里的那 8 张，判据就是
`map_name in MAP_TO_IDX` —— 和 `tasks/aim.py` 里 `MAP_TO_IDX.get(map_name)` 查不到就
`return` 一模一样，只是提前到了抽样阶段。其余地图（真实目录里混着 `de_Ancient` /
`de_cache_d` / `de_cbble_d` / `de_inferno_d_prefab` / `nip_aim_map`）**整个 csz 都不参与**，
连名额都不占：没有地图几何、坐标对不上 `MAP_CENTER`，`aim` 更是一条样本都不产出。
名字精确比较，`de_Ancient` 不是 `de_ancient`，照筛不误。

开跑前会把筛掉了哪几张图、共几个 csz 印出来。要改：

    --maps all                       全都要（旧行为）
    --maps de_anubis,de_mirage       只要这两张

为什么要有这一条：爬下来的比赛里地图很不均匀（热门图比冷门图多几倍），直接随机抽
`count` 个会让训练集被常打的图淹没。按图均分至少保证每张图都在里面。

⚠ 名额是**向下取整**的：`count` 比地图数还小时（比如 8 张图只抽 5 个），配额是
`5 // 8 = 0`，等于每张图一个都分不到，最后全靠"随机补" —— 这时平衡就退化成纯随机了。
开跑时会打一行 `[!]` 说清楚，别以为是 bug。

⚠ 这两条都要求**先知道每个 csz 是哪张图**，而地图名在 `meta` 里 —— 为这个把每个
csz 全量 load 一遍，等于把 30 小时变成 60 小时。所以走 `store.map_name_fast()`：
只解文件头的 `format` + `meta`，实测 **6 ms/个**（`peek_meta()`，见 `store.py`），
几千个文件几十秒。这就是 `count` 唯一多出来的成本。

开跑前会印出**每张图有几个 csz 参与计算**（`选中 / 该图总数`）；`task.json` 里则记下
`count` / `map_balance` / 每张图的 `used`+`available`，以及 `train` / `valid` / `test`
**各用了哪些 csz**（文件名列表 + 个数）和各自的最终样本数 —— 拿到产物就能复盘
"这份数据集到底是哪几个 csz 造的"。

同一个 csz 造多个任务
---------------------
任务名只影响"样本长什么样"，不影响划分：`demo_split()` 只吃文件名和 `seed`，所以
`--task example` 和 `--task mytask` 用同一个 `seed` 切出来的 train/valid/test **是
同一批 demo**（val loss 可以跨任务比）。别给不同任务用不同的 `seed`。

对应的，**每个任务写自己的 out_dir**（`prepared/<task>`），不要往同一个目录里写：
样本混在一起之后 `__len__` 和文件名都说不清是谁的。`task.json` 会挡住这种混写。

体检过滤（框架做的，任务不用管）
--------------------------------
`round_health()` 不合格的回合直接跳过、不写盘。实测现在 89 个回合**全是健康的**
（切碎的 fire/smoke 在 load 时已经合并回一条了，见 README §5.2），所以一个都不会滤掉。

留着这道闸是因为过滤是个便宜且必要的习惯：哪天切片器真出了修不好的碎，你不希望脏
标签被**永久写进磁盘** —— 那时连"原始 csz 还在，重抽一遍"这条退路都没了。

重跑：`--overwrite`，没有断点续跑
----------------------------------
**没有 `--resume`。** 目标目录里有上次的样本就拒绝开工，理由是人得对自己按下的键
负责 —— 一次误传 `--out-dir` 就删掉 1.9 TB 太容易了。要重跑，二选一：

    python -m scripts.training_data.prepare --task mytask --overwrite   # 删掉重写
    python -m scripts.training_data.prepare --task mytask --out-dir <新目录>

`--overwrite` 只删**这个工具自己认识的东西**：`train/` `valid/` `test/` 三个子目录和
`task.json`（外加早年那个 `.done/`）。**别的文件和目录一律不碰** —— 实测目录里混着
`notes.txt` 和 `extra_dir/keep.bin` 时，清完之后它们原样还在。所以就算把 `--out-dir`
误写成父目录 `prepared/`，删掉的也只是那个目录下的 `train/valid/test/task.json`，
碰不到各任务自己的子目录；而如果那个目录的 `task.json` 写的是别的任务，上面那道
检查会先拒绝。

⚠ 这意味着**跑 30 小时的任务打断了就是从零重来**。这是 2026-09 明确选择的取舍：
不用为"跳过已完成文件"维护一套断点标记（它得记 csz 名、任务名、任务代码指纹、
划分配置、样本数，还要处理"样本被手工删了但标记还在"这类对不上的情况）。
代价知情：解析 csz 占了整个任务 95% 以上的时间，所以重来就是重付那 95%。

真要能续跑，跑的时候按文件切片、每个进程一个 `--out-dir` 是更省事的办法
（划分只看文件名，多进程各算各的，见上面"划分"一节的第 2 条好处）。

不管有没有 `--overwrite`，有一条不变：**不要吞掉单文件解析错误**。哪个 csz 坏了/
格式不对就直接崩，最后一行进度就是崩溃点 —— 修好那个文件再整体重跑。

实测数字
--------
* `codec.load()` 一个 csz 要 **11–15 s**，写样本只占其中一小截（实测 4 个文件
  46.4 s 里写 89 个样本约 1–2 s，**2%~5%**）。所以这个任务的成本几乎全是"解析"。
  `CszStore` 保证一个进程只 load 一次，但 `clear_cache()` 之后再问同一个文件会
  **重新 load** —— 所以别在同一批里回头。
* 一个回合 14 个 key 合计 **平均 9.2 MB**，89 回合全 dump 是 **817 MB**。按这个
  比例，100 GB 的 csz 落盘后约 **1.9 TB** —— 先确认盘够。想省盘就只挑要用的 key
  （`tasks/<名>.py` 里的 `KEYS`），或者把 bool 数组用 `np.packbits` 压（numpy 的
  bool 是 **1 byte/elem**，不是 1 bit）。
* 落盘之后每个 epoch 都要把这份数据从磁盘读一遍（内存装不下，页缓存帮不上）。
  所以"存多少 key"直接等于"每 epoch 要搬多少字节"。
* npz vs `npy`+mmap：从一个 `(5410,10,20)` 的数组里取 T=64 的窗口，npz **2199 us**，
  mmap **4.3 us**（506×）—— 同一个文件要反复取不同片段就别用 npz。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Sequence

try:                                    # 进度条而已，不是依赖：装不上就安静地不画
    from tqdm import tqdm
except ImportError:                     # pragma: no cover
    tqdm = None

from ._paths import REPO_ROOT
from .constants import MAP_TO_IDX
from .store import CszStore
from .tasks import TASK_META, Task, available, get_task, recorded_task

#: split 名字就是子文件夹名。`dataset.py` / `selftest.py` 用的也是这三个。
SPLITS = ("train", "valid", "test")


def demo_split(
    name: str,
    *,
    seed: int = 0,
    train: float = 0.9,
    valid: float = 0.05,
) -> str:
    """把一个 demo（= 一个 csz = 一张地图）稳定地分到 train / valid / test。

    参数是**文件名**而不是 `file_idx`：csz 目录里多一个文件就会让所有下标平移，
    已经分好的 demo 会整体换 split，两次实验的 val loss 就没法比了。

    哈希是 `blake2b(f"{seed}:{name}")` 的前 64 位映射到 `[0, 1)`：

        u < train            -> train
        u < train + valid    -> valid
        否则                  -> test

    因为不需要知道总共有多少个 demo，`build()` 才能单遍流式跑；又因为它是纯函数，
    多个进程各管一批文件时各算各的就行，不用通信。换 `seed` 就换一套划分。

    `valid` 传 0 就只分 train / test；`train + valid == 1` 时 test 恒为空。

    ⚠ 只看文件名，**不看任务名** —— 所以同一份 csz、同一个 seed，不管跑哪个任务，
    切出来的 train/valid/test 都是同一批 demo。
    """
    digest = hashlib.blake2b(f"{seed}:{name}".encode("utf-8"), digest_size=8).digest()
    u = int.from_bytes(digest, "big") / 2**64
    if u < train:
        return "train"
    if u < train + valid:
        return "valid"
    return "test"


def _parse_maps(spec: str | Sequence[str] | None) -> set[str] | None:
    """`maps` 参数 → 允许参与的地图名集合，`None` = 不筛。

    默认（`spec is None`）就是 `MAP_TO_IDX` 的那 8 张。用户显式给的字符串只认两种：
         "all" / "*" / ""        → None（不筛）
         "de_anubis,de_mirage"   → {"de_anubis", "de_mirage"}

    名字**原样比较**，不转小写 —— 和 `MAP_TO_IDX` 的查表方式一样。
    """
    if spec is None:
        return set(MAP_TO_IDX)
    if isinstance(spec, str):
        text = spec.strip()
        if text.lower() in {"", "all", "*"}:
            return None
        items: Sequence[str] = text.split(",")
    else:
        items = spec
    out = {str(s).strip() for s in items if str(s).strip()}
    if not out:
        raise ValueError("maps 给了一个空列表 —— 要「不筛」请显式写 all")
    return out


def _map_index(
    store: CszStore, allow: set[str] | None
) -> tuple[dict[str, list[int]], dict[str, int]]:
    """地图名 → 该图的 csz 下标（升序），按地图名排序；同时返回被筛掉的地图。

    只走 `CszStore.map_name_fast()`（= 解文件头，实测 6 ms/个），**不 load 整个文件**。
    它返回 `meta["map"]` 的原样字符串，和 `allow` **精确比较**：`de_Ancient` 不在
    `MAP_TO_IDX` 里，就整个 csz 跳过，和 `tasks/aim.py` 里查不到 `map_id` 就 `return`
    是同一条判据。

    Args:
        allow: 允许的地图名集合；`None` = 不筛（所有出现过的图都要）。

    Returns:
        `(kept, dropped)`：`kept` 是 `{地图: [fi, ...]}`；`dropped` 是
        `{地图: 个数}` —— 这些 csz **一个都不参与**，也不占名额。
    """
    kept: dict[str, list[int]] = {}
    dropped: dict[str, int] = {}
    for fi in range(len(store.paths)):
        name = store.map_name_fast(fi)
        if allow is not None and name not in allow:
            dropped[name] = dropped.get(name, 0) + 1
        else:
            kept.setdefault(name, []).append(fi)
    return dict(sorted(kept.items())), dict(sorted(dropped.items()))


def _pick_files(
    pool: list[int],
    maps: dict[str, list[int]],
    *,
    count: int | None,
    seed: int,
    map_balance: bool,
) -> list[int]:
    """从候选池 `pool` 里挑出这次要用哪几个 csz，返回**升序**的下标。

    `pool` 是已经过地图筛选的候选（升序），`maps` 是它按地图分好的组。

    * `count` 为 None 或 >= 候选数 → 全用（这条参数不生效，见模块 docstring）。
    * `count < 候选数` → 从 `pool` 里随机抽 `count` 个。
    * `map_balance=True` → 每张图先拿 `count // 图数`（向下取整），图纸本身不够就全拿，
      不够 `count` 的名额从**还没选中的**文件里随机补。`count < 图数` 时配额是 0，
      于是退化成"随机补满"（`_print_map_table()` 会打 `[!]` 提醒）。

    随机数用 `random.Random(f"csz-count:{seed}")`：字符串种子的哈希是稳定的
    （不依赖 `PYTHONHASHSEED`），所以**同 seed + 同目录 = 同一批文件**，两次实验的
    val loss 才能比。`maps` 是按地图名排好序的 dict，遍历顺序固定，抽的序也就固定。
    """
    n = len(pool)
    if count is None or int(count) >= n:
        return list(pool)

    want = int(count)
    rng = random.Random(f"csz-count:{seed}")
    if not map_balance or not maps:
        return sorted(rng.sample(pool, want))

    quota = want // len(maps)          # 每张图 count/图数；图纸不够就全拿
    picked: list[int] = []
    for fis in maps.values():
        picked.extend(fis if len(fis) <= quota else rng.sample(fis, quota))

    if len(picked) < want:             # 有的图不够 quota → 名额没用完，从剩下的里随机补
        have = set(picked)
        rest = [fi for fi in pool if fi not in have]
        picked.extend(rng.sample(rest, min(want - len(picked), len(rest))))
    return sorted(picked)


def _print_map_table(
    maps: dict[str, list[int]],
    dropped: dict[str, int],
    used: set[int],
    *,
    n_files: int,
    n_candidates: int,
    limit: int | None,
    count: int | None,
    map_balance: bool,
) -> None:
    """开跑前把"每张图有几个 csz 参与计算"印出来（`选中 / 该图候选数`）。

    走 **stdout**，和每个文件那行总结一起留在日志里。为什么要在开跑前印：跑完才
    发现"某张图一个样本都没有"就白等 30 小时了 —— 一眼能看出 sampling 是不是抽歪了。

    被筛掉的非标准地图（`de_Ancient` / `nip_aim_map` …）单独列一行：它们**不占名额**，
    但必须让人看见"我确实扔了这些东西"，而不是悄悄少掉几个 csz。
    """
    if limit is not None:
        mode = f"limit={int(limit)}，排序后取前 N 个"
    elif count is not None and int(count) < n_candidates:
        mode = f"count={int(count)}" + ("，按地图平衡" if map_balance else "，随机抽")
    elif count is not None:
        mode = f"count={int(count)} ≥ 候选数 {n_candidates}，全用"
    else:
        mode = "全部"

    noise = f"（另有 {len(dropped)} 张非标准图当噪声筛掉）" if dropped else ""
    print(
        f"[地图] {len(maps)} 张图参与{noise}；参与计算 {len(used)}/{n_candidates} "
        f"个候选 csz（{mode}），目录里一共 {n_files} 个",
        flush=True,
    )
    width = max((len(m) for m in maps), default=0)
    for name, fis in maps.items():
        n = sum(1 for fi in fis if fi in used)
        print(f"       {name:<{width}}  {n:>5} / {len(fis)}", flush=True)

    if dropped:
        n_noise = sum(dropped.values())
        print(
            f"[!] 另有 {len(dropped)} 张非标准地图共 {n_noise} 个 csz 当噪声跳过"
            f"（要全要：--maps all；只挑几张：--maps de_anubis,de_mirage）：",
            flush=True,
        )
        w = max(len(m) for m in dropped)
        for name, n in dropped.items():
            print(f"       {name:<{w}}  {n:>5}", flush=True)

    if map_balance and count is not None and 0 <= int(count) < len(maps):
        # 每张图的配额是 count // 图数（向下取整）= 0，等于条目一个都分不到，
        # 最后全靠"随机补"拿满 —— 说清楚，别让人以为平衡没生效。
        print(
            f"[!] count={int(count)} 小于地图数 {len(maps)}：每张图的配额 "
            f"{int(count)} // {len(maps)} = 0，按地图平衡退化成了随机抽",
            flush=True,
        )


#: 每个**进程**一个 `CszStore`。它只是"目录里有哪些 csz"这份清单 + 懒加载内容，
#: 建起来是毫秒级；留着它，一个 worker 连着跑好几个文件时不用重复扫目录。
_WORKER_STORES: dict[str, CszStore] = {}


def _worker_store(csz_dir: str) -> CszStore:
    store = _WORKER_STORES.get(csz_dir)
    if store is None:
        store = _WORKER_STORES[csz_dir] = CszStore.from_dir(csz_dir)
    return store


def _run_file(
    csz_dir: str, out_dir: str, task: str, fi: int, split: str
) -> tuple[str, str, int, int, float]:
    """处理**一个** csz：load → 每个回合写样本 → `clear_cache()`。

    返回 `(文件名, split, 写的回合数, 体检滤掉, 耗时秒)`。

    必须是**模块级**函数、只收可 pickle 的参数 —— `workers > 1` 时它跑在
    `ProcessPoolExecutor` 的子进程里。每个子进程同一时刻只 load 一个 csz
    （~1.7 GB），所以内存 ≈ `workers × 1.7 GB`，和 csz 总量无关。
    """
    t0 = time.perf_counter()
    store = _worker_store(csz_dir)
    t = get_task(task)
    name = store.paths[fi].name
    n_round = n_bad = 0
    for rid in store.round_ids(fi):
        if store.round_health(fi, rid) is not None:
            n_bad += 1
            continue  # 体检不合格的回合不写进磁盘（理由见模块 docstring）
        path = Path(out_dir) / split / t.sample_name(store, fi=fi, rid=rid)
        t.write_round(store, fi=fi, rid=rid, split=split, path=path)
        n_round += 1
    store.clear_cache()  # ← 丢掉这个文件的 ~1 GB
    return name, split, n_round, n_bad, time.perf_counter() - t0


def _make_bar(total: int):
    """一个 csz 一格。没装 tqdm 就返回 None（造数据照跑，只是没有进度条）。"""
    if tqdm is None or total <= 0:
        return None
    # disable=None：stderr 不是终端（重定向到日志）时自动关掉，不往日志里刷
    return tqdm(total=total, desc="csz", unit="csz", dynamic_ncols=True, disable=None)


def build(
    csz_dir: str | Path,
    out_dir: str | Path,
    *,
    task: str,
    train: float = 0.9,
    valid: float = 0.05,
    test: float = 0.05,
    seed: int = 0,
    overwrite: bool = False,
    limit: int | None = None,
    count: int | None = None,
    map_balance: bool = False,
    maps: str | Sequence[str] | None = None,
    workers: int = 4,
) -> dict[str, int]:
    """csz 文件夹 → `out_dir/{train,valid,test}/`。返回每个 split 里有多少个样本。

    Args:
        csz_dir: 原始 `.csz` 目录。
        out_dir: 产物根目录，三个子文件夹建在这下面。**一个任务一个目录**。
        task: 任务名，就是 `tasks/<名>.py` 的文件名（`tasks/example.py` → `"example"`）。
            它决定一个回合怎么写成一个样本。
        train, valid, test: 三个 split 的比例，**必须加起来等于 1**，可以给 0。
        seed: 划分的哈希种子。**别省** —— 没有它，每跑一次 build 就换一套
            train/valid，两次实验的 val loss 根本没法比。换任务时**别换 seed**。
            它同时决定 `count` 随机抽的是哪几个 csz（同 seed + 同目录 = 同一批）。
        overwrite: 重跑。先删掉 `out_dir` 下上次的产物（三个 split 子目录 + `task.json`）
            再从头写。默认 False = 目标目录非空就拒绝，免得手滑删掉一天半的产物。
        limit: 只处理**排序后的前 N 个** csz（默认全部）。给冒烟测试和"先跑一小批看看
            产出对不对"用 —— 一个 csz 要 11–15 s，全量 30 小时，先跑 2 个再全跑很划算。
            划分只跟文件名有关，所以前 N 个的 split 和全量跑时**完全一致**，不是随机抽样。
            **不能和 `count` 同时给**（一个是"前 N 个"，一个是"随机 N 个"，别猜）。
        count: 只用 **N 个** csz（默认全部）。`N >= 文件数` 时这条不生效；`N <` 时按
            `seed` 随机抽 N 个。为什么要有它：全量跑是 30 小时 + 1.9 TB，先按 `count`
            抽一小批把 loss 曲线和 pipeline 都跑通，再决定值不值得全量。
        map_balance: 按**地图**均分名额，**只在 `count < 文件数` 时生效**。目录里有 K 张
            图 → 每张图抽 `count // K` 个；某张图不够就全拿；还差的名额从没被选中的
            文件里随机补齐。地图不均是真实存在的（热门图比冷门图多几倍），直接随机抽
            会被热门图淹没。它需要每张图是谁 —— 走 `store.map_name_fast()`
            （只解文件头，6 ms/个），不会为了地图名把 csz 全量 load 一遍。
            ⚠ `count < K` 时配额 `count // K` 是 0，平衡退化成纯随机（会打 `[!]` 提醒）。
        maps: 哪些地图的 csz 才参与。默认 `None` = 只跑 `constants.MAP_TO_IDX` 里的
            那 8 张（`name in MAP_TO_IDX`，判据同 `tasks/aim.py`），别的地图
            （`de_Ancient` / `de_cache_d` / `nip_aim_map` …）**整个 csz 都不参与**，
            连名额都不占。名字精确比较。给 `"all"` = 不筛；给
            `"de_anubis,de_mirage"` 或名字列表 = 只跑这几张。
        workers: 并行处理几个 csz（默认 4，`1` = 单进程）。每个 worker 同时只 load 一个
            csz，**内存 ≈ workers × 1.7 GB**；每个 csz 写自己的文件、各算各的 split，
            互不通信。`--limit` / `--count` 小于 workers 时多出来的 worker 空着。

    `count` / `map_balance` 选出来的这批 csz、以及每张图用了几个、哪些图被当噪声筛掉，
    都会写进 `task.json`（`train` / `valid` / `test` 各自用了哪些 csz 也在里面），
    开跑前还会把每张图的数量印到 stdout。

    **没有断点续跑** —— 被打断就是从零重来，理由和取舍见模块 docstring 的"重跑"一节。

    单个 worker 一次只 load 一个 csz（峰值内存 ≈ 最大的那个文件，实测 1674 MB）。
    进度条是"跑完几个 csz"（tqdm，默认画在 stderr；不是终端时自动关掉），
    每个文件跑完另外往 stdout 打一行总结。
    """
    # ── 校验 + 建目录 ──────────────────────────────────────────────────────
    ratios = {"train": train, "valid": valid, "test": test}
    for k, r in ratios.items():
        if not 0.0 <= r <= 1.0:
            raise ValueError(f"{k} 比例要在 0..1 之间，现在是 {r!r}")
    total = sum(ratios.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"三个比例要加起来等于 1，现在 train+valid+test = "
            f"{train} + {valid} + {test} = {total!r}"
        )
    if limit is not None and count is not None:
        raise ValueError(
            f"limit={limit} 和 count={count} 别同时给：\n"
            f"  limit  排序后的**前** N 个（冒烟测试用，不看内容）\n"
            f"  count  **随机**抽 N 个（配合 map_balance 按地图均分）"
        )
    if count is not None and int(count) < 0:
        raise ValueError(f"count 要 >= 0，现在是 {count!r}")
    out_dir = Path(out_dir)

    # 任务名先解析 —— 写错任务名不该等到第一个回合才发现
    t = get_task(task)

    # 烟雾凸包占造数据的大头（每颗烟每个 journal 记录重建一次）。装了 scipy 走 Qhull，
    # 没装就回落到纯 Python，**输出完全一样但慢 ~40×**。这种"结果对、只是慢"的差别
    # 最难发现，所以开跑前打一行：没装的话这里会明说。
    from demo_extract import voxel as _voxel

    if _voxel.HULL_BACKEND == "qhull":
        import scipy

        print(f"[烟雾] 凸包后端 qhull（scipy {scipy.__version__}）", flush=True)
    else:
        print(
            "[烟雾] ⚠ 凸包后端 = 纯 Python（没装 scipy）：结果一样但慢 ~40×。\n"
            "       想要快的：pip install scipy",
            flush=True,
        )

    # `task.json` 记着这个目录是哪个任务造的。挡住"把 A 的样本当 B 往里写"。
    recorded = recorded_task(out_dir)
    if recorded is not None and recorded != task:
        raise FileExistsError(
            f"{out_dir} 是任务 {recorded!r} 造出来的，现在要往里写 {task!r} 的样本。\n"
            f"  一个任务一个 out_dir（默认就是 prepared/<task>）：--out-dir {out_dir}-{task}\n"
            f"  确认要混在一起：先删掉 {out_dir / TASK_META}"
        )

    # 只数**文件**，不数目录 —— 这样空跑一次（上面那个 raise 之前已经把三个空目录
    # 建出来了）不会让第二次调用误判成"上次的残留"。
    existing = _count_samples(out_dir)
    if existing and not overwrite:
        raise FileExistsError(
            f"{out_dir} 下已经有 {existing} 个样本文件了 —— 会和这次的混在一起。\n"
            f"  删掉重写：--overwrite\n"
            f"  写到别处：--out-dir {out_dir}-2（推荐，旧的那份留着还能对比）"
        )
    if overwrite:
        # 只删这个工具自己认识的东西（三个 split 子目录 + task.json）。上面已经确认过
        # `task.json` 要么不存在、要么写的就是当前任务，所以删不到别人的目录。
        n_old = _wipe(out_dir)
        if n_old:
            print(
                f"[!] --overwrite：清掉 {out_dir} 下原有的 {n_old} 个样本文件，从头写",
                flush=True,
            )
    for split in SPLITS:
        (out_dir / split).mkdir(parents=True, exist_ok=True)

    store = CszStore.from_dir(csz_dir)     # 父进程只用它的 paths 列表（不 load 内容）
    n_files = len(store.paths)
    workers = max(1, int(workers))

    # 每张图各有哪些 csz。只解文件头（`peek_meta()`，6 ms/个），**不 load 整个文件** ——
    # 这里还在父进程、还没开始跑，为地图名全量 load 一遍等于把 30 小时变成 60 小时。
    # `maps=None` → 只跑 MAP_TO_IDX 里的 8 张，别的地图（de_Ancient / nip_aim_map …）
    # 整个 csz 都不参与，理由见 `_parse_maps()` 的 docstring。
    allow = _parse_maps(maps)
    map_groups, dropped = _map_index(store, allow)
    pool = sorted(fi for fis in map_groups.values() for fi in fis)
    if not pool:
        # 一个候选都没有：要么 --maps 写错了，要么这批 csz 的地图全不在 MAP_TO_IDX 里。
        # 继续跑只会产出空数据集，还得等下一次才发现，所以直接报错。
        raise ValueError(
            f"{csz_dir} 里没有一个 csz 的地图在允许列表里。\n"
            f"  允许：{'、'.join(sorted(allow)) if allow else '（不筛）'}\n"
            f"  目录里出现："
            + "、".join(f"{k}({v})" for k, v in dropped.items())
            + "\n  要么加 --maps all，要么把缺的图补进 constants.MAP_CENTER / MAP_TO_IDX"
        )

    # 这次到底用哪几个 csz。`n_files` 记**目录里一共几个**，`files_used` 记这次真跑了
    # 几个 —— 否则只跑 2 个的产物里写着 n_files=4，看的人会以为样本丢了。
    if limit is not None:
        picked = pool[: min(len(pool), max(int(limit), 0))]
    else:
        picked = _pick_files(pool, map_groups, count=count, seed=seed, map_balance=map_balance)
    used = set(picked)

    _print_map_table(
        map_groups, dropped, used,
        n_files=n_files, n_candidates=len(pool),
        limit=limit, count=count, map_balance=map_balance,
    )

    n_round = 0  # 写出去的**回合**数 —— 一个回合可以写多个样本文件，所以它可能 ≠ 样本数

    # 划分只跟文件名有关，跟 load 出来的内容无关，所以先算好（父进程算，worker 只干活）
    jobs = [
        (fi, demo_split(store.paths[fi].name, seed=seed, train=train, valid=valid))
        for fi in picked
    ]

    meta = _build_meta(
        task=task, t=t, seed=seed, ratios=ratios, n_files=n_files,
        n_candidates=len(pool), maps=allow, dropped=dropped,
        count=count, map_balance=map_balance, map_groups=map_groups, used=used, jobs=jobs,
        names=[p.name for p in store.paths],
    )
    _write_meta(out_dir, meta)

    bar = _make_bar(len(jobs))

    def say(text: str) -> None:
        # 进度条在画的时候把总结行"顶"到条上方，别让两行互相覆盖
        if bar is not None:
            bar.write(text)
        else:
            print(text, flush=True)

    def report(res: tuple[str, str, int, int, float]) -> None:
        nonlocal n_round
        name, split, n_file, bad, secs = res
        n_round += n_file
        if bar is not None:
            bar.update(1)
            bar.set_postfix_str(f"{n_file} 回合 {secs:.1f}s")
        say(
            f"{name}  {split:<5}  rounds={n_file:<3} 体检滤掉 {bad:<3} {secs:5.1f}s  "
            f"累计 {n_round} 回合"
        )

    try:
        if workers == 1 or len(jobs) <= 1:
            for fi, split in jobs:                       # 单进程：好调试，栈是干净的
                report(_run_file(str(csz_dir), str(out_dir), task, fi, split))
        else:
            with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as ex:
                futs = [
                    ex.submit(_run_file, str(csz_dir), str(out_dir), task, fi, split)
                    for fi, split in jobs
                ]
                for fut in as_completed(futs):
                    try:
                        report(fut.result())
                    except BaseException:                # 一个文件崩了就别再跑剩下的
                        for f in futs:
                            f.cancel()
                        raise
    finally:
        if bar is not None:
            bar.close()

    # 最后数一遍磁盘：报出去的必须是**样本文件数**，和 `SampleDataset.__len__`
    # 用同一套 glob，保证两边永远一致。（进度行里的"回合"数是另一回事。）
    counts = {s: _count_samples(out_dir, s) for s in SPLITS}
    # 样本数只有跑完才知道，所以 `task.json` 写第二遍（第一遍在开跑前，那时先记下
    # 选了哪些 csz）。两次写的是同一个 dict，别在中间改别的字段。
    for split, n in counts.items():
        meta["splits"][split]["n_samples"] = n
    _write_meta(out_dir, meta)
    return counts


def _count_samples(out_dir: Path, split: str | None = None) -> int:
    """样本**文件**数。`split=None` 就数三个 split 加起来的。

    这是"样本数"的唯一定义 —— `dataset.py` 那边 `len(ds)` 数的也是它（同一套 glob）。
    **不能**用"调了几次 `write_round`"来数：一个回合可以写多个样本文件。
    """
    dirs = SPLITS if split is None else (split,)
    return sum(1 for s in dirs for p in (out_dir / s).glob("*") if p.is_file())


def _wipe(out_dir: Path) -> int:
    """删掉这个工具自己的产物，返回删掉的文件数。

    只认四个名字：`train/` `valid/` `test/` `task.json`。**不碰别的东西** ——
    这样即使 `--out-dir` 写成了别的目录，也不会把不认识的文件删掉。
    （早年还有个 `.done/` 断点标记目录，一起清掉。）
    """
    n = 0
    for name in (*SPLITS, ".done"):
        d = out_dir / name
        if d.is_dir():
            n += sum(1 for p in d.rglob("*") if p.is_file())
            shutil.rmtree(d)
    meta = out_dir / TASK_META
    if meta.is_file():
        meta.unlink()
    return n


def _build_meta(
    *,
    task: str,
    t: Task,
    seed: int,
    ratios: dict[str, float],
    n_files: int,
    n_candidates: int,
    maps: set[str] | None,
    dropped: dict[str, int],
    count: int | None,
    map_balance: bool,
    map_groups: dict[str, list[int]],
    used: set[int],
    jobs: list[tuple[int, str]],
    names: list[str],
) -> dict:
    """拼出 `task.json` 的内容。**写盘在 `_write_meta()`**，开跑前后各写一次。

    为什么要有"用了哪些 csz"这一节：`--count` / `--map-balance` 是随机抽的，光看
    样本文件名（`de_mirage_f3_r7.npz`，只有 `f{fi}`）回不到"这批数据是抽了哪 8 个
    csz"。半年后想复现/解释一次 val loss，这份记录就是唯一的线索。

    里面有什么：
        task / desc / ext       谁造的（`recorded_task()` 读的就是 task）
        seed / ratios           划分用的输入
        n_files                 目录里一共有几个 csz
        n_candidates            过了地图筛选、有资格参与的候选数
        files_used              这次真用了几个
        count / map_balance     抽样参数原样记下来（None = 没给）
        map_filter              允许的地图；None = 不筛（`--maps all`）
        maps                    {地图: {"used": 参与个数, "available": 该图候选数}}
        maps_excluded           {地图: 个数} —— 被当噪声筛掉的非标准地图
        splits                  {train/valid/test: {"n_csz", "n_samples", "csz": [...]}}
                                `n_samples` 先写 None，跑完再补（见 build()）
    """
    by_split: dict[str, list[str]] = {s: [] for s in SPLITS}
    for fi, split in jobs:
        by_split[split].append(names[fi])       # jobs 已经是升序，所以这里天然是有序的

    return {
        "task": task,
        "desc": t.desc,
        "ext": t.ext,
        "seed": seed,
        "ratios": ratios,
        "n_files": n_files,
        "n_candidates": n_candidates,
        "files_used": len(jobs),
        "count": None if count is None else int(count),
        "map_balance": bool(map_balance),
        "map_filter": None if maps is None else sorted(maps),
        "maps": {
            name: {"used": sum(1 for fi in fis if fi in used), "available": len(fis)}
            for name, fis in map_groups.items()
        },
        "maps_excluded": dropped,
        "splits": {
            s: {"n_csz": len(by_split[s]), "n_samples": None, "csz": by_split[s]}
            for s in SPLITS
        },
        "created": datetime.now().isoformat(timespec="seconds"),
    }


def _write_meta(out_dir: Path, meta: dict) -> None:
    """把 `meta` 写成 `out_dir/task.json`。给 `SampleDataset` 校验任务名用，人也能看。

    调两次：开跑前（记下选了哪些 csz），跑完（补上每个 split 的样本数）。
    中间崩了也不会留下"半个目录说不清来源"的状态 —— 第一遍已经写了抽样信息。
    """
    (out_dir / TASK_META).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    names = available()
    ap = argparse.ArgumentParser(
        prog="scripts.training_data.prepare",
        description="csz → 训练样本（按 demo 分 split；一个回合怎么写由任务决定）",
        epilog=(
            "任务就是一个文件（tasks/<名>.py），新任务复制 tasks/example.py 改。\n"
            + (
                "现有任务：" + "、".join(names)
                if names
                else "tasks/ 下还没有任务文件。"
            )
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--task", required=True, help=f"任务名 = tasks/<名>.py 的文件名。现有：{names or '无'}"
    )
    ap.add_argument("--csz-dir", default=str(REPO_ROOT / "data" / "demos" / "csz"))
    ap.add_argument(
        "--out-dir",
        default=None,
        help="产物根目录，默认 data/demos/prepared/<task>（一个任务一个目录）",
    )
    ap.add_argument("--train", type=float, default=0.9, help="训练集比例（默认 0.9）")
    ap.add_argument("--valid", type=float, default=0.05, help="验证集比例（默认 0.05）")
    ap.add_argument("--test", type=float, default=0.05, help="测试集比例（默认 0.05）")
    ap.add_argument("--seed", type=int, default=0, help="划分哈希种子（默认 0）")
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="重跑：先删掉 out_dir 下上次的产物（三个 split 子目录 + task.json）再从头写；"
        "默认目标非空就拒绝（不想覆盖就换 --out-dir）",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="只处理排序后的前 N 个 csz（默认全部）。先跑一小批看看产出对不对时用；"
        "和 --count 二选一（这个不看内容，按顺序取）",
    )
    ap.add_argument(
        "--count",
        type=int,
        default=None,
        metavar="N",
        help="只用 N 个 csz（默认全部）。N >= 目录里文件数时等于全用；N 小则按 --seed "
        "随机抽 N 个（可复现）。配合 --map-balance 按地图均分名额",
    )
    ap.add_argument(
        "--map-balance",
        dest="map_balance",
        action="store_true",
        help="按地图均分名额（只在 --count 小于候选数时生效）：每张图抽 count/图数 个，"
        "该图不够就全拿，名额还差就从没选中的文件里随机补",
    )
    ap.add_argument(
        "--maps",
        default=None,
        metavar="LIST",
        help="哪些地图的 csz 才参与，逗号分隔。默认只跑 MAP_TO_IDX 里的 8 张"
        "（" + "/".join(sorted(MAP_TO_IDX)) + "）；别的地图整个 csz 跳过、不占名额。"
        "`all` = 不筛",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        metavar="N",
        help="并行处理几个 csz（默认 1；1 = 单进程）。内存 ≈ N × 1.7 GB",
    )
    args = ap.parse_args(argv)

    out_dir = args.out_dir or str(REPO_ROOT / "data" / "demos" / "prepared" / args.task)

    try:
        n = build(
            args.csz_dir,
            out_dir,
            task=args.task,
            train=args.train,
            valid=args.valid,
            test=args.test,
            seed=args.seed,
            overwrite=args.overwrite,
            limit=args.limit,
            count=args.count,
            map_balance=args.map_balance,
            maps=args.maps,
            workers=args.workers,
        )
    except (ValueError, TypeError, FileExistsError) as e:
        print(f"[!] {e}")
        return 2

    counts = "  ".join(f"{s}={n.get(s, 0)}" for s in SPLITS)
    if args.limit is not None:
        picked = f"（只跑了排序后的前 {args.limit} 个 csz）"
    elif args.count is not None:
        picked = f"（count={args.count}" + (
            "，按地图平衡）" if args.map_balance else "，随机抽）"
        )
    else:
        picked = ""
    print(f"[OK] task={args.task}  {counts} 个样本文件{picked}  -> {out_dir}")
    print(f"     用到的 csz 和每张图的个数都记在 {(Path(out_dir) / TASK_META).as_posix()}")
    print(f"     下一步：SampleDataset({(Path(out_dir) / 'train').as_posix()!r}, "
          f"task={args.task!r})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
