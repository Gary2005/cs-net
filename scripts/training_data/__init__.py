"""
CS2 demo → 训练数据（v5：读 `.csz`）。

**一个任务一个文件。** `prepare.py`（造数据）和 `dataset.py`（读数据）都是框架，
不用改；"一份 csz 要变成什么数据集"写在 `tasks/<任务名>.py` 里：

    tasks/example.py    ★★★ 你改这个 ★★★  完整可跑的示例（一个回合 → 一个 .npz）
    tasks/_template.py  ★★★ 你改这个 ★★★  空白模板，复制成 tasks/<你的任务名>.py

同一份 csz 可以造出任意多个任务的数据集（`--task example` / `--task mytask` …），
各自写自己的目录 `data/demos/prepared/<task>/{train,valid,test}/`。

    prepare.py     造数据  逐文件单遍、处理完就丢；按 demo（= 一个 csz = 一张地图）
                           整个分到 train/valid/test；唯一 import demo_extract 的地方
    dataset.py     读数据  SampleDataset(data_dir, task=...)：glob 文件 + 转发给任务
    tasks/         任务    ★ 你写：一个回合怎么写（write_round）、怎么读（read_sample）
    constants.py   常量：特征列、地图中心、武器词表、体检阈值
    _paths.py      把仓库 tools/ 加进 sys.path（让 demo_extract 可导入）
    store.py       第 1 层  CszStore：加载 .csz、合并切碎的 fire/smoke、缓存张量、体检
    features.py    事件过滤 + 字段语义的参考实现（窗口形状，照用坑和说明即可）
    spatial.py     fire / smoke 空间特征（可选，很贵）
    selftest.py    跑通检查：python -m scripts.training_data

快速开始：

    python -m scripts.training_data --task example          # 体检 csz + 任务 + 数据
    python -m scripts.training_data.prepare --task example  # 造数据
    python -m scripts.training_data --todos                 # 列出还没写的 TODO

    from scripts.training_data import SampleDataset

    ds = SampleDataset("data/demos/prepared/example/train", task="example")
    len(ds)
    ds[0]

写自己的任务：

    copy tasks\\_template.py tasks\\mytask.py     # 文件名就是 --task 的值
    python -m scripts.training_data --todos      # 看还差哪几处
    python -m scripts.training_data.prepare --task mytask
    python -m scripts.training_data --task mytask

任务契约（要写哪三个方法、能用什么、坑在哪）在 `tasks/base.py` 的 docstring 里；
完整例子在 `tasks/example.py`。**不用改** `prepare.py` / `dataset.py`。

**为什么 `SampleDataset` 的参数是"造好的数据文件夹 + 任务名"**：样本数、以及"哪边是
验证集"，都由造数据那一侧决定（丢哪些不健康的回合、一个回合切几个样本、怎么划分），
csz 侧算不出来；硬在 `__len__` 里复现一遍预处理逻辑，两边迟早静默不一致。任务名是
必填的，因为换一个任务，样本文件的布局就变了，读错不会报错、只会读出别的字段。

**参数也不是 `prepared` 根目录**：样本文件在 `<task>/train/` `.../valid/` `.../test/`
三个子文件夹里，`glob` 不递归。传错会报"没有样本文件"，报错信息会把子文件夹列出来。

⚠ **实测：只要 import 了这个包里的任何东西，就会连带 `demo_extract`** —— 包括
`import scripts.training_data.dataset` 和 `from ...dataset import SampleDataset`。
原因是 Python 导入子模块前一定先执行父包的 `__init__.py`，而它就是下面那几行急切的
`from .store import ...`。想真的做到"训练侧不碰 `demoparser2` / 数据文件夹能单独搬走"，
得把本文件改成惰性导入（PEP 562 的模块级 `__getattr__`），现在还没有。

对照：dev 仓库 `cs-net/scripts/training_data/` 里有完整的 feature_builder /
label_builder / depth_map / wds_writer，但它是从 `.json.gz` 读的，数据源接口
不同不能直接搬；可复用的是地图中心、武器词表、特征/标签定义那些常量。
"""

from .constants import (
    COL,
    FEATURES,
    HU_TO_METERS,
    MAP_CENTER,
    MAP_TO_IDX,
    N_PLAYERS,
    N_WEAPONS,
    WEAPON_NAMES,
    WEAPON_TO_IDX,
    weapon_name_to_idx,
)
from .dataset import HAS_TORCH, INT_KEYS, SampleDataset, to_torch
from .features import events_in_window, normalize_xyz, round_events
from .store import CszStore, WindowRef
from .tasks import Task, available, get_task

__all__ = [
    # 任务（写自己的数据集 = 写一个 Task 子类）
    "Task",
    "get_task",
    "available",
    # 造数据层
    "CszStore",
    "WindowRef",
    # 第 2 层（读数据）
    "SampleDataset",
    "to_torch",
    "INT_KEYS",
    "HAS_TORCH",
    # 事件 → 采样下标（做标签用）
    "round_events",
    "events_in_window",
    "normalize_xyz",
    # 常量
    "FEATURES",
    "COL",
    "MAP_CENTER",
    "MAP_TO_IDX",
    "N_PLAYERS",
    "N_WEAPONS",
    "WEAPON_NAMES",
    "WEAPON_TO_IDX",
    "weapon_name_to_idx",
    "HU_TO_METERS",
]
