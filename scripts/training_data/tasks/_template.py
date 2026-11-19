"""空白任务模板：**复制成 `tasks/<你的任务名>.py` 再改**。

    copy tasks\\_template.py tasks\\mytask.py      # Windows
    cp   tasks/_template.py tasks/mytask.py       # bash

文件名去掉 `.py` 就是任务名，也就是 `--task` 的值：

    python -m scripts.training_data.prepare --task mytask
    python -m scripts.training_data --task mytask
    python -m scripts.training_data --todos      # 看这个新任务还差哪几处

⚠ 这个文件以 `_` 开头，所以它**不会被注册成任务**（`--task _template` 会报没有这个
任务），也不会出现在 `--todos` 里 —— 它是给你复制的，不是给你填的。复制出去、
去掉下划线，`# ── TODO` 那几行才会被 `--todos` 数到。

写过一遍就懂了：契约和坑都在 `tasks/base.py` 的 docstring 里，完整可跑的例子在
`tasks/example.py`。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .base import Task

#: 造数据时 dump 哪些 key（可用清单见 tasks/example.py 里的注释）。
KEYS: tuple[str, ...] = ("players", "alive", "teams", "ticks")


class MyTask(Task):
    #: 样本文件的扩展名。`SampleDataset` 的 glob 默认就是 `*{ext}`，改这一个就够。
    #: 一个回合一个文件 → ".npz"；想 mmap / 一个文件装多个窗口 → ".npy"。
    ext = ".npz"

    #: 一行说明，会出现在 `--help` 和未知任务的报错里。
    desc = "一句话说清这个任务要什么"

    # 文件名规则不用写：Task.sample_name() 给的 `{地图}_f{fi}_r{rid}{ext}` 够用。
    # 要改就覆盖 def sample_name(self, store, *, fi, rid)。

    # ── TODO 1 ── 一共多少个样本（必须便宜：只看文件名，不许开样本文件）────────
    # `samples` 是**样本文件**（prepared/<task>/<split>/*.npz），不是 csz 文件 ——
    # csz 那一侧只在 write_round 里、靠 store + fi 访问。详见 tasks/base.py 开头。
    def n_samples(self, samples: list[Path]) -> int:
        # 一个样本文件 = 一个样本：
        #     return len(samples)
        #
        # 一个回合切 K 个窗口（记得和 read_sample 里的换算配套）：
        #     return len(samples) * K
        #
        # 这个 split 一个文件都没有时返回 0（合法：整张图都分到别的 split 去了）。
        #
        # ⚠ `len(ds)` 必须覆盖 read_sample 接受的**所有** i，一个不多一个不少。
        raise NotImplementedError("TODO 1：n_samples 还没写")

    # ── TODO 2 ── 落盘：把一个回合写成 path 这一个文件 ─────────────────────────
    def write_round(self, store, *, fi: int, rid: int, split: str, path: Path) -> None:
        # 取数（可用方法见 tasks/base.py 的 docstring）：
        #     tensors = store.round_tensors(fi, rid)      # 14 个预计算张量
        #     rnd = store.round(fi, rid)                  # 原始 dict（事件/火/烟）
        #     labels = round_events(rnd, "kills")         # ⚠ 事件必须过滤
        #
        # 落盘：
        #     np.savez(path, **{k: tensors[k] for k in KEYS})
        #
        # 要标签 / 派生特征都在这里算好再写 —— 训练时再算就是每个 epoch 白烧一遍。
        raise NotImplementedError("TODO 2：write_round 还没写")

    # ── TODO 3 ── 读回来：第 i 个样本 ──────────────────────────────────────────
    # `samples` 和上面 n_samples 收到的是同一批样本文件。
    def read_sample(self, samples: list[Path], i: int) -> Any:
        #     with np.load(samples[i]) as z:      # ⚠ npz 上 mmap_mode 无效
        #         return {k: z[k] for k in z.files}
        #
        # 返回 dict[str, ndarray] 即可，框架会自动 to_torch()。要在 GPU 上更省事，
        # 也可以在这里就返回 tuple / 直接拼成一个大张量，随你。
        raise NotImplementedError("TODO 3：read_sample 还没写")


TASK = MyTask()


# ── 冒烟测试 ───────────────────────────────────────────────────────────────
# 写完上面三个 TODO 之后，直接跑这个文件验证一遍"造 → 读"：
#
#     python -m scripts.training_data.tasks.<你的任务名>   # 在仓库根目录执行，必须 -m
#
# **每个任务的冒烟测试各写各的**（检查什么、要不要画图都随任务），下面是能跑通的最小
# 骨架：用前 N_CSZ 个真 csz 造一份小数据集到 `tmp/training_data/<任务名>/`（每次自动
# 覆盖旧的），再把每个 split 逐个下标读回来。`example.py` 末尾那份注释更全，可以对着改。
#
# ⚠ import 全写在 `if` 里面 —— 任务模块会被 DataLoader 的每个 worker import，写在顶部
#   会让每个 worker 都连带 import `..prepare`（`CszStore` / `demo_extract`）。
if __name__ == "__main__":
    import shutil

    from .._paths import REPO_ROOT
    from ..dataset import SampleDataset
    from ..prepare import SPLITS, build

    TASK = Path(__file__).stem
    N_CSZ = 2  # 1 个约 13 s
    CSZ_DIR = REPO_ROOT / "data" / "demos" / "csz"
    OUT = REPO_ROOT / "tmp" / "training_data" / TASK

    shutil.rmtree(OUT, ignore_errors=True)  # 每次从零开始，不用手动清
    counts = build(CSZ_DIR, OUT, task=TASK, limit=N_CSZ)
    print(f"[冒烟] 造出来 {counts}")

    for split in SPLITS:
        ds = SampleDataset(OUT / split, task=TASK)
        n = len(ds)
        print(f"[冒烟] {split:<5} len={n}")
        if n == 0:
            continue  # 空 split 合法
        for i in range(n):
            s = ds[i]

            # ── TODO 4 ── 你这个任务该满足什么？在这里 assert（也可以画图存盘）──
            # 例：按你上面 dump 的 key 检查形状/取值范围/标签分布
            #     assert sorted(s) == sorted(KEYS), f"{split}[{i}]: keys 不对"
            #     assert s["players"].ndim == 3
            # 画图（要 pip install matplotlib）：
            #     import matplotlib; matplotlib.use("Agg")
            #     import matplotlib.pyplot as plt
            #     plt.savefig(OUT / "demo.png")
            raise NotImplementedError("TODO 4：写你这个任务的检查（不想写就删掉整段）")

        print(f"[冒烟]   {ds.samples[0].name}: { {k: tuple(v.shape) for k, v in ds[0].items()} }")

    print("[冒烟] ✅ 通过")
