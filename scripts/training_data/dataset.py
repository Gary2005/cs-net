"""Dataset 层：`SampleDataset` —— 参数是**造好的数据文件夹** + **任务名**。

    from scripts.training_data import SampleDataset

    ds = SampleDataset("data/demos/prepared/example/train", task="example")
    len(ds)     # -> task.n_samples(ds.samples)
    ds[0]       # -> to_torch(task.read_sample(ds.samples, 0))

**这个文件是框架，不用改。** 一个样本里装什么、怎么读出来，写在 `tasks/<任务名>.py` 里
（完整例子见 `tasks/example.py`，空白模板见 `tasks/_template.py`）。这里只做四件事：

1. 构造时 glob 样本文件名（**不读样本内容**，所以是瞬时的）
2. 校验这个文件夹确实是 `task` 造出来的（读 `task.json`）
3. `__len__` → `task.n_samples(self.samples)`
4. `__getitem__` → `task.read_sample(self.samples, i)`，返回 dict 就自动过一遍 `to_torch()`

⚠ 注意 `self.samples` 里装的是**样本文件（.npz）**，不是 csz 文件 —— 这个类的参数
`data_dir` 也永远是造好的数据文件夹。csz 只在 `prepare.py` 那侧出现，
详见 `tasks/base.py` 开头那张表。

数据文件夹从哪来：先跑一次 `prepare.py`

    python -m scripts.training_data.prepare --task example

产出（`out_dir` 默认就是 `prepared/<task>`）：

    data/demos/prepared/example/
    ├── task.json                           {"task": "example", ...}
    ├── train/    de_mirage_f0_r3.npz  ...
    ├── valid/    ...
    └── test/     ...

**参数是具体的那个 split 子文件夹，不是 `prepared`，也不是 `prepared/example`** ——
前两者下面没有样本文件，拿它们构造会报错（报错信息会把子文件夹列出来提醒你）：

    ds_train = SampleDataset("data/demos/prepared/example/train", task="example")
    ds_valid = SampleDataset("data/demos/prepared/example/valid", task="example")
    ds_test  = SampleDataset("data/demos/prepared/example/test",  task="example")

哪个 split 里装什么、比例怎么算，全部由 `prepare.py` 决定，这里不关心 ——
你传哪个文件夹，它就只看那个。

一个 split 里 0 个样本是**合法**的
----------------------------------
样本数是造数据那一侧定的，可能真的是 0：

* 整个 split 一张图都没分到（实测 4 个 demo 按 0.9/0.05/0.05 切，valid/test 常常空）；
* 分到的图里所有回合都没通过体检，一个文件都没写。

这两种情况下 `len(ds) == 0`，构造**不会报错**（只要旁边的 `task.json` 确认了任务名对得上）。
但要小心：`DataLoader(ds, shuffle=True)` 会因为 `num_samples=0` 报错 ——
**训练脚本里记得 `if len(ds) == 0: 跳过这个 split`**。

⚠ 和"参数指错目录"的区别：指到 `prepared`、`prepared/example` 这种没有样本文件的
目录仍然会报错（那是参数错了）；只有"目录是某个任务的 split 子目录、而且真的是空的"
才当成 0。

为什么 `task` 是必填的
----------------------
任务名不一样，样本文件的**内容布局就不一样**（`read_sample` 是两个不同函数）。
填错不会报错，只会静静地读出别的任务的字段 —— 所以：

* `task` 没有默认值，必须显式传
* `prepare.py` 在 `out_dir/task.json` 里记下任务名，这里对不上就直接报错

**这个文件不 import `store.py`，也不 import `demo_extract`** —— 训练侧只认磁盘上的
样本，数据文件夹可以单独搬走。（前提是你自己的 `read_sample` 里也别 import
`..constants` / `..store`，见 `tasks/base.py` 的规矩 3。）

为什么分两层：`__len__` 的定义权在造数据那一侧
---------------------------------------------
样本数不是 csz 目录决定的，是造数据那侧决定的——丢哪些不健康的回合、一个回合切几个
样本，都是那边的选择。如果 `dataset.py` 的参数还是 csz 目录，`__len__` 就只能
**把预处理逻辑再实现一遍去"预判"**。两边逻辑迟早不一致，而且是静默不一致：
`len(ds)` 报 89、磁盘上其实只有 86 个文件，shuffle 到越界才炸。

现在这个"两边"被收进了**同一个任务文件**（`n_samples` 和 `read_sample` 就在隔壁），
对不上时你一眼能看到两处定义。

三条别踩的（详见 README §5）
---------------------------
1. `n_samples()` 必须便宜：它在构造时就被调一次，而且每个 DataLoader worker 各一份。
   只许看文件名，不许打开样本文件。
2. Windows 的 DataLoader 用 spawn，会把 dataset 整个 pickle 给每个 worker。
   别在 `Task` 上挂大对象；要缓存就在 `read_sample` 里惰性建。
3. `__init__` 只 glob 文件名 + 读很小的元数据。把整个样本目录读进内存，
   乘以 N 个 worker 就是 N 份，直接 OOM。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .tasks import get_task, read_meta

try:
    import torch
    from torch.utils.data import Dataset as _Dataset

    HAS_TORCH = True
except ImportError:  # 数据层不需要 torch
    torch = None  # type: ignore[assignment]
    _Dataset = object  # type: ignore[assignment,misc]
    HAS_TORCH = False


# 需要转成 torch.long 的 key（其余整型也会自动走 long，见 to_torch）
INT_KEYS: set[str] = {"weapon", "place", "team", "action_buttons"}


def to_torch(
    sample: dict[str, Any], int_keys: set[str] | None = None
) -> dict[str, Any]:
    """按约定把 numpy 数组转成 torch 张量。没有 torch 时原样返回。

        key 以 `_mask` 结尾 或 bool 数组 → torch.bool
        int_keys 里 / 整型数组           → torch.long
        其它                             → torch.float32

    `__getitem__` 已经自动对你的 `read_sample()` 结果调了一次；返回的不是
    `dict[str, ndarray]`（比如 tuple）时不会被调用。
    """
    if not HAS_TORCH:
        return sample
    int_keys = INT_KEYS if int_keys is None else int_keys
    out: dict[str, Any] = {}
    for k, v in sample.items():
        if not isinstance(v, np.ndarray):
            out[k] = v
        elif k.endswith("_mask") or v.dtype == bool:
            out[k] = torch.from_numpy(np.ascontiguousarray(v)).bool()
        elif k in int_keys or np.issubdtype(v.dtype, np.integer):
            out[k] = torch.from_numpy(np.ascontiguousarray(v)).long()
        else:
            out[k] = torch.from_numpy(np.ascontiguousarray(v, dtype=np.float32))
    return out


class SampleDataset(_Dataset):  # type: ignore[misc]
    """参数是造好的数据文件夹 + 任务名；取数逻辑全在 `tasks/<任务名>.py` 里。"""

    def __init__(
        self,
        data_dir: str | Path,
        *,
        task: str,
        pattern: str | None = None,
    ) -> None:
        """扫文件夹 + 解析任务。**不读样本内容**，所以构造是瞬时的。

        Args:
            data_dir: `prepare.py` 写出来的那个 split 目录（不是 csz 目录，也不是
                `prepared` 根目录）。里面装的是**样本文件（.npz）**。
            task: 任务名，就是 `tasks/<名>.py` 的文件名。**必填** —— 填错不会报错，
                只会读出别的任务的字段，所以这里要求显式写出来。
            pattern: 样本文件的 glob。默认用任务的 `pattern`（= `*{ext}`），
                **必须和造数据时写文件用的扩展名一致**，否则会报"没有样本文件"。

        构造完之后：`self.samples` 是排好序的样本文件列表，`len(self)` 是样本数
        （可能为 0，见模块 docstring —— 那是合法的，但别拿去 shuffle）。
        """
        self.data_dir = Path(data_dir)
        self.task = get_task(task)
        self.task_name = task
        self.pattern = pattern or self.task.pattern

        if not self.data_dir.is_dir():
            raise FileNotFoundError(
                f"{self.data_dir} 不存在。这是**造好的数据文件夹**，不是 csz 目录。\n"
                f"  先跑一次：python -m scripts.training_data.prepare --task {task} "
                f"--out-dir {self.data_dir.parent}"
            )

        # 这个文件夹是哪个任务造的？对不上就是"拿 A 的样本按 B 读"，必须拦住：
        # read_sample 换了，报错信息不会出现，只会静默读出错的 key。
        # 顺便：`recorded == task` 也是"空目录合法"的判据（见下面那段）。
        recorded = read_meta(self.data_dir.parent).get("task")
        if recorded and recorded != task:
            raise ValueError(
                f"{self.data_dir} 是任务 {recorded!r} 造出来的，你却按 {task!r} 去读。\n"
                f"  样本布局跟着任务走，读错了不会报错，只会读出别的字段。\n"
                f"  要读这个目录就用 task={str(recorded)!r}；要用 {task!r} 就先造它：\n"
                f"    python -m scripts.training_data.prepare --task {task}"
            )

        self.samples: list[Path] = sorted(self.data_dir.glob(self.pattern))
        if not self.samples and recorded != task:
            # 空目录分两种：① 合法产物（整个 split 没分到图），② 参数指错了。
            # 只有旁边 task.json 确认"这个目录就是 task 造的"才算 ①，否则一律报错 ——
            # 不然打错一个路径就会静默拿到 len(ds)==0 的空数据集。
            subs = sorted(p.name for p in self.data_dir.iterdir() if p.is_dir())
            hint = ""
            if subs:
                hint = (
                    f"\n  但下面有子文件夹 {subs}。如果这是 train/valid/test，"
                    f"参数要指到具体的那个：\n"
                    f"    SampleDataset({self.data_dir.as_posix()!r} + '/train', "
                    f"task={task!r})"
                )
            raise FileNotFoundError(
                f"{self.data_dir} 下没有 {self.pattern}（glob 不递归，只看这一层）。{hint}\n"
                f"  目录是空的？先跑 python -m scripts.training_data.prepare --task {task}\n"
                f"  或者造数据时用的扩展名和 {self.pattern!r} 对不上，"
                f"改 tasks/{task}.py 的 ext（或传 pattern=...）"
            )

        # 便宜（任务的规矩），所以这里调一次定死：`len(ds)` 之后就是读一个 int。
        # samples 为空时它也必须是 0（没有任何文件可供 read_sample 返回）。
        self._n = int(self.task.n_samples(self.samples))
        if self._n < 0:
            raise ValueError(
                f"tasks/{task}.py 的 n_samples() 返回了负数 {self._n}，肯定是写错了"
            )
        if not self.samples and self._n:
            raise ValueError(
                f"{self.data_dir} 里一个样本文件都没有，tasks/{task}.py 的 n_samples() "
                f"却说有 {self._n} 个 —— `samples` 为空时要返回 0"
            )

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, i: int) -> Any:
        sample = self.task.read_sample(self.samples, i)
        # 返回 dict 就统一转成张量；返回 tuple / 自定义对象就原样交给 collate。
        return to_torch(sample) if isinstance(sample, dict) else sample

    def __repr__(self) -> str:
        return (
            f"SampleDataset({self.data_dir.as_posix()!r}, task={self.task_name!r}, "
            f"n={self._n}, samples={len(self.samples)})"
        )


__all__ = ["SampleDataset", "to_torch", "INT_KEYS", "HAS_TORCH"]
