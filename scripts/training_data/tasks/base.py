"""任务契约：一个数据集 = 一个 `Task` 子类。

`prepare.py`（造数据）和 `dataset.py`（读数据）都不认识具体任务，只认这个类的四个钩子：

    n_samples(samples)                   一共多少个样本（只看文件名，不读内容）
    write_round(store, *, fi, rid, ...)  把一个回合写成一个样本文件
    read_sample(samples, i)              把第 i 个样本读回来
    sample_name(store, *, fi, rid)       样本文件叫什么（有默认实现，一般不用改）

一个任务 = `tasks/` 下的**一个文件**：继承 `Task`，在模块末尾写 `TASK = 你的类()`。
文件名去掉 `.py` 就是这个任务的名字 —— `--task example` 指的就是 `tasks/example.py`。

    tasks/example.py    完整可跑的示例，照抄它
    tasks/_template.py  空白模板（带 TODO），复制它改名更省事

先分清三个"文件"，名字很像但完全不是一回事
------------------------------------------
这份契约里最容易搞混的就是"文件"指哪一个。**造和读拿到的文件是两种不同的东西**，
参数名刻意分开了：带 `s` 的复数形参一律是**读侧**（造好的样本）。

    csz 文件    原始 .csz（data/demos/csz/*.csz）   只在 write_round（造）
                用 `fi` 下标访问：store.paths[fi]、store.round_tensors(fi, rid)……
                ⚠ 任务永远拿不到它的 Path —— 造数据侧传进来的只有 store 和 fi

    样本文件    造出来的 .npz（prepared/<task>/<split>/*.npz）  只在 n_samples / read_sample（读）
                就是形参 `samples`，即 samples[i]

    样本        逻辑上的第 i 条数据                  read_sample(samples, i)
                i 的取值范围由 n_samples(samples) 定义

所以：**`n_samples` / `read_sample` 收到的永远是样本文件（.npz），不是 csz**；csz 那一侧
的信息（回合号、地图名、原始事件）全部靠 `store` + `fi` 在 `write_round` 里取。两个参数
集合不重叠，`files` 这种两边都能套的叫法已经全部改名了。

任务文件里除了 `Task` 子类，还能写什么
--------------------------------------
**什么都能写** —— 它就是个普通 Python 模块，框架只从里面拿两样东西：模块末尾的
`TASK = <你的类>()`（必须是 `Task` 子类），以及文件名（= 任务名）。其余随便：

    # 模块 docstring 随便写（这里省略）
    from __future__ import annotations

    import math                      # 标准库
    from pathlib import Path
    from typing import Any

    import numpy as np               # 第三方库（读侧只有 numpy 一定会被装上）
    from ._util import rel_xy         # 自己写的共用模块（见下）
    from .base import Task

    KEYS = ("players", "alive")       # 模块级常量：example.py 就是这么写的
    SCALE = 1.0 / 64                  # 别的常量、查表、小数组都行

    def _pick(store, fi, rid):        # 模块级辅助函数，几个都行
        ...

    class _Window:                    # 辅助类也行（不是 Task 子类也没关系）
        ...

    class MyTask(Task):
        ...

    TASK = MyTask()                   # ← 必须有，且只能有一个

几条实际的坑：

* **模块级代码在每个进程里各执行一次。** 实测：同一个进程里 import 两次只执行一次，
  换个进程就再来一次。Windows 的 DataLoader 是 spawn，**每个 worker 都是新进程** →
  你的模块级代码在 N 个 worker 里各跑一遍。所以模块级只放"常量 + 函数定义"这种便宜的
  东西；想要大查表/大数组，写成函数里的惰性缓存（`def _lut(): global ...`），别在
  import 时就建。也别在模块级 print / 写文件（会在每个 worker 里各来一次）。
* **多个任务共用的辅助代码放 `tasks/_util.py`**（下划线开头 → `task_files()` 会跳过，
  不会被当成任务，也不会出现在 `--list` / `--todos` 里）。`base.py` 也享受同等待遇。
  只有真的想让某个文件变成"一个任务"时，它才需要不带下划线 + 有 `TASK`。
* **别在模块级改全局状态**（`sys.path`、环境变量、随机种子）—— 每个 worker 都会再来
  一遍，而且和别的任务混在一起时顺序不确定。
* `store`（造数据侧的东西）**不能**在模块级 import；它只是 `write_round` 的第一个参数，
  类型标注写成字符串或 `TYPE_CHECKING` 里 import（`base.py` 自己就是这么干的）。
* **可以写 `if __name__ == "__main__":` 冒烟测试**，而且推荐写、**每个任务各写各的**
  （检查哪几个不变量、要不要画图存盘，只有你知道）。它只在直接执行时跑，`get_task()`
  import 任务时不会触发：

      python -m scripts.training_data.tasks.<你的任务名>     # 在仓库根目录执行，必须 -m

  `example.py` 末尾是完整的一份（前 N 个真 csz → `tmp/training_data/<任务>/` → 读回来
  → 断言 → 想画图就画），`_template.py` 末尾是留了 TODO 的骨架。**框架不提供共用的
  冒烟测试** —— 共用的只有 `build(limit=N)` 和 `SampleDataset` 这两个入口。
  ⚠ import 全写在 `if` 里面：任务模块会被每个 worker import，写在文件顶部会让每个
  worker 都连带 import `..prepare`（`CszStore` / `demo_extract`）。

四条硬规矩
----------
1. **造和读必须对得上。** `write_round` 写哪些 key / 什么 dtype / 什么文件名，
   `read_sample` 就得按同一套读回来。两边都在你这一个文件里，别拆到两处去维护。
2. **`n_samples()` 必须便宜。** 它会在 dataset 构造时被调一次（每个 DataLoader worker
   各一次）。只许看文件名，不许打开样本文件。
3. **`read_sample()` 里别 import `..constants` / `..store`。** 那会把 `demo_extract`
   拖进训练进程、数据文件夹就不能单独搬走了。列下标（`COL`）这类东西在
   `write_round` 里用掉就行；真需要词表就在 `write_round` 时写进样本或旁边的小 json。
   ⚠ 实测现状：这条**还做不到真正的隔离** —— `scripts/training_data/__init__.py` 是
   急切的（自己就 import 了 `constants` / `store`），而 Python 导入任何子模块前都会先
   执行父包的 `__init__.py`，所以 `import scripts.training_data.dataset` 一样会带上
   `demo_extract.codec/extract/...`。规矩先留着别破坏：等 `__init__.py` 改成惰性导入
   （PEP 562 的模块级 `__getattr__`）它才真的生效。
4. **别在 `write_round` 里攒东西。** 一个回合的原始张量约 6 MB，20 万个回合就是 TB 级；
   算完立刻写盘、立刻丢。也不要缓存到 `self` 上 —— 同一个进程要过几千个文件。

一个 csz 出几个样本，由你的实现决定
-----------------------------------
框架不预设"一个 csz 几个样本"，它只是**对每个回合调一次 `write_round`**。所以：

    一个回合 = 一个样本文件        write_round 里存一份；n_samples = len(samples)
                                  ← tasks/example.py 就是这个，实测一个 csz 出 20～24 个

    一个回合 = K 个样本文件        write_round 里自己派生兄弟文件名（见下一节）
                                  n_samples 用默认的 len(samples) 就自动对得上
                                  ← 一个回合一个玩家一个样本、多种时间尺度分开落盘

    一个回合读的时候切 K 个样本     造的时候仍是一回合一个文件，读的时候按
                                  `n_samples = len(samples) * K`、`samples[i // K]` + `i % K` 拆
                                  ← 推荐这种：滑窗、预测未来 N tick 都这么做

    0 个                          完全合法。整张图的回合全被体检过滤掉，就一个文件都不写，
                                 该 csz 的样本数记 0
                                 整个 split 一个样本都没有也是合法的，见 dataset.py

一个回合写 K 个文件：能写，而且读侧天生就对
------------------------------------------
`write_round` 每次只拿到一个 `path`，但**没有任何东西限制你写几个** —— 兄弟名字自己
派生就行：

    def write_round(self, store, *, fi, rid, split, path):
        tensors = store.round_tensors(fi, rid)
        for k in range(K):
            p = path.with_name(f"{path.stem}_p{k}{path.suffix}")
            np.savez(p, ...)          # 框架传进来的 path 本身不写也行，没人检查它

    def n_samples(self, samples):     # ← 框架**没有**默认实现，不写这句会抛 NotImplementedError
        return len(samples)           #    一个样本文件 = 一个样本时，就是这一句

为什么这是安全的：框架**不数 `write_round` 被调了几次**，它只数磁盘上的样本文件
（`prepare.py` 最后 glob 一遍报出来，`dataset.py` 的 `len(ds)` 用同一套 glob）。实测一个
回合写 3 个文件：20 个回合 → 磁盘 60 个文件 → `len(ds) == 60`，`ds[0]`/`ds[1]`/`ds[2]`
取到的是同一个回合的三个不同窗口。

三条要注意的：

* **文件名的唯一性由你负责**：`sample_name()` 只有 `(fi, rid)` 两个维度，一个回合算不出
  K 个名字，所以兄弟名字在 `write_round` 里派生（`f"{path.stem}_p{k}"` 之类）。
* **`n_samples` 别乱改**：默认 `len(samples)` 就是对的，写成 `len(samples) // K` 反而错。
* **想让框架来驱动这个循环**（`write_round` 每个分片被调一次、名字统一由 `sample_name`
  给）才需要加 `part: int` 参数 + `n_parts(store, *, fi, rid)`；在 `write_round` 里自己
  派生名字的话，一行框架代码都不用动。

加之前想清楚：**读侧的窗口切分能解决的事，不要搬到写侧**（写侧多一份文件就多一份
磁盘 I/O，而且每个 epoch 都要多读一遍）。

`write_round` 里你能用的东西（都在传进来的 `store` 上，不用 import 任何东西）：

    store.round_tensors(fi, rid)   一个回合预计算好的 14 个张量（首选，最快）
                                   键和形状见 store.py 的 round_tensors() docstring
    store.round(fi, rid)           原始回合 dict：events / kills / fire / smoke / winner ...
                                   ⚠ 事件列表里混着别的回合的，必须用 features.round_events()
    store.map_name(fi)             地图名，如 'de_mirage'
    store.weapon_lut(fi)           武器下标 → 全局下标（本文件下标跨 demo 不可比）
    store.weapon_names(fi)         武器下标 → 名字
    store.meta(fi) / store.data(fi) 整个文件的 meta / 全部内容（后者很占内存，慎用）

`read_sample` 里你能用的只有 `samples[i]`（样本文件）和 numpy —— 这是故意的：那时候
csz 早就解析完了，训练进程里不该有 `demo_extract`。
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # 只为类型标注，运行时**不**导入 → 训练侧不会拖进 demo_extract
    from ..store import CszStore


class Task:
    """一个数据集任务。子类实现 `n_samples` / `write_round` / `read_sample` 三个方法。"""

    #: 样本文件的扩展名。`dataset` 侧的 glob 默认就是 `*{ext}`，改这一个就够。
    #: 换成分片 `.npy` + mmap 的布局时写 `".npy"`。
    ext: str = ".npz"

    #: 一行说明。`--help` / 未知任务的报错里会打出来，写清楚这个任务是干什么的。
    desc: str = "（没写 desc）"

    @property
    def pattern(self) -> str:
        """`SampleDataset` 用来 glob 样本文件的通配符。默认 `*{ext}`。

        需要更复杂的布局（比如只认 `part-*.npy`）就覆盖这个属性。
        """
        return f"*{self.ext}"

    # ── 一般不用改的默认实现 ───────────────────────────────────────────────

    def sample_name(self, store: "CszStore", *, fi: int, rid: int) -> str:
        """样本文件名（不含目录）。默认 `{地图}_f{文件下标}_r{回合号}{ext}`。

        **这是造和读之间唯一的契约**：`read_sample` 只能从文件名 + 文件内容反推，
        所以名字里至少要能看出 `(fi, rid)` —— 将来某个样本出问题，你得能回到
        原始 csz 去对。

        ⚠ 别用 csz 的完整文件名当前缀（实测那名字有 60+ 字符），`f{fi}` 就够：
        第 fi 个 = csz 目录里按文件名排序的第 fi 个。
        """
        return f"{store.map_name(fi)}_f{fi}_r{rid}{self.ext}"

    # ── 子类必须实现的三个 ─────────────────────────────────────────────────

    def n_samples(self, samples: list[Path]) -> int:
        """一共多少个样本。⚠ 必须便宜：只许看文件名，不许打开样本文件。

        Args:
            samples: **样本文件（.npz）**，`SampleDataset` 在自己的 split 目录下 glob 好、
                排好序的那一批 —— 不是 csz 文件。`samples[i]` 就是 `read_sample` 拿到的
                同一个 `i`。csz 那边的东西这里看不到，也不该看到。

        一个样本文件 = 一个样本（最常见）：

            return len(samples)

        一个回合切 K 个窗口（造数据时一个回合写一个文件，读的时候切）：

            return len(samples) * K     # 同时 read_sample 里要 samples[i // K] + i % K

        `samples` 是空列表（这个 split 一张图都没分到）时返回 0 —— 这是正常情况，不是错误。

        ⚠ 它必须和 `read_sample()` 接受的 i **一个不多一个不少**。对不上的话
        要 shuffle 到越界才报错，很难查。两边的换算写在同一个文件里就是为了这个。
        """
        raise NotImplementedError(
            f"{type(self).__name__}.n_samples() 还没写 —— 在 tasks/ 里你的任务文件里实现"
        )

    def write_round(
        self,
        store: "CszStore",
        *,
        fi: int,
        rid: int,
        split: str,
        path: Path,
    ) -> None:
        """把第 `(fi, rid)` 个回合写成 `path` 这一个文件。**造数据侧唯一的出口。**

        Args:
            store: 已经 load 好的 `CszStore`（可用方法见模块 docstring）。
            fi: csz 文件下标 —— `store.paths[fi]`、`store.map_name(fi)` 都用它。
            rid: 回合号。
            split: `"train"` / `"valid"` / `"test"`，`path` 已经在这个子目录下了。
                一般用不上，需要按 split 写不同的东西时才有用。
            path: 目标文件**完整路径**，父目录已建好。扩展名和 `ext` 一致。

        典型写法就是一个 `np.savez`：

            tensors = store.round_tensors(fi, rid)
            np.savez(path, **{k: tensors[k] for k in KEYS})

        ⚠ 别把 `tensors` 攒进 list、也别挂到 `self` 上：一个回合约 6 MB。算完就丢。
        """
        raise NotImplementedError(
            f"{type(self).__name__}.write_round() 还没写 —— 在 tasks/ 里你的任务文件里实现"
        )

    def read_sample(self, samples: list[Path], i: int) -> Any:
        """第 i 个样本。`samples` 是**样本文件（.npz）**，和 `n_samples` 收到的是同一批。

        最省事的写法就是"第 i 个样本 = 第 i 个文件"：

            with np.load(samples[i]) as z:
                return {k: z[k] for k in z.files}

        一个文件里装多个样本时，`i` 到文件的映射自己定（要和 `n_samples` 一致）：

            with np.load(samples[i // K]) as z:
                ...  # 用 i % K 确定取第几个窗口

        返回 `dict[str, np.ndarray]` 的话，`SampleDataset` 会自动过一遍 `to_torch()`
        （bool/`*_mask` → bool，整型 → long，其余 → float32），然后 torch 默认的
        collate 会把同名字段 stack 成 batch。返回别的类型（tuple / 自定义对象）
        就原样交给 collate。

        ⚠ `.npz` 是 zip，`np.load(..., mmap_mode="r")` 会被**静默忽略**：取一个
        成员就把整个成员读出来。同一个文件要反复取不同片段就别用 npz（实测取一个
        T=64 的窗口：npz 2199 us vs `npy` + mmap 4.3 us，见 `prepare.py` 开头）。
        """
        raise NotImplementedError(
            f"{type(self).__name__}.read_sample() 还没写 —— 在 tasks/ 里你的任务文件里实现"
        )


__all__ = ["Task"]
