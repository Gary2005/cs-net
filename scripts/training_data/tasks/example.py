"""示例任务：**一个回合 → 一个 `.npz`**。照抄这个文件写你自己的任务。

跑通（造 + 读 + 体检，一共三条命令）：

    python -m scripts.training_data.prepare --task example
    python -m scripts.training_data --task example
    python -m scripts.training_data --todos

它刻意写得很小（只 dump 4 个 key），但每个选择都写了理由和坑。你自己那版只需要
改三处：`KEYS`（要哪些字段）、`write_round`（怎么派生成你要的样子）、
`read_sample`（怎么读回来）。文件名的规则在 `Task.sample_name()` 里，一般不用动。

产出的一个样本（一个回合）：

    players  (T, 10, 20)  float32  位置/视角/速度/血量……，列含义见 constants.FEATURES
    alive    (T, 10)      bool
    teams    (10,)        int8     0=CT  1=T  -1=未知
    ticks    (T,)         int64    该回合每 tick 的绝对 tick 号

T = 这个回合的 tick 数（实测平均 ~3700，约 58 秒）。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import Task

#: 这个任务 dump 哪些 key。`store.round_tensors()` 一共给 14 个（见 store.py）：
#:
#:   players alive yaw_sincos seconds ticks teams inventory
#:   spotted on_ground jump land crouch crouching eye_z
#:
#: 全存最省事，只存要用的省盘 —— 但**存多少 = 每个 epoch 要从磁盘搬多少字节**。
#: 实测一个回合 14 个 key 合计平均 9.2 MB，只这 4 个约 3 MB。
#:
#: ⚠ 取舍要按"训练时会不会用"来定，不是按"省不省盘"：少存一个之后要用到的 key，
#:   你得重新跑一遍 30 小时的预处理。
KEYS = ("players", "alive", "teams", "ticks")


class Example(Task):
    """一个回合一个 npz。"""

    ext = ".npz"
    desc = "示例：一个回合一个 .npz，只装 players/alive/teams/ticks"

    def n_samples(self, samples: list[Path]) -> int:
        # `samples` 是**样本文件**（prepared/example/<split>/*.npz）——不是 csz 文件。
        # 一个样本文件 = 一个样本，所以就是文件个数。
        #
        # 想一个回合切 K 个窗口（滑窗、预测未来 N tick 之类）：造数据时一回合仍只写
        # 一个文件，读的时候切 —— 那么这里 return len(samples) * K，read_sample 里用
        #     path = samples[i // K]; k = i % K
        # 定位。**两处换算必须配套改**，所以把它们放在同一个文件里。
        #
        # 想一个回合写 K 个文件：**能写**（以前这里写成"不支持"，是错的）。框架不数
        # write_round 被调了几次，只数磁盘上的样本文件，所以上面这一行
        # `return len(samples)` 不用改 —— 磁盘上有几个文件就是几个样本。
        # 兄弟名字在 write_round 里自己派
        #     p = path.with_name(f"{path.stem}_p{k:02d}{path.suffix}")
        # （补零，免得 _p10 按字典序排到 _p2 前面）。见 base.py 的"一个回合写 K 个文件"。
        return len(samples)

    def write_round(self, store, *, fi: int, rid: int, split: str, path: Path) -> None:
        # ① 取数。round_tensors() 是预计算好的（按 LRU 缓存），比在窗口里反复调
        #    recipes.* 快得多。这一步不 load csz —— 文件在 prepare.py 的循环里已经
        #    load 好了，这里只是从内存里的对象图算张量。
        tensors = store.round_tensors(fi, rid)

        # ② 落盘。只写这一个文件，写完全局变量里不留引用（下一个回合覆盖 tensors）。
        #
        #    - np.savez 不压缩：快，实测一个回合 3 MB 左右。
        #    - np.savez_compressed 压缩：bool 能压得很狠（inventory 一个回合 1.6 MB
        #      全是 1 byte/elem 的 bool），但慢，而且每个 epoch 都要解一遍。
        #    - 想 mmap 就必须换成分片 `.npy` + 自己记偏移，npz 是 zip 读不了局部。
        np.savez(path, **{k: tensors[k] for k in KEYS})

        # 想加派生特征，就在上面这一行之前算，比如：
        #
        #   武器：players[:, :, 12] 存的是**本文件下标**，跨 demo 不可比，先转全局：
        #       lut = store.weapon_lut(fi)
        #       w = tensors["players"][:, :, COL["weapon"]].astype(np.int64)
        #       valid = w >= 0                      # ⚠ -1 = 无武器，直接 lut[w] 会
        #       glob = np.full(w.shape, -1, np.int16)  #   取到最后一个元素（负索引）
        #       glob[valid] = lut[w[valid]]
        #   （要 COL 就得 import ..constants，那会拖进 demo_extract —— 造数据这侧无所谓，
        #     但别在 read_sample 里 import。）
        #
        #   事件/标签（谁打死了谁、下包、投掷物轨迹）：
        #       rnd = store.round(fi, rid)
        #       kills = round_events(rnd, "kills")   # ⚠ 必须过滤，原始列表里混着别的
        #                                            #   回合的事件（见 README §5.8）
        #   空间特征（火/烟的体素）：spatial.py，很贵，见 README §7。
        #
        # ⚠ 别在这里 print：服务器上这是几万次调用，日志会淹掉一切（框架每个文件打一行）。

    def read_sample(self, samples: list[Path], i: int) -> dict[str, np.ndarray]:
        # 读回来。`samples` 和 n_samples 收到的是同一批**样本文件**。
        # **返回 dict[str, ndarray] 就够了** —— SampleDataset 会自动过一遍
        # to_torch()（bool → bool，整型 → long，其余 → float32），torch 的默认
        # collate 再把同一个 key stack 成 batch。
        #
        # ⚠ np.load(..., mmap_mode="r") 对 .npz **无效且不报错**（zip 不是裸数组）：
        #   取一个成员就把整个成员读出来。真需要 mmap 就得换成分片 .npy 布局。
        with np.load(samples[i]) as z:
            return {k: z[k] for k in z.files}


#: 注册入口：`tasks/__init__.py` 里 `get_task()` 找的就是这个名字。
TASK = Example()


# ── 冒烟测试 ───────────────────────────────────────────────────────────────
# 直接跑这个文件就能"从 csz 造一份小数据集 → 再从磁盘读回来"验证一遍：
#
#     python -m scripts.training_data.tasks.example        # 在仓库根目录执行
#     （必须用 -m：直接 python xxx/example.py 会炸在相对 import 上）
#
# **每个任务的冒烟测试各写各的** —— 检查哪几个不变量、要不要打印张量、要不要画图，
# 都随这个任务。这一段就是给你改的：删掉不关心的、加上只有你这个任务才该满足的断言。
#
# ⚠ import 全写在 `if` 里面。任务模块会被 DataLoader 的**每个 worker** import，写在
#   文件顶部的话每个 worker 都会连带 import `..prepare`（`CszStore` / `demo_extract`）。
# ⚠ 被 `get_task()` import 时 `__name__` 是 `scripts.training_data.tasks.example`，
#   不是 `"__main__"`，所以下面这段完全不执行。
if __name__ == "__main__":
    import shutil

    from .._paths import REPO_ROOT
    from ..constants import FEATURES, N_PLAYERS
    from ..dataset import SampleDataset
    from ..prepare import SPLITS, build

    TASK = Path(__file__).stem  # 从文件名取任务名 —— 复制成 aim.py 时这行不用改
    N_CSZ = 2  # 用几个 csz 造（1 个约 13 s；全量 4 个约 50 s，全量正式跑请用 prepare.py）
    CSZ_DIR = REPO_ROOT / "data" / "demos" / "csz"
    OUT = REPO_ROOT / "tmp" / "training_data" / TASK

    print(f"[冒烟] task={TASK}\n[冒烟] csz ={CSZ_DIR}\n[冒烟] 产出={OUT}")

    # 1. 每次从零开始：旧的整体删掉，不用手动清（`tmp/` 在 .gitignore 里）
    shutil.rmtree(OUT, ignore_errors=True)

    # 2. 造数据。`limit=N_CSZ` 是专门给这种情况用的（划分只跟文件名有关，所以前 N 个的
    #    split 和全量跑时完全一致，不是随机抽样）
    counts = build(CSZ_DIR, OUT, task=TASK, limit=N_CSZ)
    print(f"[冒烟] 造出来 {counts}")

    # 3. 读回来：每个非空 split，**逐个下标**都读一遍（能抓到"第 37 个回合特别短"这类）
    for split in SPLITS:
        ds = SampleDataset(OUT / split, task=TASK)
        n = len(ds)
        print(f"[冒烟] {split:<5} len={n}")
        if n == 0:
            continue  # 空 split 是合法的：这几张图都分到别的 split 去了

        for i in range(n):
            s = ds[i]
            where = f"{split}[{i}] {ds.samples[i].name}"

            # ★★★ 你这个任务真正该满足的东西，写在这里 ★★★
            assert sorted(s) == sorted(KEYS), f"{where}: keys 不对：{sorted(s)}"
            p = s["players"]
            assert p.shape[1] == N_PLAYERS, f"{where}: players 应是 (T,{N_PLAYERS},F)，现在 {tuple(p.shape)}"
            assert p.shape[2] == len(FEATURES), f"{where}: 特征列应是 {len(FEATURES)}，现在 {p.shape[2]}"
            assert s["alive"].shape == p.shape[:2], f"{where}: alive 形状和 players 对不上"
            assert s["ticks"].shape[0] == p.shape[0], f"{where}: ticks 和 players 的 T 对不上"

        print(
            f"[冒烟]   {ds.samples[0].name}: "
            + "  ".join(f"{k}{tuple(v.shape)}" for k, v in ds[0].items())
        )

    # 4. 要画图/存档就画在这（这台机器的 conda 环境还没装 matplotlib：pip install matplotlib）
    #      import matplotlib
    #      matplotlib.use("Agg")                    # 服务器上没有显示，必须 Agg
    #      import matplotlib.pyplot as plt
    #      s = SampleDataset(OUT / "train", task=TASK)[0]
    #      plt.figure(figsize=(12, 4))
    #      plt.imshow(s["players"][:, 0, :].T, aspect="auto")   # 玩家 0：20 个特征 × T
    #      plt.colorbar(); plt.tight_layout()
    #      plt.savefig(OUT / "player0.png", dpi=120)
    #      print(f"[冒烟] 图在 {OUT / 'player0.png'}")

    print("[冒烟] ✅ 通过")
