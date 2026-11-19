"""用 **t-SNE**（或 `--method pca`）把一个 run 的 `weapon_emb` 降维画出来，看"相似的枪是不是靠在一起"。

    python -m training.aim.weapon_embedding --run models/aim/20261108-131429
    python -m training.aim.weapon_embedding --run models/aim/20261108-131429 --method pca
    python -m training.aim.weapon_embedding --run models/aim/20261108-131429 \
        --compare models/aim/20261027-231512 --out runs/aim/weapon-embedding.png

依赖 **scikit-learn**（只有这个可视化脚本用；训练 / 推理路径不需要它）。

## 为什么 t-SNE 一个人说了不算

`weapon_emb` 是 `(44, 1024)`，本地样本只喂过其中 26 把枪。t-SNE 对这种小样本
**非常乐观**：它按每点的 perplexity 把局部尺度归一化，然后把"谁离谁近"摊到平面上 ——
输入哪怕全是随机噪声，它也会还你几个漂亮的团。所以图里固定放一个对照面板：

* **A** 训练权重（去掉 padding 行）的 t-SNE，余弦距离，带名字、按类别上色；
* **B** 同形状的**随机初始化**（`nn.Embedding` 默认 N(0,1)）走**同一套参数** ——
  如果 A 里的团和 B 里的团看起来一样热闹，那团就是 t-SNE 画的，不是学到的。
  （给 B 套上**同一组类别标签**，算"同类 − 跨类"应该 ≈ 0；A 应该明显为正。）

定量结论不看图，看**余弦矩阵上的置换检验**（`--perm`，默认 5000 次）：同类两两
相似度比跨类高多少 σ、把类别标签随机重排后还能不能有这么大。t-SNE 负责"看"，
σ 和 p 负责"信"。

## 三个坑

1. **没出现过的武器还是随机初始化**（`nn.Embedding` 默认 N(0,1)）。共 43 把，
   本地只有 26 把过 `--min-count`，剩下 17 把的相似度是纯噪声 —— 图里不画：它们会把
   t-SNE 的邻域抢走，把真正的关系挤扁；只在"没画出来"那行字里交代。
2. **单份权重的绝对值没有意义**：cos 的**排序**在两次独立训练之间只有 r≈0.53 的
   一致性，所以"这两把枪很像"这类结论要看共识矩阵（`--compare` 把两份权重各自
   z 分数化后平均），不能看单份。
3. **t-SNE 的坐标没有单位**：只能读"谁和谁一伙"，不能读距离大小，也不能跟另一次
   运行的图逐点比。`init` 固定成 `pca`（向量）/ `random+seed`（预计算距离，sklearn
   不允许前者），同一个 `--seed` 重跑结果一致；换 `--perplexity` 看结构稳不稳
   （同类的枪应该一直在一起，而不是这次一伙、下次换一伙）。

## 这个 run（20261108-131429）的结果

本地 449 条样本喂过 **26/43** 把枪（`--min-count 100` 人·帧）：

* **结构是真的**：同类 +0.50σ / 跨类 -0.13σ，差 **+0.63σ**，置换检验 **p=0.0002**
  （5000 次）；把同一组类别标签套到随机初始化上，差只有 **+0.12σ**。
* **但"枪族"不是主轴**：nade +1.39σ、gear +1.78σ 抱得最紧，**smg 反而 -1.54σ**
  （SMG 之间互不相似），手枪×步枪 **-0.36σ**；留一最近类心 0.385 对多数类基线
  0.308 只高 8 个点。这一版的表更像"道具 / 装备 / 手枪"一坨 + "长枪"一坨。
* **别读几何**：perplexity 8 → 14，5 近邻只重合 **0.60**（随机对照 0.59）。图适合看
  "谁和谁近"，"学到了类别"这件事要靠上面的 σ 和 p。
* 最像的几对里有明显的跨类配对（P250–Smoke +3.4σ、Deagle–Molotov +2.6σ），说明单份
  权重的排序里噪声不小 —— 这正是要 `--compare` 做共识矩阵的原因。

## t-SNE 和 PCA 各看什么

`--method tsne`（默认）和 `--method pca` 共用同一套统计 / 对照 / 热图，所以两张图能
直接对比：

| | t-SNE | PCA |
|---|---|---|
| 轴 | 没有单位，只读"谁和谁近" | 有单位：前两主成分的方差占比（这个 run：6% + 5%） |
| 对噪声 | 会把噪声放大成团（所以要有 B 对照） | 不放大，画不出来就是画不出来 |
| 邻域保持（trustworthiness, k=5） | 0.777（随机对照 0.745） | 0.750（随机对照 0.696） |

两维 PCA 只留下 **11%** 的方差，26 个点挤在中间、好几对几乎重合（AK-47 和 M4A4 基本
落在同一点），只有一条淡淡的"手枪/道具 ↔ 长枪"斜向分离 —— 1024 维里的结构**不是
线性的**。所以：看分组用 t-SNE（别数团），"分组不是 t-SNE 的幻觉"用 PCA 里也能看到
同样的两块来佐证。
"""

from __future__ import annotations

import argparse
import inspect
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from scripts.training_data.constants import WEAPON_NAMES

#: 武器 → 类别。只用来上色和算"同类/跨类"，不参与任何训练。
CATEGORY: dict[str, str] = {
    "Desert Eagle": "pistol", "Dual Berettas": "pistol", "Five-SeveN": "pistol",
    "Glock-18": "pistol", "P2000": "pistol", "USP-S": "pistol", "P250": "pistol",
    "CZ75 Auto": "pistol", "Tec-9": "pistol", "R8 Revolver": "pistol",
    "MAC-10": "smg", "MP9": "smg", "MP7": "smg", "MP5-SD": "smg",
    "UMP-45": "smg", "P90": "smg", "PP-Bizon": "smg",
    "AK-47": "rifle", "AUG": "rifle", "FAMAS": "rifle", "Galil AR": "rifle",
    "M4A4": "rifle", "M4A1-S": "rifle", "SG 553": "rifle",
    "AWP": "sniper", "SCAR-20": "sniper", "G3SG1": "sniper", "SSG 08": "sniper",
    "Nova": "shotgun", "XM1014": "shotgun", "MAG-7": "shotgun", "Sawed-Off": "shotgun",
    "M249": "mg", "Negev": "mg",
    "Zeus x27": "gear", "C4 Explosive": "gear", "Knife": "gear",
    "High Explosive Grenade": "nade", "Flashbang": "nade", "Smoke Grenade": "nade",
    "Molotov": "nade", "Incendiary Grenade": "nade", "Decoy Grenade": "nade",
}

CAT_ORDER = ("pistol", "smg", "rifle", "sniper", "shotgun", "mg", "nade", "gear")
CAT_COLOR = {
    "pistol": (0xE6, 0x7E, 0x22), "smg": (0x2E, 0x9E, 0x4F),
    "rifle": (0xC0, 0x39, 0x2B), "sniper": (0x7B, 0x4E, 0xA3),
    "shotgun": (0x2A, 0x84, 0xA8), "mg": (0xA8, 0x8B, 0x2A),
    "nade": (0x8D, 0x6E, 0x63), "gear": (0x55, 0x55, 0x55),
}

#: 画图用的短名（长名字在 26 个点的散点里会互相压住）。
SHORT = {
    "Desert Eagle": "Deagle", "Dual Berettas": "Duals", "Glock-18": "Glock",
    "High Explosive Grenade": "HE Nade", "Incendiary Grenade": "Incendiary",
    "Smoke Grenade": "Smoke", "C4 Explosive": "C4", "Zeus x27": "Zeus",
    "Five-SeveN": "Five-SeveN", "Galil AR": "Galil", "Sawed-Off": "Sawed-Off",
}
SHORT = {n: SHORT.get(n, n) for n in WEAPON_NAMES}      # 其余用原名


# ── 降维 ────────────────────────────────────────────────────────────────────


def _tsne(perplexity: float, *, metric: str, seed: int, iters: int) -> Any:
    """构造 `sklearn.manifold.TSNE`，顺手抹平 `max_iter` / `n_iter` 的版本差异。

    `metric="precomputed"` 时不给 `init="pca"`：sklearn 不支持（距离矩阵没有"方向"，
    拿它做 PCA 初始化是没有意义的）。
    """
    try:
        from sklearn.manifold import TSNE
    except ImportError as exc:                        # pragma: no cover
        raise SystemExit("这个脚本要 scikit-learn：pip install scikit-learn") from exc

    kw: dict[str, Any] = {
        "n_components": 2,
        "early_exaggeration": 12.0,
        "learning_rate": "auto",
        "random_state": seed,
        "metric": metric,
        "init": "random" if metric == "precomputed" else "pca",
    }
    params = inspect.signature(TSNE.__init__).parameters
    kw["max_iter" if "max_iter" in params else "n_iter"] = iters
    return TSNE(perplexity=float(perplexity), **kw)


def _clip_perplexity(perplexity: float, n: int | None) -> float:
    """sklearn 要求 `perplexity < n`；小样本按 n/3 收一下（否则退化成"人人都是邻居"）。"""
    if n is None:
        return float(perplexity)
    return float(np.clip(perplexity, 2.0, max(2.0, (n - 1) / 3.0)))


def tsne_from_vectors(x: np.ndarray, perplexity: float, seed: int = 0,
                      iters: int = 1000) -> np.ndarray:
    """`(n, d)` 向量 → `(n, 2)`，余弦距离。"""
    model = _tsne(_clip_perplexity(perplexity, len(x)), metric="cosine",
                  seed=seed, iters=iters)
    return np.asarray(model.fit_transform(np.asarray(x, dtype=np.float64)), dtype=np.float64)


def tsne_from_dist(dist: np.ndarray, perplexity: float, seed: int = 0,
                   iters: int = 1000) -> np.ndarray:
    """`(n, n)` 距离矩阵 → `(n, 2)`（对角线必须是 0）。"""
    d = np.array(dist, dtype=np.float64, copy=True)
    np.fill_diagonal(d, 0.0)
    model = _tsne(_clip_perplexity(perplexity, len(d)), metric="precomputed",
                  seed=seed, iters=iters)
    return np.asarray(model.fit_transform(d), dtype=np.float64)


def cos_dist(e: np.ndarray) -> np.ndarray:
    """余弦距离 `1 - cos`，对角线 0。"""
    n = np.linalg.norm(e, axis=1, keepdims=True).clip(min=1e-9)
    u = e / n
    return np.clip(1.0 - u @ u.T, 0.0, 2.0)


def sim_to_dist(s: np.ndarray) -> np.ndarray:
    """相似度（σ 单位的 z 分数）→ 距离。t-SNE 只用排序，所以减一个常数就够了。"""
    d = float(np.max(s)) - np.asarray(s, dtype=np.float64)
    np.fill_diagonal(d, 0.0)
    return np.clip(d, 0.0, None)


def pca2(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """中心化后的前两个主成分，返回 `(xy, 两轴各自的方差占比)`。

    PCA 和 t-SNE 的差别就在这两轴：PCA 的轴有**单位**（能说"只留下 x% 的方差"），
    而且它是线性的 —— 它看不见"弯的"结构，也**不会**把噪声放大成团。
    """
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(0, keepdims=True)
    _, s, vt = np.linalg.svd(x, full_matrices=False)
    evr = (s ** 2) / max(float((s ** 2).sum()), 1e-12)
    return x @ vt[:2].T, evr[:2]


def trustworthiness(dist: np.ndarray, xy: np.ndarray, k: int = 5) -> float:
    """2D 里保住了多少"高维里的近邻"。

    衡量的是 **t-SNE 有没有把这个距离矩阵画歪**，不是"结构真不真"：随机对照也有
    它自己的值，通常还不低。所以别拿它当"学到了东西"的证据。
    """
    from sklearn.manifold import trustworthiness as tw

    k = int(min(k, len(dist) - 1))
    return float(tw(dist, xy, n_neighbors=k, metric="precomputed"))


def knn_overlap(xy1: np.ndarray, xy2: np.ndarray, k: int = 5) -> float:
    """两个 2D 布局的 k 近邻重合率（1 = 每个点的近邻完全一样）。

    换一个 perplexity 再跑一遍 t-SNE，看"谁和谁一伙"是不是同一条结论 —— 这是 t-SNE
    唯一能自证的地方（坐标没意义，但邻域应当稳定）。随机对照也会有一个值。
    """
    if len(xy1) < k + 1:
        return float("nan")

    def nn(xy: np.ndarray) -> list[set[int]]:
        d = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=-1)
        np.fill_diagonal(d, np.inf)
        return [set(np.argsort(r)[:k].tolist()) for r in d]

    return float(np.mean([len(x & y) / k for x, y in zip(nn(xy1), nn(xy2))]))


# ── 统计 ────────────────────────────────────────────────────────────────────


def same_cross_gap(m: np.ndarray, labels: np.ndarray) -> float:
    """同类两两相似度的均值 − 跨类的均值。

    `m` 是**已经按 idx 取过子矩阵**的相似度（`n × n`），`labels` 是那 n 个点的类别 ——
    内部一律用局部下标，免得把全局的武器下标套到位数更小的矩阵上。
    """
    iu, ju = np.triu_indices(len(m), 1)
    same = labels[iu] == labels[ju]
    vals = m[iu, ju]
    if not same.any() or same.all():
        return float("nan")
    return float(vals[same].mean() - vals[~same].mean())


def perm_pvalue(m: np.ndarray, labels: np.ndarray,
                n_perm: int = 5000, seed: int = 0) -> tuple[float, float]:
    """类别标签随机重排后，"同类 − 跨类"还能不能有观测值那么高。返回 `(gap, p)`。

    这是"这张表学到了类别"的唯一硬证据：t-SNE 的团、最近类心准确率都可能是 26 个点
    的巧合，置换检验不是（p 的分辨率 = 1/(n_perm+1)）。
    """
    iu, ju = np.triu_indices(len(m), 1)
    vals = m[iu, ju]
    obs = same_cross_gap(m, labels)
    if n_perm <= 0:
        return obs, float("nan")

    # 一次生成所有置换：`(n_perm, n)` 的标签矩阵，再按 pair 展开，避免 Python 循环。
    rng = np.random.default_rng(seed)
    perms = rng.permuted(np.broadcast_to(labels, (n_perm, len(labels))).copy(), axis=1)
    same = perms[:, iu] == perms[:, ju]
    n_same = same.sum(axis=1)
    n_diff = same.shape[1] - n_same
    ok = (n_same > 0) & (n_diff > 0)
    gaps = np.full(n_perm, -np.inf)
    gaps[ok] = ((same * vals).sum(axis=1)[ok] / n_same[ok]
                - ((~same) * vals).sum(axis=1)[ok] / n_diff[ok])
    return obs, float((np.sum(gaps >= obs) + 1) / (n_perm + 1))


def centroid_acc(m: np.ndarray, labels: np.ndarray) -> float:
    """留一最近类心：只用相似度矩阵，不看任何向量。"""
    n = len(m)
    ok = 0
    for i in range(n):
        best, best_cat = -9.0, None
        for cat in set(labels.tolist()):
            mem = [j for j in range(n) if j != i and labels[j] == cat]
            if not mem:
                continue
            v = float(np.mean(m[i, mem]))
            if v > best:
                best, best_cat = v, cat
        ok += best_cat == labels[i]
    return ok / n


def zscore(c: np.ndarray) -> np.ndarray:
    """按自己的非对角分布标准化 —— 两份权重的 cos 绝对值尺度不同，只有 σ 可比。"""
    off = c[np.triu_indices(len(c), 1)]
    return (c - off.mean()) / off.std()


# ── 权重 / 样本 ─────────────────────────────────────────────────────────────


def load_embedding(run_dir: str | Path) -> np.ndarray:
    """读 `weapon_emb.weight`，去掉 padding 行 0，返回 `(43, d)`。"""
    ck = torch.load(Path(run_dir) / "best.pt", map_location="cpu", weights_only=False)
    return ck["model"]["weapon_emb.weight"].float().numpy()[1:]


def count_weapons(root: str | Path) -> Counter:
    """每把枪在样本里出现过多少「人·帧」。

    新版样本是 `ctx_player_weapon (T,10)` 逐帧全场（-1 = 无武器 / 已死）；
    旧样本是 `atk_weapon` / `tgt_weapon` 两条全局下标。两种都认。
    """
    c: Counter = Counter()
    for f in sorted(Path(root).glob("*.npz")):
        with np.load(f) as z:
            if "ctx_player_weapon" in z.files:
                vals = z["ctx_player_weapon"].ravel()
                vals = vals[vals >= 0]
                for i, n in zip(*np.unique(vals, return_counts=True)):
                    c[int(i)] += int(n)
            else:
                for key in ("atk_weapon", "tgt_weapon"):
                    for w in z[key].ravel():
                        c[int(w)] += 1
    return c


def weapon_ablation(run_dir: str | Path, data_root: str | Path, mode: str,
                    bs: int = 16) -> tuple[float, float]:
    """把 `weapon_emb` 动一下再算准确率：`zero` = 全清零，`shuffle` = 打乱行。

    返回 `(末帧 acc, 全列 acc)`，两个方向都算（和训练时一样对调 A/B）。
    """
    from training.aim.dataset import collate_aim, load_arrays, swap_ab
    from training.aim.model import AimConfig, AimModel

    ck = torch.load(Path(run_dir) / "best.pt", map_location="cpu", weights_only=False)
    model = AimModel(AimConfig(**dict(ck["cfg"]["model"])))
    model.load_state_dict(ck["model"])
    with torch.no_grad():
        if mode == "zero":
            model.weapon_emb.weight.zero_()
        elif mode == "shuffle":
            perm = torch.randperm(len(WEAPON_NAMES),
                                  generator=torch.Generator().manual_seed(0)) + 1
            model.weapon_emb.weight[1:] = model.weapon_emb.weight[perm].clone()

    base = [load_arrays(f) for f in sorted(Path(data_root).glob("*.npz"))]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(dev).eval()
    hits = []
    with torch.inference_mode():
        for flip in (False, True):
            items = [{k: (v.clone() if torch.is_tensor(v) else v) for k, v in it.items()}
                     for it in base]
            if flip:
                for it in items:
                    swap_ab(it)
            order = np.argsort([int(it["seconds"].shape[0]) for it in items], kind="stable")
            for s in range(0, len(order), bs):
                ids = order[s:s + bs]
                batch = collate_aim([items[i] for i in ids])
                batch = {k: (v.to(dev) if torch.is_tensor(v) else v)
                         for k, v in batch.items()}
                p = model(batch)["probs"].float().cpu().numpy()
                for r, i in enumerate(ids):
                    t = int(items[i]["seconds"].shape[0])
                    hits.append(((p[r, :t] > 0.5) == (not flip),
                                 (p[r, t - 1] > 0.5) == (not flip)))
    all_acc = np.concatenate([h[0] for h in hits]).mean()
    last_acc = np.mean([h[1] for h in hits])
    return float(last_acc), float(all_acc)


# ── 画图 ────────────────────────────────────────────────────────────────────


def _font(size: int) -> ImageFont.FreeTypeFont:
    for name in ("msyh.ttc", "msyhbd.ttc", "simhei.ttf", "arial.ttf"):
        p = Path("C:/Windows/Fonts") / name
        if p.exists():
            return ImageFont.truetype(str(p), size)
    return ImageFont.load_default()


def _place(dr, pts, names, colors, box, font, obstacles=()) -> int:
    """画点 + 名字，名字躲开已经放下的框和所有点（简单贪心）。返回没放下的个数。"""
    x0, y0, x1, y1 = box
    used: list[tuple[float, float, float, float]] = list(obstacles)
    dropped = 0
    for (px, py), name, col in zip(pts, names, colors):
        dr.ellipse([px - 5, py - 5, px + 5, py + 5], fill=col, outline=(255, 255, 255))
        tw = dr.textlength(name, font=font)
        # 候选位置从右边开始，一圈圈往外找；都不行就丢掉这个名字，不叠着写。
        cand = [(9, -8), (9, 6), (-tw - 9, -8), (-tw - 9, 6)]
        for dy in (-22, 20, -36, 34, -50, 48):
            cand += [(9, dy), (-tw - 9, dy), (-tw / 2, dy)]
        placed = False
        for dx, dy in cand:
            bx0, by0 = px + dx, py + dy
            bx1, by1 = bx0 + tw + 4, by0 + 15
            if not (x0 <= bx0 and bx1 <= x1 and y0 <= by0 and by1 <= y1):
                continue
            if any(not (bx1 < u0 or bx0 > u1 or by1 < v0 or by0 > v1)
                   for u0, v0, u1, v1 in used):
                continue
            used.append((bx0 - 1, by0 - 1, bx1 + 1, by1 + 1))
            dr.rectangle([bx0 - 1, by0 - 1, bx1, by1], fill=(255, 255, 255))
            dr.text((bx0 + 1, by0), name, fill=(30, 30, 30), font=font)
            placed = True
            break
        if not placed:
            dropped += 1
    return dropped


def _scatter(dr, box, xy: np.ndarray, colors, names=None, font=None) -> int:
    """把一个 2D 布局等比例铺进 `box`（t-SNE 的两个轴没有单位，必须等比）。"""
    x0, y0, x1, y1 = box
    dr.rectangle(box, outline=(232, 232, 232))
    span = float(np.abs(xy).max()) if len(xy) else 1.0
    sc = min(x1 - x0, y1 - y0) * 0.42 / max(span, 1e-9)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    pts = [(cx + float(p[0]) * sc, cy - float(p[1]) * sc) for p in xy]
    if names is None:
        for (px, py), col in zip(pts, colors):
            dr.ellipse([px - 5, py - 5, px + 5, py + 5], fill=col, outline=(255, 255, 255))
        return 0
    obstacles = [(px - 7, py - 7, px + 7, py + 7) for px, py in pts]
    return _place(dr, pts, names, colors,
                  (x0 + 4, y0 + 4, x1 - 4, y1 - 4), font, obstacles)


def _draw_heatmap(dr, s_full, idx, counts, cats, n_seen: int, x0: int, y0: int,
                  cell: int, font) -> None:
    """共识余弦矩阵，按类别分块；只画训过的枪（其余是随机初始化，纯噪声）。"""
    v = 3.0
    rows = sorted(range(n_seen),
                  key=lambda kk: (CAT_ORDER.index(cats[idx[kk]]), -counts.get(idx[kk], 0)))
    for rr, kk in enumerate(rows):
        i = idx[kk]
        nm = SHORT[WEAPON_NAMES[i]]
        dr.text((x0 - 10 - dr.textlength(nm, font=font), y0 + rr * cell + cell / 2 - 6),
                nm, fill=(60, 60, 60), font=font)
        dr.rectangle([x0 - 6, y0 + rr * cell, x0 - 2, y0 + (rr + 1) * cell - 1],
                     fill=CAT_COLOR[cats[i]])
        for cc, kk2 in enumerate(rows):
            j = idx[kk2]
            if i == j:
                col = (236, 236, 236)
            else:
                t = float(np.clip(s_full[i, j] / v, -1.0, 1.0))
                if t >= 0:      # 白 → 红（+3σ）
                    col = (int(255 - 72 * t), int(255 - 227 * t), int(255 - 227 * t))
                else:           # 白 → 蓝（-3σ）
                    col = (int(255 - 234 * -t), int(255 - 154 * -t), int(255 - 63 * -t))
            dr.rectangle([x0 + cc * cell, y0 + rr * cell,
                          x0 + (cc + 1) * cell - 1, y0 + (rr + 1) * cell - 1], fill=col)
    prev = None
    for rr, kk in enumerate(rows):
        g = cats[idx[kk]]
        if prev is not None and g != prev:
            y = y0 + rr * cell
            dr.line([x0 - 8, y, x0 + n_seen * cell, y], fill=(255, 255, 255), width=3)
        prev = g
    dr.text((x0 - 70, y0 + n_seen * cell + 8),
            "红 = 更像，蓝 = 更不像（单位 σ；对角线灰；只标左轴）",
            fill=(120, 120, 120), font=font)


def main() -> None:                                    # noqa: C901 - 一个报告脚本
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True, help="主 run（含 best.pt）")
    ap.add_argument("--compare", default=None, help="第二个 run，用来做共识矩阵（可选）")
    ap.add_argument("--data", default="data/demos/prepared/aim/train", help="数覆盖率的 npz 目录")
    ap.add_argument("--min-count", type=int, default=100, help="出现多少「人·帧」以上才算训过")
    ap.add_argument("--method", choices=("tsne", "pca"), default="tsne",
                    help="降维方法：t-SNE（默认，看局部结构）或 PCA（线性基线，轴有方差单位）")
    ap.add_argument("--perplexity", type=float, default=8.0, help="t-SNE 的 perplexity（26 个点用 5–10）")
    ap.add_argument("--seed", type=int, default=0, help="t-SNE / 置换检验的随机种子")
    ap.add_argument("--perm", type=int, default=5000, help="置换检验次数（0 = 跳过）")
    ap.add_argument("--out", default=None, help="默认 runs/aim/weapon-embedding-<method>.png")
    ap.add_argument("--ablate", action="store_true",
                    help="额外跑两次前向：把 weapon_emb 清零 / 打乱，看这张表到底重不重要")
    a = ap.parse_args()

    runs = [a.run] + ([a.compare] if a.compare else [])
    embs = [load_embedding(r) for r in runs]
    counts = count_weapons(a.data) if Path(a.data).is_dir() else Counter()
    seen = np.array([counts.get(i, 0) >= a.min_count for i in range(len(WEAPON_NAMES))])
    idx = np.where(seen)[0]
    n_seen, d = len(idx), embs[0].shape[1]
    cats = [CATEGORY.get(n, "gear") for n in WEAPON_NAMES]
    labels = np.array([cats[i] for i in idx])
    print(f"{' + '.join(runs)}：d={d}，训过 {n_seen}/43 把"
          f"（>= {a.min_count} 人·帧）")
    if n_seen < 5:
        raise SystemExit(f"只有 {n_seen} 把枪过 --min-count，t-SNE 画不出东西；调小 --min-count")

    sims = [1.0 - cos_dist(e) for e in embs]
    s = np.mean([zscore(m) for m in sims], axis=0)
    s_full = s.copy()
    np.fill_diagonal(s_full, 0.0)

    # 之后所有统计量都只在训过的枪上算：局部下标（0..n_seen-1）↔ `labels` 一一对应。
    sub = s_full[np.ix_(idx, idx)]
    iu, ju = np.triu_indices(n_seen, 1)
    same = labels[iu] == labels[ju]
    vals = sub[iu, ju]
    seen_sd = float(vals.std())
    gap = float(vals[same].mean() - vals[~same].mean())
    print(f"  同类 {vals[same].mean():+.2f}σ   跨类 {vals[~same].mean():+.2f}σ"
          f"   差 {gap:+.2f}σ（{int(same.sum())} / {int((~same).sum())} 对）")

    def within(cat: str) -> float | None:
        mem = [k for k in range(n_seen) if labels[k] == cat]
        if len(mem) < 2:
            return None
        return float(np.mean([sub[a, b] for a in mem for b in mem if a > b]))

    cats_bits = []
    for cat in ("nade", "gear", "pistol", "rifle", "sniper", "smg", "shotgun", "mg"):
        v = within(cat)
        if v is not None:
            cats_bits.append(f"{cat} {v:+.2f}σ")
    pr = [k for k in range(n_seen) if labels[k] == "pistol"]
    rf = [k for k in range(n_seen) if labels[k] == "rifle"]
    cross_pr = (float(np.mean([sub[i, j] for i in pr for j in rf]))
                if pr and rf else float("nan"))
    print("  各类内部: " + "  ".join(cats_bits) + f"   （手枪×步枪 {cross_pr:+.2f}σ）")

    acc_cons = centroid_acc(sub, labels)
    acc1 = centroid_acc(sims[0][np.ix_(idx, idx)], labels)
    base = Counter(labels.tolist()).most_common(1)[0][1] / n_seen
    print(f"  最近类心分类（留一）: 单 run {acc1:.3f} → 共识 {acc_cons:.3f}"
          f"（多数类基线 {base:.3f}）")

    r = float("nan")
    if len(embs) > 1:
        r = float(np.corrcoef(sims[0][np.ix_(idx, idx)][iu, ju],
                              sims[1][np.ix_(idx, idx)][iu, ju])[0, 1])
        print(f"  两份权重的一致性: r={r:+.3f}（单份的 cos 噪声 σ ≈ 1/√d = {1 / np.sqrt(d):.3f}，"
              f"训过的 {n_seen} 把实测 σ = {seen_sd:.3f}）")

    p_val = float("nan")
    if a.perm > 0:
        _, p_val = perm_pvalue(sub, labels, n_perm=a.perm, seed=a.seed)
        print(f"  置换检验（{a.perm} 次）: 同类−跨类 {gap:+.2f}σ，p={p_val:.4g}")

    order = sorted(zip(iu.tolist(), ju.tolist()), key=lambda p: -sub[p[0], p[1]])
    print("\n  最像的 8 对:")
    for i, j in order[:8]:
        print(f"    {WEAPON_NAMES[idx[i]]:<22} {WEAPON_NAMES[idx[j]]:<22} {sub[i, j]:+.2f}σ")
    print("  最不像的 5 对:")
    for i, j in order[-5:]:
        print(f"    {WEAPON_NAMES[idx[i]]:<22} {WEAPON_NAMES[idx[j]]:<22} {sub[i, j]:+.2f}σ")

    # ---------------------------------------------------------------- 降维
    # A/B 用**同一套参数**：同一种方法、同一个种子。t-SNE 的对照能看出"随机也会结团"，
    # PCA 的对照能看出"随机矩阵的前两主成分长什么样"（平坦、没有团）。
    dist_emb = cos_dist(embs[0][idx])
    ctrl = np.random.default_rng(a.seed).normal(size=(n_seen, d))
    dist_ctrl = cos_dist(ctrl)

    def project(x: np.ndarray) -> tuple[np.ndarray, str | None]:
        """`(n, 2)` + 轴上的说明（PCA 有方差占比，t-SNE 的轴没有单位）。"""
        if a.method == "pca":
            xy, evr = pca2(x)
            return xy, f"PC1 {evr[0] * 100:.0f}% · PC2 {evr[1] * 100:.0f}%"
        return tsne_from_vectors(x, a.perplexity, a.seed), None

    xy_a, axis_a = project(embs[0][idx])
    xy_b, axis_b = project(ctrl)
    tw_a = trustworthiness(dist_emb, xy_a)
    tw_b = trustworthiness(dist_ctrl, xy_b)
    gap_ctrl = same_cross_gap(zscore(1.0 - dist_ctrl), labels)

    hi_p = a.perplexity * 1.75
    stab_a = stab_b = float("nan")
    if a.method == "tsne":
        # 稳定性：同一个种子、换一个 perplexity，近邻结构应该基本不变（坐标本来就没意义）
        stab_a = knn_overlap(xy_a, tsne_from_vectors(embs[0][idx], hi_p, a.seed))
        stab_b = knn_overlap(xy_b, tsne_from_vectors(ctrl, hi_p, a.seed))
        print(f"\n  t-SNE（cosine，perp {a.perplexity:g}，seed {a.seed}）:"
              f" trustworthiness(k=5) 训练 {tw_a:.3f} / 随机对照 {tw_b:.3f}")
        print(f"  换 perp {hi_p:g} 后 5 近邻重合: 训练 {stab_a:.3f} / 随机对照 {stab_b:.3f}")
    else:
        print(f"\n  PCA（中心化，前两主成分）: 训练 {axis_a}；随机对照 {axis_b}")
        print(f"  trustworthiness(k=5): 训练 {tw_a:.3f} / 随机对照 {tw_b:.3f}"
              "（2 维装不下 1024 维的邻域，比 t-SNE 低是正常的）")
    print(f"  同一套类别标签下的「同类−跨类」: 训练 {gap:+.2f}σ · 随机对照 {gap_ctrl:+.2f}σ")

    proj_name = ("PCA（中心化，前两主成分）" if a.method == "pca"
                 else f"t-SNE（余弦，perp {a.perplexity:g}）")
    panels: list[tuple[np.ndarray, str, bool, str, str | None]] = [
        (xy_a, f"A 训练权重 · {proj_name}", True, "cat", axis_a),
        (xy_b, "B 对照 · 随机初始化（同形状 N(0,1)）", False, "gray", axis_b),
    ]
    if len(embs) > 1:
        if a.method == "pca":
            xy_c, evr_c = pca2(sub)
            panels.append((xy_c, "C 共识矩阵 · PCA（线性 MDS）", True, "cat",
                           f"PC1 {evr_c[0] * 100:.0f}% · PC2 {evr_c[1] * 100:.0f}%"))
        else:
            xy_c = tsne_from_dist(sim_to_dist(sub), a.perplexity, a.seed)
            panels.append((xy_c, "C 共识矩阵 · t-SNE", True, "cat", None))

    ab = None
    if a.ablate:
        print("\n  这张表到底重不重要（本地样本，两个方向）:")
        ab = {"base": weapon_ablation(a.run, a.data, "base")}
        for mode in ("zero", "shuffle"):
            ab[mode] = weapon_ablation(a.run, a.data, mode)
        for mode in ("base", "zero", "shuffle"):
            last, allc = ab[mode]
            print(f"    {mode:<8} 末帧 acc {last:.4f}   全列 acc {allc:.4f}")

    # ---------------------------------------------------------------- 画图
    cell = int(np.clip(760 / n_seen, 14, 22))
    heat_x0, heat_y0 = 1150, 118
    W = heat_x0 + n_seen * cell + 90
    H = 1020
    img = Image.new("RGB", (W, H), (255, 255, 255))
    dr = ImageDraw.Draw(img)
    f_title, f_lab, f_small = _font(25), _font(14), _font(11)
    dr.text((36, 18), f"武器 embedding · {'PCA' if a.method == 'pca' else 't-SNE'} · "
                      f"{Path(a.run).name} · d={d} · 训过 {n_seen}/43",
            fill=(20, 20, 20), font=f_title)
    dr.text((36, 52), "A 训练权重（带名字，颜色 = 类别）    "
                      "B 随机初始化对照（同形状 N(0,1)，同一套参数）    "
                      "右：共识余弦矩阵（红 = 更像）", fill=(90, 90, 90), font=f_lab)

    k = len(panels)
    left, right, top, bottom = 40, 1040, 152, 700
    gap_px = 14
    pw = (right - left - gap_px * (k - 1)) / k
    dropped = 0
    for t, (xy, title, with_labels, color_mode, note) in enumerate(panels):
        bx0 = left + t * (pw + gap_px)
        box = (bx0, top, bx0 + pw, bottom)
        dr.text((bx0 + 2, top - 34), title, fill=(110, 110, 110), font=f_small)
        if note:
            dr.text((bx0 + 2, top - 19), note, fill=(150, 150, 150), font=f_small)
        names = [SHORT[WEAPON_NAMES[i]] for i in idx] if with_labels else None
        colors = ([(150, 150, 150)] * n_seen if color_mode == "gray"
                  else [CAT_COLOR[cats[i]] for i in idx])
        dropped += _scatter(dr, box, xy, colors, names, f_small)

    _draw_heatmap(dr, s_full, idx, counts, cats, n_seen, heat_x0, heat_y0, cell, f_small)

    # 底部说明
    legend_y = 732
    y = legend_y
    for cat in CAT_ORDER:
        if not any(cats[i] == cat for i in idx):
            continue
        dr.rectangle([44, y + 3, 56, y + 15], fill=CAT_COLOR[cat])
        dr.text((62, y), cat, fill=(70, 70, 70), font=f_lab)
        y += 24
    tx = 200
    lines = [
        f"同类 {vals[same].mean():+.2f}σ   跨类 {vals[~same].mean():+.2f}σ   差 {gap:+.2f}σ"
        + (f"   置换检验 p={p_val:.3g}（{a.perm} 次）" if a.perm > 0 else ""),
        f"同一组类别标签下的差：训练 {gap:+.2f}σ vs 随机对照 {gap_ctrl:+.2f}σ"
        "（对照≈0 才说明 A 里的结构不是画出来的）",
        "各类内部：" + "   ".join(cats_bits) + f"（手枪×步枪 {cross_pr:+.2f}σ）",
        f"最近类心（留一）：单 run {acc1:.3f} → 共识 {acc_cons:.3f}"
        f"（多数类基线 {base:.3f}）",
        f"trustworthiness(k=5)：训练 {tw_a:.3f} · 随机对照 {tw_b:.3f}"
        + (f"（只说明图没画歪）；换 perp {hi_p:g} 后 5-近邻重合 {stab_a:.2f} / {stab_b:.2f}"
           if a.method == "tsne"
           else f"（2 维留下的方差：训练 {axis_a}，对照 {axis_b}）"),
        "两份权重余弦矩阵的相关 r = " + (f"{r:+.2f}" if len(embs) > 1 else "—")
        + f"；单份 cos 噪声 σ ≈ 1/√d = {1 / np.sqrt(d):.3f}"
        f"，训过的 {n_seen} 把实测 σ = {seen_sd:.3f}",
        "最像：" + "   ".join(f"{SHORT[WEAPON_NAMES[idx[i]]]}–{SHORT[WEAPON_NAMES[idx[j]]]} "
                             f"{sub[i, j]:+.1f}σ" for i, j in order[:4]),
        "最不像：" + "   ".join(f"{SHORT[WEAPON_NAMES[idx[i]]]}–{SHORT[WEAPON_NAMES[idx[j]]]} "
                              f"{sub[i, j]:+.1f}σ" for i, j in order[-3:]),
        f"没画出来的 {43 - n_seen} 把武器本地没出现过（或不足 {a.min_count} 人·帧），"
        "embedding 还是随机初始化 —— 两两相似度是纯噪声。",
    ]
    if dropped:
        lines.append(f"（有 {dropped} 个名字没找到空位没写上去）")
    if ab:
        lines.append(f"把 weapon_emb 清零 → 末帧 acc {ab['base'][0]:.3f} → {ab['zero'][0]:.3f}；"
                     f"只打乱行（保住统计、毁掉身份）→ {ab['shuffle'][0]:.3f}")
    for n, line in enumerate(lines):
        dr.text((tx, legend_y + n * 25), line, fill=(40, 40, 40), font=f_lab)

    out = Path(a.out or f"runs/aim/weapon-embedding-{a.method}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    print(f"\n  图写到 {out}")


if __name__ == "__main__":
    main()
