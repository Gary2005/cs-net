"""把训练产物（`<run>/last.pt` + `<run>/config.yaml`）压成"只带权重"的一对文件。

训练存档里真正占地方的从来不是模型：

    aim      854 MB = 271.5 MB 权重 + 542 MB AdamW 一阶/二阶动量 + 40 MB 杂项
    spatial 1,180 MB = 393.5 MB 权重 + 787 MB 动量

网页推理只读 `ckpt["model"]`（`app/ai/models.py::Bundle.load`），优化器状态是
"接着训"才需要的东西，发给人下载纯属浪费 —— 去掉后两个文件合计 665 MB，
是原来的 1/3。

产物（和训练目录**同级**，不塞进 run 目录里，目录留给 wandb 之类）：

    models/aim/20261119-013008/last.pt  →  models/aim/20261119-013008.pt
    models/aim/20261119-013008/config.yaml →  models/aim/20261119-013008.yaml

`app/ai/models.py::pinned_run` 认两种布局：扁平（`<run>.pt` + `<run>.yaml`，
推荐，网页用这个）和训练目录（`<run>/last.pt` + `<run>/config.yaml`，兜底）。

用法：

    python scripts/export_weights.py                    # 导 PINNED_WEIGHTS 那两个
    python scripts/export_weights.py models/aim/xxx/last.pt
    python scripts/export_weights.py ... --force        # 覆盖已有的产物
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

#: 留下来的键。`cfg` 是训练时的完整配置（1 KB），留着让 `.pt` 自己也能说明来历；
#: `optimizer` / `scheduler` 明确丢掉 —— 那才是体积的来源。
_KEEP = ("model", "epoch", "step", "best", "cfg")


def _mb(n: int | float) -> float:
    return round(float(n) / 2**20, 1)


def targets_of(weights: Path) -> tuple[Path, Path]:
    """`<...>/<run>/last.pt` → (`<...>/<run>.pt`, `<...>/<run>.yaml`)。

    输入已经是扁平布局（`<run>.pt`）时，产物就是它自己 —— 调用方据此跳过重导。
    """
    weights = Path(weights)
    run_dir = weights.parent
    # 训练产物的权重文件名固定是 last.pt / best.pt；别的名字说明已经是扁平文件了
    if weights.stem in ("last", "best") and (run_dir / "config.yaml").is_file():
        return run_dir.parent / f"{run_dir.name}.pt", run_dir.parent / f"{run_dir.name}.yaml"
    return weights, weights.with_suffix(".yaml")


def _optimizer_bytes(ckpt: dict[str, Any]) -> int:
    """优化器状态占多少字节（只为打印一句"省了多少"）。"""

    def walk(obj: Any) -> int:
        if isinstance(obj, dict):
            return sum(walk(v) for v in obj.values())
        if isinstance(obj, (list, tuple)):
            return sum(walk(v) for v in obj)
        numel = getattr(obj, "numel", None)
        element = getattr(obj, "element_size", None)
        if callable(numel) and callable(element):
            try:
                return int(numel()) * int(element())
            except (TypeError, RuntimeError):
                return 0
        return 0

    return sum(walk(ckpt.get(k)) for k in ("optimizer", "scheduler"))


def export(weights: Path, *, force: bool = False) -> int:
    """导出一个 checkpoint。返回 0 成功 / 1 跳过 / 2 失败。"""
    import torch

    weights = Path(weights)
    if not weights.is_file():
        print(f"[x] 没有这个文件：{weights}", flush=True)
        return 2
    out_pt, out_yaml = targets_of(weights)
    if out_pt.resolve() == weights.resolve():
        print(f"[=] {weights.name} 已经是扁平布局，跳过", flush=True)
        return 1
    if out_pt.exists() and not force:
        print(f"[=] {out_pt} 已存在（要覆盖加 --force）", flush=True)
        return 1

    before = weights.stat().st_size
    print(f"[…] {weights}（{_mb(before)} MB）", flush=True)
    ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "model" not in ckpt:
        print(f"[x] {weights} 里没有 'model'，不是训练存档？", flush=True)
        return 2

    state = ckpt["model"]
    params = sum(int(t.numel()) for t in state.values())
    slim = {k: ckpt[k] for k in _KEEP if k in ckpt}
    dropped = _optimizer_bytes(ckpt)

    out_pt.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_pt.with_suffix(out_pt.suffix + ".tmp")
    torch.save(slim, tmp)
    tmp.replace(out_pt)

    src_yaml = weights.parent / "config.yaml"
    if src_yaml.is_file():
        shutil.copyfile(src_yaml, out_yaml)

    after = out_pt.stat().st_size
    print(
        f"[ok] {out_pt.name}：{_mb(after)} MB"
        f"（原 {_mb(before)} MB，丢掉优化器/调度器 {_mb(dropped)} MB）"
        f" · {params / 1e6:.1f} M 参数",
        flush=True,
    )
    if out_yaml.is_file():
        print(f"[ok] {out_yaml.name}：{out_yaml.stat().st_size} B", flush=True)
    else:
        print(f"[!] {src_yaml} 不存在，没复制配置", flush=True)
    return 0


def pinned_paths() -> list[Path]:
    from app.ai import models

    return list(models.PINNED_WEIGHTS.values())


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="导出「只带权重」的 .pt + .yaml")
    ap.add_argument("checkpoints", nargs="*", type=Path,
                    help="last.pt / best.pt（默认：PINNED_WEIGHTS 那两个）")
    ap.add_argument("--force", action="store_true", help="覆盖已存在的产物")
    args = ap.parse_args(argv)

    paths = list(args.checkpoints) or pinned_paths()
    codes = [export(p, force=args.force) for p in paths]
    return 0 if all(c in (0, 1) for c in codes) else 2


if __name__ == "__main__":
    raise SystemExit(main())
