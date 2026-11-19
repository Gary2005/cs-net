"""用 `best.pt` 给一条样本算一遍预测 —— 纯推理层，不碰 HTTP。

    from . import predict
    p = predict.Predictor("models/aim/20261108-131429")
    out = p.predict(path)        # {"probs": [...], "probs_rev": [...], ...}

为什么 torch 是懒加载的
----------------------
这个工具本来只依赖 numpy（读 npz）+ Pillow（出 PNG），见 `data.py` 的说明。
模型预测是**可选**功能：`import torch` 要好几秒，没装的时候整个页面也该照常能开。
所以 torch / yaml / `training.aim.model` 全在 `Predictor.load()` 里才 import，
缺了就把原因包成 `ModelUnavailable` 抛出去，由 `server.py` 变成一句人话。

预测点是什么
-----------
模型是 decoder-only + 因果注意力，一条样本**一次前向**就吐出 `T` 个预测点（见
`training/aim/model.py` 的 `forward`）。每帧拼的是 `[A_t] [B_t] [PRED_t]`：

    列 t (0..T-1)  看完了第 0..t 帧（同帧的 A、B 状态都在这个预测点前面），
                   时间戳 = `seconds[t]`

所以"每一帧一个预测"是**不用循环**的：一次前向拿到整条曲线，x 轴就是帧号，前端把
"拖到第 i 帧"直接对应到**第 i 列**。「一帧都没看到」时模型不知道 A、B 是谁（身份只
存在于逐帧 token 里），所以这一版**没有首帧预测**。

为什么两个方向一起算
-------------------
`dataset.py` 训练时把 50% 的样本 A/B 对调（标签跟着取反），所以两个方向模型都学过。
这里一次 batch 塞两条（A = atk / A = tgt），两条曲线应当关于 0.5 镜像
（`p + p' ≈ 1`：这对人**恰好**有一个击杀了另一个）—— 白送的校准检查，代价只是
batch 从 1 变成 2。

**这里不随机对调**：固定 A = `atk`（磁盘上存的方向），所以主曲线永远是
`P(ATK 击杀 TGT)`，和页面上 ATK/TGT 的标注一致。反方向看第二条曲线。

和训练的一致性
-------------
输入走的还是 `training/aim/dataset.py` 的 `load_arrays` + `collate_aim`（和
`DataLoader` 用的是同一段代码），不在这里重写一份 collate —— 否则哪天特征表或
补齐逻辑改了，训练和可视化就会悄悄看不同的东西。区别只有两点：batch = 2（两个
方向）、取的是 `probs` 而不是 loss。
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Sequence
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Any, Optional

#: `scripts/training_data_visualizer/aim/predict.py` → 仓库根
PROJECT_ROOT = Path(__file__).resolve().parents[3]

#: 自动找训练产物的地方：每个**直接子目录**算一个 run（里面要有 config.yaml + 权重）。
#: 两个都扫，因为 `train.save_dir` 怎么填两种都见过：
#:   `models/aim/<时间戳>/`  `save_dir: models/aim`（本仓库现在的产物在这里）
#:   `models/<时间戳>/`      `save_dir: models`（老习惯，保留兼容）
DEFAULT_RUNS_DIRS = (PROJECT_ROOT / "models", PROJECT_ROOT / "models" / "aim")

#: 优先 `best.pt`（`train.py` 只在 valid loss 变好时写），没有才退到 `last.pt`。
_WEIGHTS = ("best.pt", "last.pt")


class ModelUnavailable(RuntimeError):
    """预测功能用不了：没权重 / 没装 torch / 权重和 config 对不上。

    这不是"服务器出错了"，是一条**给人看**的说明，`server.py` 按 503 返回。
    """


def weights_of(run_dir: Path) -> Optional[Path]:
    """这个目录里能找到的权重文件（`best.pt` 优先）。"""
    for name in _WEIGHTS:
        path = run_dir / name
        if path.is_file():
            return path
    return None


def find_runs(roots: Sequence[Path] = DEFAULT_RUNS_DIRS) -> list[Path]:
    """默认目录里"有 config.yaml + 有 best.pt/last.pt"的子目录，新的在前。

    只看文件名，不 import torch —— 启动打 banner 时要用，不能为它等几秒。
    `models/` 和 `models/aim/` 都扫：`models/aim/` 自己虽然是 `models/` 的子目录，
    但里面没有 config.yaml，会被自然跳过，所以两个根不会互相干扰。
    """
    seen: set[Path] = set()
    runs: list[Path] = []
    for root in roots:
        if not root.is_dir():
            continue
        for d in root.iterdir():
            if not d.is_dir():
                continue
            key = d.resolve()
            if key in seen or not (d / "config.yaml").is_file():
                continue
            if weights_of(d) is not None:
                seen.add(key)
                runs.append(d)
    runs.sort(key=lambda d: (d.stat().st_mtime, d.name), reverse=True)
    return runs


def resolve_run_dir(spec: str | Path | None = None) -> Path:
    """`--model-dir` → 一个确定的模型目录（目录里必须有 config.yaml + 权重）。

    `spec` 为空时取默认目录下**最新**的一个 run（按目录 mtime）——本地工具，
    刚跑完的那个就是你想看的那个。

    `spec` 的写法（按顺序试，第一个存在的赢）：

        models/aim/20261108-131429  相对路径，先按仓库根、再按当前目录
        D:/project/.../20261108-131429   绝对路径
        20261108-131429             只给 run 名字，去 `models/` 和 `models/aim/` 里找
    """
    if spec:
        raw = Path(spec).expanduser()
        if raw.is_absolute():
            # 绝对路径：就算不存在也不去别处猜，直接报错更清楚。
            candidates = [raw]
        else:
            # 顺序即优先级：仓库根 → 当前目录 → `models/` → `models/aim/`（`cwd` 和仓库根
            # 常常是同一个，后面去重，报错信息才不至于把同一条路径列两遍）。
            candidates = [PROJECT_ROOT / raw, Path.cwd() / raw]
            candidates += [root / raw for root in DEFAULT_RUNS_DIRS]
        tried_paths: list[Path] = []
        for c in candidates:
            c = c.resolve()
            if c not in tried_paths:
                tried_paths.append(c)
        run = next((c for c in tried_paths if c.is_dir()), None)
        if run is None:
            raise ModelUnavailable(
                f"模型目录不存在：{spec}（找过 {'、'.join(str(c) for c in tried_paths)}）"
            )
        if not (run / "config.yaml").is_file():
            raise ModelUnavailable(f"{run} 里没有 config.yaml（训练会把它存在 run 目录）")
        if weights_of(run) is None:
            raise ModelUnavailable(f"{run} 里没有 best.pt / last.pt")
        return run

    runs = find_runs()
    if not runs:
        where = "、".join(str(p) for p in DEFAULT_RUNS_DIRS)
        raise ModelUnavailable(
            f"没在 {where} 下找到带 config.yaml + best.pt 的训练产物；"
            "用 --model-dir 指定目录"
        )
    return runs[0]


class Predictor:
    """一个 run 目录 = 一个模型。`load()` 只做一次，推理串行（各一把锁）。"""

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.weights = weights_of(self.run_dir)
        self._load_lock = threading.Lock()
        #: 推理也串行：`ThreadingHTTPServer` 是多线程的，而模型只有一份；
        #: 两条请求同时 forward 会让计时和显存分配互相干扰，收益也基本没有。
        #: 和 `_load_lock` 分开是为了 `load()` 里能安全地拿它之外的那把。
        self._infer_lock = threading.Lock()
        self._model: Any = None
        self._torch: Any = None
        self._device = "cpu"
        self._cfg: dict[str, Any] = {}
        self._meta: dict[str, Any] = {}

    # ------------------------------------------------------------------ 加载

    def load(self) -> Any:
        """建模型 + 读权重（幂等）。失败抛 `ModelUnavailable`，成功返回 model。"""
        with self._load_lock:
            if self._model is not None:
                return self._model
            if self.weights is None:
                raise ModelUnavailable(f"{self.run_dir} 里没有 best.pt / last.pt")
            try:
                import yaml
                import torch
                from training.aim.model import AimConfig, AimModel
            except ImportError as exc:  # 没装 torch / pyyaml，或者不在仓库根起
                raise ModelUnavailable(
                    f"模型预测需要 torch + pyyaml（{exc}）；"
                    "只想看数据就加 --no-model"
                ) from exc

            try:
                raw = yaml.safe_load((self.run_dir / "config.yaml").read_text(encoding="utf-8"))
                model_cfg = dict((raw or {}).get("model") or {})
                # 只留 `AimConfig` 认识的字段：旧 run 的 config 里还有 `view_grid` /
                # `view_channels` / `n_layers` 这些已经删掉的键，直接 unpack 会 TypeError。
                known = {f.name for f in dataclass_fields(AimConfig)}
                model_cfg = {k: v for k, v in model_cfg.items() if k in known}
                cfg = AimConfig(**model_cfg)
            except (OSError, TypeError, ValueError) as exc:
                raise ModelUnavailable(f"{self.run_dir}/config.yaml 里的 model 段读不了：{exc}")

            model = AimModel(cfg)
            try:
                ckpt = torch.load(self.weights, map_location="cpu", weights_only=False)
                state = ckpt["model"] if isinstance(ckpt, dict) else ckpt
                model.load_state_dict(state)
            except (OSError, RuntimeError, KeyError, TypeError) as exc:
                if "pred_start" in str(exc):
                    # 旧权重是 `T+1` 列的布局（打头的首帧预测 + 末尾那个预测点），
                    # token 顺序和参数集都变了，只能重训，不是改 config 能救的。
                    raise ModelUnavailable(
                        f"{self.weights.name} 是旧版布局（T+1 个预测点、有 pred_start）"
                        "的权重，和现在的模型不兼容 —— 用新代码重训一个 run 再看"
                    ) from exc
                raise ModelUnavailable(
                    f"{self.weights.name} 和 config.yaml 对不上：{str(exc).splitlines()[0]}"
                )

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(self._device).eval()
            self._torch = torch
            self._model = model
            self._cfg = model_cfg
            self._meta = {
                "epoch": _int_or_none(ckpt.get("epoch") if isinstance(ckpt, dict) else None),
                "step": _int_or_none(ckpt.get("step") if isinstance(ckpt, dict) else None),
                "best": _json_safe(ckpt.get("best") if isinstance(ckpt, dict) else None),
            }
            return model

    # ------------------------------------------------------------------ 元信息

    def info(self) -> dict[str, Any]:
        """给 `/api/model` 用的说明：哪份权重、在哪个设备、多少参数、训到第几个 epoch。"""
        model = self.load()
        params = sum(p.numel() for p in model.parameters())
        return {
            "ok": True,
            "dir": str(self.run_dir),
            "name": self.run_dir.name,
            "weights": self.weights.name if self.weights else None,
            "device": self._device,
            "params": params,
            "epoch": self._meta.get("epoch"),
            "step": self._meta.get("step"),
            "best": self._meta.get("best") or {},
            "config": _json_safe(self._cfg),
        }

    # ------------------------------------------------------------------ 推理

    def predict(self, path: str | Path) -> dict[str, Any]:
        """一条 npz → `T` 个预测点（每帧一个，两个方向各一条曲线）。

        返回的 `probs` 是 `P(ATK 击杀 TGT)`（A = `atk`），`probs_rev` 是把 A/B
        对调后的 `P(TGT 击杀 ATK)`；`time[t]` 是第 t 列看完的那一帧的秒数。
        """
        model = self.load()
        torch = self._torch
        from training.aim.dataset import collate_aim, load_arrays, swap_ab

        item = load_arrays(path)
        rev = swap_ab(dict(item))        # 同一个样本，A/B 对调 + 标签取反
        batch = collate_aim([item, rev])
        batch = {k: v.to(self._device) for k, v in batch.items()}

        with self._infer_lock, torch.inference_mode():
            t0 = time.perf_counter()
            try:
                out = model(batch)
            except (KeyError, RuntimeError) as exc:
                # 数据集这一版换了布局（逐帧全场 + 遮挡矩阵），而 models/ 里的权重还是
                # 旧布局（atk/tgt + 视角图）。权重救不回来，只能重训 —— 这里给一句人话，
                # 别让页面上出现一段 KeyError。
                raise ModelUnavailable(
                    f"{self.weights.name} 是按旧样本布局（atk/tgt + 视角图）训的，"
                    f"和新数据 / 新模型对不上：{str(exc).splitlines()[0]}"
                ) from exc
            ms = (time.perf_counter() - t0) * 1000.0

        probs = out["probs"].float().cpu()
        if not bool(out["pred_mask"].all()):
            # 两条样本的 T 相同，所以两个方向都该是满列。出现 False 说明 collate 或
            # forward 的 mask 逻辑变了，宁可报错也别画一条少一截的曲线。
            raise RuntimeError("预测列数少于 T（pred_mask 里有 False），collate/forward 逻辑变了？")

        T = int(item["seconds"].shape[0])
        seconds = [round(float(s), 4) for s in item["seconds"].tolist()]
        a, b = probs[0].tolist(), probs[1].tolist()
        return {
            "frames": T,
            # 列 t = 看完第 0..t 帧；没有"一帧都没看到"那一列。
            "seen": list(range(1, T + 1)),
            "time": seconds,
            "probs": [round(p, 4) for p in a],
            "probs_rev": [round(p, 4) for p in b],
            "first": round(a[0], 4),
            "last": round(a[-1], 4),
            "first_rev": round(b[0], 4),
            "last_rev": round(b[-1], 4),
            "label": True,               # 磁盘上 A = atk 就是击杀者（样本都是正样本）
            "device": self._device,
            "weights": self.weights.name if self.weights else None,
            "ms": round(ms, 1),
        }


def _int_or_none(v: Any) -> Optional[int]:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _json_safe(v: Any) -> Any:
    """ckpt 里的 `best` / config → JSON 能吃的（`json.dumps(allow_nan=False)`）。

    指标可能是 numpy 标量或 `inf`（还没验证过），`inf` 会让整个响应 500。
    """
    if isinstance(v, dict):
        return {str(k): _json_safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_json_safe(x) for x in v]
    if isinstance(v, bool) or v is None or isinstance(v, str):
        return v
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return round(v, 6) if math.isfinite(v) else None
    item = getattr(v, "item", None)          # numpy 标量
    if callable(item):
        try:
            return _json_safe(item())
        except (TypeError, ValueError):
            return str(v)
    return str(v)
