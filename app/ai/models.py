"""网页固定使用哪两份权重（**路径写死**），并把它们装上 GPU。

版本管理
--------
`PINNED_WEIGHTS` 是唯一的版本开关 —— 改那里的路径就换版本，刷新页面生效
（缓存键跟着权重的**内容**走，换文件才作废；把同样的文件重下一遍不会）：

    aim                 models/aim/20261119-013008.pt                 + 同名 .yaml
    spatial_prediction  models/spatial_prediction/20261113-112930.pt  + 同名 .yaml

这两份 `.pt` 由 `scripts/export_weights.py` 从训练产物里抽出来：只留
`ckpt["model"]`（外加 `epoch` / `step` / `best` / `cfg` 几个小字段），没有
AdamW 的一阶/二阶动量 —— 所以两份合计 665 MB 而不是 2 GB，发给别人下载时
不必连带传优化器。模型结构读权重**旁边**的 `.yaml`（两者必须配套）。

也认训练目录的老布局（`<run>/last.pt` + `<run>/config.yaml`），方便把
`PINNED_WEIGHTS` 直接指到刚训完的 run 上试效果。

torch / yaml / `training.*` 全部在函数里 import：`app.server` 启动时不该为了
两个模型等 3 秒，也不该在没装 torch 的机器上直接起不来。缺东西时抛
`ModelUnavailable`（一句人话），由路由变成 503。
"""

from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass, fields as dataclass_fields
from pathlib import Path
from typing import Any, Callable, Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

#: 两个任务各自的产物根目录（`train.save_dir` 的两种写法都见过，这里只认这个）。
#: 训练脚本默认是 `models/<task>/<时间戳>/`；网页用的扁平文件就摆在它旁边。
MODELS_DIR = _REPO_ROOT / "models"

#: 网页固定使用的两个权重 —— **改这里 = 换版本**，不跟"哪个目录最新"走。
#: 页面拿到的永远是这两个文件，没有模型选择器。配置读同名的 `.yaml`。
PINNED_WEIGHTS: dict[str, Path] = {
    "aim": MODELS_DIR / "aim" / "20261119-013008.pt",
    "spatial_prediction": MODELS_DIR / "spatial_prediction" / "20261113-112930.pt",
}


class ModelUnavailable(RuntimeError):
    """AI 功能用不了：没权重 / 没装 torch / 权重和 config 对不上。

    这不是"服务器内部错误"，是一条**给人看**的说明。
    """


#: 内容指纹读多少：头尾各 1 MB（见 :func:`file_fingerprint`）。
_FP_CHUNK = 1 << 20


def file_fingerprint(path: str | Path) -> str:
    """文件的**内容**短指纹：大小 + 头 1 MB + 尾 1 MB 的 sha1。

    缓存键（模型指纹 / `.csz` 的 demo 指纹）都用它，而不是 `mtime`。

    为什么：同一个文件被**原样重写一遍**是常态 —— `hf_hub_download` 每次都会把
    权重重写，`scripts/download_models.py` 跑的也是它，重新提取一份一模一样的
    `.csz` 同理。mtime 一进缓存键，重下一遍权重就会让**全部历史 AI 缓存集体
    miss**：文件还在 `app/data/ai/<demo>/` 里，只是文件名前缀（模型指纹）变成
    了老的那个，一个都读不到，表现就是"AI 缓存全没了"。

    只读 2 MB 是因为全读一遍几百 MB 会把"开页面"变成"等几秒"；大小 + 头尾足够
    区分不同的训练产出（导出时中间任何改动都会改变尾部或总大小）。

    **不缓存结果**：文件系统可能延迟更新 mtime（Windows 上连续写两次经常拿到
    同一个 mtime），拿 `stat` 当缓存键会返回过期指纹 —— 这个坑比读 2 MB 贵得多。
    页缓存里读 2 MB 约 1 ms，和一次几十秒的预测比可以忽略。
    """
    p = Path(path)
    size = p.stat().st_size
    with open(p, "rb") as fh:
        head = fh.read(_FP_CHUNK)
        tail = b""
        if size > _FP_CHUNK:
            fh.seek(max(0, size - _FP_CHUNK))
            tail = fh.read(_FP_CHUNK)
    return hashlib.sha1(head + tail).hexdigest()[:16]


@dataclass(frozen=True)
class RunRef:
    """一份权重（扁平 `<run>.pt` 或训练目录 `<run>/last.pt`）和它的 config。

    `name` 是缓存键里的"哪个 run"：扁平布局取文件名（`20261119-013008`），
    训练目录取目录名 —— 换权重就换 key，旧缓存自然读不到。
    """

    task: str
    name: str
    weights: Path
    config: Path

    def signature(self) -> dict[str, Any]:
        """缓存键要的"这份权重是什么"：任务 + run + 大小 + **内容**指纹。

        **不带 mtime**，理由见 :func:`file_fingerprint`：重新下载 / 拷贝一遍同样
        的权重会刷新 mtime，而 mtime 进缓存键会让全部历史 AI 缓存失效。
        """
        st = self.weights.stat()
        cfg = self.config.stat()
        return {
            "task": self.task,
            "run": self.name,
            "weights": self.weights.name,
            "bytes": st.st_size,
            "digest": file_fingerprint(self.weights),
            "config_digest": file_fingerprint(self.config),
        }


def pinned_weights(task: str) -> Path:
    """这个任务写死的那份权重（只查表，不碰磁盘）。"""
    try:
        return PINNED_WEIGHTS[task]
    except KeyError:
        raise ModelUnavailable(
            f"`{task}` 没有写死权重（见 app/ai/models.py::PINNED_WEIGHTS）"
        ) from None


def pinned_run(task: str, weights: Optional[Path] = None) -> RunRef:
    """`PINNED_WEIGHTS[task]` → `RunRef`（`weights` 参数只给测试用）。

    缺权重 / 缺 config 就抛 `ModelUnavailable`，消息里带**具体哪条路径** ——
    这类故障基本都是"改了 `PINNED_WEIGHTS` 忘了同步文件"。
    """
    path = Path(weights) if weights is not None else pinned_weights(task)
    if not path.is_file():
        raise ModelUnavailable(
            f"权重不存在：{path}（app/ai/models.py::PINNED_WEIGHTS 写死的路径）"
        )
    # 扁平布局优先（网页用的就是它）：`<run>.pt` 旁边一个 `<run>.yaml`
    flat = path.with_suffix(".yaml")
    if flat.is_file():
        return RunRef(task=task, name=path.stem, weights=path, config=flat)
    config = path.parent / "config.yaml"
    if not config.is_file():
        raise ModelUnavailable(
            f"{path} 旁边既没有 {flat.name} 也没有 {config.name}，读不出模型结构"
        )
    return RunRef(task=task, name=path.parent.name, weights=path, config=config)


def _filtered_config(raw: dict, cfg_cls: Any) -> dict[str, Any]:
    """`config.yaml` 的 `model:` 段 → 只留 dataclass 认识的字段。

    旧 run 的 config 里还留着已经删掉的超参（`view_grid` / `spatial_layers` …），
    直接 `**` 展开会 TypeError。
    """
    known = {f.name for f in dataclass_fields(cfg_cls)}
    return {k: v for k, v in (raw.get("model") or {}).items() if k in known}


def _json_safe(v: Any) -> Any:
    """ckpt 里的 `best` 指标 → JSON 能吃的（numpy 标量 / `inf` 都要处理）。"""
    if isinstance(v, dict):
        return {str(k): _json_safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_json_safe(x) for x in v]
    if isinstance(v, bool) or v is None or isinstance(v, str):
        return v
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return v if v == v and v not in (float("inf"), float("-inf")) else None
    item = getattr(v, "item", None)
    if callable(item):
        try:
            return _json_safe(item())
        except (TypeError, ValueError):
            return str(v)
    return str(v)


class Bundle:
    """`aim` + `spatial_prediction` 两个模型，一份 GPU 显存。

    `load()` 幂等、线程安全（一把锁把"加载"和"推理"串起来：Flask 是多线程的，
    两条请求同时 forward 只会互相抢显存，收益为零）。
    """

    def __init__(
        self,
        aim: RunRef,
        spatial: RunRef,
        *,
        device: Optional[str] = None,
    ) -> None:
        self.aim_ref = aim
        self.spatial_ref = spatial
        self._device_hint = device
        self.device = "cpu"
        self.torch: Any = None
        self.aim: Any = None
        self.spatial: Any = None
        self.meta: dict[str, Any] = {}
        self.loaded_ms: float = 0.0

    # ── 加载 ───────────────────────────────────────────────────────────────

    def load(self, progress: Optional[Callable[[float, str], None]] = None) -> "Bundle":
        """读两个 checkpoint（幂等）。

        `progress(0..1, 一句话)` 是给网页进度条用的：两个 checkpoint 各占一半，
        一声 "读取 aim 权重…" 之后要等 ~0.7 s（1.2 GB 从磁盘进内存）。
        """
        import time

        say = progress or (lambda _frac, _msg: None)
        if self.aim is not None and self.spatial is not None:
            return self
        t0 = time.perf_counter()
        try:
            import torch
            import yaml

            from training.aim.model import AimConfig, AimModel
            from training.spatial_prediction.model import SpatialConfig, SpatialModel
        except ImportError as exc:  # pragma: no cover - 环境问题
            raise ModelUnavailable(
                f"AI 预测需要 torch + pyyaml（{exc}）。用 conda env cs-net-v5 启动"
            ) from exc

        params = {
            "aim": (self.aim_ref, AimConfig, AimModel),
            "spatial": (self.spatial_ref, SpatialConfig, SpatialModel),
        }
        built: dict[str, Any] = {}
        meta: dict[str, Any] = {}
        for n, (task, (ref, cfg_cls, model_cls)) in enumerate(params.items()):
            say(n / 2, f"读取 {task} 权重 {ref.weights.name}…")
            try:
                raw = yaml.safe_load(ref.config.read_text(encoding="utf-8")) or {}
                cfg = cfg_cls(**_filtered_config(raw, cfg_cls))
            except (OSError, TypeError, ValueError) as exc:
                raise ModelUnavailable(
                    f"{ref.config} 的 model 段读不了：{exc}"
                ) from exc
            model = model_cls(cfg)
            try:
                ckpt = torch.load(ref.weights, map_location="cpu", weights_only=False)
            except (OSError, RuntimeError) as exc:
                raise ModelUnavailable(f"读不了 {ref.weights}：{exc}") from exc
            state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
            try:
                model.load_state_dict(state)
            except (RuntimeError, KeyError, TypeError) as exc:
                raise ModelUnavailable(
                    f"{ref.weights.name} 和 config.yaml 对不上：{str(exc).splitlines()[0]}"
                ) from exc
            built[task] = model
            meta[task] = {
                "epoch": _json_safe((ckpt or {}).get("epoch") if isinstance(ckpt, dict) else None),
                "step": _json_safe((ckpt or {}).get("step") if isinstance(ckpt, dict) else None),
                "best": _json_safe((ckpt or {}).get("best") if isinstance(ckpt, dict) else None) or {},
                "params": int(sum(p.numel() for p in model.parameters())),
            }

        device = self._device_hint or ("cuda" if torch.cuda.is_available() else "cpu")
        say(0.95, f"搬到 {device}…")
        for model in built.values():
            model.to(device).eval()
        self.aim, self.spatial = built["aim"], built["spatial"]
        self.torch, self.device = torch, device
        self.meta = meta
        self.loaded_ms = (time.perf_counter() - t0) * 1000.0
        return self

    # ── 说明 ───────────────────────────────────────────────────────────────

    def info(self) -> dict[str, Any]:
        """给 `/api/ai/status` 用：哪份权重、什么设备、多少参数、训到第几个 epoch。"""
        out: dict[str, Any] = {
            "device": self.device if self.aim is not None else self._device_hint or "?",
            "loaded": self.aim is not None,
            "runs": {
                "aim": {**self.aim_ref.signature(), "path": str(self.aim_ref.weights)},
                "spatial": {
                    **self.spatial_ref.signature(),
                    "path": str(self.spatial_ref.weights),
                },
            },
        }
        if self.aim is not None:
            out["device"] = self.device
            out["meta"] = self.meta
            out["gpu"] = self._gpu_name()
            out["load_ms"] = round(self.loaded_ms, 1)
            # 推理时给页面看的短标识：`aim 20261119-013008 · spatial 20261113-112930`
            out["label"] = f"aim {self.aim_ref.name} · spatial {self.spatial_ref.name}"
        return out

    def _gpu_name(self) -> Optional[str]:
        try:
            return str(self.torch.cuda.get_device_name(0))
        except Exception:  # pragma: no cover - 没 GPU / 老驱动
            return None


def signature_key(aim: RunRef, spatial: RunRef) -> str:
    """两份权重 + config 的短哈希（缓存目录名的一部分）。

    跟着**内容**走，不跟 mtime 走（见 :func:`file_fingerprint`）。
    """
    import json

    payload = json.dumps(
        [aim.signature(), spatial.signature()], sort_keys=True, ensure_ascii=False
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
