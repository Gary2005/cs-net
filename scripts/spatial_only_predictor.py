"""
spatial-only 下游任务推理器（公共模块，wds_viewer 与 visualizer 共用）。

加载多个 spatial-only 下游任务 checkpoint（winrate / alive_end / future_kill，
每个 = embedder + spatial + head，无 temporal/decoder），对**单个 tick 的
完整局面**直接推理，输出每玩家概率。不做路径/历史/窗口。

checkpoint 格式（finetune_spatial_only.py 保存）:
    {task, global_step, model_state, head_state, config}
    - model_state:  embedder + spatial 权重（剥离 _orig_mod.）
    - head_state:   fc: d_model → 1
    - config:       架构字段（d_model 等，用于构建底座）
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_PROJECT_ROOT), str(_PROJECT_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scripts.pretrain_model import CS2PretrainModel, PretrainConfig  # noqa: E402
from scripts.training_data.torch_dataset import (  # noqa: E402
    augment_depth_with_angles,
    pretrain_collate_fn,
    sample_to_torch,
)

TASKS = ("winrate", "alive_end", "future_kill")


def _finite_list(arr) -> List[Optional[float]]:
    """把数组转成 JSON 安全的 float 列表：NaN/Inf → None（前端显示 '—'）。

    NaN 不是合法 JSON（Flask jsonify 默认 emit 成字面 NaN，浏览器解析直接报错），
    MPS/个别局面下模型输出可能出现非有限值，这里统一清洗，保证接口永远返回合法 JSON。
    """
    out = []
    for x in np.asarray(arr).reshape(-1).tolist():
        try:
            out.append(float(x) if math.isfinite(float(x)) else None)
        except (TypeError, ValueError):
            out.append(None)
    return out

# spatial-only 模块所在目录带连字符，无法常规 import，用 importlib 加载一次
_MOD_PATH = _PROJECT_ROOT / "scripts" / "downstream-spatial-only" / "finetune_spatial_only.py"
_spec = importlib.util.spec_from_file_location("finetune_spatial_only", str(_MOD_PATH))
_fso = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fso)


class SpatialOnlyPredictor:
    """加载多个 spatial-only 任务模型，对单 tick 完整局面推理。

    device 支持 cpu / mps / cuda。**torch < 2.13 的 MPS 已知数值问题**：
    间歇性 NaN + 有限但错误的值（内存损坏，非数据问题，CPU 上完全相同
    输入输出确定且干净；已在 torch 2.13 修复，实测 5/5 独立实例干净）。
    为兼容旧 torch，推理时若检测到非有限输出，自动用 CPU 副本重算该批。
    """

    def __init__(self, model_dir: str | Path, device: str = "cpu",
                 tasks: Optional[List[str]] = None):
        self.device = torch.device(device)
        self._SpatialOnlyModel = _fso.SpatialOnlyModel
        self._TaskHead = _fso.TaskHead
        self._input_keys = _fso._INPUT_KEYS
        self._last_kill_ticks = _fso.last_kill_ticks
        self._collate = pretrain_collate_fn
        self._augment_depth = augment_depth_with_angles
        self._sample_to_torch = sample_to_torch
        self._cpu_models: Optional[Dict[str, Tuple]] = None   # MPS NaN 兜底用
        self._warned_nan_fallback = False

        # 发现任务 checkpoint（按 ckpt 内的 task 字段，不依赖文件名）
        model_dir = Path(model_dir)
        want = set(tasks) if tasks else set(TASKS)
        found: Dict[str, Tuple[Path, dict]] = {}
        for p in sorted(model_dir.glob("*.pt")):
            try:
                ck = torch.load(p, map_location="cpu", weights_only=False)
            except Exception:
                continue
            task = ck.get("task")
            if task in want and "model_state" in ck and "head_state" in ck:
                found[task] = (p, ck)
        if not found:
            raise ValueError(
                f"模型目录中没有 spatial-only 任务 checkpoint: {model_dir}"
                f"（需要 task 字段为 {sorted(want)} 的 .pt）")
        self.tasks = sorted(found)
        print(f"[spatial] 发现任务模型: {self.tasks}")

        # 每个任务：随机初始化底座 → 只加载 embedder+spatial 权重 → 包装 + head
        self.models: Dict[str, Tuple] = {}
        for task in self.tasks:
            path, ck = found[task]
            cfg = ck.get("config", {})
            model_cfg = PretrainConfig(
                d_model=int(cfg.get("d_model", 768)),
                n_spatial_layers=int(cfg.get("n_spatial_layers", 5)),
                n_temporal_layers=int(cfg.get("n_temporal_layers", 3)),
                n_decoder_layers=int(cfg.get("n_decoder_layers", 6)),
                n_heads=int(cfg.get("n_heads", 12)),
                d_ff=int(cfg.get("d_ff", 3072)),
                dropout=float(cfg.get("dropout", 0.1)),
                n_depth_ray_layers=int(cfg.get("n_depth_ray_layers", 3)),
                n_ticks=int(cfg.get("n_ticks", 16)),
                move_range=float(cfg.get("move_range", 128.0)),
                move_grid_size=float(cfg.get("move_grid_size", 4.0)),
                angle_grid_size=float(cfg.get("angle_grid_size", 5.0)),
                use_residual_correction=bool(cfg.get("use_residual_correction", True)),
            )
            base = CS2PretrainModel(model_cfg)
            base.load_state_dict(ck["model_state"], strict=False)  # temporal/decoder 缺失，忽略
            model = self._SpatialOnlyModel(base).to(self.device).eval()
            head = self._TaskHead(model_cfg.d_model).to(self.device).eval()
            head.load_state_dict(ck["head_state"])
            self.models[task] = (model, head)
            print(f"  [spatial] {task}: {path.name}（d_model={model_cfg.d_model}）")

    # ── 数据处理 ──────────────────────────────────────────────────────────

    def _round_arrays(self, sample: dict) -> Tuple[dict, int, np.ndarray, np.ndarray]:
        """深度增强 + 每玩家队伍(0=CT 1=T -1=未知) + last_kill 表。"""
        if "player_depth" in sample and sample["player_depth"].ndim == 3:
            sample = augment_depth_with_angles(sample)
        T = int(sample["player_pos"].shape[0])
        meta = sample.get("meta", {}) if isinstance(sample.get("meta"), dict) else {}
        teams_list = meta.get("teams") or []
        player_teams = np.array(
            [0 if t == "CT" else (1 if t == "T" else -1) for t in teams_list],
            dtype=np.int64,
        )
        last_kill = self._last_kill_ticks(
            sample.get("label_nxt_kill"), sample.get("label_nxt_death"))
        return sample, T, player_teams, last_kill

    def _tick_batch(self, sample: dict, t: int, T: int,
                    player_teams: np.ndarray, last_kill: np.ndarray,
                    device: torch.device) -> dict:
        """切出 tick t 的完整局面 → 与训练同款 batch dict（[1, 1, 10, ...]）。"""
        st: dict = {}
        for key in self._input_keys:
            if key in sample:
                st[key] = sample[key][t:t + 1]               # 保持 [1, ...] 维
        if "tick_times_input" not in st:
            st["tick_times_input"] = np.asarray(
                sample.get("round_seconds", np.zeros(T, dtype=np.float32))[t],
                dtype=np.float32).reshape(1)
        st["label_alive_end"] = sample["label_alive_end"][t:t + 1]   # [1,10]
        st["label_future_kill"] = (last_kill > t).astype(np.float32)[None]  # [1,10]
        st["label_winrate"] = sample["label_winrate"][t:t + 1]       # [1]
        st["player_teams"] = player_teams.copy()                     # [10]
        batch = self._collate([sample_to_torch(st)])
        return {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()}

    @staticmethod
    def _agg_winrate(p10: np.ndarray, player_teams: np.ndarray) -> Dict[str, Optional[float]]:
        """聚合胜率（与 finetune_spatial_only.compute_ct_winrate 同口径）：
        p10 = P(T 胜)；CT 玩家用 1-p，T 玩家用 p，ct = CT/(CT+T)。
        模型输出含 NaN/Inf 时返回 None（表示该 tick 聚合结果不可用）。"""
        p10 = np.asarray(p10, dtype=np.float64)
        if not np.all(np.isfinite(p10)):
            return {"ct": None, "t": None, "ct_winrate": None}
        is_ct = player_teams == 0
        is_t = player_teams == 1
        ct_avg = float(((1.0 - p10) * is_ct).sum() / max(int(is_ct.sum()), 1))
        t_avg = float((p10 * is_t).sum() / max(int(is_t.sum()), 1))
        denom = ct_avg + t_avg
        ct = ct_avg / denom if denom > 0 else 0.5
        return {"ct": ct_avg, "t": t_avg, "ct_winrate": ct}

    # ── 推理 ──────────────────────────────────────────────────────────────

    def _get_cpu_models(self) -> Dict[str, Tuple]:
        """懒加载 CPU 副本（MPS 兜底重算用，CPU 确定且干净）。"""
        if self._cpu_models is None:
            import copy as _copy
            self._cpu_models = {}
            for task, (model, head) in self.models.items():
                m = _copy.deepcopy(model).to("cpu").eval()
                h = _copy.deepcopy(head).to("cpu").eval()
                self._cpu_models[task] = (m, h)
        return self._cpu_models

    @staticmethod
    def _cat_batches(batches: List[dict]) -> dict:
        """把多个单 tick batch [1, ...] 沿 dim0 拼成 [C, ...]。

        ⚠ 仅供测试/参考：MPS 上逐 tick cat 会触发内存损坏（间歇性
        NaN/垃圾值），正式推理请用 _chunk_batch（torch.stack）。
        """
        big = {}
        for k in batches[0]:
            if isinstance(batches[0][k], torch.Tensor):
                big[k] = torch.cat([b[k] for b in batches], dim=0)
            else:
                big[k] = batches[0][k]
        return big

    def _chunk_batch(self, sample: dict, ticks_range, T: int,
                     player_teams: np.ndarray, last_kill: np.ndarray,
                     device: torch.device) -> dict:
        """把一段 tick 一次性 collate 成 [C, 1, 10, ...] batch。

        与训练同款：每个 tick 一个 torch sample，pretrain_collate_fn 用
        torch.stack 拼成 batch。**用 stack 而非逐 tick cat** —— torch < 2.13
        的 MPS 对多个 MPS 张量的 cat 有内存损坏 bug（间歇性 NaN / 有限错误
        值，chunk 越大越严重；已在 2.13 修复，stack/cat 路径实测都干净）。
        保持 stack 与训练管线完全一致，双保险。
        """
        sts = []
        for t in ticks_range:
            st: dict = {}
            for key in self._input_keys:
                if key in sample:
                    st[key] = sample[key][t:t + 1]          # 保持 [1, ...] 维
            if "tick_times_input" not in st:
                st["tick_times_input"] = np.asarray(
                    sample.get("round_seconds", np.zeros(T, dtype=np.float32))[t],
                    dtype=np.float32).reshape(1)
            st["player_teams"] = player_teams.copy()
            sts.append(st)
        big = self._collate([self._sample_to_torch(st) for st in sts])
        return {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                for k, v in big.items()}

    @torch.no_grad()
    def _run_tasks(self, big: dict, device: torch.device,
                   models: Dict[str, Tuple]) -> Dict[str, np.ndarray]:
        """对所有任务跑一次前向，返回 per_task: {task: [C, 10] numpy}。"""
        n = int(big["player_pos"].shape[0])
        per_task: Dict[str, np.ndarray] = {}
        for task in self.tasks:
            model, head = models[task]
            emb = model(big)                                 # [C, 1, 10, d]
            logits = head(emb.reshape(-1, emb.shape[-1]))    # [C*10]
            per_task[task] = torch.sigmoid(logits).reshape(n, 10).cpu().numpy()
        return per_task

    def _run_tasks_safe(self, big: dict,
                        rebuild_cpu_big=None) -> Dict[str, np.ndarray]:
        """前向 + MPS 损坏自动 CPU 兜底（兼容 torch < 2.13）。

        torch < 2.13 的 MPS 在这台机器上会**内存级损坏**：间歇性把输出
        变成 NaN、甚至把输入张量（如 int64 map_idx）变成垃圾（非数据
        问题，CPU 上同输入完全确定且干净；已在 2.13 修复）。检测到非
        有限输出 / 前向异常时，用 rebuild_cpu_big() 从**原始 numpy 数据**
        在 CPU 重建 batch 重算（绝不能 .to("cpu") 拷贝可能已损坏的 MPS
        张量）。2.13+ 上兜底基本不会触发。
        """
        if self.device.type != "cpu" and rebuild_cpu_big is not None:
            per_task = None
            try:
                per_task = self._run_tasks(big, self.device, self.models)
            except Exception:
                per_task = None
            if per_task is not None and \
                    all(np.isfinite(p).all() for p in per_task.values()):
                return per_task
            # MPS 损坏 → 从原始 numpy 在 CPU 重建 batch 重算
            per_task = self._run_tasks(rebuild_cpu_big(), torch.device("cpu"),
                                       self._get_cpu_models())
            if not self._warned_nan_fallback:
                self._warned_nan_fallback = True
                print("[spatial] ⚠ 检测到 MPS 前向异常（NaN/崩溃，torch MPS "
                      "内存损坏，非数据问题），该批已自动用 CPU 重算保证正确；"
                      "如频繁出现建议 --device cpu", flush=True)
            return per_task
        # CPU 或未提供重建函数：直接跑（CPU 无此问题；异常向上抛）
        return self._run_tasks(big, self.device, self.models)

    @torch.no_grad()
    def predict_tick(self, sample: dict, tick: int) -> dict:
        """单 tick：每玩家三任务概率 + 聚合胜率 + 每玩家队伍胜率。"""
        sample, T, player_teams, last_kill = self._round_arrays(sample)
        t = min(max(int(tick), 0), T - 1)
        batch = self._tick_batch(sample, t, T, player_teams, last_kill, self.device)
        # alive_mask 从原始 numpy 取（MPS 张量可能损坏）
        alive_mask = None
        if "player_alive_mask" in sample:
            alive_mask = sample["player_alive_mask"][t].astype(bool)

        def rebuild():
            return self._tick_batch(sample, t, T, player_teams, last_kill,
                                    torch.device("cpu"))

        per_task = self._run_tasks_safe(batch, rebuild)

        out: Dict[str, object] = {"tick": t, "T": T}
        for task in self.tasks:
            p10 = per_task[task][0]                              # [10]
            out[task] = _finite_list(p10)
            if task == "winrate":
                out["winrate_agg"] = self._agg_winrate(p10, player_teams)
                # 每玩家"队伍胜率"：CT 玩家 = 1−P(T胜)，T 玩家 = P(T胜)
                team_prob = np.where(player_teams == 0, 1.0 - p10, p10)
                out["winrate_team"] = _finite_list(team_prob)
        out["player_teams"] = player_teams.tolist()
        out["alive_mask"] = alive_mask.tolist() if alive_mask is not None else None
        meta = sample.get("meta", {}) if isinstance(sample.get("meta"), dict) else {}
        out["winner"] = meta.get("winner")
        return out

    @torch.no_grad()
    def predict_round_curve(self, sample: dict, chunk: int = 32) -> dict:
        """整回合 winrate 曲线：对每个 tick 批量推理，返回聚合 CT 胜率。"""
        sample, T, player_teams, last_kill = self._round_arrays(sample)
        curve: List[Dict[str, float]] = []
        for t0 in range(0, T, chunk):
            t1 = min(t0 + chunk, T)
            ticks_range = list(range(t0, t1))
            big = self._chunk_batch(sample, ticks_range, T, player_teams,
                                    last_kill, self.device)

            def rebuild():
                return self._chunk_batch(sample, ticks_range, T, player_teams,
                                         last_kill, torch.device("cpu"))

            per_task = self._run_tasks_safe(big, rebuild)
            p = per_task["winrate"]
            for i in range(t1 - t0):
                curve.append(self._agg_winrate(p[i], player_teams))
        meta = sample.get("meta", {}) if isinstance(sample.get("meta"), dict) else {}
        return {"curve": curve, "T": T, "winner": meta.get("winner"),
                "player_teams": player_teams.tolist()}

    @torch.no_grad()
    def predict_round_full(self, sample: dict, chunk: int = 32) -> dict:
        """整回合全任务逐 tick 推理（前端"切回合自动预测 + 缓存"用）。

        与 predict_tick 完全同口径，只是对整回合批量算一遍：
        每个 tick 输出每玩家各任务概率 + 聚合胜率 + 每玩家队伍胜率 +
        存活掩码。批量按 chunk 分块（默认 32 tick/块），用 torch.stack
        拼装（避开 MPS 的 torch.cat 内存损坏 bug），同一块内对所有已
        加载任务复用输入 batch（每个任务跑一次前向）。MPS 前向仍异常时
        该块自动从原始 numpy 在 CPU 重建重算（见 _run_tasks_safe）。
        """
        sample, T, player_teams, last_kill = self._round_arrays(sample)
        ticks: List[Dict[str, object]] = []
        for t0 in range(0, T, chunk):
            t1 = min(t0 + chunk, T)
            ticks_range = list(range(t0, t1))
            big = self._chunk_batch(sample, ticks_range, T, player_teams,
                                    last_kill, self.device)

            def rebuild():
                return self._chunk_batch(sample, ticks_range, T, player_teams,
                                         last_kill, torch.device("cpu"))

            per_task = self._run_tasks_safe(big, rebuild)
            # alive_mask 从原始 numpy 取（MPS 张量可能损坏）
            alive_mask = None
            if "player_alive_mask" in sample:
                alive_mask = sample["player_alive_mask"][t0:t1].astype(bool)
            for i in range(t1 - t0):
                out: Dict[str, object] = {"tick": t0 + i}
                for task, p in per_task.items():
                    out[task] = _finite_list(p[i])
                if "winrate" in per_task:
                    p = per_task["winrate"][i]
                    out["winrate_agg"] = self._agg_winrate(p, player_teams)
                    # 每玩家"队伍胜率"：CT 玩家 = 1−P(T胜)，T 玩家 = P(T胜)
                    team_prob = np.where(player_teams == 0, 1.0 - p, p)
                    out["winrate_team"] = _finite_list(team_prob)
                out["alive_mask"] = alive_mask[i].tolist() \
                    if alive_mask is not None else None
                ticks.append(out)
        meta = sample.get("meta", {}) if isinstance(sample.get("meta"), dict) else {}
        return {
            "T": T,
            "tasks": list(self.tasks),
            "ticks": ticks,
            "winner": meta.get("winner"),
            "player_teams": player_teams.tolist(),
        }
