"""自检：`python -m scripts.training_data_visualizer.spatial_prediction.selftest`

不打开浏览器也能验证四件事：

1. **逆变换是对的**：用 `SCALE`/`BIAS` 把一个已知的物理量正向编码成 [0,1]，再
   `decode_player()` 回来，必须落在原来的值上（yaw / pitch 在 ±180 附近除外）。
   `SCALE`/`BIAS` 就是 `tasks/aim.py` 的 `_SCALE`/`_BIAS` 的镜像 —— 这条钉住的是
   "两边的列序和半宽没有漂移"。
2. **合成样本的完整载荷**：在内存里造一份 3 行的 npz 数组，`build_payload()` 之后逐项
   检查存活掩码 / 队伍 / 5 个标签 / 道具 / JSON 可序列化。再故意造几份**坏样本**，
   确认标签体检（ct_win 变化、next_death 时间倒流、can_kill 变多、死人拿到击杀）
   真的会**报警告**而不是静默通过。
3. **真实样本**（有 `prepared/spatial_prediction` 时）：扫第一个文件，读第 0 行，
   检查形状 / 列数 / 数值范围 / 玩家表能在 phys 里查到每个特征列。
4. **HTTP 端到端**：优先用真实数据目录；没有就把合成样本写进临时目录再起一个真
   server，拉一遍 `/api/samples`、`/api/sample/...?k=`、`/api/radar/...`、
   `/api/weapon-icons`、三个静态文件，外加越界 / 目录穿越必须被拒。没有可写目录
   （只读挂载 / 沙箱）时这一项会**跳过并打印原因**，不算失败。

任何一条失败都抛 AssertionError，退出码非 0。
"""

from __future__ import annotations

import json
import math
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

from . import data as D
from . import server as S

#: 合成样本用的地图（`MAP_NAMES` 里的下标就是 `map_id`）。
_MAP = "de_mirage"
_MAP_ID = D.MAP_NAMES.index(_MAP)


# ═══════════════════════════════════════════════════════════════════════════
# ① 逆变换
# ═══════════════════════════════════════════════════════════════════════════


def test_roundtrip() -> None:
    phys = {
        "x": 1234.5, "y": -567.25, "z": 128.0,
        "yaw": 37.5, "pitch": -12.25,
        "vx": 123.0, "vy": -45.0, "vz": 678.0,
        "hp": 84.0, "crouch": 0.25, "armor": 92.0, "helmet": 1.0,
        "team": 1.0, "shot": 1.0, "since_shot": 1.25, "has_kill": 1.0,
        "blinded": 1.0, "defuser": 1.0,
    }
    raw = D.encode_row(phys)
    assert raw.min() >= 0.0 and raw.max() <= 1.0, "编码后必须落在 [0,1]"
    out = D.decode_player(raw[None, :], _MAP)
    for key in ("x", "y", "z", "yaw", "pitch", "vx", "vy", "vz", "hp", "armor",
                "crouch", "helmet", "team", "shot", "since_shot", "has_kill",
                "blinded", "defuser"):
        got, want = out[key][0], phys[key]
        assert abs(got - want) < 1e-2, f"{key}: 还原出 {got}，应为 {want}"
    for key, want in (
        ("yaw_sin", math.sin(math.radians(37.5))),
        ("yaw_cos", math.cos(math.radians(37.5))),
        ("pitch_sin", math.sin(math.radians(-12.25))),
        ("pitch_cos", math.cos(math.radians(-12.25))),
    ):
        assert abs(out[key][0] - want) < 1e-3, (key, out[key][0], want)
    assert abs(out["eye_z"][0] - (D.STANDING_EYE_HU - D.CROUCH_EYE_DROP_HU * 0.25)) < 1e-3
    assert abs(out["box_h"][0] - (D.HULL_STAND_HU - 18.0 * 0.25)) < 1e-3

    # 18 列旧样本：blinded / defuser 都按 0 处理，不抛
    old = D.decode_player(raw[None, :18], _MAP)
    assert old["blinded"] == [0.0] and old["defuser"] == [0.0]

    # 角度跨 ±180 处：atan2 恢复的是 [-180,180]，回绕后必须一致
    yaw2 = D.decode_player(D.encode_row({**phys, "yaw": -179.5})[None, :], _MAP)["yaw"][0]
    assert abs(yaw2 - (-179.5)) < 1e-2, yaw2
    print("  逆变换往返 OK")


def test_constants() -> None:
    """镜像常量对拍：本文件的 `MAP_CENTER` / `WEAPON_NAMES` / `FEATURES` / `SCALE`/`BIAS`
    必须和 `scripts.training_data` 里那几份**逐值一致**。

    ⚠ 只在**能 import 到**它们时才做：这个工具故意只依赖 numpy，而
    `scripts/training_data/__init__.py` 会把 `demo_extract`（demoparser2 / numba）拖进来。
    没装解析器的机器上对拍就跳过，不算失败 —— 那正是这个工具存在的意义。
    """
    try:
        from scripts.training_data import constants as C
        from scripts.training_data.tasks import aim as A
    except Exception as exc:                     # noqa: BLE001 —— 缺依赖是正常情况
        print(f"  常量对拍跳过（{type(exc).__name__}: {exc}）")
        return

    assert tuple(C.FEATURES)  # 只是确认模块真的加载了
    assert tuple(A.AIM_FEATURES) == D.FEATURES, "FEATURES 和 tasks/aim.py 的 AIM_FEATURES 不一致"
    assert np.allclose(A._SCALE, D.SCALE), "SCALE 和 tasks/aim.py 的 _SCALE 不一致"
    assert np.allclose(A._BIAS, D.BIAS), "BIAS 和 tasks/aim.py 的 _BIAS 不一致"
    assert set(D.MAP_CENTER) == set(C.MAP_CENTER), "MAP_CENTER 的图名对不上"
    for name, xyz in C.MAP_CENTER.items():
        assert np.allclose(D.MAP_CENTER[name], xyz, atol=1e-9), f"{name} 的地图中心不一致"
    assert tuple(C.WEAPON_NAMES) == D.WEAPON_NAMES, "WEAPON_NAMES 不一致"
    assert D.MAP_NAMES == tuple(sorted(C.MAP_CENTER)), "MAP_NAMES 的排序规则变了"
    assert all(D.MAP_NAMES[C.MAP_TO_IDX[n]] == n for n in C.MAP_TO_IDX), "MAP_TO_IDX 对不上"
    print(f"  常量镜像和 scripts.training_data 一致 OK（{len(D.MAP_CENTER)} 张图 / "
          f"{len(D.WEAPON_NAMES)} 种武器 / {len(D.FEATURES)} 列）")


# ═══════════════════════════════════════════════════════════════════════════
# ② 合成样本 → 载荷
# ═══════════════════════════════════════════════════════════════════════════


def _player_phys(p: int, team: int) -> dict[str, float]:
    """第 p 号玩家的物理量（和 `encode_row` 的入参对应）。"""
    return {
        "x": -300.0 + 140.0 * p, "y": -200.0 + 90.0 * p, "z": 64.0,
        "yaw": (p * 36.0) % 360.0 - 180.0, "pitch": -3.0 + p,
        "vx": 10.0 * p, "vy": -8.0 * p, "vz": 0.0,
        "hp": 100.0 if p < 7 else 42.0, "crouch": 0.0,
        "armor": 100.0 if team == 0 else 65.0, "helmet": 1.0 if team == 0 else 0.0,
        "team": float(team), "shot": 0.0, "since_shot": 12.5, "has_kill": 1.0 if p in (2, 4) else 0.0,
        "blinded": 0.0, "defuser": 1.0 if p == 0 else 0.0,
    }


def synthetic_arrays(
    n_rows: int = 3,
    *,
    ct_win: Any = None,
    next_death: Any = None,
    next_kill: Any = None,
    can_kill: Any = None,
    alive_end: Any = None,
    dead_from: int | None = None,
) -> dict[str, np.ndarray]:
    """造一份**结构合法**的 `spatial_prediction` npz 数组（默认 3 行、10 人）。

    默认值是一个自洽的小故事：前三号 CT（含击杀者 2）、后五号 T，第 3 行时才死掉
    7 号（所以前两行的 `next_death` 是 7、最后一行是 10 = 没有下一次），`can_kill`
    随时间收缩。
    """
    n_pl = 10
    n_feat = len(D.FEATURES)
    ctx = np.zeros((n_rows, n_pl, n_feat), np.float32)
    alive = np.ones((n_rows, n_pl), dtype=bool)
    team = np.zeros((n_rows, n_pl), np.int8)
    weapon = np.full((n_rows, n_pl), -1, np.int8)
    for p in range(n_pl):
        t = 0 if p < 5 else 1
        ctx[:, p, :] = D.encode_row(_player_phys(p, t))
        team[:, p] = t
        weapon[:, p] = 4 if p % 3 == 0 else (9 if t == 0 else 7)   # AK / M4A4 / FAMAS
    if dead_from is not None:
        # `dead_from` 之后的行里 7 号是死的（武器必须是 -1，和造数据那侧一致）
        alive[dead_from:, 7] = False
        weapon[dead_from:, 7] = -1

    n_rows = alive.shape[0]
    if ct_win is None:
        ct_win = np.ones(n_rows, dtype=bool)
    if next_death is None:
        next_death = np.array([7, 7, D.NO_EVENT][:n_rows] + [D.NO_EVENT] * max(0, n_rows - 3), np.int8)
    if next_kill is None:
        next_kill = np.array([2, 2, D.NO_EVENT][:n_rows] + [D.NO_EVENT] * max(0, n_rows - 3), np.int8)
    if can_kill is None:
        can_kill = np.zeros((n_rows, n_pl), dtype=bool)
        can_kill[0, [2, 4]] = True
        can_kill[1, [2, 4]] = True
    if alive_end is None:
        alive_end = np.zeros((n_rows, n_pl), dtype=bool)
        alive_end[:] = np.array([True] * 4 + [False] + [True] * 3 + [False] * 2)

    seconds = np.array([10.0 + 10.0 * i for i in range(n_rows)], np.float32)
    global_feat = np.zeros((n_rows, 5), np.float32)
    global_feat[:, :3] = 0.5                      # 炸弹在地图中心（归一化后）
    global_feat[:, 3] = 0.0                       # 未安放
    global_feat[:, 4] = seconds / 200.0           # 回合时间

    return {
        "ctx_player": ctx,
        "ctx_player_mask": alive,
        "ctx_player_team": team,
        "ctx_player_weapon": weapon,
        "obstruction": np.zeros((n_rows, n_pl, n_pl), np.int8),
        "ctx_projectile": np.zeros((n_rows, 32, 4), np.float32),
        "ctx_projectile_id": np.full((n_rows, 32), -1, np.int8),
        "global_feat": global_feat,
        "seconds": seconds,
        "frame_index": np.arange(n_rows, dtype=np.int32) * 640 + 640,
        "ct_win": np.asarray(ct_win, dtype=bool),
        "alive_end": np.asarray(alive_end, dtype=bool),
        "can_kill": np.asarray(can_kill, dtype=bool),
        "next_death": np.asarray(next_death, dtype=np.int8),
        "next_kill": np.asarray(next_kill, dtype=np.int8),
        "map_id": np.int8(_MAP_ID),
        "round_id": np.int16(3),
    }


def _payload(arrays: dict[str, np.ndarray], k: int = 1, name: str = "") -> dict[str, Any]:
    return D.build_payload(
        arrays,
        split="train",
        name=name or f"de_mirage_f0_r3_n{np.asarray(arrays['ctx_player']).shape[0]}.npz",
        k=k,
        path_label="<memory>",
        file_bytes=1234,
    )


def test_payload() -> None:
    arrays = synthetic_arrays()
    d = _payload(arrays, k=1)
    assert d["row"] == 1 and d["rows"] == 3
    assert d["map_name"] == _MAP and d["map_id"] == _MAP_ID and d["round_id"] == 3
    assert abs(d["seconds"] - 20.0) < 1e-4, d["seconds"]
    assert d["n_players"] == 10 and d["n_features"] == len(D.FEATURES)
    assert d["features"] == list(D.FEATURES)

    # mask / 队伍
    assert d["players"]["alive"] == [True] * 10
    assert d["players"]["team"] == [D.TEAM_CT] * 5 + [D.TEAM_T] * 5
    assert d["players"]["teams"] == ["CT"] * 5 + ["T"] * 5

    # 物理量：第 3 号玩家的血量就是 encode 进去的那个
    assert abs(d["players"]["phys"]["hp"][3] - 100.0) < 1e-3
    assert abs(d["players"]["phys"]["hp"][7] - 42.0) < 1e-3
    assert d["players"]["raw"][0][0] == round(float(arrays["ctx_player"][1, 0, 0]), 4) \
        or abs(d["players"]["raw"][0][0] - float(arrays["ctx_player"][1, 0, 0])) < 1e-3

    # 5 个标签
    L = d["labels"]
    assert L["ct_win"] is True
    assert L["next_death"] == 7 and L["next_kill"] == 2
    assert L["next_death_team"] == "T" and L["next_kill_team"] == "CT"
    assert L["alive_end_n"] == sum(arrays["alive_end"][1].tolist())
    assert L["can_kill_n"] == 2
    assert L["alive_end"][4] is False and L["can_kill"][4] is True

    # 没有警告（合成样本是自洽的，雷达图也在）
    assert not d["warnings"], d["warnings"]
    assert d["radar"] is not None and d["radar"]["url"] == f"/api/radar/{_MAP}"

    # 全局 / 道具 / files
    assert d["global"]["planted"] is False
    assert d["global"]["round_seconds"] is not None
    assert d["projectiles"] == []
    assert {f["key"] for f in d["files"]} >= set(D.REQUIRED_KEYS)

    # JSON 必须可序列化（server 里 allow_nan=False）
    json.dumps(d, ensure_ascii=False, allow_nan=False)

    # 越界的行号必须报错
    for bad_k in (-1, 3, 99):
        try:
            _payload(arrays, k=bad_k)
        except ValueError:
            pass
        else:
            raise AssertionError(f"k={bad_k} 越界竟然没报错")

    # 缺 key 的 npz（比如 aim 的样本）必须明确拒绝
    try:
        D.build_payload({"ctx_player": arrays["ctx_player"]}, split="train",
                        name="x.npz", k=0)
    except ValueError as exc:
        assert "spatial_prediction" in str(exc)
    else:
        raise AssertionError("缺 key 的 npz 没有被拒绝")

    print(f"  合成样本载荷 OK（{d['rows']} 行 × {d['n_players']} 人 × {d['n_features']} 列）")


def test_label_checks() -> None:
    """故意造坏标签，确认体检会报警告（而不是静默通过）。"""
    cases: list[tuple[str, dict[str, Any], str]] = [
        ("ct_win 在文件内变了", {"ct_win": [True, False, True]}, "ct_win"),
        (
            "next_death 时间倒流",
            {"next_death": [D.NO_EVENT, 3, D.NO_EVENT], "next_kill": [D.NO_EVENT] * 3,
             "can_kill": np.zeros((3, 10), bool)},
            "next_death",
        ),
        (
            "can_kill 变多",
            {"can_kill": np.array([
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ], bool), "next_kill": [D.NO_EVENT, 0, D.NO_EVENT]},
            "can_kill",
        ),
        (
            "alive_end 在文件内变了",
            {"alive_end": np.array([
                [True] * 10, [True] * 9 + [False], [True] * 10,
            ], bool)},
            "alive_end",
        ),
    ]
    for title, override, needle in cases:
        arrays = synthetic_arrays(**override)
        d = _payload(arrays, k=1)
        hits = [w for w in d["warnings"] if needle in w]
        assert hits, f"{title}: 没有报警告（{d['warnings']}）"

    # 死人拿击杀 / 下一个死的人是死人：这两条是"行级"检查，构造一个当帧死人的样本
    arrays = synthetic_arrays()
    arrays["ctx_player_mask"][1, 7] = False
    arrays["ctx_player_weapon"][1, 7] = -1
    d = _payload(arrays, k=1)
    assert any("已经死了" in w for w in d["warnings"]), d["warnings"]

    arrays = synthetic_arrays()
    arrays["can_kill"][1, 8] = True                 # 8 号当帧活着 → 先造一行"死人拿击杀"
    arrays["ctx_player_mask"][1, 8] = False
    arrays["ctx_player_weapon"][1, 8] = -1
    d = _payload(arrays, k=1)
    assert any("can_kill[8]" in w for w in d["warnings"]), d["warnings"]

    print("  标签体检 OK（4 条序列不变式 + 2 条行级检查都能触发）")


# ═══════════════════════════════════════════════════════════════════════════
# ③ 真实样本
# ═══════════════════════════════════════════════════════════════════════════


def test_real_samples(data_dir: Path) -> str | None:
    files = D.scan_files(data_dir)
    assert files, f"在 {data_dir} 下没扫到 *_n{{K}}.npz"
    first = files[0]
    path = D.sample_path(data_dir, first["split"], first["name"])
    payload = D.load_sample(path, split=first["split"], name=first["name"], k=0)

    n = payload["n_players"]
    assert n == 10, n
    assert payload["n_features"] == len(payload["features"]) == len(D.FEATURES)
    assert payload["rows"] == first["n"] > 0
    assert 0 <= payload["row"] < payload["rows"]
    for key in D.FEATURES:
        assert key in payload["players"]["phys"], f"phys 缺列: {key}"
        assert len(payload["players"]["phys"][key]) == n
    assert len(payload["players"]["alive"]) == n
    assert len(payload["players"]["raw"]) == n
    assert len(payload["obstruction"]) == n and len(payload["obstruction"][0]) == n

    raw = np.asarray(payload["players"]["raw"], dtype=np.float64)
    assert raw.min() >= 0.0 and raw.max() <= 1.0, (raw.min(), raw.max())

    # 武器：死人的那一格必须是 -1；队伍只能是 0/1（-1 会飘警告，这里不要求）
    mask = np.asarray(payload["players"]["alive"], dtype=bool)
    team = np.asarray(payload["players"]["team"])
    wp = np.asarray(payload["players"]["weapon"])
    assert (wp[~mask] == -1).all(), "死人的武器格不是 -1"
    assert set(np.unique(team).tolist()) <= {D.TEAM_CT, D.TEAM_T, D.TEAM_UNKNOWN}

    # 标签：11 类的取值范围 + "下一个死亡 / 击杀的人当帧必须活着"
    L = payload["labels"]
    for key in ("next_death", "next_kill"):
        idx = L[key]
        assert 0 <= idx <= D.NO_EVENT, (key, idx)
        if idx < n:
            assert mask[idx], f"{key}={idx} 当帧是死的"
    assert (L["next_death_team"] is None) == (L["next_death"] >= n)
    assert (L["next_kill_team"] is None) == (L["next_kill"] >= n)
    assert len(L["alive_end"]) == n and len(L["can_kill"]) == n
    assert not (np.asarray(L["can_kill"]) & ~mask).any(), "死人还能拿击杀"
    assert isinstance(L["ct_win"], bool)

    json.dumps(payload, ensure_ascii=False, allow_nan=False)
    print(f"  真实样本 {first['split']}/{first['name']} 第 0 行："
          f"{n} 人 × {payload['n_features']} 列 · map={payload['map_name']} · "
          f"t={payload['seconds']:.2f}s · next_death={L['next_death']} · "
          f"next_kill={L['next_kill']}")
    if payload["warnings"]:
        print(f"  提醒: {payload['warnings']}")
    return first["name"]


# ═══════════════════════════════════════════════════════════════════════════
# ④ HTTP 端到端
# ═══════════════════════════════════════════════════════════════════════════


def _write_fixture(root: Path) -> Path:
    """在 `root/{train,valid,test}` 下写一份合成样本（只用于自检）。"""
    for split in D.SPLITS:
        (root / split).mkdir(parents=True, exist_ok=True)
    arrays = synthetic_arrays()
    name = "de_mirage_f0_r3_n3.npz"
    np.savez_compressed(root / "train" / name, **arrays)
    return root


def test_http(data_dir: Path, name: str) -> None:
    state = S.AppState(data_dir)
    # 自检时把访问日志关掉，输出只留下断言结果
    S.SpatialHandler.log_message = lambda *args, **kwargs: None  # noqa: E731
    httpd = S.SpatialHTTPServer(("127.0.0.1", 0), state)
    port = int(httpd.server_address[1])
    thread = threading.Thread(target=httpd.serve_forever,
                              kwargs={"poll_interval": 0.02}, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{port}"

    def get(path: str, expect: str = "application/json") -> tuple[bytes, str, int]:
        req = urllib.request.Request(base + path, headers={"Accept-Encoding": "identity"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            return resp.read(), resp.headers.get("Content-Type", ""), resp.status

    def expect_error(path: str, codes: tuple[int, ...]) -> None:
        try:
            with urllib.request.urlopen(base + path, timeout=10) as resp:
                raise AssertionError(f"{path} 竟然返回了 {resp.status}")
        except urllib.error.HTTPError as exc:
            assert exc.code in codes, f"{path} → {exc.code}，期望 {codes}"

    try:
        body, ctype, code = get("/api/samples")
        assert code == 200 and "application/json" in ctype
        listing = json.loads(body.decode("utf-8"))
        assert listing["total_files"] == len(listing["files"]) >= 1
        assert listing["total_samples"] == sum(f["n"] for f in listing["files"])
        assert listing["labels"] and listing["density"] > 0

        target = next(f for f in listing["files"] if f["name"] == name)
        quoted = "/".join(urllib.parse.quote(p, safe="") for p in
                          (target["split"], target["name"]))
        body, _ctype, code = get(f"/api/sample/{quoted}?k=1")
        payload = json.loads(body.decode("utf-8"))
        assert code == 200 and payload["id"] == f"{target['split']}/{name}"
        assert payload["row"] == 1 and payload["rows"] == target["n"]
        assert payload["players"]["alive"] and payload["labels"]["next_death"] is not None

        if payload["radar"]:
            body, ctype, _ = get(f"/api/radar/{payload['map_name']}", "image/png")
            assert body[:8] == b"\x89PNG\r\n\x1a\n", ctype

        # 页面和静态资源
        body, ctype, _ = get("/", "text/html")
        assert b"style.css" in body
        body, _ctype, _ = get("/static/app.js", "javascript")
        assert b"/api/sample/" in body
        body, _ctype, _ = get("/static/style.css", "text/css")
        assert b"--panel" in body

        # 武器图标：素材目录是共享的
        body, _ctype, _ = get("/api/weapon-icons")
        icons = json.loads(body.decode("utf-8"))["icons"]
        assert icons, "武器图标表是空的（visualizer/static/assets/weapon_icons 不见了？）"
        assert "AK-47" in icons, f"AK-47 没有图标，只有 {sorted(icons)[:8]}"
        body, _ctype, _ = get(icons["AK-47"], "image/png")
        assert body[:8] == b"\x89PNG\r\n\x1a\n"

        # 错误路径
        expect_error(f"/api/sample/{quoted}?k=9999", (400,))          # 行号越界
        expect_error(f"/api/sample/{quoted}?k=abc", (400,))           # k 不是整数
        expect_error("/api/sample/train/..%2F..%2Ftask.json", (400, 404))
        expect_error("/api/nope", (404,))
        expect_error("/weapon-icons/..%2Fapp.js", (400, 404))

        body, _ctype, _ = get("/api/rescan")
        assert json.loads(body.decode("utf-8"))["ok"] is True
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)

    print(f"  HTTP {base} OK（samples / sample?k= / radar / 静态资源 / "
          f"越界与穿越拒绝 / rescan）")


# ═══════════════════════════════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════════════════════════════


def main() -> int:
    print("spatial_prediction 可视化自检")
    print("─" * 62)
    print("[1/4] 逆变换往返 + 常量对拍 …")
    test_roundtrip()
    test_constants()
    print("[2/4] 合成样本载荷 + 标签体检 …")
    test_payload()
    test_label_checks()

    print("[3/4] 真实样本 …")
    real: str | None = None
    if D.DEFAULT_DATA_DIR.is_dir() and any(D.DEFAULT_DATA_DIR.glob("*/" + "*_n*.npz")):
        real = test_real_samples(D.DEFAULT_DATA_DIR)
    else:
        print(f"  跳过：{D.DEFAULT_DATA_DIR} 里还没有样本")
        print("       先跑 python -m scripts.training_data.prepare "
              "--task spatial_prediction")

    print("[4/4] HTTP 端到端 …")
    data_dir = D.DEFAULT_DATA_DIR
    name = real
    tmp_root: Path | None = None
    if name is None:
        import tempfile

        try:
            tmp_root = Path(tempfile.mkdtemp(prefix="spatial_pred_selftest_"))
            data_dir = _write_fixture(tmp_root)
            name = "de_mirage_f0_r3_n3.npz"
        except OSError as exc:                     # 只读挂载 / 沙箱：跳过，不算失败
            print(f"  跳过：建不出临时夹具目录（{exc}）")
            tmp_root = None
    if name is not None and data_dir.is_dir():
        try:
            test_http(data_dir, name)
        finally:
            if tmp_root is not None:
                import shutil

                shutil.rmtree(tmp_root, ignore_errors=True)

    print("─" * 62)
    print("全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
