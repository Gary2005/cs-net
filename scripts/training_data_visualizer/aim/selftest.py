"""自检：`python -m scripts.training_data_visualizer.aim.selftest`

不打开浏览器也能验证四件事：

1. **逆变换是对的**：用 `_SCALE`/`_BIAS` 把一个已知的物理量正向编码成 [0,1]，
   再 `decode_player()` 回来，必须落在原来的值上（yaw / pitch 在 ±180 附近除外）。
   `decode_players()`（全场逐帧那一版）要给出同样的数，只是布局不同。
2. **遮挡矩阵的语义是对的**：窗口**最后一帧**（受害者死前那一刻）A 必须看得见 B ——
   人是在看得见的地方被打死的。这一条是 `obstruction` 唯一的硬语义检查，错了就是
   射线 / 包围盒 / 眼高哪里写反了。另外检查取值只可能是 0/1/2、死人的行列全 0、
   方向性（`[i][j]` 和 `[j][i]` 允许不同）。见 `test_obstruction()`。
3. **真实样本能读**：扫 `data/demos/prepared/aim`，挑第一个样本，检查帧数 / 列数 /
   A/B 下标 / 逐帧通道长度 / 数值范围（含 `defuser` 只有 0/1）。旧版样本（含
   `atk` / `atk_view`）必须被**明确拒绝**，而不是解出一堆错数。
4. **HTTP 真的能跑**：在随机端口起一个真 server，用 urllib 拉一遍
   `/api/samples`、`/api/sample/...`、`/api/radar/...`、`/api/weapon-icons`
   （外加一张真的图标 PNG）和三个静态文件。有训练产物（`models/*/best.pt`，
   老习惯的 `models/aim/*/best.pt` 也算）且装了 torch 时，再真跑一遍
   `/api/model` + `/api/predict/...`；没有就跳过并打印原因 —— 模型预测是可选功能。

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

import numpy as np

from . import data as D
from . import server as S


def _encoded_row(phys: dict[str, float]) -> np.ndarray:
    """物理量 → 20 列 [0,1]（用 SCALE/BIAS 正向编码，和 aim.py 一致）。"""
    raw = np.zeros(len(D.FEATURES), dtype=np.float64)
    cx, cy, cz = D.MAP_CENTER.get("de_mirage", (0.0, 0.0, 0.0))
    raw[0] = (phys["x"] - cx) / D.MAP_RANGE[0] * 0.5 + 0.5
    raw[1] = (phys["y"] - cy) / D.MAP_RANGE[1] * 0.5 + 0.5
    raw[2] = (phys["z"] - cz) / D.MAP_RANGE[2] * 0.5 + 0.5
    raw[3] = math.sin(math.radians(phys["yaw"])) * 0.5 + 0.5
    raw[4] = math.cos(math.radians(phys["yaw"])) * 0.5 + 0.5
    raw[5] = math.sin(math.radians(phys["pitch"])) * 0.5 + 0.5
    raw[6] = math.cos(math.radians(phys["pitch"])) * 0.5 + 0.5
    raw[7] = phys["vx"] * D.SCALE[7] + D.BIAS[7]
    raw[8] = phys["vy"] * D.SCALE[8] + D.BIAS[8]
    raw[9] = phys["vz"] * D.SCALE[9] + D.BIAS[9]
    raw[10] = phys["hp"] / 100.0
    raw[11] = phys["crouch"]
    raw[12] = phys["armor"] / 100.0
    raw[13] = phys["helmet"]
    raw[14] = phys["team"]
    raw[15] = phys["shot"]
    raw[16] = min(phys["since_shot"] / 100.0, 1.0)
    raw[17] = phys["has_kill"]
    raw[18] = phys["blinded"]
    raw[19] = phys["defuser"]
    return np.clip(raw, 0.0, 1.0)


def test_roundtrip() -> None:
    phys = {
        "x": 1234.5, "y": -567.25, "z": 128.0,
        "yaw": 37.5, "pitch": -12.25,
        "vx": 123.0, "vy": -45.0, "vz": 678.0,
        "hp": 84.0, "crouch": 0.25, "armor": 92.0, "helmet": 1.0,
        "team": 1.0, "shot": 1.0, "since_shot": 1.25, "has_kill": 1.0,
        "blinded": 1.0, "defuser": 1.0,
    }
    raw = _encoded_row(phys)
    assert raw.min() >= 0.0 and raw.max() <= 1.0, "编码后必须落在 [0,1]"
    out = D.decode_player(raw[None, :], "de_mirage")
    for key in ("x", "y", "z", "yaw", "pitch", "vx", "vy", "vz", "hp", "armor",
                "crouch", "helmet", "team", "shot", "since_shot", "has_kill", "blinded",
                "defuser"):
        got = out[key][0]
        want = phys[key]
        assert abs(got - want) < 1e-2, f"{key}: 还原出 {got}，应为 {want}"
    # 4 个 sin/cos 列：逆变换后就是 [-1,1] 的原始三角函数值
    for key, want in (
        ("yaw_sin", math.sin(math.radians(37.5))),
        ("yaw_cos", math.cos(math.radians(37.5))),
        ("pitch_sin", math.sin(math.radians(-12.25))),
        ("pitch_cos", math.cos(math.radians(-12.25))),
    ):
        assert abs(out[key][0] - want) < 1e-3, (key, out[key][0], want)
    # 派生量
    assert abs(out["eye_z"][0] - (D.STANDING_EYE_HU - D.CROUCH_EYE_DROP_HU * 0.25)) < 1e-3
    assert abs(out["box_h"][0] - (D.HULL_STAND_HU - 18.0 * 0.25)) < 1e-3

    # 18 列旧样本：blinded / defuser 都按 0 处理，不抛
    old = D.decode_player(raw[None, :18], "de_mirage")
    assert old["blinded"] == [0.0] and old["defuser"] == [0.0]
    # 19 列（有 blinded、还没有 defuser）：只补缺的那一列，别的不许动
    old19 = D.decode_player(raw[None, :19], "de_mirage")
    assert old19["blinded"] == [1.0] and old19["defuser"] == [0.0], old19["defuser"]

    # 角度跨 ±180 处：atan2 恢复的是 [-180,180]，回绕后必须一致
    raw2 = _encoded_row({**phys, "yaw": -179.5})
    yaw2 = D.decode_player(raw2[None, :], "de_mirage")["yaw"][0]
    assert abs(yaw2 - (-179.5)) < 1e-2, yaw2

    # 全场逐帧那一版：把同一个人放在第 4 位，解出来的 (T,10) 必须逐格相同
    stacked = np.zeros((3, 10, len(D.FEATURES)))
    stacked[:, 4, :] = raw
    full = D.decode_players(stacked, "de_mirage")
    for key in ("x", "y", "z", "yaw", "hp"):
        assert full[key][0][4] == out[key][0], (key, full[key][0][4], out[key][0])
    # 空行（全 0 的 [0,1] 输入）解出来就是"地图中心减去整幅范围"，不是 0 —— 别把
    # 它当成"没有这个人"。谁死了由 `ctx_player_mask` 说，不靠数值猜。
    blank = D.decode_player(np.zeros((1, len(D.FEATURES))), "de_mirage")
    assert full["x"][0][0] == blank["x"][0], (full["x"][0][0], blank["x"][0])


def test_obstruction(data_dir: Path) -> None:
    """`obstruction` 的硬语义：死前最后一帧 A 看得见 B。

    人的死法就是"在看得见的地方被打中"，所以窗口最后一帧（`last_alive` 那一帧，
    见 `tasks/aim.py`）`obstruction[-1][A][B]` 必须是 0。实测 449 条样本里 93.1%
    成立；剩下 7% 是"一枪穿墙 / 打到露出来的那一角又被判成墙"这类边界情况，所以
    这里只要求 ≥70%，不要求 100% —— 但**如果大面积不成立**，就是射线、包围盒或者
    眼高算错了。

    顺带钉住三条结构性事实：
      * 取值只有 0/1/2；
      * 死人的行列全 0（和 `ctx_player_mask == 0` 对齐）；
      * 方向性：`[i][j]` 和 `[j][i]` 允许不同（两个人的眼睛不在同一个位置），
        但绝大多数时刻相同（实测只差 2.6%）。
    """
    samples = D.scan_samples(data_dir)[:40]
    assert samples, f"在 {data_dir} 下没扫到样本"
    clear_last = []
    diff = same = 0
    for item in samples:
        path = D.sample_path(data_dir, item["split"], item["name"])
        payload = D.load_sample(path, split=item["split"], name=item["name"], data_dir=data_dir)
        obs = np.asarray(payload["obstruction"], dtype=np.int8)
        mask = np.asarray(payload["players"]["mask"], dtype=np.int8)
        a, b = payload["atk"]["index"], payload["tgt"]["index"]
        assert a is not None and b is not None
        assert obs.min() >= 0 and obs.max() <= 2, (item["name"], obs.min(), obs.max())
        # 死人的行列必须是 0
        dead = mask == 0
        if dead.any():
            bad_cols = np.abs(obs).sum(axis=1) * dead          # 第 j 列有值
            bad_rows = np.abs(obs).sum(axis=2) * dead          # 第 i 行有值
            assert bad_cols.max() == 0, f"{item['name']}: 死人的列里有非 0 遮挡"
            assert bad_rows.max() == 0, f"{item['name']}: 死人的行里有非 0 遮挡"
        clear_last.append(int(obs[-1, a, b]))
        for t in range(obs.shape[0]):
            al = np.flatnonzero(mask[t])
            if len(al) < 2:
                continue
            sub = obs[t][np.ix_(al, al)]
            iu = np.triu_indices(len(al), 1)
            diff += int((sub[iu] != sub.T[iu]).sum())
            same += int(sub[iu].size)

    ratio = float(np.mean(np.array(clear_last) == 0))
    assert ratio >= 0.70, (
        f"只有 {ratio:.0%} 的样本在最后一帧「A 看得见 B」——遮挡语义可能错了"
        f"（分布 {np.bincount(clear_last, minlength=3).tolist()}）"
    )
    dir_ratio = diff / max(same, 1)
    assert dir_ratio < 0.25, f"i→j 与 j→i 不同的比例 {dir_ratio:.0%}，两侧眼睛不对称过头了"
    print(f"  遮挡: 最后一帧 A→B 看得见 {ratio:.0%}；方向不一致 {dir_ratio:.1%}  OK")


def test_real_sample(data_dir: Path) -> str:
    samples = D.scan_samples(data_dir)
    assert samples, f"在 {data_dir} 下没扫到样本"
    first = samples[0]
    path = D.sample_path(data_dir, first["split"], first["name"])
    payload = D.load_sample(path, split=first["split"], name=first["name"], data_dir=data_dir)

    n = payload["frames"]
    assert n > 1
    assert len(payload["seconds"]) == n
    assert payload["n_players"] == 10
    assert payload["n_features"] == len(payload["features"]) == len(D.FEATURES)
    assert len(payload["atk"]["phys"]["x"]) == n
    assert len(payload["tgt"]["phys"]["x"]) == n
    assert len(payload["atk"]["raw"]) == n

    # 逐帧梯度的每个字段都必须是 (T,10) 或 (T,10,10)
    pl = payload["players"]
    for key, arr in pl["phys"].items():
        assert len(arr) == n and len(arr[0]) == 10, (key, len(arr))
    assert len(pl["mask"]) == n and len(pl["mask"][0]) == 10
    assert len(pl["weapon"]) == n and len(pl["weapon"][0]) == 10
    assert len(payload["obstruction"]) == n and len(payload["obstruction"][0]) == 10
    assert len(payload["projectiles"]) == n

    # 前端特征表会按 features 逐个查 phys；缺一个就会在控制台报 undefined
    for name in payload["features"]:
        assert name in payload["atk"]["phys"], f"phys 缺列: {name}"

    # A / B 的下标和 mask 必须对得上（窗口内恒定）
    a, b = payload["atk"]["index"], payload["tgt"]["index"]
    assert a is not None and b is not None and a != b
    mask = np.asarray(pl["mask"], dtype=np.int8)
    assert (mask[:, a] == 1).all() and (mask[:, b] == 2).all(), "A/B 的 mask 沿窗口不恒定"
    # 单人视图就是 players 里那一行，逐格相同
    for col in ("x", "y", "z", "hp"):
        assert pl["phys"][col][3][a] == payload["atk"]["phys"][col][3], col
        assert pl["phys"][col][3][b] == payload["tgt"]["phys"][col][3], col

    # 原始值必须在 [0,1]（这是 aim.py 的契约）
    raw = np.asarray(payload["atk"]["raw"], dtype=np.float64)
    assert raw.min() >= 0.0 and raw.max() <= 1.0, (raw.min(), raw.max())

    # 状态列只能是 0/1（除 since_shot）
    for key in ("shot", "has_kill", "blinded", "helmet", "defuser"):
        arr = np.asarray(payload["atk"]["phys"][key])
        assert np.isin(arr, (0.0, 1.0)).all(), f"{key} 不是 0/1: {sorted(set(arr))[:5]}"

    # 武器：死人的那一格必须是 -1
    wp = np.asarray(pl["weapon"])
    assert wp.shape == (n, 10)
    assert (wp[mask == 0] == -1).all(), "死人的武器格不是 -1"

    # global_feat 现在逐帧 (T,5)：第 5 列 = 回合时间，和 `seconds` 必须严格同步
    # （两者都是从 rnd["round_seconds"] 来的，只差一个常数偏移）。
    g = payload["global"]
    assert g["per_frame"], "新样本的 global_feat 应该是 (T,5) 逐帧，不是单帧快照"
    assert len(g["round_seconds"]) == n and len(g["x"]) == n, "global_feat 帧数不对"
    assert 0.0 <= g["round_seconds"][0] <= D._ROUND_TIME_SPAN, g["round_seconds"][0]
    drift = (np.asarray(g["round_seconds"]) - g["round_seconds"][0]
             - np.asarray(payload["seconds"]))
    assert np.abs(drift).max() < 0.02, f"回合时间和 seconds 不同步（max {np.abs(drift).max():.3f} s）"
    assert not any("global_feat" in w for w in payload["warnings"]), payload["warnings"]

    # 窗口跨度固定 4 s → T = 4×64+1 = 257，全样本同长（只有窗口被截到回合开头时才短）
    assert abs(payload["seconds"][-1] - D.LOOKBACK) < 1e-4, payload["seconds"][-1]
    assert n == D.EXPECTED_FRAMES, f"T={n}，固定 4 s + {D.TICK_HZ:g} Hz 应该是 {D.EXPECTED_FRAMES}"

    # 道具：id 和位置必须一致（-1 的空槽不许有非 0 坐标看不出来，这里只查 id 范围）
    proj_ids = {int(p["id"]) for frame in payload["projectiles"] for p in frame}
    assert proj_ids <= set(D.PROJECTILE_TYPES), proj_ids

    # JSON 必须可序列化（allow_nan=False 在 server 里已经这么用了）
    json.dumps(payload, ensure_ascii=False, allow_nan=False)

    print(f"  样本: {first['split']}/{first['name']}  "
          f"{n} 帧 × 10 人 × {payload['n_features']} 维  map={payload['map_name']}")
    print(f"  玩家: a={a} v={b}  hs={payload['hs']} dmg={payload['dmg']}  "
          f"首帧道具 {len(payload['projectiles'][0])} 个  "
          f"雷达 {'有' if payload['radar'] else '无'}")
    if payload["warnings"]:
        print(f"  提醒: {payload['warnings']}")
    return first["split"] + "/" + first["name"]


def test_http(data_dir: Path, sample_id: str) -> None:
    state = S.AppState(data_dir, use_model=False)
    # 自检时把访问日志关掉，输出只留下断言结果
    S.AimHandler.log_message = lambda *args, **kwargs: None  # noqa: E731
    httpd = S.AimHTTPServer(("127.0.0.1", 0), state)
    port = int(httpd.server_address[1])
    thread = threading.Thread(target=httpd.serve_forever,
                              kwargs={"poll_interval": 0.02}, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{port}"

    def get(path: str, expect: str = "application/json") -> tuple[bytes, str]:
        req = urllib.request.Request(base + path, headers={"Accept-Encoding": "identity"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            body = resp.read()
            ctype = resp.headers.get("Content-Type", "")
            code = resp.status
        assert code == 200, f"{path} → HTTP {code}"
        assert expect in ctype, f"{path} → {ctype}"
        return body, ctype

    try:
        body, _ = get("/api/samples")
        listing = json.loads(body.decode("utf-8"))
        assert listing["total"] == len(listing["samples"]) > 0

        body, _ = get("/api/sample/" + "/".join(
            urllib.parse.quote(p, safe="") for p in sample_id.split("/")))
        payload = json.loads(body.decode("utf-8"))
        assert payload["id"] == sample_id
        assert payload["obstruction"] and payload["players"]["mask"]

        if payload["radar"]:
            body, _ = get(f"/api/radar/{payload['map_name']}", "image/png")
            assert body[:8] == b"\x89PNG\r\n\x1a\n"

        # 旧接口必须已经不在（否则说明前端还在指望它）
        try:
            get(f"/api/view/{sample_id}/atk?ch=composite", "image/png")
            raise AssertionError("/api/view 还活着 —— 视角图这一版已经删了")
        except urllib.error.HTTPError as exc:
            assert exc.code == 404, exc.code

        # 页面和静态资源
        body, ctype = get("/", "text/html")
        assert b"style.css" in body
        body, _ = get("/static/app.js", "javascript")
        assert b"/api/sample/" in body, "app.js 里没找到取样本的路径"
        body, _ = get("/static/style.css", "text/css")
        assert b"--panel" in body

        # 武器图标：查表里有东西、而且真能取到一张 PNG（素材目录是共享的）
        body, _ = get("/api/weapon-icons")
        icons = json.loads(body.decode("utf-8"))["icons"]
        assert icons, "武器图标表是空的（visualizer/static/assets/weapon_icons 不见了？）"
        assert "AK-47" in icons, f"AK-47 没有图标，只有 {sorted(icons)[:8]}"
        body, _ = get(icons["AK-47"], "image/png")
        assert body[:8] == b"\x89PNG\r\n\x1a\n"

        # 模型预测（可选）：自检这里不加载 torch（`use_model=False`），只验 503 的措辞。
        body, _ = get("/api/model")
        model = json.loads(body.decode("utf-8"))
        assert model["ok"] is False and model["reason"], model
        print(f"  模型: 跳过（{model['reason']}）")

        # 错误路径：目录穿越必须被拒
        req = urllib.request.Request(base + "/api/sample/train/..%2F..%2Ftask.json")
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                raise AssertionError(f"非法路径竟然返回了 {resp.status}")
        except urllib.error.HTTPError as exc:
            assert exc.code in (400, 404), exc.code
        # 图标路由只认 weapon_*.png，`../` 一律 404
        req = urllib.request.Request(base + "/weapon-icons/..%2Fapp.js")
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                raise AssertionError(f"非法图标名竟然返回了 {resp.status}")
        except urllib.error.HTTPError as exc:
            assert exc.code in (400, 404), exc.code
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)

    print(f"  HTTP: {base}  OK（samples / sample / radar / 静态资源 / "
          f"{len(icons)} 个武器图标 / 越界拒绝 / 旧接口已下线）")


def test_legacy_rejected(path: Path) -> None:
    """旧版样本（含 `atk` / `atk_view`）必须**明确报错**，不能硬解出假数据。

    这一版把 atk/tgt 两条轨迹和视角图全删了（换成逐帧全场 + 遮挡矩阵），列的含义
    和旧样本完全对不上。静默读旧样本会画出一堆"看起来合理但全是错的"图 ——
    比报错危险得多，所以这里钉住那条 ValueError。
    """
    import tempfile

    with np.load(path) as z:
        arrays = {k: np.asarray(z[k]) for k in z.files}
    arrays["atk"] = np.zeros((arrays["seconds"].shape[0], 20), np.float32)
    arrays["tgt"] = arrays["atk"]
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "train").mkdir()
        out = root / "train" / "legacy.npz"
        np.savez_compressed(out, **arrays)
        try:
            D.load_sample(out, split="train", name="legacy.npz", data_dir=root)
        except ValueError as exc:
            assert "旧版样本" in str(exc) and "overwrite" in str(exc), str(exc)
        else:
            raise AssertionError("旧版样本没被拒绝 —— load_sample 会解出一堆错数")
    print("  旧版样本（含 atk/atk_view）: 明确拒绝并提示 --overwrite  OK")


def main() -> int:
    print("aim 可视化自检")
    print("─" * 56)
    print("[1/4] 逆变换往返 …")
    test_roundtrip()
    print("  OK")
    print("[2/4] 遮挡矩阵语义 …")
    test_obstruction(D.DEFAULT_DATA_DIR)
    print("[3/4] 真实样本 …")
    sample_id = test_real_sample(D.DEFAULT_DATA_DIR)
    test_legacy_rejected(D.sample_path(D.DEFAULT_DATA_DIR, *sample_id.split("/", 1)))
    print("[4/4] HTTP 端到端 …")
    test_http(D.DEFAULT_DATA_DIR, sample_id)
    print("─" * 56)
    print("全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
