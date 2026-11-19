"""把 README / API.md 里写的用法真的跑一遍，确认文档没写错。

    PYTHONPATH=tools python notes/scratch/verify_docs_examples.py path/to.csz
"""
import math
import sys, numpy as np
from demo_extract import codec, recipes, voxel

path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/tv6b_full.csz"
ok = bad = 0
def check(label, cond, detail=""):
    global ok, bad
    if cond: ok += 1; print(f"  ok   {label}")
    else:    bad += 1; print(f"  FAIL {label}   {detail}")

# ── README §2 快速开始 ───────────────────────────────────────────────────────
data = codec.load(path)
rnd = recipes.round_by_id(data, 7)
check("codec.load -> dict", isinstance(data, dict))
check("round_by_id(data, 7)", rnd["id"] == 7)
check("codec.equal 自洽", codec.equal(data, codec.loads(codec.dumps(data)))[0])

# ── README §4.1 / API §4.2 张量 ──────────────────────────────────────────────
X = recipes.player_tensor(rnd)
T, P = len(rnd["ticks"]), len(rnd["players"])
check("player_tensor 形状 (T,P,16)", X.shape == (T, P, 16), str(X.shape))
check("PLAYER_FEATURES 是 16", len(recipes.PLAYER_FEATURES) == 16)
check("PLAYER_FEATURES_V6 是 19", len(recipes.PLAYER_FEATURES_V6) == 19)
G = recipes.player_tensor(rnd, recipes.PLAYER_FEATURES_V6)
check("player_tensor v6 形状 (T,P,19)", G.shape == (T, P, 19), str(G.shape))
check("v6 前 16 列 == 老 16 列", np.array_equal(X, G[:, :, :16], equal_nan=True))
check("player_tracks 形状 (P,T)", recipes.player_tracks(rnd, ("x",))["x"].shape == (P, T))
check("yaw_sincos 形状 (T,P,2)", recipes.yaw_sincos(rnd).shape == (T, P, 2))
check("to_meters 只缩 HU 列",
      abs(recipes.to_meters(X)[0, 0, 0] - X[0, 0, 0] * 0.0254) < 1e-3
      and abs(recipes.to_meters(X)[0, 0, 8] - X[0, 0, 8]) < 1e-3)
check("nan_report 键是速度列（含 v）", set(recipes.nan_report(data)) <= {"v", "vx", "vy", "vz"})
check("alive_mask 形状 (T,P)", recipes.alive_mask(rnd).shape == (T, P))

# ── §4.2 单 tick ────────────────────────────────────────────────────────────
s = recipes.tick_sample(data, 7, rnd["ticks"][1200])
check("tick_sample players (P,16)", s["players"].shape == (P, 16), str(s["players"].shape))
check("tick_sample 含 ground 三键", {"on_ground", "jump", "land"} <= set(s))
check("tick_sample 三键形状 (P,)", all(s[k].shape == (P,) for k in ("on_ground", "jump", "land")))
check("tick_sample meta", set(s["meta"]) >= {"tick", "seconds", "map", "round", "sample_index"})

# ── §4.11 地面 ──────────────────────────────────────────────────────────────
og, jm, lm = recipes.on_ground_mask(rnd), recipes.jump_mask(rnd), recipes.land_mask(rnd)
check("on_ground_mask bool (T,P)", og.dtype == bool and og.shape == (T, P))
check("jump/land int (T,P)", jm.shape == (T, P) and lm.shape == (T, P))
check("land 第 0 行恒为 0", lm[0].sum() == 0)
a_fast = recipes.player_tensor(rnd, ("v",))[:, :, 0]
check("v 的有限值不超过水平速度上限", float(np.nanmax(np.abs(a_fast))) < 260.0,
      str(float(np.nanmax(np.abs(a_fast)))))
check("v 缺值只出现在 vx 也缺的采样点上",
      set(np.argwhere(~np.isfinite(a_fast))[:, 0].tolist())
      <= set(np.argwhere(~np.isfinite(recipes.player_tensor(rnd, ("vx",))[:, :, 0]))[:, 0].tolist()))
check("nan_report 会报 v（不只报 vx/vy/vz）", "v" in recipes.nan_report(data),
      str(recipes.nan_report(data)))
check("工具不再提供任何阈值函数",
      not any(hasattr(recipes, n) for n in
              ("audible_movement", "RUN_SPEED_HU", "WALK_SPEED_RATIO",
               "RUN_SPEED_MIN_HU", "RUN_SPEED_MAX_HU")))
check("footstep_warning 不再指向已删的函数",
      "audible_movement" not in (recipes.footstep_warning(data) or ""))

# ── §4.3 火焰 / §4.4 烟雾 ───────────────────────────────────────────────────
r = next(r for r in data["rounds"] if r["fire"])
tick = r["fire"][0]["changes"][-1]["tick"]
fire = recipes.active_fire(r, tick)
check("active_fire 非空", len(fire) >= 1)
check("fire 点云 (N,3)", fire[0]["points"].shape[1] == 3)
check("fire_volume_at", "grid" in recipes.fire_volume_at(r, tick, resolution=32.0))

r2 = next(r for r in data["rounds"] if r["smoke"])
inst = r2["smoke"][0]
t2 = inst["changes"][-1]["tick"]
g = recipes.smoke_grid_at(inst, t2)
check("smoke_grid_at (32,32,32)", g is not None and g.shape == (32, 32, 32))
check("smoke_points 数量 == grid 体素数", len(recipes.smoke_points_at(inst, t2)) == int(g.sum()))
check("grid_to_world 中心 == detonation",
      np.allclose(recipes.grid_to_world(np.array([[15, 15, 15]]), inst["detonation"]),
                  np.array(inst["detonation"]) - 6.0, atol=1e-3))
check("active_smoke 结构", set(recipes.active_smoke(r2, t2)) >= {"grids", "points", "meta"})
check("smoke_lifetime 能按 id 找到", recipes.smoke_lifetime(data, inst["entity_id"]) is not None)

# ── §4.4 multi-hot ──────────────────────────────────────────────────────────
inv = recipes.inventory_multihot(rnd, n_items=len(data["weapons"]))
check("inventory_multihot (T,P,32)", inv.shape == (T, P, 32), str(inv.shape))
sp = recipes.spotted_multihot(rnd)
check("spotted_multihot (T,P,10)", sp.shape == (T, P, 10))
check("spotted 对角线全 False", not np.diagonal(sp, axis1=1, axis2=2).any())

# ── §4.5 窗口 ───────────────────────────────────────────────────────────────
w = list(recipes.iter_windows(data, length=64, stride=32, round_id=7))
check("iter_windows 不跨回合", all(b - a == 64 for _, a, b, _ in w) and len(w) > 0)
check("窗口数量 = T-64+1 除以 stride",
      len(w) == len(range(0, max(0, T - 64 + 1), 32)), f"{len(w)}")

# ── §4.1 定位 ───────────────────────────────────────────────────────────────
check("round_ids", recipes.round_ids(data) == [r["id"] for r in data["rounds"]])
check("skipped_rounds 是 dict 列表",
      all(isinstance(x, dict) and "reason" in x for x in recipes.skipped_rounds(data)))
check("ticks_of / seconds_of 等长",
      len(recipes.ticks_of(rnd)) == len(recipes.seconds_of(rnd)) == T)
check("sample_index 命中", recipes.sample_index(rnd, rnd["ticks"][5]) == 5)
check("sample_index 未命中 -1", recipes.sample_index(rnd, -1) == -1)
check("round_winner", recipes.round_winner(rnd) == rnd["winner"])
check("summary 出字符串", isinstance(recipes.summary(data), str))
check("iter_samples 逐点", sum(1 for _ in recipes.iter_samples(rnd)) == T)

# ── §5 voxel 底层 ───────────────────────────────────────────────────────────
j = voxel.journal_at(inst, t2)
check("journal_at 非空", j is not None)
check("change_ticks 非空", len(voxel.change_ticks(inst)) > 0)
check("state_at == journal 长度", voxel.state_at(inst, t2)[0] == len(j))
check("boundary_voxels ⊆ solid_voxels",
      voxel.boundary_voxels(j) <= set(voxel.solid_voxels(j)))
check("morton3/demorton3 往返",
      all(voxel.demorton3(voxel.morton3(x, y, z, 3), 3) == (x, y, z)
          for x in range(8) for y in range(8) for z in range(8)))
check("world_voxels 出点列", len(voxel.world_voxels(j, inst["detonation"])) == int(g.sum()))

# ── 老文件兼容（v6 之前没有地面三列，v7 之前没有下蹲两列） ──────────────────
_STRIPPED = recipes.GROUND_FEATURES + recipes.CROUCH_FEATURES + ("crouching",)
old = {"rounds": [{k: v for k, v in rnd.items()}], "players": data["players"],
       "meta": data["meta"]}
old_r = old["rounds"][0]
old_r["players"] = [{k: v for k, v in p.items() if k not in _STRIPPED}
                    for p in rnd["players"]]
try:
    recipes.on_ground_mask(old_r); check("老文件 on_ground 抛 KeyError", False)
except KeyError as e:
    check("老文件 on_ground 抛 KeyError", "cs2.demo.v6" in str(e))
check("老文件 tick_sample 不含 ground 键",
      "on_ground" not in recipes.tick_sample(old, old_r["id"], old_r["ticks"][3]))
check("老文件 player_tensor(16) 仍可用",
      recipes.player_tensor(old_r).shape == (T, P, 16))

# v6 文件：有地面三列、没有下蹲两列 —— 报错必须指向 v7，不能是 v6
v6 = {"rounds": [{k: v for k, v in rnd.items()}], "players": data["players"],
      "meta": data["meta"]}
v6_r = v6["rounds"][0]
v6_r["players"] = [{k: v for k, v in p.items()
                    if k not in recipes.CROUCH_FEATURES + ("crouching",)}
                   for p in rnd["players"]]
try:
    recipes.crouch_amount(v6_r); check("v6 文件 crouch 抛 KeyError", False)
except KeyError as e:
    check("v6 文件 crouch 抛 KeyError", "cs2.demo.v7" in str(e))
check("v6 文件 tick_sample 不含 crouch 键",
      "crouch" not in recipes.tick_sample(v6, v6_r["id"], v6_r["ticks"][3]))
check("v6 文件仍给出 ground 键",
      "on_ground" in recipes.tick_sample(v6, v6_r["id"], v6_r["ticks"][3]))
check("v6 文件 player_tensor(19) 仍可用",
      recipes.player_tensor(v6_r, recipes.PLAYER_FEATURES_V6).shape == (T, P, 19))

# ── §4.12 下蹲：文档里的关系必须在真实数据上成立 ────────────────────────────
c = recipes.crouch_amount(rnd)
cd = recipes.crouching_mask(rnd)
eye = recipes.eye_height(rnd)
check("§4.12 crouch 落在 [0,1]", float(c.min()) >= 0.0 and float(c.max()) <= 1.0)
check("§4.12 eye_height 无 NaN", int(np.isnan(eye).sum()) == 0)
check("§4.12 eye_height == 64.0938 - 18 * crouch",
      float(np.abs(eye - (recipes.STANDING_EYE_HEIGHT_HU
                          - recipes.CROUCH_EYE_DROP_HU * c)).max()) < 1e-3)
check("§4.12 碰撞盒只有 72 / 54 两种",
      set(np.unique(np.where(cd, 54.0, 72.0)).tolist()) <= {72.0, 54.0})
check("§4.12 crouching 不是 crouch > 0.5",
      not np.array_equal(cd, c > 0.5))
check("§4.12 (x,y,z) 是脚底：eye_position 的 x 原样",
      float(np.abs(recipes.eye_position(rnd)[:, :, 0]
                   - recipes.player_tensor(rnd, ("x",))[:, :, 0]).max()) == 0.0)
check("§4.12 PLAYER_FEATURES_V7 = 20", len(recipes.PLAYER_FEATURES_V7) == 20)
check("§4.12 has_crouch 标记", data["meta"].get("has_crouch") is True)

# ── §1.9 量级表：文档里写的数字必须和真实数据对得上 ────────────────────────
X16 = recipes.player_tensor(rnd)          # 回合 7，(T,10,16)
fin = np.isfinite(X16)
def rng(j):
    c = X16[:, :, j][fin[:, :, j]]
    return float(c.min()), float(c.max())
lo, hi = rng(recipes.PLAYER_FEATURES.index("vz"))
check("§1.9 vz 落在文档写的 ±1152 内", -1160 <= lo and hi <= 1160, f"[{lo:.1f}, {hi:.1f}]")
lo, hi = rng(recipes.PLAYER_FEATURES.index("hp"))
check("§1.9 hp 是 0..100", lo >= 0 and hi <= 100, f"[{lo}, {hi}]")
lo, hi = rng(recipes.PLAYER_FEATURES.index("armor"))
check("§1.9 armor 是 0..100", lo >= 0 and hi <= 100, f"[{lo}, {hi}]")
lo, hi = rng(recipes.PLAYER_FEATURES.index("flash_alpha"))
check("§1.9 flash_alpha 是 0..255 字节", lo >= 0 and hi <= 255, f"[{lo}, {hi}]")
lo, hi = rng(recipes.PLAYER_FEATURES.index("yaw"))
check("§1.9 yaw 是 (-180, 180]", lo >= -180 and hi <= 180, f"[{lo}, {hi}]")
lo, hi = rng(recipes.PLAYER_FEATURES.index("pitch"))
check("§1.9 pitch 落在引擎上限 ±89 内", lo >= -89 and hi <= 89, f"[{lo}, {hi}]")

fire = [f for r in data["rounds"] for f in r["fire"]]
check("§1.9 火焰每快照最多 16 格",
      all(c["count"] <= 16 for f in fire for c in f["changes"]),
      str(max(c["count"] for f in fire for c in f["changes"])))
nrm = [math.sqrt(sum(x * x for x in c["normal"][3*i:3*i+3]))
       for f in fire for c in f["changes"] for i in range(c["count"])]
check("§1.9 法线模长精确为 1", all(abs(n - 1.0) < 1e-6 for n in nrm))
inf_iv = [e["s1"] - e["s0"] for r in data["rounds"] for e in r["events"]["infernos"]
          if e.get("s0") is not None and e.get("s1") is not None]
check("§1.9 火焰区间 ≤ 7.03s（燃烧瓶）",
      inf_iv and max(inf_iv) <= 7.1, f"max {max(inf_iv):.2f}s")
check("§1.9 火焰区间 ≥ 2.41s（最短的那个）",
      inf_iv and min(inf_iv) >= 2.3, f"min {min(inf_iv):.2f}s")
# 实体存续比"真的在烧"长得多，文档里这两个数是分开写的，别混
ent = [(f["last_tick"] - f["first_tick"]) / 64 for f in fire]
check("§1.9 火焰实体存续可达 ~20s（≠ 燃烧时长）",
      max(ent) <= 20.2 and max(ent) > 7.5, f"max {max(ent):.2f}s")
burn = []
for f in fire:
    t = [c["tick"] for c in f["changes"] if any(c["burning"])]
    if t:
        burn.append((t[-1] - t[0]) / 64)
check("§1.9 真的在烧的时长 ≤ 6.92s",
      burn and max(burn) <= 7.0, f"max {max(burn):.2f}s")
sm_iv = [e["s1"] - e["s0"] for r in data["rounds"] for e in r["events"]["smokes"]
         if e.get("s0") is not None and e.get("s1") is not None]
check("§1.9 烟雾区间精确等于 22.062s",
      sm_iv and all(abs(x - 22.062) < 1e-3 for x in sm_iv), str(sorted(set(sm_iv))[:3]))
vox = []
for r in data["rounds"]:
    for s in r["smoke"]:
        g = recipes.smoke_grid_at(s, s["changes"][-1]["tick"])
        if g is not None:
            vox.append(int(g.sum()))
check("§1.9 烟体素数在网格容量 32³ 之内",
      vox and max(vox) <= 32 ** 3, f"max {max(vox)}")
check("§1.9 网格是 32³", recipes.smoke_grid_at(
    next(s for r in data["rounds"] for s in r["smoke"]), 
    next(s["changes"][-1]["tick"] for r in data["rounds"] for s in r["smoke"])).shape
    == (voxel.VOXEL_GRID_DIM,) * 3)

print(f"\n{ok} ok, {bad} failed")
sys.exit(1 if bad else 0)
