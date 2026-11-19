"""量出每个字段的实测区间，用来写进文档（不猜）。

    PYTHONPATH=tools python notes/scratch/measure_ranges.py path/to.csz
"""
import sys, math
import numpy as np
from demo_extract import codec, recipes

path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/tv6b_full.csz"
d = codec.load(path)
rnds = d["rounds"]
print(f"# {path}  rounds={len(rnds)}  samples={sum(len(r['ticks']) for r in rnds)}")

def stats(name, vals, per_sample_note=""):
    a = np.asarray(vals, dtype=np.float64)
    fin = a[np.isfinite(a)]
    nan = int((~np.isfinite(a)).sum())
    if not len(fin):
        print(f"{name:14s} 全 NaN ({nan})"); return
    q = np.percentile(fin, [0, 1, 50, 99, 100])
    kind = "int" if np.allclose(fin, np.round(fin)) else "float"
    print(f"{name:14s} {kind:5s} min {q[0]:>12.4g}  p1 {q[1]:>12.4g}  "
          f"p50 {q[2]:>10.4g}  p99 {q[3]:>12.4g}  max {q[4]:>12.4g}  nan {nan}"
          + (f"  {per_sample_note}" if per_sample_note else ""))

print("\n== 玩家列（所有回合所有玩家所有采样点合并）==")
cols = ["x","y","z","yaw","pitch","v","vx","vy","vz","hp","armor",
        "flash","flash_alpha","weapon","place","shots","footsteps",
        "on_ground","jump","land"]
for c in cols:
    vals = [v for r in rnds for p in r["players"] for v in p[c]]
    stats(c, vals)
for c in ("helmet","defuser","alive"):
    vals = [v for r in rnds for p in r["players"] for v in p[c]]
    print(f"{c:14s} bool  True {sum(bool(x) for x in vals)}/{len(vals)}")

print("\n== 列表列 ==")
inv = [len(v) for r in rnds for p in r["players"] for v in p["inventory"]]
sp  = [len(v) for r in rnds for p in r["players"] for v in p["spotted"]]
stats("inventory_len", inv); stats("spotted_len", sp)
print("= 每个采样窗口的计数 =")
for c in ("shots","footsteps","jump","land"):
    vals = [v for r in rnds for p in r["players"] for v in p[c]]
    u, cts = np.unique(np.asarray(vals), return_counts=True)
    mx = [(int(a), int(b)) for a, b in zip(u, cts)][-4:]
    print(f"{c:14s} max {max(vals):>3}  >0 的占比 {np.mean(np.asarray(vals)>0):.4%}  分布尾部 {mx}")

print("\n== 回合级 ==")
stats("round_seconds", [s for r in rnds for s in r["round_seconds"]])
stats("ticks", [t for r in rnds for t in r["ticks"]])
stats("samples/round", [len(r["ticks"]) for r in rnds])
stats("missing_samples", [len(r["missing_samples"]) for r in rnds])
stats("bomb_planted_time", [r["bomb_planted_time"] for r in rnds if r["bomb_planted_time"] is not None])
bp = [c for r in rnds for c in r["bomb_position"]]
stats("bomb_pos.x", [p[0] for p in bp]); stats("bomb_pos.z", [p[2] for p in bp])

print("\n== 事件 ==")
k = [e for r in rnds for e in r["events"]["kills"]]
dm = [e for r in rnds for e in r["events"]["damage"]]
gr = [e for r in rnds for e in r["events"]["grenades"]]
sn = [e for r in rnds for e in r["events"]["sound"]]
print(f"kills/round        {len(k)/len(rnds):.2f}   dmg min/med/max "
      f"{min(e['dmg'] for e in k)}/{int(np.median([e['dmg'] for e in k]))}/{max(e['dmg'] for e in k)}")
stats("damage hp", [e["hp"] for e in dm])
stats("grenade x", [e["x"] for e in gr])
stats("grenade z", [e["z"] for e in gr])
stats("grenade v(z)", [e["vz"] for e in gr] if gr and "vz" in gr[0] else [])
from collections import Counter
print("sound ty          ", dict(Counter(e["ty"] for e in sn)))
print("sound per round    ", {t: round(sum(1 for e in sn if e["ty"]==t)/len(rnds), 2)
                               for t in {e["ty"] for e in sn}})

print("\n== 火焰 / 烟雾 ==")
stats("fire cells/snap", [c["count"] for r in rnds for f in r["fire"] for c in f["changes"]])
stats("fire snaps/inf", [len(f["changes"]) for r in rnds for f in r["fire"]])
stats("fire life(s)", [(f["last_tick"]-f["first_tick"])/64 for r in rnds for f in r["fire"]])
fp = np.concatenate([np.asarray(c["pos"], float).reshape(-1,3)
                     for r in rnds for f in r["fire"] for c in f["changes"]]) if any(r["fire"] for r in rnds) else np.zeros((0,3))
if len(fp):
    print(f"fire point xyz    min {fp.min(axis=0)}  max {fp.max(axis=0)}")
    off = np.concatenate([np.linalg.norm(np.asarray(c['pos'],float).reshape(-1,3)
                          - np.asarray(f['origin'],float), axis=1)
                          for r in rnds for f in r["fire"] if f.get("origin") for c in f["changes"]])
    stats("fire->origin", off)
    nrm = np.asarray([np.linalg.norm(c["normal"][3*i:3*i+3])
                      for r in rnds for f in r["fire"] for c in f["changes"]
                      for i in range(c["count"])])
    stats("normal |n|", nrm)
stats("smoke recs/smoke", [len(s["changes"]) for r in rnds for s in r["smoke"]])
stats("smoke life(s)", [(s["last_tick"]-s["first_tick"])/64 for r in rnds for s in r["smoke"]])
stats("smoke journal B", [c["size"] for r in rnds for s in r["smoke"] for c in s["changes"]])

print("\n== 体素覆盖 / 网格 ==")
tot = []
for r in rnds:
    for s in r["smoke"]:
        for t in (s["changes"][len(s["changes"])//2]["tick"], s["changes"][-1]["tick"]):
            g = recipes.smoke_grid_at(s, t)
            if g is not None:
                tot.append(int(g.sum()))
stats("voxels/smoke", tot)
gz = [int(np.argwhere(recipes.smoke_grid_at(s, s["changes"][-1]["tick"])).max(axis=0)[2])
      for r in rnds for s in r["smoke"] if recipes.smoke_grid_at(s, s["changes"][-1]["tick"]) is not None]
stats("smoke grid max z", gz)
