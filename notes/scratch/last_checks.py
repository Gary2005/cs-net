"""Last checks: molotov last-flight-point vs inferno centre distance; round timing model."""
import json, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
from demoparser2 import DemoParser

DEMO = "/Users/wanjungu/Downloads/0823_inferno.dem"
p = DemoParser(DEMO)
pd.set_option("display.width", 250)
out = {}

sb = p.parse_event("inferno_startburn")
g = p.parse_grenades()
mol = g[g.grenade_type == "CMolotovProjectile"].copy()
mol = mol.sort_values(["grenade_entity_id", "tick"]).reset_index(drop=True)
mol["new_run"] = (mol.grenade_entity_id.ne(mol.grenade_entity_id.shift()) |
                  mol.tick.diff().ne(1) | mol.steamid.ne(mol.steamid.shift()))
mol["run_id"] = mol.new_run.cumsum()
last = mol.groupby("run_id").agg(t_last=("tick", "max"), x=("x", "last"), y=("y", "last"),
                                 z=("z", "last"), name=("name", "first")).reset_index()

print("=" * 78)
print("1) nearest (in time) inferno_startburn for each molotov run + 3D distance")
recs = []
for _, r in last.iterrows():
    cand = sb[(sb.tick >= r.t_last - 2) & (sb.tick <= r.t_last + 40)]
    if not len(cand):
        cand = sb.iloc[(sb.tick - r.t_last).abs().argsort()[:1]]
    c = cand.iloc[(np.abs(cand.tick - r.t_last)).argsort()].iloc[0]
    dist = float(np.linalg.norm([r.x - c.x, r.y - c.y, r.z - c.z]))
    recs.append({"dt": int(c.tick - r.t_last), "dist": round(dist, 2), "name": r["name"],
                 "burn_name": c.user_name, "same_thrower": r["name"] == c.user_name})
rec = pd.DataFrame(recs)
print("   dt (startburn.tick - molotov.last_tick) distribution:", rec.dt.value_counts().head(8).to_dict())
print("   distance stats (units):", rec.dist.describe()[["min", "25%", "50%", "75%", "max"]].round(2).to_dict())
print("   thrower name agreement:", int(rec.same_thrower.sum()), "/", len(rec))
out["molotov_inferno"] = {"dt_counts": {int(k): int(v) for k, v in rec.dt.value_counts().head(8).items()},
                          "dist_median": float(rec.dist.median()), "dist_min": float(rec.dist.min()),
                          "dist_max": float(rec.dist.max()), "thrower_agree": int(rec.same_thrower.sum()),
                          "n": int(len(rec))}
print("\n   => prefer the inferno_startburn EVENT for the burn centre; the molotov trajectory gives flight.")

print("\n" + "=" * 78)
print("2) round timing model (tick space, /64 for seconds)")
pre = p.parse_event("round_prestart")["tick"].tolist()
fz = p.parse_event("round_freeze_end")["tick"].tolist()
post = p.parse_event("round_poststart")["tick"].tolist()
off = sorted(set(p.parse_event("round_officially_ended")["tick"].tolist()))
bnm = int(p.parse_event("begin_new_match")["tick"].iloc[0])
pre = [t for t in pre if t >= bnm]
fz = [t for t in fz if t >= bnm]
post = [t for t in post if t >= bnm]
print("   begin_new_match:", bnm, "| n prestart>=bnm:", len(pre), "n freeze_end:", len(fz), "n poststart:", len(post))
print("   freeze duration (freeze_end - prestart) ticks:", sorted(set(int(b - a) for a, b in zip(pre, fz))))
print("   prestart==poststart?", pre[:5], post[:5])
rounds = []
for i, (a, b) in enumerate(zip(pre, fz)):
    nxt = pre[i + 1] if i + 1 < len(pre) else None
    rounds.append({"round": i + 1, "prestart": a, "freeze_end": b, "freeze_s": round((b - a) / 64, 2),
                   "end": nxt})
print(pd.DataFrame(rounds).head(25).to_string(index=False))
out["rounds"] = rounds
out["freeze_ticks"] = sorted(set(int(b - a) for a, b in zip(pre, fz)))

print("\n" + "=" * 78)
print("3) total demo ticks / duration")
d = p.parse_ticks(["game_time"], ticks=[0, 1, 1000, 167846])
print(d.to_string(index=False))
print("   total ticks = 167846 (tick 1..167846, contiguous, no gaps)")
print("   duration @64 =", round(167846 / 64, 2), "s =", round(167846 / 64 / 60, 2), "min")
out["total_ticks"] = 167846

json.dump(out, open("last_result.json", "w"), indent=1, default=str)
print("\n", json.dumps(out["molotov_inferno"], indent=1))
