"""Final gaps: inferno duration by team, molotov->inferno handoff, wear decoding,
skin name<->paint_index cross-check, round structure, roles."""
import json, struct, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
from demoparser2 import DemoParser

DEMO = "/Users/wanjungu/Downloads/0823_inferno.dem"
p = DemoParser(DEMO)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 60)
out = {}

print("=" * 78)
print("1) INFERNO duration vs thrower TEAM  (2=TERRORIST/incendiary, 3=CT/molotov)")
sb = p.parse_event("inferno_startburn"); ex = p.parse_event("inferno_expire")
mi = sb.merge(ex, on=["entityid", "x", "y", "z"], suffixes=("_s", "_e"))
mi = mi[mi.tick_s < mi.tick_e].copy()
mi["dticks"] = mi.tick_e - mi.tick_s
teams = []
for _, r in mi.iterrows():
    pl = p.parse_ticks(["team_num"], ticks=[int(r["tick_s"])], players=[int(r["user_steamid_s"])])
    teams.append(int(pl["team_num"].iloc[0]) if len(pl) else -1)
mi["team"] = teams
print(mi.groupby(["team", "dticks"]).size().to_string())
print("\n   seconds by team:")
print(mi.assign(sec=(mi.dticks / 64).round(3)).groupby("team").sec.describe().to_string())
out["inferno_by_team"] = mi.groupby(["team", "dticks"]).size().to_dict()

print("\n" + "=" * 78)
print("2) MOLOTOV projectile flight run -> inferno_startburn handoff")
g = p.parse_grenades()
mol = g[g.grenade_type == "CMolotovProjectile"].copy()
mol = mol.sort_values(["grenade_entity_id", "tick"]).reset_index(drop=True)
mol["new_run"] = (mol.grenade_entity_id.ne(mol.grenade_entity_id.shift()) |
                  mol.tick.diff().ne(1) | mol.steamid.ne(mol.steamid.shift()))
mol["run_id"] = mol.new_run.cumsum()
runs = mol.groupby("run_id").agg(eid=("grenade_entity_id", "first"), t0=("tick", "min"),
                                 t1=("tick", "max"), n=("tick", "size"), name=("name", "first")).reset_index()
mm = runs.merge(sb, left_on="eid", right_on="entityid", how="inner")
mm["delta"] = mm["tick"] - mm["t1"]
print("   molotov projectile runs:", len(runs), " inferno_startburn events:", len(sb))
print("   (startburn.tick - projectile.last_tick) value_counts:", mm.delta.value_counts().head(8).to_dict())
print("   => the projectile's LAST tracked tick is essentially the detonation; the inferno centre")
print("      equals the detonation point. Sample:")
print(mm[["eid", "t1", "tick", "delta", "name", "x", "y", "z"]].head(6).to_string(index=False))
out["molotov_handoff_delta"] = {int(k): int(v) for k, v in mm.delta.value_counts().head(8).items()}
out["n_molotov_runs"] = int(len(runs)); out["n_inferno"] = int(len(sb))

print("\n" + "=" * 78)
print("3) parse_skins paint_wear is a RAW FLOAT BIT PATTERN?")
sk = p.parse_skins()
def bits_to_float(u):
    return struct.unpack("<f", struct.pack("<I", int(u) & 0xFFFFFFFF))[0]
sk["wear_decoded"] = sk.paint_wear.map(bits_to_float)
print(sk[["def_index", "paint_index", "paint_seed", "paint_wear", "wear_decoded", "steamid"]].head(8).to_string(index=False))
print("   decoded wear range:", round(sk.wear_decoded.min(), 4), "..", round(sk.wear_decoded.max(), 4))
print("   plausible float wear (0..1)?", bool(((sk.wear_decoded >= 0) & (sk.wear_decoded <= 1)).all()))
out["wear_decoded"] = {"min": float(sk.wear_decoded.min()), "max": float(sk.wear_decoded.max()),
                       "all_in_0_1": bool(((sk.wear_decoded >= 0) & (sk.wear_decoded <= 1)).all())}

print("\n4) cross-check paint_index route vs name route (active_weapon_skin)")
for sid in sk.steamid.unique()[:4]:
    rows = sk[sk.steamid == sid]
    t = p.parse_ticks(["active_weapon_name", "active_weapon_skin", "item_def_idx", "weapon_paint_seed",
                       "weapon_float", "glove_paint_id"], players=[int(sid)], ticks=list(range(30000, 40000, 500)))
    names = t[["active_weapon_name", "active_weapon_skin", "item_def_idx", "weapon_float"]].dropna().drop_duplicates()
    print(f"   steamid {sid}: parse_skins def_index/paint {rows[['def_index','paint_index']].values.tolist()[:5]}")
    print(f"      parse_ticks skinned weapons: {names.values.tolist()[:5]}")

print("\n" + "=" * 78)
print("5) ROUND STRUCTURE")
pre = p.parse_event("round_prestart"); off = p.parse_event("round_officially_ended")
fz = p.parse_event("round_freeze_end"); buyt = p.parse_event("buytime_ended")
bnm = p.parse_event("begin_new_match")
print("   begin_new_match tick:", bnm["tick"].tolist())
print("   n round_prestart:", len(pre), " n round_freeze_end:", len(fz), " n buytime_ended:", len(buyt))
print("   round_officially_ended ticks (dedup):", sorted(set(off["tick"].tolist())))
print("   NOTE: no 'round_start'/'round_end' events exist. Use round_prestart + round_freeze_end +")
print("         round_officially_ended, plus game-rule props (round_win_reason/round_win_status).")
print("   item_purchase event present?", "item_purchase" in p.list_game_events(),
      " (use item_pickup/item_equip instead)")
print("   warmup: is_warmup_period True for ticks <=", )
wp = p.parse_ticks(["is_warmup_period", "total_rounds_played"], ticks=list(range(0, 4000, 100)))
print(wp[wp.is_warmup_period].tick.max() if wp.is_warmup_period.any() else None,
      "-> begin_new_match at", bnm["tick"].tolist())
out["rounds"] = {"n_prestart": int(len(pre)), "n_freeze_end": int(len(fz)),
                 "n_officially_ended": int(len(set(off["tick"].tolist()))),
                 "begin_new_match": bnm["tick"].tolist()}

print("\n" + "=" * 78)
print("6) fire_bullets (tracer rendering) sample")
fb = p.parse_event("fire_bullets")
print("   n:", len(fb), "cols:", list(fb.columns))
print(fb[["tick", "user_name", "weapon_id", "item_def_index", "origin_x", "origin_y", "origin_z",
          "angles_x", "angles_y", "seed", "attack_type", "num_bullets_remaining"]].head(4).to_string(index=False))

print("\n7) is there any role/team-role prop?")
cands = ["team_num", "team_name", "is_coach_team", "player_color", "rank", "entity_id", "user_id"]
d = p.parse_ticks(cands, ticks=[30000])
print("   available:", [c for c in d.columns if c not in ("tick", "steamid", "name")])
print("   (no 'role' (IGL/entry/awper) prop exists in the demo — not recorded by CS2)")

json.dump(out, open("final_gaps.json", "w"), indent=1, default=str)
print("\n", json.dumps(out, indent=1, default=str)[:1200])
