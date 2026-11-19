"""Verify: which grenade item produces which burn duration, and molotov->inferno by POSITION."""
import json, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
from demoparser2 import DemoParser

DEMO = "/Users/wanjungu/Downloads/0823_inferno.dem"
p = DemoParser(DEMO)
pd.set_option("display.width", 250)
out = {}

print("=" * 78)
print("1) inferno duration vs thrower team + the actual thrown ITEM (from item_pickup/equip)")
sb = p.parse_event("inferno_startburn"); ex = p.parse_event("inferno_expire")
mi = sb.merge(ex, on=["entityid", "x", "y", "z"], suffixes=("_s", "_e"))
mi = mi[mi.tick_s < mi.tick_e].copy()
mi["dticks"] = mi.tick_e - mi.tick_s
mi["sec"] = (mi.dticks / 64).round(3)

pk = pd.concat([p.parse_event("item_pickup")[["tick", "user_steamid", "item"]],
                p.parse_event("item_equip")[["tick", "user_steamid", "item"]]], ignore_index=True)
pk = pk[pk.item.isin(["molotov", "incgrenade", "Molotov", "Incendiary Grenade"])].sort_values("tick")

rows = []
for _, r in mi.iterrows():
    sid = int(r["user_steamid_s"]); t = int(r["tick_s"])
    pl = p.parse_ticks(["team_num"], ticks=[t], players=[sid])
    team = int(pl["team_num"].iloc[0]) if len(pl) else -1
    prev = pk[(pk.user_steamid.astype("int64") == sid) & (pk.tick < t)]
    item = prev.item.iloc[-1] if len(prev) else None
    rows.append({"team": team, "item": item, "dticks": int(r.dticks), "sec": r.sec})
df = pd.DataFrame(rows)
print(df.groupby(["team", "item"]).agg(n=("sec", "size"), sec_median=("sec", "median"),
                                       secs=("sec", lambda s: sorted(set(s.round(2))))).to_string())
print("\n   duration histogram by item:")
print(pd.crosstab(df.item, df.sec).to_string())
out["inferno_item_duration"] = {f"{k[0]}|{k[1]}": int(v) for k, v in df.groupby(["item", "sec"]).size().items()}

print("\n" + "=" * 78)
print("2) MOLOTOV projectile -> inferno by POSITION (entity ids are unrelated)")
g = p.parse_grenades()
mol = g[g.grenade_type == "CMolotovProjectile"].copy()
mol = mol.sort_values(["grenade_entity_id", "tick"]).reset_index(drop=True)
mol["new_run"] = (mol.grenade_entity_id.ne(mol.grenade_entity_id.shift()) |
                  mol.tick.diff().ne(1) | mol.steamid.ne(mol.steamid.shift()))
mol["run_id"] = mol.new_run.cumsum()
last = mol.groupby("run_id").agg(t_last=("tick", "max"), x=("x", "last"), y=("y", "last"),
                                 z=("z", "last"), name=("name", "first")).reset_index()
m = last.merge(sb, left_on=["x", "y", "z"], right_on=["x", "y", "z"], how="inner")
m["delta"] = m["tick"] - m["t_last"]
print("   molotov runs:", len(last), " inferno_startburn:", len(sb), " position-matched:", len(m))
print("   (startburn.tick - molotov.last_tick) value_counts:", m.delta.value_counts().head(8).to_dict())
print("   => molotov projectile's last tick IS the detonation tick; inferno centre = that position")
print(m[["run_id", "t_last", "tick", "delta", "name", "x", "y", "z"]].head(6).to_string(index=False))
out["molotov_pos_match"] = {"n_runs": int(len(last)), "n_matched": int(len(m)),
                            "delta_counts": {int(k): int(v) for k, v in m.delta.value_counts().head(8).items()}}

print("\n3) inferno: does the burning footprint ever move / how long does a molotov's fire emit cells?")
print("   (cannot be answered: fire cell props are not exposed by demoparser2)")

print("\n4) round-count sanity: rounds with a plant, and freeze-time length")
fz = p.parse_event("round_freeze_end"); pre = p.parse_event("round_prestart")
d = (fz["tick"].iloc[:len(pre)].to_numpy() - pre["tick"].iloc[:len(pre)].to_numpy())
print("   prestart->freeze_end ticks:", sorted(set(d.tolist())), "=> seconds:", sorted(set((d / 64).round(2).tolist())))
out["freeze_ticks"] = sorted(set(int(x) for x in d.tolist()))

json.dump(out, open("verify_result.json", "w"), indent=1, default=str)
print("\n", json.dumps(out, indent=1, default=str)[:1000])
