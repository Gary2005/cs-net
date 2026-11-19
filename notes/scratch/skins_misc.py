"""parse_skins() format, player+weapon -> skin mapping, item drops, voice, bomb sites."""
import json, warnings
import pandas as pd
warnings.filterwarnings("ignore")
from demoparser2 import DemoParser

DEMO = "/Users/wanjungu/Downloads/0823_inferno.dem"
p = DemoParser(DEMO)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 60)
out = {}

print("=" * 78)
print("1) parse_skins()")
sk = p.parse_skins()
print("   type:", type(sk).__name__, "shape:", sk.shape)
print("   columns:", list(sk.columns))
print("   dtypes:\n", sk.dtypes.to_string())
print(sk.head(12).to_string())
out["skins"] = {"cols": list(sk.columns), "rows": int(len(sk)),
                "dtypes": {c: str(t) for c, t in zip(sk.columns, sk.dtypes)}}
if "weapon" in sk.columns:
    print("\n   distinct weapons:", sk.weapon.nunique())
if "skin" in sk.columns:
    print("   distinct skins:", sk.skin.nunique(), "eg:", sk.skin.dropna().unique()[:8].tolist())
if "steamid" in sk.columns:
    print("   players:", sk.steamid.nunique())

print("\n   --- per-player sample (one player's rows) ---")
if "steamid" in sk.columns:
    one = sk[sk.steamid == sk.steamid.iloc[0]]
    print(one.head(20).to_string(index=False))

print("\n" + "=" * 78)
print("2) parse_item_drops()")
try:
    idd = p.parse_item_drops()
    print("   cols:", list(idd.columns), "rows:", len(idd))
    print(idd.head(8).to_string(index=False))
    out["item_drops"] = {"cols": list(idd.columns), "rows": int(len(idd))}
except BaseException as e:
    print("   ERR", type(e).__name__, str(e)[:150]); out["item_drops"] = {"ERR": str(e)[:150]}

print("\n" + "=" * 78)
print("3) parse_voice()")
try:
    v = p.parse_voice()
    print("   type:", type(v).__name__, "len:", len(v))
    if len(v):
        print("   first entries:", [{k: (f'<{len(x)} bytes>' if k == 'bytes' else x) for k, x in e.items()} for e in v[:3]])
        print("   n speakers:", len({e['steamid'] for e in v}))
    out["voice"] = {"n": len(v), "keys": list(v[0].keys()) if len(v) else []}
except BaseException as e:
    print("   ERR", type(e).__name__, str(e)[:150]); out["voice"] = {"ERR": str(e)[:150]}

print("\n" + "=" * 78)
print("4) bomb_planted 'site' -> A/B, and how to get the PLANTED BOMB POSITION")
bp = p.parse_event("bomb_planted")
print("   bomb_planted:\n", bp.to_string(index=False))
print("\n   NOTE: bomb_planted carries NO x/y/z. Deriving the bomb position from the planter's")
print("   position at the plant tick:")
for _, r in bp.iterrows():
    t = int(r["tick"])
    pl = p.parse_ticks(["X", "Y", "Z", "last_place_name", "which_bomb_zone", "is_alive"],
                       ticks=[t], players=[int(r["user_steamid"])])
    if len(pl):
        row = pl.iloc[0]
        print(f"   tick={t:7d} site={r['site']} planter={r['user_name']:18s} "
              f"pos=({row['X']:8.1f},{row['Y']:8.1f},{row['Z']:7.1f}) place={row['last_place_name']}")
print("\n   which_bomb_zone available?", "which_bomb_zone" in p.parse_ticks(["which_bomb_zone"], ticks=[41461]).columns)

print("\n" + "=" * 78)
print("5) inferno duration vs thrower TEAM (molotov CT=3 vs incendiary T=2)")
sb = p.parse_event("inferno_startburn"); ex = p.parse_event("inferno_expire")
mi = sb.merge(ex, on=["entityid", "x", "y", "z"], suffixes=("_s", "_e"))
mi = mi[mi.tick_s < mi.tick_e].copy()
mi["dticks"] = mi.tick_e - mi.tick_s
teams = []
for _, r in mi.iterrows():
    pl = p.parse_ticks(["team_num"], ticks=[int(r["tick_s"])], players=[int(r["user_steamid"])])
    teams.append(int(pl["team_num"].iloc[0]) if len(pl) else -1)
mi["team"] = teams
print(mi.groupby(["team", "dticks"]).size().to_string())
print("\n   team 2 = TERRORIST, 3 = CT")
print(mi.groupby("team").dticks.describe().to_string())

print("\n" + "=" * 78)
print("6) molotov projectile: does the run end at the inferno start?")
g = p.parse_grenades()
mol = g[g.grenade_type == "CMolotovProjectile"].copy()
# segment runs
mol = mol.sort_values(["grenade_entity_id", "tick"]).reset_index(drop=True)
mol["new_run"] = (mol.grenade_entity_id.ne(mol.grenade_entity_id.shift()) |
                  mol.tick.diff().ne(1) | mol.steamid.ne(mol.steamid.shift()))
mol["run_id"] = mol.new_run.cumsum()
runs = mol.groupby("run_id").agg(eid=("grenade_entity_id", "first"), t0=("tick", "min"),
                                 t1=("tick", "max"), n=("tick", "size"), name=("name", "first")).reset_index()
print("   molotov projectile runs:", len(runs), " vs inferno_startburn:", len(sb))
mm = runs.merge(sb, left_on="eid", right_on="entityid", how="inner")
mm["delta"] = mm.tick - mm.t1
print("   inferno_startburn.tick - molotov projectile last tick:")
print("   ", mm.delta.value_counts().head(8).to_dict())
print("   n matched:", len(mm))
out["molotov_runs"] = int(len(runs)); out["inferno_startburn"] = int(len(sb))
out["molotov_last_to_startburn_delta"] = {int(k): int(v) for k, v in mm.delta.value_counts().head(8).items()}

json.dump(out, open("skins_result.json", "w"), indent=1, default=str)
