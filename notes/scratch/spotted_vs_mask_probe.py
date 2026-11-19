"""调研 9：`spotted`(m_bSpotted, bool) 和 `approximate_spotted_by`(m_bSpottedByMask)
到底是不是同一个信息？差在哪？

跑法: python notes/scratch/spotted_vs_mask_probe.py <dem>
"""
from __future__ import annotations

import sys
from collections import Counter

import pandas as pd

from demoparser2 import DemoParser

PROPS = ["spotted", "approximate_spotted_by", "team_num", "is_alive", "name"]


def key(v):
    return tuple(sorted(int(i) for i in (v or []))) if isinstance(v, list) else v


def main(demo):
    p = DemoParser(demo)
    hdr = p.parse_header()
    print(f"demo {demo}  map={hdr.get('map_name')} patch={hdr.get('patch_version')}")
    df = p.parse_ticks(PROPS)
    print(f"rows={len(df)}")
    df = df.sort_values(["steamid", "tick"]).reset_index(drop=True)

    team_of = {}
    for sid, g in df.groupby("steamid"):
        team_of[sid] = g["team_num"].mode().iloc[0]
    sid_name = {}
    for sid, g in df.groupby("steamid"):
        sid_name[sid] = g["name"].mode().iloc[0]
    print("\n玩家 -> 队伍")
    for sid, tm in sorted(team_of.items(), key=lambda kv: kv[1]):
        print(f"   {sid_name[sid]:12s} sid={sid} team={tm}")

    n = len(df)
    both = only_bool = only_mask = neither = 0
    mask_lens = Counter()
    team_mix = Counter()          # 被看到的人 vs 看到的人 的队伍关系
    alive_mask = Counter()
    ex_only_bool = []
    ex_only_mask = []

    for row in df.itertuples(index=False):
        sids = set(row.approximate_spotted_by or [])
        b = bool(row.spotted)
        me = row.steamid
        my_team = team_of.get(me)
        for s in sids:
            if s not in team_of:
                team_mix["unknown_sid"] += 1
                continue
            team_mix["same_team" if team_of[s] == my_team else "enemy"] += 1
        if b and sids:
            both += 1
        elif b and not sids:
            only_bool += 1
            if len(ex_only_bool) < 5:
                ex_only_bool.append((row.tick, sid_name[me], my_team))
        elif (not b) and sids:
            only_mask += 1
            if len(ex_only_mask) < 5:
                ex_only_mask.append((row.tick, sid_name[me], sorted(sids),
                                     [sid_name.get(s, s) for s in sids]))
        else:
            neither += 1
        mask_lens[len(sids)] += 1
        alive_mask[(bool(row.is_alive), b, len(sids) > 0)] += 1

    print(f"\n=== m_bSpotted(bool)  vs  mask 是否非空   （n={n}）")
    print(f"   bool=1 且 mask 非空 : {both:9d} ({both/n*100:6.3f}%)")
    print(f"   bool=1 但 mask 空   : {only_bool:9d} ({only_bool/n*100:6.3f}%)")
    print(f"   bool=0 但 mask 非空 : {only_mask:9d} ({only_mask/n*100:6.3f}%)")
    print(f"   两个都是 0          : {neither:9d} ({neither/n*100:6.3f}%)")
    print(f"   -> 不一致合计 {(only_bool+only_mask)/n*100:.4f}%")

    print("\n=== mask 长度分布（被几个人看到）")
    for k in sorted(mask_lens):
        print(f"   len={k}: {mask_lens[k]:9d} ({mask_lens[k]/n*100:6.3f}%)")

    print("\n=== mask 里那些人的队伍关系（相对被看的 pawn）")
    tot = sum(team_mix.values())
    for k, v in team_mix.most_common():
        print(f"   {k:12s}: {v:9d} ({v/tot*100:6.3f}%)")

    print("\n=== (alive, m_bSpotted, mask非空) 组合")
    for k, v in sorted(alive_mask.items()):
        print(f"   {k}: {v}")

    print("\n=== bool=1 但 mask 空的例子")
    for e in ex_only_bool:
        print("   ", e)
    print("=== bool=0 但 mask 非空的例子")
    for e in ex_only_mask:
        print("   ", e)


if __name__ == "__main__":
    main(sys.argv[1])
