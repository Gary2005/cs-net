"""调研 6：在第二个 demo（不同地图 / 不同服务器）上复核 33-tick 网格。

跑法: python notes/scratch/spotted_grid_crosscheck.py <dem>
"""
from __future__ import annotations

import sys
from collections import Counter

from demoparser2 import DemoParser

P = 33
PROPS = ["approximate_spotted_by", "spotted", "is_walking", "shots_fired", "yaw", "X"]


def key(v):
    if isinstance(v, list):
        return tuple(sorted(str(i) for i in (v or [])))
    return v


def main(demo):
    p = DemoParser(demo)
    hdr = p.parse_header()
    print(f"demo {demo}")
    print(f"  map={hdr.get('map_name')} patch={hdr.get('patch_version')}")
    df = p.parse_ticks(PROPS)
    print(f"  rows={len(df)}")

    for col in PROPS:
        if col not in df.columns:
            continue
        changes = []
        for sid, g in df.groupby("steamid"):
            g = g.sort_values("tick")
            v = [key(x) for x in g[col].tolist()]
            tk = g["tick"].tolist()
            for i in range(1, len(v)):
                if v[i] != v[i - 1]:
                    changes.append(tk[i])
        if not changes:
            print(f"  {col:26s} (no change)")
            continue
        c = Counter(t % P for t in changes)
        top, cnt = c.most_common(1)[0]
        exp = len(changes) / P
        print(f"  {col:26s} n={len(changes):6d}  tick%33 max bucket {cnt:5d} "
              f"({cnt/len(changes)*100:5.1f}%, uniform={exp:.1f}, x{cnt/exp:6.2f})  "
              f"residues={len(c)}/33")


if __name__ == "__main__":
    main(sys.argv[1])
