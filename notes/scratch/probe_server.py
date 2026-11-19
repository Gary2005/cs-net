"""Probe the local visualizer server for the newly staged assets.

Checks three things that were all real failure modes during this work:
  1. the manifest the browser will get actually lists the new weapons
  2. the new GLBs are reachable and served as model/gltf-binary
  3. textures inside a weapon dir are served as image/webp, not octet-stream

Usage: python notes/scratch/probe_server.py [base_url]
"""

import json
import os
import sys
import urllib.request
from pathlib import Path

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:5099"
ASSETS = Path("visualizer/static/assets")


def get(path):
    try:
        with urllib.request.urlopen(BASE + path, timeout=15) as r:
            return r.status, r.headers.get("Content-Type"), r.read()
    except Exception as e:  # noqa: BLE001
        return "ERR", str(e), b""


fails = []


def check(ok, label):
    print(f"  {'PASS' if ok else 'FAIL'}  {label}")
    if not ok:
        fails.append(label)


print(f"probing {BASE}\n")

status, ctype, body = get("/static/assets/manifest.json")
manifest = json.loads(body) if status == 200 else {}
check(status == 200, f"manifest.json 可访问 ({status} {ctype})")
weapons = manifest.get("weapons", [])
check(len(weapons) >= 60, f"manifest 列出 {len(weapons)} 把武器（之前只有 6）")
check(len(manifest.get("sprites", {})) == 5,
      f"manifest 有 {len(manifest.get('sprites', {}))} 组 sprite 序列")

# every manifest entry must actually be fetchable
missing = []
for w in weapons:
    d = w["dir"].replace("\\", "/")
    s, _, _ = get(f"/static/assets/{d}/{w['glb']}")
    if s != 200:
        missing.append(f"{w['key']}({s})")
check(not missing, f"{len(weapons)} 个 GLB 全部可取" +
      (f" —— 失败: {', '.join(missing[:5])}" if missing else ""))

# spot-check the two weapons the user reported as missing
for key, glb in (("Five-SeveN", "fiveseven/weapon_pist_fiveseven.glb"),
                 ("Knife(默认刀)", "knife_default_ct/weapon_knife_default_ct.glb"),
                 ("Karambit", "knife_karambit/weapon_knife_karambit.glb")):
    s, c, b = get(f"/static/assets/weapons/{glb}")
    check(s == 200 and c == "model/gltf-binary",
          f"{key:<16} {s} {c} {len(b)/1e6:.1f}MB")

# textures inside a weapon dir: Windows' registry mimetypes has no .webp, so this
# used to come back as application/octet-stream and the browser refused to decode
d = ASSETS / "weapons" / "fiveseven"
tex = sorted(p.name for p in d.iterdir() if p.suffix == ".webp")
if tex:
    s, c, b = get(f"/static/assets/weapons/fiveseven/{tex[0]}")
    check(s == 200 and c == "image/webp", f"武器贴图 {s} {c} {len(b)/1024:.0f}KB")

print()
print("全部通过" if not fails else f"{len(fails)} 项失败")
sys.exit(0 if not fails else 1)
