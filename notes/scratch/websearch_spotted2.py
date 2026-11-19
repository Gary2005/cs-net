import json
import os
import urllib.parse
import urllib.request

os.environ.setdefault("HTTPS_PROXY", "http://127.0.0.1:7890")
os.environ.setdefault("HTTP_PROXY", "http://127.0.0.1:7890")


def get(u, raw=False):
    req = urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"})
    d = urllib.request.urlopen(req, timeout=40).read()
    return d if raw else d.decode("utf-8", "replace")


print("### grep.app: CCSEntitySpotting")
try:
    d = json.loads(get("https://grep.app/api/search?q=CCSEntitySpotting"))
    hits = d.get("hits", {}).get("hits", [])
    print("   hits", len(hits))
    for h in hits[:10]:
        print("   ", h.get("repo", {}).get("raw"), h.get("path", {}).get("raw"))
except Exception as e:  # noqa: BLE001
    print("   ERR", type(e).__name__, e)

print("### grep.app: m_bSpottedByMask")
try:
    d = json.loads(get("https://grep.app/api/search?q=m_bSpottedByMask"))
    hits = d.get("hits", {}).get("hits", [])
    print("   hits", len(hits))
    for h in hits[:10]:
        print("   ", h.get("repo", {}).get("raw"), h.get("path", {}).get("raw"))
except Exception as e:  # noqa: BLE001
    print("   ERR", type(e).__name__, e)

print("### SteamDatabase GameTracking-CS2: 找 Spotting 相关文件")
for path in [
    "DumpSource2/schemas/server/CCSEntitySpotting.hpp",
    "DumpSource2/schemas/server/EntitySpottedState_t.hpp",
    "DumpSource2/schemas/client/C_CSPlayerPawn.hpp",
]:
    u = "https://raw.githubusercontent.com/SteamDatabase/GameTracking-CS2/master/" + path
    try:
        d = get(u)
        print(f"   OK {path} ({len(d)} B)")
        print("   " + d[:600].replace("\n", "\n   "))
    except Exception as e:  # noqa: BLE001
        print(f"   -- {path}: {type(e).__name__} {e}")
