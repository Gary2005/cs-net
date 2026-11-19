import os
import urllib.request

os.environ.setdefault("HTTPS_PROXY", "http://127.0.0.1:7890")
os.environ.setdefault("HTTP_PROXY", "http://127.0.0.1:7890")
BASE = "https://raw.githubusercontent.com/SteamDatabase/GameTracking-CS2/master/"
OUT = r"D:\project\cs-net-v5\notes\scratch\gtt"


def get(p):
    req = urllib.request.Request(BASE + p, headers={"User-Agent": "Mozilla/5.0"})
    return urllib.request.urlopen(req, timeout=60).read().decode("utf-8", "replace")


os.makedirs(OUT, exist_ok=True)
for p in [
    "DumpSource2/schemas/server/EntitySpottedState_t.h",
    "DumpSource2/schemas/server/CCSEntitySpotting.h",
    "DumpSource2/schemas/client/EntitySpottedState_t.h",
    "DumpSource2/convars.txt",
    "DumpSource2/commands.txt",
]:
    try:
        d = get(p)
        name = p.rsplit("/", 1)[-1]
        open(os.path.join(OUT, name), "w", encoding="utf-8").write(d)
        print(f"OK {p} ({len(d)} B)")
        if name.endswith(".h"):
            print(d)
    except Exception as e:  # noqa: BLE001
        print(f"-- {p}: {type(e).__name__} {e}")
