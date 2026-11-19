import json
import os
import urllib.request

os.environ.setdefault("HTTPS_PROXY", "http://127.0.0.1:7890")
os.environ.setdefault("HTTP_PROXY", "http://127.0.0.1:7890")


def api(u):
    req = urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0",
                                             "Accept": "application/vnd.github+json"})
    return json.loads(urllib.request.urlopen(req, timeout=40).read().decode("utf-8", "replace"))


for path in ["", "DumpSource2", "DumpSource2/schemas", "DumpSource2/schemas/server"]:
    u = "https://api.github.com/repos/SteamDatabase/GameTracking-CS2/contents/" + path
    try:
        d = api(u)
        print(f"--- {path or '/'} : {len(d)} entries")
        for it in d[:40]:
            print(f"    {it['type']:4s} {it['name']}")
    except Exception as e:  # noqa: BLE001
        print(f"--- {path}: ERR {type(e).__name__} {e}")
