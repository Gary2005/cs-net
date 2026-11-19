import json
import os
import urllib.request

os.environ.setdefault("HTTPS_PROXY", "http://127.0.0.1:7890")
os.environ.setdefault("HTTP_PROXY", "http://127.0.0.1:7890")


def api(u):
    req = urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0",
                                             "Accept": "application/vnd.github+json"})
    return json.loads(urllib.request.urlopen(req, timeout=40).read().decode("utf-8", "replace"))


base = "https://api.github.com/repos/SteamDatabase/GameTracking-CS2/contents/DumpSource2/module_metadata"
try:
    d = api(base)
    for it in d:
        print(f"  {it['type']:4s} {it['name']} {it.get('size','')}")
except Exception as e:  # noqa: BLE001
    print("ERR", type(e).__name__, e)
