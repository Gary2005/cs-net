import os
import re
import urllib.request

os.environ.setdefault("HTTPS_PROXY", "http://127.0.0.1:7890")
os.environ.setdefault("HTTP_PROXY", "http://127.0.0.1:7890")
BASE = "https://raw.githubusercontent.com/SteamDatabase/GameTracking-CS2/master/"
OUT = r"D:\project\cs-net-v5\notes\scratch\gtt"

for name in ["server.kv3", "client.kv3"]:
    req = urllib.request.Request(BASE + "DumpSource2/module_metadata/" + name,
                                 headers={"User-Agent": "Mozilla/5.0"})
    d = urllib.request.urlopen(req, timeout=90).read().decode("utf-8", "replace")
    p = os.path.join(OUT, name)
    open(p, "w", encoding="utf-8").write(d)
    print(f"=== {name} {len(d)} B -> {p}")
    for m in re.finditer(r"[^\n]*[Ss]potting[^\n]*", d):
        print("   ", m.group(0).strip()[:200])
    for m in re.finditer(r"[^\n]*Spotted[^\n]*", d):
        print("   S:", m.group(0).strip()[:200])
