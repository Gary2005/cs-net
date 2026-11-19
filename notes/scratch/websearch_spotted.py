import html
import os
import re
import urllib.parse
import urllib.request

os.environ.setdefault("HTTPS_PROXY", "http://127.0.0.1:7890")
os.environ.setdefault("HTTP_PROXY", "http://127.0.0.1:7890")


def get(u):
    req = urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"})
    return urllib.request.urlopen(req, timeout=30).read().decode("utf-8", "replace")


QUERIES = [
    "CS2 m_bSpottedByMask update interval spotted delay",
    "CCSEntitySpotting CS2 spotted update rate",
    "cs2 spotted by mask updated every 0.5 seconds radar",
    "m_bSpotted CSGO update rate server",
]

for q in QUERIES:
    print("=" * 70)
    print("Q:", q)
    for engine in ("https://html.duckduckgo.com/html/?q=",
                   "https://lite.duckduckgo.com/lite/?q="):
        try:
            d = get(engine + urllib.parse.quote(q))
        except Exception as e:  # noqa: BLE001
            print("   ERR", engine, type(e).__name__, e)
            continue
        links = re.findall(r'href="(https?://[^"]+)"[^>]*>(.*?)</a>', d, re.S)
        seen = set()
        n = 0
        for u, t in links:
            if "duckduckgo" in u:
                continue
            txt = html.unescape(re.sub("<[^>]+>", "", t)).strip()
            if not txt or u in seen:
                continue
            seen.add(u)
            print("   -", txt[:110])
            print("     ", u[:150])
            n += 1
            if n >= 6:
                break
        if n:
            break
