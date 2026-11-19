"""Is `fov 90` in CS2/Source horizontal-at-4:3 (widened on widescreen) or
horizontal-at-current-aspect?  Try to find an authoritative statement."""

import json
import os
import urllib.parse
import urllib.request

PROXY = "http://127.0.0.1:7890"
os.environ.setdefault("HTTPS_PROXY", PROXY)
os.environ.setdefault("HTTP_PROXY", PROXY)
opener = urllib.request.build_opener(
    urllib.request.ProxyHandler({"http": PROXY, "https": PROXY}))
opener.addheaders = [("User-Agent", "Mozilla/5.0 (research)")]

QUERIES = [
    "Source engine fov command horizontal 4:3 aspect ratio widescreen Hor+",
    "CS2 m_iFOV CameraServices default 90 horizontal field of view",
    "Source engine default FOV 90 4:3 16:9 106 horizontal",
]

for q in QUERIES:
    print("=" * 70)
    print("Q:", q)
    url = "https://html.duckduckgo.com/html/?q=" + urllib.parse.quote(q)
    try:
        html = opener.open(url, timeout=25).read().decode("utf-8", "replace")
    except Exception as e:
        print("  fetch failed:", type(e).__name__, e)
        continue
    # pull out result links + snippets crudely
    import re
    links = re.findall(r'result__a[^>]*href="([^"]+)"[^>]*>(.*?)</a>', html, re.S)
    snips = re.findall(r'result__snippet[^>]*>(.*?)</a>', html, re.S)
    clean = lambda s: re.sub(r"<[^>]+>", "", s).replace("&amp;", "&").strip()
    if not links:
        print("  (no results parsed; len=%d)" % len(html))
    for i, (href, title) in enumerate(links[:5]):
        print(f"  - {clean(title)[:110]}")
        print(f"    {href[:130]}")
        if i < len(snips):
            print(f"    {clean(snips[i])[:300]}")

# Also try the Valve developer wiki directly -- it is the canonical source for
# the `fov` console variable semantics.
print("=" * 70)
for url in ("https://developer.valvesoftware.com/wiki/Field_of_view",
            "https://developer.valvesoftware.com/wiki/FOV"):
    print("FETCH", url)
    try:
        html = opener.open(url, timeout=25).read().decode("utf-8", "replace")
        import re
        text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.S | re.I)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"&nbsp;?", " ", text)
        text = re.sub(r"\s+", " ", text)
        print("  len:", len(text))
        for kw in ("4:3", "16:9", "horizontal", "aspect", "Hor+", "90"):
            for m in re.finditer(re.escape(kw), text, re.I):
                a, b = max(0, m.start() - 220), m.start() + 260
                print(f"  [{kw}] …{text[a:b]}…")
                break
    except Exception as e:
        print("  fetch failed:", type(e).__name__, e)
