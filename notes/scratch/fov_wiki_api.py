import json, os, re, urllib.parse, urllib.request
PROXY = "http://127.0.0.1:7890"
op = urllib.request.build_opener(urllib.request.ProxyHandler({"http": PROXY, "https": PROXY}))
op.addheaders = [("User-Agent", "Mozilla/5.0 (research)")]

def get(u):
    return op.open(u, timeout=30).read().decode("utf-8", "replace")

# MediaWiki API on the Valve developer wiki
api = ("https://developer.valvesoftware.com/w/api.php?action=query&list=search"
       "&srsearch=" + urllib.parse.quote("field of view fov aspect") + "&format=json&srlimit=10")
try:
    j = json.loads(get(api))
    for r in j.get("query", {}).get("search", []):
        print("HIT:", r["title"])
except Exception as e:
    print("search API failed:", type(e).__name__, e)

for title in ("Field of view", "FOV", "List of console commands", "Console commands"):
    u = ("https://developer.valvesoftware.com/w/api.php?action=parse&page="
         + urllib.parse.quote(title) + "&prop=wikitext&format=json&redirects=1")
    try:
        j = json.loads(get(u))
        wt = j.get("parse", {}).get("wikitext", {}).get("*", "")
        print(f"\n=== {title}  (wikitext {len(wt)} chars)")
        if not wt:
            print("   ", json.dumps(j)[:300])
            continue
        for kw in ("4:3", "16:9", "aspect", "horizontal", "Hor+"):
            hits = [m.start() for m in re.finditer(re.escape(kw), wt, re.I)]
            if hits:
                a = max(0, hits[0] - 300)
                print(f"  [{kw}] x{len(hits)}: …{wt[a:hits[0]+400]}…".replace("\n", " ")[:700])
    except Exception as e:
        print(f"\n=== {title}: failed {type(e).__name__} {e}")
