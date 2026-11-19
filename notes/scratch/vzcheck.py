import numpy as np
from demo_extract import extract, codec
data = extract("/Users/wanjungu/Downloads/test.dem", interval=0.5, verbose=False)
r = data["rounds"][0]
V = np.array([p["vz"] for p in r["players"]], dtype=np.float64)
print("IN-MEMORY round0 vz min/med/max", np.nanmin(V), np.nanmedian(V), np.nanmax(V))
print("IN-MEMORY abs>600 count", int((np.abs(V)>600).sum()), "of", V.size)
blob = codec.dumps(data)
back = codec.loads(blob)
print("codec.equal:", codec.equal(data, back))
B = np.array([p["vz"] for p in back["rounds"][0]["players"]], dtype=np.float64)
print("ROUNDTRIP vz min/med/max", np.nanmin(B), np.nanmedian(B), np.nanmax(B))
print("identical:", np.array_equal(V, B, equal_nan=True))
codec.save(data, "/tmp/vzcheck.csz")
L = codec.load("/tmp/vzcheck.csz")
Lv = np.array([p["vz"] for p in L["rounds"][0]["players"]], dtype=np.float64)
print("FILE roundtrip identical:", np.array_equal(V, Lv, equal_nan=True), "max diff", np.nanmax(np.abs(V-Lv)))
