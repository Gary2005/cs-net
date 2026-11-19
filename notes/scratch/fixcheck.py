import numpy as np, time
from demo_extract import extract, recipes
for iv in (1.0, 0.25):
    t0=time.perf_counter()
    d = extract("/Users/wanjungu/Downloads/test.dem", interval=iv, verbose=False)
    worst_id = worst_vz = 0.0
    for r in d["rounds"]:
        v  = np.asarray(recipes.player_tensor(r,("v",))[:,:,0], float)
        vx = np.asarray(recipes.player_tensor(r,("vx",))[:,:,0], float)
        vy = np.asarray(recipes.player_tensor(r,("vy",))[:,:,0], float)
        vz = np.asarray(recipes.player_tensor(r,("vz",))[:,:,0], float)
        sp = np.hypot(vx,vy); ok = np.isfinite(v)&np.isfinite(sp)
        if ok.any(): worst_id = max(worst_id, float(np.abs(v-sp)[ok].max()))
        f = np.isfinite(vz)
        if f.any(): worst_vz = max(worst_vz, float(np.abs(vz[f]).max()))
    print(f"interval={iv}  {time.perf_counter()-t0:5.1f}s  samples={sum(len(r['ticks']) for r in d['rounds'])}"
          f"  worst |v-hypot(vx,vy)|={worst_id:.4f}  worst |vz|={worst_vz:.1f}  nan={recipes.nan_report(d)}")
    # airborne vs grounded vz sanity
    a=[];g=[]
    for r in d["rounds"]:
        og=recipes.on_ground_mask(r); al=recipes.alive_mask(r)
        vz=np.abs(np.asarray(recipes.player_tensor(r,("vz",))[:,:,0],float))
        a.append(vz[al&~og]); g.append(vz[al&og])
    A=np.concatenate(a); G=np.concatenate(g)
    print(f"   mean|vz| airborne {np.nanmean(A):8.2f}  grounded {np.nanmean(G):8.2f}  ratio {np.nanmean(A)/np.nanmean(G):.1f}x")
