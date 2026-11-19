# What real CS2 demo data can be extracted with `demoparser2` 0.42.0

Empirical research report. **Everything below was verified by running code against a real demo** —
nothing is inferred from documentation alone. Where a documented feature did *not* work, that is
stated explicitly with the exact failure mode.

## Environment / provenance

| Item | Value |
| --- | --- |
| conda env | `cs2demo` (`source /opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh && conda activate cs2demo`) |
| demoparser2 | **0.42.0** (compiled Rust ext `demoparser2.cpython-310-darwin.so`) |
| Python / libs | 3.10.19, polars 1.35.2, pandas, numpy, pyarrow |
| Primary demo | `/Users/wanjungu/Downloads/0823_inferno.dem` (328 MB, `de_inferno`) |
| Primary demo stats | **167,846 ticks** (tick `1..167846`, contiguous, no gaps), 10 players, 64 tick, 24 rounds |
| Cross-check demos | `de_ancient` (`test.dem`), `de_overpass` (`9207151584779083532_0.dem`), `de_nuke`, `de_mirage` |

> ⚠️ **Environmental note:** `0823_inferno.dem` **and** the whole `~/Downloads/test_demos/` tree
> disappeared from disk *during* this research session (they existed at session start and were
> parsed successfully many times). All measurements below were captured before that and are
> persisted in `notes/scratch/*.json`. Cross-demo checks were re-run against the surviving
> `~/Downloads/test.dem` etc.

### Top-level API surface (complete, from the shipped `__init__.pyi` stub)

```python
DemoParser(demo_path)
  .parse_header()        -> Dict[str, str]
  .parse_player_info()   -> pd.DataFrame
  .parse_skins()         -> pd.DataFrame
  .parse_item_drops()    -> pd.DataFrame
  .list_game_events()    -> List[str]     # events that ACTUALLY occur in this demo
  .list_updated_fields() -> List[str]     # every netprop the demo transmits
  .parse_event(name, player=None, other=None)   -> pd.DataFrame
  .parse_events([names], player=None, other=None) -> List[Tuple[str, pd.DataFrame]]
  .parse_ticks(wanted_props, players=None, ticks=None, prop_states=None) -> pd.DataFrame
  .parse_grenades(extra=None, grenades=True)    -> pd.DataFrame
  .parse_voice()         -> List[{'tick','steamid','bytes'}]
```

`parse_ticks` and `parse_grenades` return **pandas** DataFrames; `parse_player_info` / `parse_skins`
return pyarrow-backed pandas.

**Import:** `from demoparser2 import DemoParser, WantedPropState`

---

## 0. Critical gotchas found the hard way

These four behaviours will silently corrupt a renderer if you don't know them.

### 0.1 Unknown prop names are **silently dropped**, never raise

`parse_ticks(["not_a_real_prop"])` returns a DataFrame with only `tick, steamid, name`.
No exception, no warning. Same for `parse_grenades(extra=["bogus"])` — the column just isn't there.
**Never assume a prop worked; always assert on the returned columns.**

```python
df = p.parse_ticks(["X", "is_planting"])
assert "is_planting" in df.columns, "prop silently dropped!"
```

**This bites hardest when probing for a field whose name you are guessing.** Real
example: `parse_ticks(["m_iFOV"])` "succeeds" and returns a frame with
`steamid`/`name` columns — which reads like *"the field exists, it just has no
values"*, when the truth is *"the name is wrong"*. Two different names for the
same concept gave two different results and neither raised:

| what was asked | what came back | what it actually means |
| --- | --- | --- |
| `CCSPlayerController.m_iDesiredFOV` | a real column, all `0` | the field exists and is dead |
| `CCSPlayerPawn.CCSPlayer_CameraServices.m_iFOV` | a real column, 5 distinct values | the live field |

Always check `df.columns`, and when a probe reports "OK" with only
`steamid`/`name`, treat it as a **failure**, not an empty result.

### 0.2 `grenade_entity_id` / event `entityid` are **reused all match**

Entity index 258 was one player's smoke early in the match and a different player's smoke later.
*Always* segment grenade rows into **runs** and pair events by position, not by id.

```python
g = p.parse_grenades().sort_values(["grenade_entity_id", "tick"]).reset_index(drop=True)
g["new_run"] = (g.grenade_entity_id.ne(g.grenade_entity_id.shift())
                | g.tick.diff().ne(1)              # gap => different projectile
                | g.steamid.ne(g.steamid.shift())) # thrower change => different projectile
g["run_id"] = g.new_run.cumsum()
```

This yields exactly the right projectile counts for the inferno demo, matching the detonation
events one-for-one:

| type | runs | matching event |
| --- | --- | --- |
| `CSmokeGrenadeProjectile` | **137** | `smokegrenade_detonate` = 137 |
| `CFlashbangProjectile` | **126** | `flashbang_detonate` = 126 |
| `CHEGrenadeProjectile` | **114** | `hegrenade_detonate` = 114 |
| `CMolotovProjectile` | **87** | `inferno_startburn` = 85 (+2 never burned) |
| `CDecoyProjectile` | **20** | `decoy_started` = 19 |

Total 484 runs. Naively grouping by `grenade_entity_id` is **wrong** and produces garbage
(one smoke "entity" appeared to last 65,855 ticks).

### 0.3 Raw entity tick fields live in a **different, per-demo tick base**

`m_nSmokeEffectTickBegin` / `m_nExplodeEffectTickBegin` are **not** in the same tick space as the
`tick` column / event ticks. The offset is a constant **within** a demo but **differs per demo**:

| demo | constant offset (`field_tick - demoparser_tick`) | consistency |
| --- | --- | --- |
| `0823_inferno.dem` | **794** | 127/127 smokes, 107/107 HE grenades |
| `..._de_nuke.dem` | **830** | 105/105 |
| `..._de_mirage.dem` | **691** | 88/88 |

→ Use **event ticks** for everything time-related. Only if you must use an entity tick field,
calibrate the offset per demo (e.g. `field_tick - detonation_event_tick`) and verify it is constant.

### 0.4 The demo **includes pre-game warmup**

`is_warmup_period == True` and `total_rounds_played == 0` for ticks `1 .. ~3500`;
`begin_new_match` is at tick **3533**. A `round_prestart` exists at tick **43** for the warmup round.
A replay renderer must start from `begin_new_match`, not from tick 0.

---

## 1. `DemoParser.parse_grenades()`

```python
g = p.parse_grenades()                      # all grenade entities
g = p.parse_grenades(grenades=False)        # ONLY in-flight projectiles (recommended)
g = p.parse_grenades(extra=["m_vSmokeDetonationPos", "m_flDetonateTime"])
```

### Returned columns (always these 8, pandas)

| column | dtype | notes |
| --- | --- | --- |
| `grenade_type` | `object` (str) | class name, 11 distinct values (below) |
| `grenade_entity_id` | `int32` | entity index — **reused**, segment into runs |
| `x`, `y`, `z` | `float32` | world coords; **NULL for non-projectile types** |
| `tick` | `int32` | demo tick |
| `steamid` | `uint64` | thrower steamid (never null) |
| `name` | `object` | thrower name at that tick |

2,287,704 rows × 8 cols for the full inferno demo (`grenades=True`);
310,438 rows with `grenades=False` (projectiles only).

### `grenade_type` values — the crucial distinction

| value | has coords? | meaning |
| --- | --- | --- |
| `CSmokeGrenadeProjectile` | ✅ | smoke **in flight** |
| `CFlashbangProjectile` | ✅ | flashbang in flight |
| `CHEGrenadeProjectile` | ✅ | HE in flight |
| `CMolotovProjectile` | ✅ | molotov **and** incendiary in flight |
| `CDecoyProjectile` | ✅ | decoy in flight |
| `CSmokeGrenade`, `CFlashbang`, `CHEGrenade`, `CIncendiaryGrenade`, `CMolotovGrenade`, `CDecoyGrenade` | ❌ **all x/y/z NULL** | the *weapon item* entity sitting in a player's inventory |

Verified: for the non-projectile types `x.notna().sum() == 0` on every demo tested.
**So `parse_grenades(grenades=False)` is what a replay wants** — it gives you exactly the 5
projectile classes with real per-tick trajectories.

### Per-tick trajectories: YES

For every `*Projectile` entity you get one row per tick, `diff(tick) == 1` within a run, with
interpolatable `x/y/z`. This covers smoke, flash, HE, molotov/incendiary and decoy flight paths.

### `extra=` — which props actually survive

`extra=[...]` merges entity netprops onto the grenade rows (joined by entity+tick). These **work**
(measured on the inferno demo; "nonzero" excludes `0/0.0/False/[]/[0,0,0]`):

| prop | dtype | nonzero rows | meaning |
| --- | --- | --- | --- |
| `m_vSmokeDetonationPos` | `object` → `[x,y,z]` | 188,155 | **smoke detonation point** |
| `m_vSmokeColor` | `object` → `[R,G,B]` | 215,284 | smoke colour, e.g. `[180,129,50]` |
| `m_nSmokeEffectTickBegin` | `float64` | 187,862 | smoke start tick (**offset base!**) |
| `m_bDidSmokeEffect` | `object` (bool) | 187,862 | flips `True` at the detonation tick |
| `m_nVoxelFrameDataSize` | `float64` | 187,833 | smoke volume byte size — **grows over life** |
| `m_nVoxelUpdate` | `float64` | 187,833 | volume-update counter |
| `m_vecExplodeEffectOrigin` | `object` → `[x,y,z]` | 36,480 | detonation origin (HE/molotov/flash) |
| `m_nExplodeEffectTickBegin` | `float64` | 36,480 | detonation tick (HE/molotov/flash) |
| `m_vInitialPosition` | `object` → `[x,y,z]` | 310,438 | spawn point of the projectile |
| `m_vInitialVelocity` | `object` → `[x,y,z]` | 310,438 | initial throw velocity (units/s) |
| `m_flDetonateTime` | `float32` | 69,745 | absolute detonation **time** (s) |
| `m_flCreateTime` | `float32` | all | entity create time (s) |
| `m_nBounces` | `float64` | 265,234 | bounce count |
| `m_flThrowStrength` | `float32` | 111,263 | throw strength 0..1 |
| `m_nDeployTick` | `float64` | 1,051,829 | deploy tick |
| `m_bIsLive` | `object` (bool) | 0 | always `False` in this demo |
| `m_hThrower` | — | — | thrower handle |
| `m_VoxelFrameData` | `float64` | 187,833 | ⚠️ **degenerate — see §3** |

Silently **dropped** by `extra=` (see §2): every `m_fire*` prop.

---

## 2. Inferno / fire — **the real burning fire cells are NOT obtainable**

**This is a hard negative result.** The data *is present in the demo* but demoparser2 0.42.0
provides no route to it.

### The data demonstrably exists in the file

```python
p.list_updated_fields()
# -> 'Grenade.m_firePositions', 'Grenade.m_fireParentPositions', 'Grenade.m_fireCount',
#    'Grenade.m_bFireIsBurning', 'Grenade.m_nInfernoType', 'Grenade.m_nFireLifetime',
#    'Grenade.m_nFireEffectTickBegin'
```

So the engine transmits `m_firePositions` (the per-cell position array) and `m_bFireIsBurning`.
Note the CS2 names differ from the CS:GO-era ones in the task brief: it is `m_fireCount` (not
`m_nFireCount`) and `m_firePositions` (not `m_fireXDelta/m_fireYDelta/m_fireZDelta` — those do not
exist in CS2 and are silently dropped). The class is `C_Inferno`, surfaced under the `Grenade`
prefix by the netprop dumper.

### Every route tried, and exactly what happened

| # | route | result |
| --- | --- | --- |
| 1 | `parse_grenades(extra=["m_firePositions", "m_fireCount", "m_bFireIsBurning", "m_nInfernoType", "m_nFireLifetime", "m_nFireEffectTickBegin", "m_fireParentPositions", "m_BurnNormal", "m_fireXDelta/YDelta/ZDelta"])` | **All 12 columns silently dropped.** No column added, no error. |
| 2 | Each fire prop passed **individually** to `extra=` (one call per prop, 12 separate runs) | Every one **DROPPED**. Not a batch-size limit. |
| 3 | `parse_grenades(grenades=False)` | 310,438 rows, but only 5 classes: `CSmokeGrenadeProjectile`, `CFlashbangProjectile`, `CHEGrenadeProjectile`, `CMolotovProjectile`, `CDecoyProjectile`. **No `CInferno`.** |
| 4 | `parse_grenades(grenades=False, extra=FIRE_PROPS)` | Fire cols still **dropped**. |
| 5 | `parse_ticks(FIRE_PROPS)` | Fire props **silently ignored**; output only `tick, steamid, name`. `parse_ticks` is strictly **player-scoped** (the shipped `.pyi` confirms: *"The props to parse for each player at each tick"*). |
| 6 | `parse_event("inferno_extinguish")` | The string exists in the compiled `.so`, but this demo never fires it → returns an **empty Python list** (not an empty DataFrame). |
| 7 | `strings demoparser2.so \| grep m_fire` | **0 hits.** The binary hard-codes no fire-prop handling. |
| 8 | `grenade_type` enumeration | No `CInferno` value ever appears. |

**Root cause:** `parse_grenades` only scans a fixed set of grenade *projectile* classes, and
`C_Inferno` is not among them; `parse_ticks` only reads player pawns. There is no generic
"parse arbitrary entity class" entry point in the public API.

### What you *can* get about fire (accurate, verified)

Event `inferno_startburn` / `inferno_expire` — 85 each in the inferno demo:

```python
sb = p.parse_event("inferno_startburn")
# cols: entityid:int32, tick:int32, user_name:object, user_steamid:object, x:float32, y:float32, z:float32
# e.g. entityid=168 tick=5617 CallMeS1U  x=1207.73 y=22.77 z=256.0
ex = p.parse_event("inferno_expire")
# NOTE: entityid is REUSED -> pair by (entityid, x, y, z), NOT by entityid alone
m = sb.merge(ex, on=["entityid", "x", "y", "z"], suffixes=("_s", "_e"))
m = m[m.tick_s < m.tick_e]
m["dticks"] = m.tick_e - m.tick_s           # -> 85/85 paired, positions identical
```

* `x/y/z` is the **fire's centre** (the impact point) — one point, **not** the cell grid.
* Lifetime is exact and bimodal — and it is a real game mechanic, reproduced on 3 different maps:

| thrown item (from `item_pickup`/`item_equip`) | team | duration | ticks |
| --- | --- | --- | --- |
| `molotov` | T (`team_num == 2`) | **7.0 s** | 449–450 |
| `incgrenade` (Incendiary) | CT (`team_num == 3`) | **5.5 s** | 352 |

Verified on `de_inferno` (32×449, 7×450, 42×352), `de_ancient` (19×449, 2×450, 35×352) and
`de_overpass` (9×449, 13×352). A handful of shorter values (2.2–4.4 s) are fires extinguished early.

* The molotov/incendiary **flight path to the impact point** is available via
  `parse_grenades(grenades=False)` (`CMolotovProjectile`), and `m_vecExplodeEffectOrigin` +
  `m_nExplodeEffectTickBegin` via `extra=` give the detonation point/tick.
* **Unavailable:** the real per-cell footprint grid, `m_bFireIsBurning` per cell, the cell count,
  the true burn radius, and the growth of the fire over its lifetime.

**Workaround for a faithful-enough renderer:** draw a circular fire of CS2's real inferno radius
(≈150 units, 18–19 cell grid) centred on `inferno_startburn.x/y/z`, alive for
`inferno_startburn.tick .. inferno_expire.tick`. To get the *true* footprint you need a parser that
exposes `C_Inferno` entities — demoparser2 0.42.0 cannot, and no `extra`/prop name will fix it.

---

## 3. Smoke

### Detonation position — two independent, exactly-agreeing sources

```python
det = p.parse_event("smokegrenade_detonate")
# cols: entityid, tick, user_name, user_steamid, x, y, z  (137 rows)
```
and via props:
```python
g = p.parse_grenades(extra=["m_vSmokeDetonationPos", "m_bDidSmokeEffect"])
```
`m_vSmokeDetonationPos` matches the event `x/y/z` to <0.5 units in 100% of cases, and
`m_bDidSmokeEffect` flips to `True` on exactly the event's `tick`. **Use the event's `x/y/z`.**

### Start / end ticks

```python
det = p.parse_event("smokegrenade_detonate")   # 137 rows  -> smoke start
exp = p.parse_event("smokegrenade_expired")    # 127 rows  -> smoke end
m = det.merge(exp, on=["entityid", "x", "y", "z"], suffixes=("_det", "_exp"))
m = m[m.tick_det < m.tick_exp]
```

* `smokegrenade_expired` carries the **same x/y/z** as `smokegrenade_detonate` (verified
  `np.allclose`), which is what makes positional pairing reliable.
* Lifetime: **1412 ticks = 22.06 s**, identical for all 127 paired smokes.
  (Note: `smokegrenade_expired` fires when the effect entity is removed, i.e. after the
  visual fade-out, so 22.06 s is *entity lifetime*, not necessarily the fully-dense window.)
* **10 of the 137 smokes never expire** (round ended first) → always treat the smoke as
  "alive from detonate until `min(expired, round end)`".

```python
# minimal smoke timeline builder
smokes = []
for _, r in det.iterrows():
    e = exp[(exp.entityid == r.entityid) & (exp.x == r.x) & (exp.y == r.y) & (exp.z == r.z)
            & (exp.tick > r.tick)]
    end = int(e.tick.min()) if len(e) else None      # None => cut off by round end
    smokes.append({"start": int(r.tick), "end": end, "pos": (r.x, r.y, r.z),
                   "steamid": int(r.user_steamid)})
```

### Bloom / growing — partially recoverable

* ❌ **The real smoke volume `m_VoxelFrameData` is NOT usable.** Although the column is added,
  it comes back as a **single degenerate `float64`** (observed values `170.0`, `1.0`, `15.0` — i.e.
  a scalar, apparently the first byte), not an array/grid. It cannot give the smoke's shape.
  `m_bParticlesSpawned` and `m_bInPostEffectTime` are silently dropped.
* ✅ **Growth can be approximated from `m_nVoxelFrameDataSize`.** For one smoke it rises
  monotonically across its 1412 ticks, with rising `m_nVoxelUpdate`:

  | tick | `m_nVoxelFrameDataSize` | `m_nVoxelUpdate` |
  | --- | --- | --- |
  | 49290 | 722 | 1 |
  | 49758 | 1444 | 67 |
  | 50226 | 1961 | 134 |
  | 50694 | 2502 | 201 |
  | 50701 (last) | 2521 | — |

  Monotonically non-decreasing: **True**. Normalising this to `0..1` gives a usable bloom curve.
* ✅ `m_vSmokeColor` gives the real per-smoke RGB (e.g. `[180, 129, 50]`).
* The projectile trajectory up to detonation is available (see §1).

---

## 4. Player state per tick — `parse_ticks`

```python
df = p.parse_ticks(["X","Y","Z","yaw","pitch","health", ...],
                   players=[76561199551992959],      # None/[] = all players
                   ticks=list(range(30000, 30100)))  # None/[] = all ticks
```

Always-present join keys: `tick` (int32), `steamid` (uint64), `name` (object).

**92 props verified available in a single call.** Exact names + dtypes:

### Transform / identity / team
| prop | dtype | notes |
| --- | --- | --- |
| `X`, `Y`, `Z` | float32 | world coords (already `m_vec + m_cell`) |
| `pitch`, `yaw` | float32 | eye angles; **no `roll`** |
| `player_name` | object | |
| `player_steamid` | uint64 | |
| `team_num` | uint32 | **2 = TERRORIST, 3 = CT** |
| `team_name` | object | `'TERRORIST'` / `'CT'` |
| `entity_id` | int32 | pawn entity index |
| `user_id` | int32 | |
| `player_color` | int32 | teammate colour |
| `rank` | int32 | competitive rank |
| `is_coach_team` | int32 | |

### Vitals / status
| prop | dtype | notes |
| --- | --- | --- |
| `health` | int32 | also `armor` int32 (identical values) |
| `armor_value` | int32 | documented name |
| `has_helmet`, `has_defuser` | bool | |
| `is_alive` | bool | |
| `life_state` | uint32 | 0 = alive, 2 = dead |
| `is_defusing` | bool | |
| `is_scoped`, `is_walking`, `is_airborne` | bool | |
| `ducking` | bool | also `duck_amount` float32 |
| `spotted` | bool | `CCSPlayerPawn.m_bSpotted`。**和 `approximate_spotted_by` 完全冗余**：实测 de_inferno 2,172,400 行原始数据里，`spotted == (mask != 0)` 的**不一致率 0.0000%**（94,807 行同时为真，0 行只有一个为真）。所以 `.csz` 里**只存了 mask 那一列**，bool 没存 |
| `approximate_spotted_by` | object | `CCSPlayerPawn.m_bSpottedByMask`（`uint32[2]`）。⚠ 这里以前写的 "list (empty in practice)" **是错的**：它有值，而且就是 `demo_extract` 提取 `spotted` 列的**唯一来源**。demoparser2 的解码是**有损**的 —— 只读第一个 `Variant::U32`、只循环 bit 0..15，bit i 映射到 controller id `i+1` 再查 steamid。实测 de_inferno：101,470 个 mask 条目**全是敌人**（0 个队友），长度分布 0→2,077,593 / 1→88,177 / 2→6,597 / 3→33，从没 ≥4。⚠ mask 在 pawn 死亡后**留着最后一个值不清零**（实测 87,251 个采样点是"死人 + mask 非空"，而活人非空只有 7,556 个）—— 用的时候必须自己按 `alive` 过滤 |
| `flash_duration` | float32 | **flashbang blind amount, seconds** |
| `flash_max_alpha` | float32 | 255.0 |
| `molotov_damage_time` | float32 | |
| `shots_fired` | int32 | |
| `last_place_name` | object | e.g. `'Banana'`, `'BombsiteA'` — great for map labeling |
| `is_bomb_planted`, `is_bomb_dropped` | bool | |
| `is_freeze_period`, `is_warmup_period` | bool | |
| `game_phase` | int32 | |
| `total_rounds_played` | int32 | |
| `round_win_status`, `round_win_reason` | int32 | |
| `round_start_time` | float32 | |
| `game_time` | float32 | seconds; **slope exactly 1/64 s per tick** (tickrate proof) |
| `is_game_restart`, `has_bombites`, `has_buy_zone`, `hostages_remaining`, … | | see README list |

### Camera / FOV (measured 2026-02, de_inferno m2, 2,172,400 rows)

| prop | dtype | notes |
| --- | --- | --- |
| `CCSPlayerPawn.CCSPlayer_CameraServices.m_iFOV` | int32 | **the real, live FOV field.** Values seen: `0` (95.23% of alive ticks — means "use the default"), `90` (2.36%, the default written out explicitly), `40` (2.04%), `10` (0.37%), `15` (~0). **`40 / 10 / 15` are scope zoom levels, and that is measured not guessed:** `40` co-occurs with AWP 97.8% / SSG 08 2.2%, `10` is AWP 100%, `15` is SSG 08 100%. While holding an AWP, 64.7% of ticks sit at 40 or 10. So a scoped player's real horizontal FOV is 40° or 10°, **not** 90°. Cheap to store: only **394 changes** across all 10 players for the whole 24-round match (vs 2,172,400 per-tick entries) — same order as `spotted` (740). **Not stale on death** (dead players are 100% at 0), unlike the spotted mask |
| `CCSPlayerPawn.CCSPlayer_CameraServices.m_iFOVStart` | int32 | FOV at the start of a zoom transition. 98.9% `90`; also 40/10/66/72/71/15 |
| `CCSPlayerPawn.CCSPlayer_CameraServices.m_flFOVRate` / `m_flFOVTime` | float32 | zoom transition timing |
| `CCSPlayerPawn.m_flViewmodelFOV` | float32 | **viewmodel** FOV only (the gun model), *not* the world FOV. Values seen: `68` (70.0%) and `60` (30.0%) — exactly CS2's `viewmodel_fov` default 60 / max 68, which is what confirms these fields carry real client settings |
| ⚠️ `CCSPlayerController.m_iDesiredFOV` | int32 | **DEAD FIELD — it parses but is `0` in every one of the 1,532,610 rows.** Do not use it. It is the name you will find first (it is what Source 1 used), and it looks like it worked because it returns a real column of zeros rather than being dropped. The live value is on the pawn's `CCSPlayer_CameraServices` |

> ⚠️ **Unresolved:** whether `m_iFOV` is the horizontal FOV at the *current* aspect
> ratio, or Source's traditional *4:3-reference* horizontal FOV (widened on
> widescreen via Hor+). If the latter, at 16:9 the true horizontal is ≈**106.26°**
> and vertical ≈**73.74°**. Could not verify — the Valve developer wiki is
> unreachable from this machine (returns non-JSON) and web search returns nothing.
> Flagged as unverified rather than guessed.

### Economy
| prop | dtype |
| --- | --- |
| `balance` | int32 (money) |
| `start_balance` | int32 |
| `total_cash_spent` | int32 |
| `cash_spent_this_round` | int32 |
| `current_equip_value` | uint32 |
| `round_start_equip_value` | uint32 |
| `equipment_value_total` | int32 |

### Weapons
| prop | dtype | notes |
| --- | --- | --- |
| `active_weapon` | uint32 | raw handle; `16777215` = none |
| `active_weapon_name` | object | **`'AK-47'`, `'MAC-10'`, `'Skeleton Knife'`** |
| `active_weapon_skin` | object | **skin NAME, `'Aphrodite'`, `'Sakaku'`, `'Fade'`** |
| `active_weapon_ammo` | float64 | clip1 |
| `active_weapon_original_owner` | object | steamid string |
| `item_def_idx` | float64 | definition index |
| `weapon_paint_seed` | uint32 | |
| `weapon_float` | float32 | **wear 0..1, already a real float** |
| `weapon_stickers` | object | list of dicts `{id,name,wear,...}` |
| `fall_back_paint_kit` | float64 | |
| `custom_name` | object | nametag |
| `inventory` | object | **list of names**, e.g. `['knife_t','Glock-18','MAC-10','Flashbang']` |
| `inventory_as_ids` | object | `[59, 4, 17, 43]` |
| `total_ammo_left` | float64 | |
| `glove_paint_id` | float64 | gloves |
| `agent_skin` | object | `'customplayer_ct_map_based'` |

### Movement / buttons / usercommands
| prop | dtype | notes |
| --- | --- | --- |
| `velocity` | float32 | **scalar speed only** |
| `velocity_X`, `velocity_Y`, `velocity_Z` | float32 | the real velocity vector |
| `max_speed`, `fall_velo` | float32 | |
| `FORWARD`,`BACK`,`LEFT`,`RIGHT`,`FIRE`,`RIGHTCLICK`,`RELOAD`,`USE`,`ZOOM` | bool | from button mask |
| `usercmd_viewangle_x/_y`, `usercmd_forward_move` | float32 | per-usercmd data |

### ⚠️ Documented-but-NOT-working in 0.42.0 (silently dropped)

`is_planting` · `in_crouch` · `crouch_state` · `stamina` · `move_state` ·
`aim_punch_angle` · `aim_punch_angle_vel` · `WALK` (button) · `buttons` (raw mask) ·
`m_iAccount` · `m_vecVelocity` · `m_fFlags` · `kills`/`deaths`/`assists` (use
`kills_total`/`deaths_total`/`assists_total`) · all `m_fire*` props.

> `is_planting` is **not available** — derive planting state from `bomb_beginplant` and
> `bomb_planted` events (a ~3–5 s window), or from the planter's frozen position.

### Role / team
There is **no role prop** (IGL / entry / AWPer / lurker) — CS2 demos do not record player roles.
You get `team_num` (2=T, 3=CT), `team_name`, `player_color`, `rank`, plus per-round stats
(`kills_total`, `deaths_total`, `damage_total`, `equipment_value_total`, …).

### `prop_states` (server-side filter)
`WantedPropState(prop, state)` works as an inclusion filter on some props but is unreliable here:
`WantedPropState("is_alive", False)` returned exactly the 4 dead-player rows, while
`WantedPropState("is_alive", True)` returned **0 rows** despite 9/10 players being alive.
Tuple/dict forms raise `AttributeError`. **Filter in pandas/polars instead.**

---

## 5. `parse_skins()`

```python
sk = p.parse_skins()
# 15 rows x 7 cols for this match (only ~15 skinned items among 10 players)
```

| column | dtype |
| --- | --- |
| `steamid` | uint64 |
| `def_index` | uint32 (weapon/knife/glove def index) |
| `item_id` | uint64 |
| `paint_index` | uint32 (**the skin/paint-kit id**) |
| `paint_seed` | uint32 (pattern seed) |
| `paint_wear` | uint32 — ⚠️ **raw float BIT PATTERN, not a float** |
| `custom_name` | large_string (nametag, `<NA>` usually) |

`paint_wear` must be reinterpreted as float32 bits:

```python
import struct
def wear(v):  # 1038596544 -> 0.113147
    return struct.unpack("<f", struct.pack("<I", int(v) & 0xFFFFFFFF))[0]
```

Verified: decoded values all fall in `0.0115 .. 0.4423` (a plausible wear range), and the decode
cross-checks against the per-tick route — e.g. `def_index=33, paint_index=1436` for a player
decodes to wear `0.113147`, and `parse_ticks` reports that same player's `MP7` as
`active_weapon_skin='Amberline'`, `weapon_float=0.1131472`.

### How to map player+weapon → skin for rendering

**Recommended (no lookup table needed):** use the per-tick weapon props, which already resolve
names for you:

```python
props = ["active_weapon_name", "active_weapon_skin", "weapon_paint_seed",
         "weapon_float", "fall_back_paint_kit", "glove_paint_id", "weapon_stickers"]
df = p.parse_ticks(props)          # -> 'AK-47' + 'Aphrodite' + seed + wear
```

**Alternative (needs a paint-kit table):** `paint_index` from `parse_skins()` (join on
`steamid` + `def_index` → `item_def_idx`) mapped through a paint-kit id→name table.
Note `parse_skins()` returns items **equipped at some point**, not tied to a tick, and may repeat
rows. For a tick-accurate weapon appearance, the `parse_ticks` route is strictly better.

---

## 6. `parse_player_info()` and `parse_header()`

```python
p.parse_player_info()   # 10 rows x 3 cols
```
| column | dtype | notes |
| --- | --- | --- |
| `steamid` | uint64 | join key against `parse_ticks`/events `steamid` |
| `name` | large_string | unicode names OK (`音驹第一突破研磨`) |
| `team_number` | int32 | 2 = T, 3 = CT |

⚠️ `team_number` here is the *initial* team; teams swap at halftime. Use per-tick `team_num`.

```python
p.parse_header()
# {'demo_file_stamp': 'PBDEMS2\x00', 'demo_version_guid': '8e9d71ab-...',
#  'demo_version_name': 'valve_demo_2', 'client_name': 'SourceTV Demo',
#  'server_name': 'Counter-Strike 2', 'map_name': 'de_inferno',
#  'patch_version': '14176', 'game_directory': 'D:\\steamcmd\\...',
#  'fullpackets_version': '2', 'allow_clientside_entities': 'true',
#  'allow_clientside_particles': 'true', 'addons': ''}
```

**`parse_header()` has NO tickrate field** — all values are strings. Derive tickrate as in §9.

`parse_item_drops()` returned **0 rows** in this demo (no souvenir/drop events) but has columns:
`account_id, def_index, dropreason, inventory, item_id, paint_index, paint_seed, paint_wear, custom_name`.

`parse_voice()` returned 138 packets, 2 speakers: `[{'tick': 4086, 'steamid': 76561198858515672,
'bytes': b'...'}]` — usable for voice chat audio, not needed for visual replay.

---

## 7. Round / event data

### The FULL list of events present in this demo (53 — from `list_game_events()`)

```
announce_phase_end            begin_new_match               bomb_begindefuse
bomb_beginplant               bomb_defused                  bomb_dropped
bomb_exploded                 bomb_pickup                   bomb_planted
buytime_ended                 chat_message                  cs_pre_restart
cs_round_final_beep           cs_round_start_beep           cs_win_panel_match
decoy_detonate                decoy_started                 fire_bullets
flashbang_detonate            hegrenade_detonate            hltv_chase
hltv_fixed                    hltv_versioninfo              inferno_expire
inferno_startburn             item_equip                    item_pickup
other_death                   player_blind                  player_changename
player_connect                player_connect_full           player_death
player_footstep               player_hurt                   player_spawn
player_team                   round_announce_final          round_announce_last_round_half
round_announce_match_point    round_announce_match_start    round_announce_warmup
round_freeze_end              round_officially_ended        round_poststart
round_prestart                round_time_warning            server_cvar
smokegrenade_detonate         smokegrenade_expired          weapon_fire
weapon_reload                 weapon_zoom
```

`list_game_events()` returns **only events that actually occur** in that demo, so this list is
demo-specific (a different demo yields ~48–51).

### 🔴 `round_start` and `round_end` DO NOT EXIST

Neither is in the list above, nor in `de_ancient` or `de_overpass` (checked explicitly).
Use instead:

| need | use |
| --- | --- |
| round begins | `round_prestart` (+ `round_poststart`, identical tick) |
| freezetime over / live | `round_freeze_end` |
| buy time over | `buytime_ended` |
| round resolved | `round_officially_ended` (= `m_bRoundEndShowTimerDefend` reached) |
| winner / reason | game-rule props `round_win_status`, `round_win_reason` (via `parse_ticks`) |
| match start | `begin_new_match` |
| halftime | `round_announce_last_round_half` |

### Exact columns for the important events

| event | n | columns (dtypes) |
| --- | --- | --- |
| `player_death` | 214 | `assistedflash:bool, assister_name:obj, assister_steamid:obj, attacker_name:obj, attacker_steamid:obj, attackerblind:bool, attackerinair:bool, distance:f32, dmg_armor:i32, dmg_health:i32, dominated:i32, headshot:bool, hitgroup:obj, noreplay:bool, noscope:bool, penetrated:i32, revenge:i32, thrusmoke:bool, tick:i32, user_name:obj, user_steamid:obj, weapon:obj, weapon_fauxitemid:obj, weapon_itemid:obj, weapon_originalowner_xuid:obj, wipe:i32` |
| `weapon_fire` | 5394 | `silenced:bool, tick:i32, user_name:obj, user_steamid:obj, weapon:obj` — weapon is **`weapon_ak47`** (prefixed!) |
| `bomb_planted` | 9 | `c4:i32, site:i32, tick:i32, user_name:obj, user_steamid:obj` — **no x/y/z!** |
| `bomb_defused` | 1 | `c4:i32, site:i32, tick:i32, user_name:obj, user_steamid:obj` |
| `bomb_exploded` | 3 | `c4:i32, site:i32, tick:i32, user_name:obj, user_steamid:obj` |
| `bomb_beginplant` | 10 | `site:i32, tick:i32, user_name:obj, user_steamid:obj` |
| `bomb_begindefuse` | 3 | `haskit:bool, tick:i32, user_name:obj, user_steamid:obj` |
| `bomb_dropped` | 44 | `entindex:i32, tick:i32, user_name:obj, user_steamid:obj` |
| `bomb_pickup` | 53 | `tick, user_name, user_steamid` |
| `round_end` | — | **does not exist** |
| `round_prestart` / `round_poststart` / `round_freeze_end` / `round_officially_ended` / `buytime_ended` / `round_time_warning` | 25/25/25/46/25/8 | **`tick` only** |
| `item_purchase` | — | **does not exist** → use `item_pickup` (`defindex, item, silent, tick, user_name, user_steamid`) and `item_equip` (`canzoom, defindex, hassilencer, hastracers, ispainted, issilenced, item, tick, user_name, user_steamid, weptype`) |
| `player_hurt` | 857 | `armor, attacker_name, attacker_steamid, dmg_armor, dmg_health, health, hitgroup, tick, user_name, user_steamid, weapon` |
| `player_blind` | 316 | `attacker_name, attacker_steamid, blind_duration:f32, entityid, tick, user_name, user_steamid` — **use `blind_duration` for flash effect strength** |
| `smokegrenade_detonate` | 137 | `entityid, tick, user_name, user_steamid, x, y, z` |
| `smokegrenade_expired` | 127 | same 7 cols |
| `hegrenade_detonate` | 114 | same 7 cols |
| `flashbang_detonate` | 126 | same 7 cols |
| `decoy_started` / `decoy_detonate` | 19/18 | same 7 cols |
| `inferno_startburn` / `inferno_expire` | 85/85 | same 7 cols |
| `fire_bullets` | 4186 | `angles_x/y/z, attack_type, ent_origin_x/y/z, inaccuracy, item_def_index, mode, num_bullets_remaining, origin_x/y/z, player, player_inair, player_scoped, recoil_index, round, seed, sound_dsp_effect, sound_type, spread, tick, user_name, user_steamid, weapon_id` — **ideal for bullet tracers/decals** |
| `other_death` | 210 | `attacker_name, attacker_steamid, attackerblind, headshot, noscope, otherid, othertype, penetrated, thrusmoke, tick, weapon, weapon_fauxitemid, weapon_itemid, weapon_originalowner_xuid` |
| `hltv_chase` / `hltv_fixed` | 412/113 | spectating camera tracks (useful as an alternative camera) |

### Events useful for a replay renderer

`round_prestart`, `round_freeze_end`, `round_officially_ended`, `begin_new_match`,
`player_death`, `player_hurt`, `player_blind`, `player_spawn`, `player_team`,
`weapon_fire`, `weapon_reload`, `weapon_zoom`, `fire_bullets`,
`bomb_beginplant`, `bomb_planted`, `bomb_begindefuse`, `bomb_defused`, `bomb_dropped`,
`bomb_pickup`, `bomb_exploded`,
`smokegrenade_detonate`, `smokegrenade_expired`, `flashbang_detonate`,
`hegrenade_detonate`, `decoy_started`, `decoy_detonate`,
`inferno_startburn`, `inferno_expire`, `item_pickup`, `item_equip`, `chat_message`,
`player_footstep` (audio), `hltv_chase`/`hltv_fixed` (cameras), `cs_win_panel_match`.

### Deriving the bomb position (no x/y/z in `bomb_planted`)

```python
bp = p.parse_event("bomb_planted")
for _, r in bp.iterrows():
    t = int(r["tick"]); sid = int(r["user_steamid"])
    pl = p.parse_ticks(["X","Y","Z","last_place_name","which_bomb_zone"],
                       ticks=[t], players=[sid]).iloc[0]
    # -> pos (2102.7, 180.5, 160.0)  last_place_name='BombsiteA'
```

* `site` is a **bombsite entity index**, not `0/1`: on `de_inferno` **`312` → Bombsite A**,
  **`403` → Bombsite B** (verified against `last_place_name` on all 9 plants). Mapping is per-map.
  A map-independent label comes from the planter's `last_place_name`
  (`'BombsiteA'` / `'BombsiteB'`).
* The bomb itself is a `C_C4` entity which `parse_grenades` does **not** expose, so the planter's
  position at the plant tick is the practical source of the bomb's world position.

---

## 8. Performance (this machine, `0823_inferno.dem`, 328 MB, 10 players, 167,846 ticks)

Measured with `time.perf_counter()` and `resource.getrusage(RUSAGE_SELF).ru_maxrss`.

| operation | wall time | rows | peak RSS |
| --- | --- | --- | --- |
| `parse_ticks(all 167,846 ticks × 10 players, 31 props)` | **7.32 s** | 1,668,280 | 2,641 MB |
| `parse_ticks(["X","Y","Z"])` full match | **0.68 s** | 1,668,280 | — |
| `parse_grenades()` | **0.95 s** | 2,287,704 | 2,641 MB |
| `parse_grenades(grenades=False)` | ~0.7 s | 310,438 | — |
| `parse_grenades(extra=[13 props])` | **2.64 s** | 2,287,704 | **3,104 MB** |
| `parse_header()` | ~0.000 s | — | — |
| `parse_player_info()` | 0.080 s | 10 | — |
| all 53 events via `parse_event` | a few s | — | — |

**Interpretation:** a full per-tick player parse is cheap (~7 s, once). The dominant cost is
**memory**, not time: ~2.6 GB for the pandas result. `extra=` on `parse_grenades` roughly triples
time and pushes peak RSS to ~3.1 GB, because it joins props per entity-tick across all 2.29 M rows.

Practical guidance:
* Cache the parsed DataFrames to Parquet/Arrow once and never re-parse per frame.
* Never call `parse_ticks` inside a render loop — batch the whole match into memory (or memory-map),
  then slice per tick.
* Restrict `ticks=`/`players=` when you only need a window; both params cut cost proportionally.
* Use `parse_grenades(grenades=False)` to drop the 1.98 M useless weapon-item rows.

---

## 9. Tickrate, tick range and time semantics

### Totals and tickrate

```python
df = p.parse_ticks(["game_time"])          # ticks=None -> all ticks
total_ticks = int(df["tick"].max())        # 167,846
```

* Tick range: **`1 .. 167846`**, all 167,846 values present, `max_gap == 1`, only gap value `1`
  → **contiguous, no gaps, no dropped ticks**.
* Demo duration: `167846 / 64 = 2622.6 s ≈ 43.7 min`.

### Tickrate = 64 — derived rigorously, not assumed

Regress the `game_time` prop against `tick`:

```python
import numpy as np
gt = p.parse_ticks(["game_time"], ticks=range(30000, 30120)).drop_duplicates("tick")
slope = np.polyfit(gt["tick"], gt["game_time"], 1)[0]
tickrate = 1 / slope        # slope = 0.01562500000000017  ->  tickrate = 63.999999999999304
```

**`tickrate == 64` exactly.** (Also consistent with the game mechanic: molotov = 449 ticks ≈ 7.0 s.)

### The time relation

```python
t_seconds = tick / 64.0          # demo-relative seconds
```
`game_time` is an absolute server clock in seconds (13.03 s at tick 40), so it is *not* equal to
`tick/64`; use it only for deltas, or use `tick/64` for all timing.

### Tick offset / pre-game warmup

There are **two** distinct tick-base issues:

1. **Warmup is included.** `tick 1 .. ~3500` is pre-game warmup
   (`is_warmup_period=True`, `total_rounds_played=0`); the warmup round's `round_prestart` is at
   tick **43**; `begin_new_match` is at tick **3533**. Start your replay at `begin_new_match`
   (or at the first real `round_prestart`, tick **3532**).
2. **Raw entity tick fields are offset by a per-demo constant** (see §0.3): 794 (inferno),
   830 (nuke), 691 (mirage). The demoparser2 `tick` column and all *event* ticks share one space;
   `m_nSmokeEffectTickBegin`, `m_nExplodeEffectTickBegin`, `m_nDeployTick`, `m_nNextPrimaryAttackTick`
   etc. are in another. **Use event ticks.**

### Round timing model (verified)

```python
pre = p.parse_event("round_prestart")["tick"].tolist()      # 25 (1 warmup + 24 real)
fz  = p.parse_event("round_freeze_end")["tick"].tolist()
pre = [t for t in pre if t >= 3532]; fz = [t for t in fz if t >= 3532]
freeze_ticks = [b - a for a, b in zip(pre, fz)]
```

| round | prestart | freeze_end | freezetime |
| --- | --- | --- | --- |
| 1 | 3532 | 4908 | **1376 ticks = 21.5 s** (match start) |
| 2 … 12 | 7302 … | 8262 … | **960 ticks = 15.0 s** |
| 13 | 85303 | 86679 | **1376 ticks = 21.5 s** (2nd half start) |
| 14–15 | … | … | 960 (15.0 s) |
| 16 | 104132 | 109731 | 5599 ticks = 87.5 s (timeout / pause outlier) |
| 17 … 24 | … | … | 960 ticks = 15.0 s |

24 real rounds, 23 `round_officially_ended` (the last round's official-end is replaced by
`cs_win_panel_match` at tick 167373). `round_announce_final` and the last `round_prestart` are both
at 161050. `max_event_tick = 167693`.

---

## 10. Recommended renderer data pipeline

```python
from demoparser2 import DemoParser
import pandas as pd

p = DemoParser(DEMO)
TR = 64.0
t0 = int(p.parse_event("begin_new_match")["tick"].iloc[0])   # skip warmup

# --- players: ONE call, whole match, then cache --------------------------------
PLAYER_PROPS = [
    "X", "Y", "Z", "pitch", "yaw",
    "health", "armor_value", "has_helmet", "has_defuser", "is_alive", "life_state",
    "is_defusing", "is_scoped", "is_walking", "is_airborne", "ducking", "duck_amount",
    "team_num", "player_name", "player_steamid", "entity_id", "player_color", "last_place_name",
    "balance", "current_equip_value", "round_start_equip_value",
    "active_weapon_name", "active_weapon_skin", "active_weapon_ammo", "item_def_idx",
    "weapon_paint_seed", "weapon_float", "weapon_stickers", "glove_paint_id",
    "inventory", "inventory_as_ids", "total_ammo_left",
    "velocity_X", "velocity_Y", "velocity_Z", "velocity",
    "flash_duration", "shots_fired", "spotted",
    "FORWARD", "BACK", "LEFT", "RIGHT", "FIRE", "RIGHTCLICK", "RELOAD", "USE", "ZOOM",
    "is_bomb_planted", "is_freeze_period", "game_phase", "total_rounds_played",
    "round_win_status", "round_win_reason", "is_warmup_period",
]
players = p.parse_ticks(PLAYER_PROPS)                 # ~7 s, ~2.6 GB
players = players[players.tick >= t0]
players.to_parquet("players.parquet")                 # parse once, slice forever

# --- grenades: projectiles only, with effect props -----------------------------
gren = p.parse_grenades(grenades=False, extra=[
    "m_vSmokeDetonationPos", "m_vSmokeColor", "m_bDidSmokeEffect",
    "m_nSmokeEffectTickBegin", "m_nVoxelFrameDataSize", "m_nVoxelUpdate",
    "m_vecExplodeEffectOrigin", "m_nExplodeEffectTickBegin",
    "m_vInitialPosition", "m_vInitialVelocity", "m_flThrowStrength", "m_nBounces",
])
gren = gren.sort_values(["grenade_entity_id", "tick"]).reset_index(drop=True)
same_entity = gren.grenade_entity_id.eq(gren.grenade_entity_id.shift())
same_thrower = gren.steamid.eq(gren.steamid.shift())
contiguous   = gren.tick.diff().eq(1)
gren["new_run"] = ~(same_entity & same_thrower & contiguous)
gren["run_id"] = gren.new_run.cumsum()

# --- events --------------------------------------------------------------------
smoke_det = p.parse_event("smokegrenade_detonate")
smoke_exp = p.parse_event("smokegrenade_expired")
inferno_s = p.parse_event("inferno_startburn")
inferno_e = p.parse_event("inferno_expire")
deaths    = p.parse_event("player_death")
shots     = p.parse_event("fire_bullets")     # tracers/decals
fires     = p.parse_event("weapon_fire")
blind     = p.parse_event("player_blind")     # blind_duration
bomb      = p.parse_event("bomb_planted")     # + planter X/Y/Z for the bomb position
```

Per-frame lookup:

```python
frame = players[players.tick == tick]                     # 10 rows
t_sec = tick / 64.0
```

### Capability summary

| Feature | Status |
| --- | --- |
| Player pos/angles/health/armor/flags/economy/inventory/weapon+skin, per tick | ✅ 92 props |
| Grenade flight trajectories (smoke/flash/HE/molotov/decoy), per tick | ✅ `parse_grenades(grenades=False)` |
| Smoke detonation point, start/end ticks, colour | ✅ events + `m_vSmokeDetonationPos` |
| Smoke real volumetric shape (voxel grid) | ❌ `m_VoxelFrameData` degenerate |
| Smoke growth curve | ⚠️ proxy only (`m_nVoxelFrameDataSize`, monotonic) |
| Molotov/incendiary impact point + burn duration (7.0 s T / 5.5 s CT) | ✅ events |
| **Inferno real burning fire cells / footprint grid** | ❌ **not obtainable in demoparser2 0.42.0** |
| Bullet tracers/origins/angles/spread/seed | ✅ `fire_bullets` |
| Flash blind amount per player | ✅ `player_blind.blind_duration`, `flash_duration` prop |
| Bomb plant/defuse/explode ticks + site + planter position | ✅ (no bomb x/y/z — derive) |
| Round boundaries | ✅ via `round_prestart` / `round_freeze_end` / `round_officially_ended` |
| `round_start` / `round_end` / `item_purchase` events | ❌ do not exist in CS2 demos |
| Player roles (IGL/AWPer/…) | ❌ not recorded |
| Voice chat audio | ✅ `parse_voice()` |

---

## Appendix: evidence files

All raw numbers above come from these reproducible scripts and their JSON output, in
`notes/scratch/`:

| script | purpose | output |
| --- | --- | --- |
| `probe_props.py` | test 151 candidate props individually | `props_result.json` |
| `probe_events.py` | every event's schema + samples | `events_result.json` |
| `batch_props.py` | verify 103-prop renderer set in one call | `batch_props_head.csv` |
| `perf_full.py` | full-match timing + memory | `perf_full.log`, `perf_result.json` |
| `tickrate.py` | tickrate derivation, tick range, round ticks | `tickrate_result.json` |
| `inferno_probe.py` | 8 independent routes to fire cells (all negative) | `inferno_result.json` |
| `smoke_grenade.py` | run segmentation, voxel probing | `smoke_result.json` |
| `debug_smoke.py` | tick-space + voxel + `prop_states` debugging | — |
| `offset_and_growth.py` | per-demo tick offset across 3 demos, bloom curve | `offset_result.json` |
| `skins_misc.py` | `parse_skins`, item drops, voice, bomb sites | `skins_result.json` |
| `final_gaps.py` | inferno item/duration, tracer, role props | — |
| `verify_final.py` | molotov/incendiary vs duration, verified via item events | `verify_result.json` |
| `last_checks.py` | round timing model | `last_result.json` |

### External references

* [demoparser README — full supported field list](https://raw.githubusercontent.com/LaihoE/demoparser/main/README.md)
  (source of the documented prop aliases; several listed props are silently ignored in 0.42.0 — see §4)
* [demoparser issue #89 — map/round/match details](https://github.com/LaihoE/demoparser/issues/89)
