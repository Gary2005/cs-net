# High-Fidelity CS2 Replay Rendering — Smoke/Fire Geometry, Asset Extraction, and the Honest Fidelity Ceiling

**Purpose.** Research report for building a game-accurate CS2 match-replay renderer from `.dem` data, for
(a) synthetic observation generation for AI training and (b) realistic visual simulation.

**Method / evidence classes.** Every claim below is tagged with how it was established:

| Tag | Meaning |
|---|---|
| **[V]** | **Valve-authoritative** — read directly out of Valve's own machine-readable schema/time-tracking dumps (`GameTracking-CS2`), or Valve's own convar dump. These are the strongest available sources outside the engine binary. |
| **[VRF]** | **Tool-authoritative** — from the ValveResourceFormat project's own docs/source, verified against the CLI binary installed on this machine. |
| **[M]** | **Measured by us** — numbers produced by our own probes against `~/Downloads/0823_inferno.dem` (already in `notes/scratch/`). |
| **[C]** | **Community / second-hand** — journalist guides, wiki, Reddit. Directionally useful, not precise. |
| **[J]** | **Judgement** — my own estimate. Explicitly not a measurement. |

Two things are worth stating up front because they change the whole project's shape:

1. **CS2 smoke is not a sphere.** It is a server-authoritative *volumetric voxel field*, and the demo file
   genuinely *contains that field* (`m_VoxelFrameData`). Our current tool (`demoparser2`) silently drops it.
   The single highest-leverage change available is to stop using `demoparser2` for these two entities.
2. **Extracted real assets are necessary but not sufficient.** The glTF path throws away baked lighting,
   lightmaps, particle systems, the 3D skybox, and the animation state machine. Real meshes buy you
   silhouette and layout; they do not buy you the look.

---

## Part 0 — Executive verdict

| Goal | Achievable? | Best realistic fidelity |
|---|---|---|
| AI-usable **smoke occupancy volume** (exact, per-tick) | **Yes** — data is in the demo | ~99 % (it is the ground truth; needs a custom parser) |
| AI-usable **fire cell positions** (exact, per-tick) | **Yes** — data is in the demo | ~99 % (same caveat) |
| Real map **geometry + silhouette** | Yes, via VRF | **95–99 %** — it is the actual mesh |
| Real **character meshes + skeleton** | Yes, via VRF | 90–95 % static, much less in motion |
| Real **character animation state** | Partly | **50–75 %** — VRF never executes the animgraph |
| Real **smoke look** (raymarched volume) | Needs custom shader | **70–85 %** |
| Real **static lighting / GI** | No | **30–60 %** — lightmaps are not exported |
| Real **skybox / horizon** | No | 3D skybox is explicitly not exported |
| **Bit-comparable still frame** | No | ~75–90 % at distance; **50–65 % in motion** |

**Blunt verdict.** A Three.js replay built on real VRF-extracted assets with a hand-written volumetric
smoke shader will be *unmistakably CS2* in a still frame and *unmistakably not CS2* in motion. That is
good enough — and in some ways better — as a synthetic observation source for AI, because the
**gameplay-relevant geometry can be made exact** even when the pixels are not. It is not good enough to
be mistaken for engine output, and no project has achieved that (see §3.4, §4.5).

---

# Part 1 — Smoke and fire as game objects

## 1.1 How CS2 smoke actually works (the authoritative answer)

### It is a volumetric voxel field, not a particle cloud and not a sphere

The compiled entity schema for `CSmokeGrenadeProjectile` is unambiguous **[V]**:

```cpp
// DumpSource2/schemas/server/CSmokeGrenadeProjectile.h
class CSmokeGrenadeProjectile : public CBaseCSGrenadeProjectile
{
    int32                          m_nSmokeEffectTickBegin;
    bool                           m_bDidSmokeEffect;
    int32                          m_nRandomSeed;
    Vector                         m_vSmokeColor;
    VectorWS                       m_vSmokeDetonationPos;
    CNetworkUtlVectorBase< uint8 > m_VoxelFrameData;   // <-- the smoke volume, networked as raw bytes
    int32                          m_nVoxelFrameDataSize;
    int32                          m_nVoxelUpdate;
    GameTime_t                     m_flLastBounce;
    GameTime_t                     m_fllastSimulationTime;
    bool                           m_bExplodeFromInferno;
    bool                           m_bDidGroundScorch;
};
```
Source: [`CSmokeGrenadeProjectile.h`](https://github.com/SteamTracking/GameTracking-CS2/blob/master/DumpSource2/schemas/server/CSmokeGrenadeProjectile.h)
(same file exists client-side as `C_SmokeGrenadeProjectile`, plus `m_bSmokeVolumeDataReceived`
and `m_bSmokeEffectSpawned` — i.e. the client *streams the volume in* and waits for it before drawing).

**`m_VoxelFrameData` is a networked `CUtlVector<uint8>`.** That is the engine shipping the actual
smoke occupancy field to every client, frame by frame, so that everyone sees the same smoke. This is
the mechanism behind "everyone sees the same smoke" and behind smoke filling a stairwell rather than
forming a ball **[V]**.

Corroborating client cvars **[V]** (from the convar dump):

| Convar | Default | Reads as |
|---|---|---|
| `cl_smoke_volumeprop` | `true` | smoke is a **volume**, not a sprite |
| `cl_smoke_volume_growth` | `1` | volumetric growth enabled |
| `r_csgo_smoke_shadow` | `true` | the volume casts shadows |
| `r_csgo_smoke_fullres_pass` | `true` | full-res pass to cover holes/artifacts in the low-res pass |
| `r_csgo_smoke_fullres_enhance` | `false` | "Enhance edges of smokes to eliminate bad pixels" |
| `r_csgo_smoke_upscale_discard_pixels_behind` | `false` | "discard pixels behind solid depth to avoid pixelated artifacts" |
| `r_csgo_smoke_clip_sniper` | `true` | sniper scope has special smoke handling |
| `r_csgo_smoke_avoid_flat` | `true` | avoids flat/disc-shaped deployments |
| `r_csgo_smoke_overlay_min_dt` | `0.0156863` | ≈ 4/255 — per-frame compositing step |
| `smoke_use_noise_texture` | `true` | 3D noise texture drives the density field |
| `smoke_volume_lod_ratio_change` | `0.6` | volume LOD |
| `inferno_smoke_volume_density` | `0.03` | density when fire is smoked out |
| `sv_smoke_volume_blind_start` | `0.2` | density at which the volume starts blinding |

The presence of a **low-res pass plus a full-res "hole-covering" pass** and of an explicit
`upscale_discard_pixels_behind` option is a direct signature of **low-resolution volumetric raymarching
that is upsampled**, i.e. the classic froxel / 3D-texture volume-rendering pipeline. It is not billboards.

### The rendering operator is literally named "VolumetricEmitter"

The particle operator that draws it is `C_OP_RenderVolumetricEmitter` **[V]**
([schema](https://s2v.app/SchemaExplorer/cs2/particles/C_OP_RenderVolumetricEmitter),
[header](https://github.com/SteamTracking/GameTracking-CS2/blob/master/DumpSource2/schemas/particles/C_OP_RenderVolumetricEmitter.h)).
Its fields are a fluid-simulation interface, not a sprite interface:

| Field | Friendly name | Meaning |
|---|---|---|
| `m_nType` | interaction type | `PARTICLE_VOLUMETRIC_SMOKE_TYPE_EMISSION` / `_SINK` / `_REPEL` / `_TRACE` |
| `m_nCreationType` | creation type | `_CONTINUOUS` ("Continuous Injection From Particles") / `_IMPULSE` ("Instant Impulse On Each Particle Event") |
| `m_strChannelType` | channel type | `VDataChoice( scripts/volumetric_channels.vdata )` |
| `m_vecPos`, `m_vecVelocity`, `m_vPrevPosition` | position / velocity / prev position | injection & tracing |
| `m_flRadius`, `m_flDensity`, `m_flTemperature` | radius / density / temperature | volume parameters |
| `m_flMagnitude`, `m_flFalloff` | magnitude / falloff | shaping |
| `m_flKillRadius`, `m_flKillDensityScale` | kill radius / kill density scale | **erasing** volume |

And the interaction enum **[V]**
([`ParticleVolumetricSmokeType_t.h`](https://github.com/SteamTracking/GameTracking-CS2/blob/master/DumpSource2/schemas/particles/ParticleVolumetricSmokeType_t.h)):

```cpp
enum ParticleVolumetricSmokeType_t : uint32_t
{
    PARTICLE_VOLUMETRIC_SMOKE_TYPE_EMISSION = 0,  // "Emission"  -> the smoke grenade
    PARTICLE_VOLUMETRIC_SMOKE_TYPE_SINK     = 1,  // "Sink"      -> HE grenade eats the cloud
    PARTICLE_VOLUMETRIC_SMOKE_TYPE_REPEL    = 2,  // "Repel"     -> displacement
    PARTICLE_VOLUMETRIC_SMOKE_TYPE_TRACE    = 3,  // "Trace"     -> bullets carve a path
};
```

This is a **complete, mechanically-grounded explanation of the gameplay** that community guides only
describe behaviourally: the HE-grenade "hole" is a `Sink` volume with a `kill radius`; the
bullet "holes" are `Trace` volumes; the smoke itself is a `Continuous` `Emission` volume.

### Is CS2 smoke a SPHERE for visibility/occlusion? — **No.**

**Direct answer:** In CS2 the smoke is **not** a sphere, and there is no engine-accurate
"smoke radius" constant to hard-code. Occlusion is evaluated against the **voxel volume**
(`m_VoxelFrameData`), which is why the same smoke can be tall in a stairwell, long in a corridor, and
dome-shaped only in the open **[V]**.

**The exact numbers that *are* public** are client-side shape/debug parameters. Note every one of these
is flagged `developmentonly` in Valve's own dump, so treat them as *the shape the client's debug/legacy
path uses*, not as a guaranteed gameplay occluder **[V]**:

| Convar | Default | Interpretation |
|---|---|---|
| `cl_smoke_origin_height` | **68** units | vertical offset of the smoke body from the detonation point |
| `cl_smoke_torus_ring_radius` | **61** units | ring radius of the base shape |
| `cl_smoke_torus_ring_subradius` | **88** units | tube (sub)radius of the base shape |
| `cl_smoke_edge_feather` | **21** units | edge softening width |
| `cl_smoke_lower_speed` | `1` | sink rate |
| `bot_max_visible_smoke_length` | **200** units | "Bots will see players through smoke clouds up to this length" |
| `smoke_param1` … `smoke_param5` | `6.26`, `8.27`, `0.13`, `0`, `0` | undocumented density/noise shaping parameters |

**A useful derived number [J]:** ring radius + subradius = `61 + 88 = 149` units outer extent. If the
intended base shape is a torus, its outer radius is ~149 units. This is *consistent* with the
long-repeated community figure of a ~144-unit smoke radius (≈288-unit diameter ≈ 5.5 m, which is a
plausible real-world smoke-grenade diameter). **However**, I could not find a primary Valve source for
"144", and the CS2 mechanism is voxel-based, so **144 should be treated as a modelling approximation for
a fallback renderer, never as engine truth.**

> **Practical consequence for AI training:** if you need smoke occlusion geometry for an RL/IL
> environment, the *correct* move is to decode `m_VoxelFrameData` and voxelize it. Falling back to a
> 144-unit sphere introduces a systematic error exactly where CS2 differs most from CS:GO — corners,
> doorways, stairs, and partial smokes — which are precisely the situations AI agents need to learn.

### Duration

| Quantity | Value | Source |
|---|---|---|
| Nominal lifetime from detonation | **18 s** (≈17 s usable cover; final 2–3 s thins at the edges) | **[C]** [swap.gg](https://swap.gg/blog/how-long-does-smoke-last-in-cs2) |
| Radar-block duration | ~**15 s** | **[C]** same, citing the CS wiki |
| `smokegrenade_detonate` → `smokegrenade_expired` | **1412 ticks = 22.06 s** @ 64 tick, n=137 runs, **min = median = 1412** (max 124703 is a cross-round pairing artifact) | **[M]** `notes/scratch/tickrate_result.json` |
| Entity bloom-in | ~1 s to fully deploy | **[C]** swap.gg |

**Reconciliation [J]:** these are not contradictory. 18 s is the *dense-coverage plateau*; 1412 ticks is
the *entity lifetime* (bloom-in ≈1–2 s + 18 s plateau + 2–3 s fade ≈ 22 s). **For replay purposes treat
1412 ticks as the ground-truth span** — it is what the demo actually says — and 18 s as the window in
which the smoke is reliably opaque. The measured value is extremely stable (min = median), so it is a
solid constant.

### Colours — cross-validated

| Side | Convar default **[V]** | Our demo measurement **[M]** |
|---|---|---|
| T | `smoke_grenade_t_color [180, 129, 50]` | `m_vSmokeColor = [180.0, 129.0, 50.0]` |
| CT | `smoke_grenade_ct_color [75, 127, 155]` | (demo was T-side) |

This is a clean independent confirmation that our demo pipeline reads real values and that Valve's
convar defaults match shipped behaviour. Smoke converges to grey over its lifetime, which is a usable
visual timer **[C]**.

---

## 1.2 Molotov / incendiary — the inferno footprint

### The `m_fireXDelta` premise is a CS:GO fact, not a CS2 fact

This is the most important correction in this report. The layout the brief asks about — `m_fireXDelta` /
`m_fireYDelta` / `m_fireZDelta`, 64 slots — **is real, but it is CS:GO's**. The CS:GO netprop dump
shows **[C]** ([gamesense-lua `DT_Inferno`](https://gamesensical.gitbook.io/docs/developers/netprops/other/cinferno), fetched and text-extracted):

```
DT_Inferno
  ...
  m_fireXDelta          (integer[0-63])
  m_fireYDelta          (integer[0-63])
  m_fireZDelta          (integer[0-63])
  m_bFireIsBurning      (integer[0-63])
  m_fireCount           (integer)
  m_nFireEffectTickBegin(integer)
```

So in CS:GO the client received **64 integer offsets** relative to `m_vecOrigin`, not absolute
positions — a bandwidth optimisation. Note what this implies: the deltas are **arbitrary integer Hammer
units**, *not* a fixed 25- or 30-unit grid pitch. The "25 or 30 units" figure cannot be read out of the
delta layout.

### CS2's actual `CInferno` layout

CS2 replaced the deltas with world-space float vectors **[V]**
([`CInferno.h`](https://github.com/SteamTracking/GameTracking-CS2/blob/master/DumpSource2/schemas/server/CInferno.h),
[schema page](https://s2v.app/SchemaExplorer/cs2/server/CInferno)):

```cpp
// server — CInferno : CBaseModelEntity
VectorWS[64]  m_firePositions;              // 0x0770 — absolute world positions
VectorWS[64]  m_fireParentPositions;        // 0x0A70 — the surface point each cell came from
bool[64]      m_bFireIsBurning;             // 0x0D70
Vector[64]    m_BurnNormal;                 // 0x0DB0 — surface normal per cell
int32         m_fireCount;                  // 0x10B0 — how many of the 64 are live
int32         m_nInfernoType;               // molotov vs incendiary
int32         m_nFireEffectTickBegin;
float32       m_nFireLifetime;
bool          m_bInPostEffectTime;
bool          m_bWasCreatedInSmoke;
Extent        m_extent;
CountdownTimer m_damageTimer;
CountdownTimer m_damageRampTimer;           // damage ramps up over the fire's life
Vector        m_splashVelocity;
Vector        m_InitialSplashVelocity;
VectorWS      m_startPos;
VectorWS      m_vecOriginalSpawnLocation;
IntervalTimer m_activeTimer;
int32         m_fireSpawnOffset;
int32         m_nMaxFlames;
int32         m_nSpreadCount;
CountdownTimer m_BookkeepingTimer;
CountdownTimer m_NextSpreadTimer;
uint16        m_nSourceItemDefIndex;
```

**Array capacity is 64; concurrency is capped far lower** (`inferno_max_flames 16`). Do not assume
`m_fireCount == 64`.

The **client** `C_Inferno` **[V]** additionally carries the render-side data, which tells you how the
client draws it:

```cpp
// client — C_Inferno : C_BaseModelEntity
ParticleIndex_t m_nfxFireDamageEffect;
CStrongHandle<InfoForResourceTypeIParticleSnapshot> m_hInfernoPointsSnapshot;            // main flame points
CStrongHandle<...> m_hInfernoFillerPointsSnapshot;                                       // fills gaps
CStrongHandle<...> m_hInfernoOutlinePointsSnapshot;                                      // outline
CStrongHandle<...> m_hInfernoClimbingOutlinePointsSnapshot;                              // fire climbing walls
CStrongHandle<...> m_hInfernoDecalsSnapshot;                                             // scorch decals
VectorWS[64] m_firePositions;  VectorWS[64] m_fireParentPositions;
bool[64] m_bFireIsBurning;     Vector[64] m_BurnNormal;
int32 m_fireCount;  int32 m_nInfernoType;  float32 m_nFireLifetime;
bool m_bInPostEffectTime;  int32 m_lastFireCount;  int32 m_nFireEffectTickBegin;
int32 m_drawableCount;  bool m_blosCheck;  int32 m_nlosperiod;
float32 m_maxFireHalfWidth;  float32 m_maxFireHeight;
VectorWS m_minBounds;  VectorWS m_maxBounds;  float32 m_flLastGrassBurnThink;
```

So the fire is a **set of discrete points with per-point surface normals**, rendered by particle
*snapshots* (points / filler points / outline points / climbing outline / decals). That is a point-cloud
simulation with several cosmetic particle layers on top — very good news for reimplementation, because
the points themselves are the gameplay-relevant data and they are in the demo.

### The complete inferno convar set (authoritative)

All values below are from Valve's own dumped convar list **[V]**
([`DumpSource2/convars.txt`](https://github.com/SteamTracking/GameTracking-CS2/blob/master/DumpSource2/convars.txt)),
retrieved and grepped locally:

| Convar | Value | Meaning |
|---|---|---|
| `inferno_flame_lifetime` | **7** | molotov/T burn duration (seconds) |
| `inferno_flame_lifetime_incendiary` | **5.5** | CT incendiary burn duration (seconds) |
| `inferno_flame_spacing` | **42** | "distance between separate flame spawns" (units) |
| `inferno_max_range` | **150** | max spread from impact point (T molotov), units |
| `inferno_max_range_ct` | **110** | max spread (CT incendiary), units |
| `inferno_max_flames` | **16** | max simultaneous flames from one grenade |
| `inferno_damage` | **40** | base damage amount |
| `inferno_damage_ct` | **40** | same for incendiary |
| `inferno_damage_timer` | **0.2** | damage applied every 0.2 s |
| `inferno_initial_spawn_interval` | **0.02** | initial bloom-out tick interval |
| `inferno_child_spawn_interval_multiplier` | **0.1** | child-spawn interval scaling |
| `inferno_child_spawn_max_depth` | **4** | recursion depth of flame spreading |
| `inferno_max_child_spawn_interval` | **0.5** | cap on child spawn interval |
| `inferno_per_flame_spawn_duration` | **3** | seconds each flame keeps trying to spawn children |
| `inferno_spawn_angle` | **45** | spawn cone (degrees) |
| `inferno_spread_speed_mult` | **1** | T spread speed multiplier |
| `inferno_spread_speed_mult_ct` | **10** | CT spread speed multiplier (incendiary blooms much faster) |
| `inferno_surface_offset` | **15** | offset above the surface (units) |
| `inferno_velocity_decay_factor` | **0.2** | splash velocity decay |
| `inferno_velocity_factor` | **0.003** | splash velocity scaling |
| `inferno_velocity_normal_factor` | **0** | normal-direction velocity contribution |
| `inferno_forward_reduction_factor` | **0.9** | forward-bias reduction |
| `inferno_friendly_fire_duration` | **6** | team-damage reporting window |
| `inferno_max_trace_per_tick` | **16** | trace budget/tick |
| `inferno_dlights` | **30** | dynamic lights emitted |
| `inferno_dlight_spacing` | **7200** | dynamic light spacing (units²) |
| `inferno_smoke_volume_density` | **0.03** | density the fire injects into the smoke volume |
| `inferno_batched_rays` | `true` | batched ray tracing |
| `Inferno_concav_plane_threshold` | `-10` | concavity threshold for spread |
| `inferno_fire` | `2` | client fire effect level |
| `inferno_ct_experiment` | `true` | CT-side behaviour toggle |
| `molotov_throw_detonate_time` | **2** | molotov airburst time (s) |

**Damage per tick:** base `40`, applied on a `0.2 s` timer, **scaled by `m_damageRampTimer`** (the
entity-level ramp) and by proximity/occlusion **[V]**. Do not naively compute 40 ÷ 0.2 = 200 DPS —
the ramp exists precisely to prevent that. Exact per-tick damage must be measured in-game; the
authoritative *structure* is: `every 0.2 s, damage = 40 × ramp(t)`, with the ramp encoded in the
`m_damageTimer` / `m_damageRampTimer` pair on the entity.

### Independent cross-validation of the burn durations **[M]**

Our own probe's inferno lifetime histogram (`notes/scratch/tickrate_result.json`) is bimodal:

| Ticks | Seconds @64 | Count | Matches |
|---|---|---|---|
| **352** | **5.500** | 42 | `inferno_flame_lifetime_incendiary = 5.5` ✔ |
| **449 / 450** | **7.016 / 7.031** | 32 + 7 | `inferno_flame_lifetime = 7` ✔ |
| 142 / 193 / 249 | 2.22 / 3.02 / 3.89 | 1 each | fires extinguished early (smoke / round end) |
| 110120 / 118282 | — | 1 each | cross-round pairing artifacts |

Two independent modes landing **exactly** on Valve's two convar values is strong evidence that the
measurement pipeline is sound and that these constants are the real ones. **For a replay you can model
fire duration as 5.5 s (incendiary) or 7.0 s (molotov), keyed off `m_nInfernoType`.**

### Cell spacing — the honest answer

- **Spacing between flame spawns: 42 units** (`inferno_flame_spacing`) **[V]**.
- **Total radius: 150 units** (T molotov) / **110 units** (CT incendiary) **[V]**.
- **Cell count: up to 16 simultaneously burning**, in 64-slot arrays **[V]**.
- **Fixed 25/30-unit grid pitch: not supported.** Neither CS2's `m_firePositions` nor CS:GO's integer
  deltas encode a fixed pitch. A 150-unit radius with 42-unit spawn spacing is consistent with roughly
  a 1 + 6 + 12-ish arrangement of cells depending on geometry — i.e. the positions are
  **geometry-conformant scatter, not a lattice**. This matters: a replay that draws a regular hex grid
  of fire will look wrong on stairs and around corners.

---

## 1.3 Getting fire cells and smoke voxels out of a `.dem`

### The blocker, measured **[M]**

`notes/scratch/inferno_probe.py` and `smoke_grenade.py` exhaustively tested `demoparser2 0.42.0`
(the current PyPI release) against `0823_inferno.dem`. Results:

| Attempt | Result |
|---|---|
| `list_updated_fields()` | `Grenade.m_firePositions`, `m_fireParentPositions`, `m_fireCount`, `m_nFireLifetime`, `m_nInfernoType`, `m_bFireIsBurning`, `m_vSmokeDetonationPos`, `m_nSmokeEffectTickBegin`, `m_bDidSmokeEffect` **are all present in the sendtable** |
| `parse_ticks(FIRE_PROPS)` | returns only `tick, steamid, name` — props silently dropped |
| `parse_event("inferno_startburn", other=["Grenade.m_fireCount"])` | event-native columns only |
| `parse_grenades(extra=FIRE_PROPS)` | **no columns added at all** |
| Control: `parse_event("round_end", other=["CCSTeam.m_iScore"])` | ✔ works — proves `other=` works and that `Grenade.*` specifically is not wired into the output channel |
| `m_VoxelFrameData` / `m_nVoxelFrameDataSize` | **present in sendtable** (187 833 non-null rows) but **DROPPED** from output |

The root cause was identified in `demoparser`'s `prop_controller.rs`: other-props are only registered
for `CCSGameRulesProxy / CCSTeam / CCSPlayerPawn / CCSPlayerController` or types explicitly mapped by
`TYPEHM`. `Grenade.*` array-typed props (`CUtlVector<Vector>`, `CNetworkUtlVectorBase<uint8>`) are not
supported by the output channel.

**So the data is in the file. The Python wrapper just won't hand it to you.**

### What you *can* get today, and it is more than nothing **[M]**

| Data | API | Fidelity |
|---|---|---|
| Per-tick projectile trajectory (all 5 grenade types) | `parse_grenades()` | 2.28 M rows incl. `grenade_entity_id`, thrower, bounces |
| `inferno_startburn` / `inferno_expire` with **exact impact x,y,z** | `parse_event()` | 85/85 paired in our demo |
| `smokegrenade_detonate` / `_expired` with x,y,z | `parse_event()` | 484 projectile runs |
| Detonation tick, throw strength, initial velocity, explode effect origin | `parse_grenades(extra=[...])` | survived |
| Player per-tick pos / yaw / pitch / health / flags | `parse_ticks()` | full |

**Critical gotcha [M]:** `grenade_entity_id` and `inferno` `entityid` are **reused across grenades**.
Joining on entity id produces wrong pairings (our naive join produced a bogus 124 703-tick "lifetime").
Segment into tick-runs (`gap > 1 tick` or thrower change ⇒ new run) or pair **chronologically**. This is
already solved in `notes/scratch/smoke_grenade.py` and should be reused.

### Recommended path to the real volumes

`demoparser2` is a dead end for these props. In rough order of expected effort:

1. **`demoinfocs-golang`** (Go) — exposes entities as generic property maps rather than a fixed
   player-centric schema, so array netprops like `m_firePositions` and `m_VoxelFrameData` are the most
   likely to come through unmodified. See
   [Entity and Property System](https://deepwiki.com/markus-wa/demoinfocs-golang/3.3-entity-management).
   **This is the recommended next experiment.** It is a single self-contained test: parse the demo,
   iterate `Grenade`/`CInferno` entities, print `Properties()` keys.
2. **Patch / vendor `demoparser` (Rust)** — extend `prop_controller.rs` to register `Grenade.*`, and add
   array-typed output support. Higher effort, keeps the Python API. Upstream issue would be welcome.
3. **`cs2-demoparser`** (Python/JS, [yickling/cs2-demoparser](https://github.com/yickling/cs2-demoparser))
   — a newer independent implementation; worth a look for array support.
4. **`awpy`** — tick data and nav-mesh work, but no evidence of fire-cell support
   ([docs](https://deepwiki.com/pnxenopoulos/awpy/2-demo-parsing)).

**Note the schema offsets are public [V]** (`m_firePositions` @ `0x0770`, `m_VoxelFrameData` @
`0x0AB0`, `m_nVoxelFrameDataSize` @ `0x0AC8`), so worst-case a bespoke entity decoder is tractable.

### Existing open-source projects that extract/reconstruct inferno fire cells

**Honest finding: I found none that extract CS2 fire cells from a demo.** The searches
`"cs2 inferno fire cells"`, `"m_firePositions"`, `"csgo molotov fire cells parser"`, `"CS2 inferno
entity"` surface only:
- the CS:GO-era netprop *reference* [`CInferno` on gamesense-lua](https://gamesensical.gitbook.io/docs/developers/netprops/other/cinferno) (documents the fields, does not extract them),
- the schema pages above,
- the convar documentation sites.

External tools that read these props do so from a **live game process** (cheat/overlay netprop reads),
not from a demo. **This is an open niche** — and it is exactly the niche this project occupies, which
makes the `m_firePositions`/`m_VoxelFrameData` extraction work genuinely novel and worth doing properly.

---

## 1.4 How CS2 renders smoke and fire (summary)

| Element | Mechanism | Evidence |
|---|---|---|
| **Smoke** | Server-simulated **volumetric density field**, networked as `m_VoxelFrameData` bytes, rendered client-side by `C_OP_RenderVolumetricEmitter` — a **low-res 3D volume raymarch, upsampled with a full-res hole-covering pass**. Not billboards. | **[V]** schema + cvars |
| Smoke shading | 3D noise texture (`smoke_use_noise_texture`), edge feathering (21 u), shadow casting (`r_csgo_smoke_shadow`), LOD (`smoke_volume_lod_ratio_change`) | **[V]** |
| Smoke gameplay interaction | Volume primitives: `Emission` (create), `Sink` (HE hole, via `kill radius`/`kill density scale`), `Trace` (bullet path), `Repel` | **[V]** |
| **Fire** | Discrete **fire cells** (`m_firePositions[64]` + `m_BurnNormal[64]`) simulated server-side, rendered as several particle **snapshots**: points, filler points, outline points, wall-climbing outline points, and scorch decals | **[V]** client schema |
| Fire lighting | Dynamic lights (`inferno_dlights 30`), batched rays (`inferno_batched_rays`) | **[V]** |
| Particles | Source 2 `.vpcf` KV3 systems; VRF round-trips these **exactly** and simulates "a large subset" of operators in its own viewer | **[VRF]** |

**Practical read for reimplementation.** Neither effect is a textured quad. Smoke needs a real
raymarch through a 3D density texture; fire needs a point-cloud with per-point normals plus decals.
Both are implementable in Three.js with a custom `ShaderMaterial`, and both get *exact positions and
timing* from the demo once the parser work in §1.3 is done. Public reverse-engineering write-ups exist
(Acerola's "reverse-engineering CS2 smoke" raymarching + SDF reconstruction is the best-known:
[bilibili mirror](https://www.bilibili.com/video/BV1Web86YEif/); UE5 reimplementations at
[morevfxacademy](https://morevfxacademy.com/counter-strike-2-smoke-in-unreal-engine-5/)), and are a
reasonable starting point for the shader.

---

# Part 2 — Real asset extraction pipeline (Source2Viewer / ValveResourceFormat)

## 2.1 Verified local state

```
~/Downloads/Source2Viewer-CLI/Source2Viewer-CLI   version 19.2.6339+c7220835  macOS 26.6.2 (Arm64)
```
**[M]** — run on this machine. Current upstream is **20.0.6980** (NuGet, 2026-08-17) **[VRF]**, so the
local binary is one major version behind.

**Verified discrepancy [M]:** the local `--help` does **not** list `--game` or
`--gltf_compose_additive`, both of which appear in current upstream docs/source. Practical impact:
you cannot point the installed build at a `gameinfo.gi` to resolve dependencies for a *loose* file.
**Update to the current release before scripting the pipeline**, or keep every input file inside a real
CS2 install tree / VPK.

**Still missing on this machine: the actual game VPK.** `steamapps/common/Counter-Strike Global
Offensive/` is a 2022 shell; there is no `pak01_dir.vpk` anywhere on disk **[M]**. The pipeline is
tool-complete and asset-blocked (also noted in `docs/PLAN.md`).

## 2.2 Exact CLI usage (verified against the local binary and VRF docs)

All flags below were confirmed present in the local `--help` output **[M]** and documented in the
[command-line guide](https://s2v.app/ValveResourceFormat/guides/command-line.html) **[VRF]**.
**`-d` is mandatory for any glTF export** — the CLI hard-errors with
`"Exporting to glTF requires specifying -d argument."` **[verified in [`CLI/Decompiler.cs`](https://github.com/ValveResourceFormat/ValveResourceFormat/blob/master/CLI/Decompiler.cs)]**.

### (a) Player model `.vmdl_c` → GLB with skeleton + animations + materials + textures

```bash
Source2Viewer-CLI \
  -i "…/Counter-Strike Global Offensive/game/csgo/pak01_dir.vpk" \
  -f  "agents/models/ctm_fbi/ctm_fbi.vmdl_c" \
  -o  ./out/ctm_fbi.glb \
  -d \
  --gltf_export_format glb \
  --gltf_export_animations \
  --gltf_export_materials \
  --gltf_textures_adapt
```
- `-f/--vpk_filepath` accepts comma-separated paths and folder prefixes.
- `-o` may name a **file** (not just a folder) only when the input is a single file **or** `--vpk_filepath`
  matches **exactly one** file **[VRF]** — which is why `-f` is pinned to one model above.
- `--gltf_textures_adapt` performs glTF spec adaptations (e.g. splitting the metallic map). **`--gltf_export_materials` + `--gltf_textures_adapt` together are what you want** for a usable PBR result.
- `--gltf_animation_list "idle,walk,run"` restricts clips; omit to take all.
- `--gltf_mesh_list "mesh1,mesh2"` restricts meshes; omit to take all.
- **Use `.gltf`, not `.glb`, for anything large** — the glTF spec caps GLB at 2 GB **[VRF]**.

Loose file instead of VPK (needs a current build for `--game`):
```bash
Source2Viewer-CLI -i ./ctm_fbi.vmdl_c -o ./out/ctm_fbi.glb -d \
  --game "/…/game/csgo/gameinfo.gi" \
  --gltf_export_format glb --gltf_export_animations --gltf_export_materials --gltf_textures_adapt
```

### (b) Prop / any model → GLB

Identical, with the prop path. Useful CS2 paths **[VRF]**: `models/` (props, vehicles, foliage),
`weapons/models/` (weapon worldmodels and viewmodels), `agents/models/` (player/agent models).

### (c) Map / world mesh → glTF with real materials

**Documented (GUI) workflow [VRF]** — this is the *supported* path:
> Open the map's VPK file in Source 2 Viewer → navigate to `maps/<map>.vmap_c` → right-click →
> **Decompile & Export** → choose **glTF** (GLB also offered but 2 GB-limited) → Save.

Output contains: map geometry (brushes and meshes), textures as **PNG files alongside the glTF**, props
and models placed in the map, and **basic** material information. For a typical CS2 competitive map
expect **hundreds of texture files** **[VRF]**
([exporting maps](https://s2v.app/ValveResourceFormat/guides/exporting-maps.html)).

**Batched/CLI [VRF]:** the docs' map section gives only
```bash
Source2Viewer-CLI -i "maps/de_dust2.vpk" -o "exported/" -d
```
which decompiles to `.vmap`. The [format-support matrix](https://s2v.app/ValveResourceFormat/guides/format-support.html)
does list **`vmap` → `glTF/GLB`** and **`vwrld` → `.vmap` (Hammer); `glTF/GLB`**, and the CLI wires
`--gltf_export_format` into `GltfModelExporter` behind a `CanExport(resource)` gate. So:
```bash
Source2Viewer-CLI -i "…/game/csgo/maps/de_mirage.vpk" -o ./out_de_mirage -d \
  --gltf_export_format gltf --gltf_export_materials --gltf_textures_adapt --threads 8
```
**Caveat [J]:** CLI *map* glTF export is not explicitly documented — the docs only demonstrate `-d`
(vmap) and reserve the GUI for glTF. **Treat the above as expected-to-work-but-verify**, and fall back to
the GUI if the CLI silently emits only `.vmap`. Maps are stored one-per-VPK in the game's `maps/`
directory, and some have a truncated `_vanity` variant used for main-menu backgrounds — **use the
version without the suffix** **[VRF]**.

### (d) `.vtex_c` → PNG

```bash
Source2Viewer-CLI -i "materials/models/player/ct_swat/ct_swat_upperbody_color.vtex_c" -o ./out.png \
  --texture_decode_flags auto
```
The docs' own example is exactly `-i "file.vtex_c" -o exported.png` **[VRF]**. `--texture_decode_flags`
accepts `none` / `auto` / `ForceLDR` (default `auto`) **[M: local help]**.
**Limitation:** file extraction always takes the **largest mip only** — the mip chain has no
single-file-preserving export path **[VRF]**. HDR textures go to EXR; cubemaps export one image per
face; LDR cubemaps can be combined to equirectangular.

### (e) Print a `.vpcf` particle

```bash
# dump every resource block as KV3 text
Source2Viewer-CLI -i "particles/smoke_grenade.vpcf_c" --all
# or just the data block
Source2Viewer-CLI -i "particles/smoke_grenade.vpcf_c" -b DATA
# decompile to an editable .vpcf (exact round-trip for KV3-era files)
Source2Viewer-CLI -i "particles/smoke_grenade.vpcf_c" -o ./out.vpcf -d
```
**[VRF]** — `vpcf` is listed as "Dump: yes; Decompiles/exports to: `.vpcf` (exact round-trip for KV3-era
files)"; particle extraction re-emits the original stored KV3 tree, not VRF's upgraded in-memory version.

Also useful for a *census* of what CS2 actually uses:
```bash
Source2Viewer-CLI -i "…/pak01_dir.vpk" --stats --stats_particles
```
collects particle **operators, renderers, emitters, initializers** across the archive **[M: local help]**,
and
```bash
Source2Viewer-CLI -i "…/pak01_dir.vpk" --vpk_list -e "vpcf_c" -f "particles/"
```
lists matching resources. Other handy inventory flags **[M]**: `--vpk_dir` (with metadata),
`--stats_vbib` (vertex attributes), `--dump_unknown_entity_keys`.

## 2.3 Does VRF export Source 2 materials into glTF correctly?

**No — they are deliberate approximations, and VRF documents this precisely.** From
[Format Support and Export Fidelity](https://s2v.app/ValveResourceFormat/guides/format-support.html) **[VRF]**,
glTF models/maps export with:

> "meshes, up to 8 bone weights per vertex, morph targets, skeletal/additive/morph animations,
> **PBR-approximated materials with ORM repacking**, and collision shapes as a separate
> `<name>_physics` file."

| Source 2 feature | glTF export result | Why |
|---|---|---|
| Complex shader graph | **Flattened to glTF PBR metallic-roughness** | Format limitation |
| Layered blend materials | **Only ONE layer exported** ([#1161](https://github.com/ValveResourceFormat/ValveResourceFormat/issues/1161)) | Format limitation |
| Cloth sheen | No accurate glTF equivalent ([#576](https://github.com/ValveResourceFormat/ValveResourceFormat/issues/576)) | Format limitation |
| Vertex-colour-driven shaders (foliage) | **Do not translate** ([#1180](https://github.com/ValveResourceFormat/ValveResourceFormat/issues/1180)) | Format limitation |
| Original shader identity | **Preserved** in a `vmat` **extras** block on every exported material | — |

**Mitigation worth using [VRF]:** the `vmat` extras block carries "the original shader name and
parameters" — so a custom Three.js material layer can branch on shader name and re-implement the
important cases (detail textures, tint masks, self-illum, blend layers) instead of accepting the flat
PBR. Also note VRF can **decompile `.vmat` → `.vmat` + unpacked texture inputs**, recovering shader name,
all int/float/vector params including `F_` feature choices, and dynamic expressions **decompiled from
bytecode back to readable text** — that is a genuinely rich source for hand-porting materials.

**On maps specifically, material fidelity is the weak point:** CS2 surfaces lean heavily on multi-layer
blends, and only one layer survives. Expect visibly flatter, more uniform surfaces than the game.

## 2.4 What is lost in glTF export (models) — complete list

### Not exported at all
| Lost | Reason |
|---|---|
| **Lower LODs** | Intentional — only the highest-detail LOD is exported ([#535](https://github.com/ValveResourceFormat/ValveResourceFormat/issues/535)) |
| Hitboxes | glTF has no collision-volume concept |
| **Cloth simulation** | No cloth solver; cloth bones are **rigidly pinned** to anchor bones |
| Morph **wrinkle weights** | Position/normal deltas export; wrinkle weight has no glTF equivalent |
| Additive animations | No additive concept; composed over bind pose, or delta tracks via a VRF-specific extras convention **that generic viewers play back incorrectly** |
| Physical properties (mass, friction, joints) | Only visualization geometry + per-shape surface property/collision tags as node extras |

### Recoverable only via the decompiled `.vmdl` route, or partially
| Lost | Reason |
|---|---|
| Cloth simulation data | Partly not in compiled files; `FeModel` cloth block not parsed ([#653](https://github.com/ValveResourceFormat/ValveResourceFormat/issues/653)) |
| Physics constraints / joint motors | `m_constraints2` not parsed |
| Animations from external animation groups | Only embedded sequences and Animgraph 2 clips get DMX; sequences in referenced `vagrp` files are **skipped** |
| **Blend sequences (blend spaces)** | **Collapse to their first referenced animation** |
| Extra skin materials | Extras beyond the default group silently dropped |
| Stereo flex controls | Compiler split them; come back as two sliders |
| **Vertical root motion** | **Intentionally zeroed** on export |
| Additional external physics files | Only the first `m_refPhysicsData` reference is extracted |

### What *does* survive well **[VRF]**
Render meshes with all vertex streams, skeleton, attachments, bodygroups, LOD groups, hitbox sets,
material groups (skins), static collision shapes, physics joints and body properties, bone constraints,
IK chains and control rigs, face flexes, breakable pieces, embedded sequences with events/layers/root
motion, Animgraph 2 clips and references, and prop_data / particle attachments. Rigging survives with
**up to 8 bone weights per vertex**.

## 2.5 Are CS2 player models rigged with usable animation sequences in the exported GLB?

**Partly — and the honest answer is "not the ones you need, on the path you'd expect."**

What VRF gives you **[VRF]**:
- Legacy embedded sequences decode through all common per-bone compression types (unknown decoder types
  are silently skipped).
- Animgraph 2 clips (`vnmclip`) decode fully: compressed poses, 3D root motion (position + **yaw only** —
  root pitch and roll are dropped), float curves, events, secondary skeleton tracks.
- Both feed **viewer playback, glTF export, and DMX reconstruction**, including **retargeting** of clips
  authored on a different skeleton.

The critical limitation **[VRF]**:
> "Animation **graphs** are decompiled to editable documents and visualized, for both Animgraph 1 and
> Animgraph 2, but are **never executed**: the viewer plays explicitly selected clips, not what the
> graph's state machine would choose. **Graph-driven behavior (IK solving, bone constraint evaluation
> such as constraint-driven eyelid morphs [#756]) does not affect playback or exports.**"

**So the exported GLB contains a rig and a bag of clips, but no state machine, no blending logic, no IK
and no constraints.** Your replay code must implement locomotion selection, blend spaces, additive
layering and turn-in-place itself. This is a substantial, easily-underestimated chunk of work, and it is
the single largest *character* fidelity gap.

### Typical sequence names — measured, not guessed

I downloaded and parsed a publicly hosted agent GLB to get real numbers instead of relying on memory
**[M]** (`rabbytesoftware/cs2.agents`, `agents/ctm_swat_variantd.glb`, 1 758 272 bytes):

```
meshes: 2   skins: 1 (83 joints)   nodes: 86   materials: 4   images: 4   textures: 4
animations: 7
  - celebrate_drop_down_noweap_idle01
  - celebrate_drop_down_noweap_idle02
  - celebrate_drop_down_noweap_idle03
  - celebrate_drop_down_noweap_idle04
  - celebrate_drop_down_noweap_start
  - ~An7xw29E/legacy/ctm_swat_variantd_anims/default
  - ~An7xw29E/legacy/ctm_swat_variantd_anims/ragdoll
```

Two things to take from this:
1. **The names are opaque.** VRF emits the raw clip/animset names, including compiler-mangled names
   (`~An7xw29E/legacy/...`). There is a `legacy/..._anims/default` entry that appears to be an entire
   legacy animset packed into one clip. **You must inventory and map clip names yourself** — do not
   expect `idle`/`run`/`walk`.
2. **83 joints.** That is a CS:GO-era rig. CS2's canonical Animgraph-2 skeleton is **74 bones rooted at
   `root_motion`** (established independently in this repo's `NOTES.md`). Mixing the two requires
   retargeting, which VRF supports but which adds a failure mode.

**Recommendation:** export agents from CS2's own `agents/models/*.vmdl_c` and enumerate the animation
list **first** (`--gltf_animation_list` forces you to know the names; run once without it and read the
GLB's `animations[].name`), rather than adopting a public CS:GO-era GLB.

## 2.6 Licensing / ToS reality

**Short version: extracting and rendering Valve assets for private, non-commercial research is
tolerated; redistributing them is not.**

| Source | What it says |
|---|---|
| [Steam Subscriber Agreement](https://store.steampowered.com/subscriber_agreement/) §2.A | Grants only "a non-exclusive license and right, to use the Content and Services for your **personal, non-commercial use**"; "The Content and Services are **licensed, not sold**." |
| SSA §2.G | You may not "copy… reproduce, publish, distribute… reverse engineer, derive source code from, modify, disassemble, decompile, create derivative works based on… the Content and Services" without prior written consent, nor "exploit the Content and Services or any of its parts for **any commercial purpose**." |
| SSA §2.C (Developer Tools), §2.D (Fan Art) | Both allow distributing what you create **"solely on a non-commercial basis"**. Commercial Source SDK use routes to `sourceengine@valvesoftware.com`. |
| [Valve Video Policy](https://store.steampowered.com/video_policy) | "We're not fine with taking assets from our games (e.g. voice, music, items) and **distributing those separately**." Videos and YouTube-partner monetisation of *gameplay* are explicitly fine. |
| [Steam Online Conduct](https://store.steampowered.com/online_conduct/Steam?l=english) | Prohibits "uploading materials that you do not own the rights to." |
| [CS2 map icon repo](https://github.com/MurkyYT/cs2-map-icons) | States plainly that radar/icon/overview data "are property of Valve Corporation." |
| [VRF](https://github.com/ValveResourceFormat/ValveResourceFormat) | **Ships no Valve assets** — MIT, except `Tests/Files` which "contains files that have likely come from Valve's games". |

**Three buckets:**

- ✅ **Clearly permitted:** local, private, non-commercial extraction and rendering you perform yourself
  on assets you own via Steam; non-commercial video/streams of the result.
- ⚠️ **Gray:** decompilation/reverse engineering is nominally barred by §2.G, yet it is the entire
  tolerated basis of VRF and SourceIO, and VRF is not merely tolerated but widely used and MIT-licensed.
  Also gray: the **MIT licence on a repo full of Valve models** (e.g. `cs2.agents` is MIT — but that MIT
  covers the *porter's* contribution, not Valve's underlying copyright; you cannot treat those GLBs as
  MIT-licensed art).
- ❌ **Clearly not permitted:** publicly hosting or serving extracted Valve meshes/textures — a CDN of
  `de_dust2.glb`, or committing exported maps/agents into a public repo — and any commercial use.

**Design implication (important, and it constrains the architecture):** ship **code plus a user-side
extraction step**, never hosted Valve geometry. For a public demo, fall back to radar PNGs or a
procedurally authored map. This is the same conclusion the repo's `docs/PLAN.md` reaches for other
reasons.

---

# Part 3 — Getting the real CS2 look without the game engine

## 3.1 Is there an open-source Source 2 renderer? — Yes, and it is better than expected

### `ValveResourceFormat.Renderer` — the strongest option

**[VRF]** [NuGet: ValveResourceFormat.Renderer 20.0.6980](https://www.nuget.org/packages/ValveResourceFormat.Renderer/),
**MIT**, **.NET 10.0**, depends on `OpenTK.Graphics 4.9.4`. Self-description:

> "**OpenGL-based rendering engine for Source 2 game assets.** Powers the Source 2 Viewer."
>
> - **3D Models & Maps** — renders compiled models (`.vmdl_c`), maps (`.vwrld_c`, `.vmap_c`)
> - **Physically-Based Rendering** — full PBR pipeline with metalness/roughness workflow
> - **Advanced Lighting** — dynamic shadows, **lightmaps**, environment maps, **light probes**, fog systems
> - **Animation** — skeletal animation, morph targets, flex animation
> - **Particle Systems** — particle system with many operators supported
> - **Materials & Shaders** — material system supporting complex shader features
> - Debug render modes (normals, UVs, lightmaps, wireframe)

This is significant: **it renders the things glTF export destroys** — Source 2's real material shaders,
baked lightmaps, light probes, fog, and particles. It is the only public renderer I found that renders a
Source 2 map close to how the engine does.

**Can it be driven by external state (i.e. could it play back a demo)?** Encouragingly, yes in
principle. The [API docs](https://s2v.app/ValveResourceFormat/api/ValveResourceFormat.Renderer.html)
expose `WorldLoader`, `EntitySystem`, `Entities.PlayerEntity`, `Input.UserInput` /
`PlayerMovement`, `Entities.CS2Projectile`, `LogicRelay`, `IPlayerController`, `CameraPlacement` — a
real library API, not just a GUI. **But demo parsing is explicitly out of scope**

- [issue #1122 "Support for .dem replay files"](https://github.com/ValveResourceFormat/ValveResourceFormat/issues/1122)
  is open, and [format-support](https://s2v.app/ValveResourceFormat/guides/format-support.html) states
  "Parsing network/demo formats such as `.dem` is out of scope". So **you feed it ticks yourself** —
  which is a clean separation of concerns and exactly the shape of the problem.

**Caveats [VRF]:** the **GUI is Windows-only**; the CLI is cross-platform; the renderer library itself
targets .NET 10 and is OpenGL/OpenTK-based, so macOS/Linux should be feasible but is not the tested
path. The package README carries an explicit breaking-change warning (no API stability guarantee). **No
JS/WASM build exists.**

**Strategic read [J]:** if the goal is *maximum visual fidelity* and the audience can run a desktop app
or a local server, **driving `ValveResourceFormat.Renderer` from parsed demo state is a strictly better
fidelity proposition than Three.js** — you keep lightmaps and real materials. Three.js wins only on
*browser deliverability*. This is worth a deliberate decision rather than defaulting to the browser.

### Other renderer-adjacent projects

| Project | What it is | Verdict |
|---|---|---|
| [SourceIO](https://github.com/REDxEYE/SourceIO) — Blender addon | Imports Source 2 `.vmdl`/`.vmap`/`.vtex`/`.vmat`. CS2 = "Partial support: maps (not all entities), textures, materials (not all shaders)" | **Import-only DCC path.** No real-time renderer. License not stated in README *(uncertain)* |
| [Facepunch/sbox-public](https://github.com/Facepunch/sbox-public) | Source 2-based engine; engine source MIT but native `game/bin` binaries under the s&box EULA | Needs Steam + engine download; not a CS2-asset player |
| [StrataSource/Engine](https://github.com/StrataSource/Engine) | ⚠️ **FALSE POSITIVE** — a community **Source 1** branch (Portal 2: CE, Momentum Mod), not Source 2 | Not applicable |
| [Open-S2/s2maps-gpu](https://github.com/Open-S2/s2maps-gpu) | ⚠️ **FALSE POSITIVE** — a MapLibre-style WebGPU **vector map** engine; "S2" here means S2 geometry, not Source 2 | Not applicable |

**Negative result: no mature Godot or Unreal Source 2 importer exists.** Anyone claiming a "Source 2 in
Godot/Unreal" pipeline is doing it by hand.

## 3.2 Public glTF/GLB conversions of CS2 maps — none that are legitimate

**Honest finding: there is no legitimately public converted CS2 (or CS:GO) map GLB.** After repeated
searching, the closest things are:

| Resource | What it actually is |
|---|---|
| [MurkyYT/cs2-map-icons](https://github.com/MurkyYT/cs2-map-icons) | **2D** radar/icon/thumbnail PNGs scraped from the depot; states they are Valve property. Useful for a 2D fallback viewer. |
| [StarKnightt/counter-strike-in-browser](https://github.com/StarKnightt/counter-strike-in-browser) | Ships a 34 MB `dust2_a.glb` (85 MB source) — but it is **procedurally authored in Blender** with Poly Haven CC0 textures and explicitly states **"No Valve assets"**. Read the label before citing. |
| [rabbytesoftware/cs2.agents](https://github.com/rabbytesoftware/cs2.agents) | MIT-licensed repo of **CS:GO-era agent** GLBs (`ctm_*`, `tm_*`), ~1.75–1.87 MB each, web-compressed, built for the .ROUNDS project. **Not maps, not CS2 models.** MIT covers the porter's work, not Valve's copyright (§2.6). |

**So the pipeline must be: ship `Source2Viewer-CLI` invocation + let the user extract on their own
machine.** Sizes are not really the constraint — legality is. For calibration: a CS2 competitive map
glTF export is *hundreds of textures plus geometry*, and GLB cannot exceed 2 GB, so **you must use
`.gltf` + separate textures** for maps regardless **[VRF]**.

## 3.3 Most practical way to get a `de_mirage`/`de_inferno` that looks like the real game into Three.js

Ranked by (fidelity × effort × legality), with the honest caveat that **nothing reaches "looks like the
real game"**:

**Option 1 — Real map, Three.js, own lighting. (Recommended if browser delivery is required.)**
1. User extracts locally: `Source2Viewer-CLI -i maps/de_mirage.vpk -o ./out -d --gltf_export_format gltf --gltf_export_materials --gltf_textures_adapt --threads 8` (verify CLI map-glTF per §2.2c; GUI is the guaranteed path).
2. Ship `.gltf` + textures (never GLB for maps). Convert textures to KTX2/Basis for browser delivery size.
3. Load in Three.js with `GLTFLoader`; re-implement the *few* material features that matter by branching
   on the `vmat` extras shader name (§2.3). Expect to hand-write detail-texture/tint-mask support —
   this is where most of the "looks like CS2" comes from.
4. **Bake your own lighting.** No lightmaps are exported, and only `light_environment` comes across as a
   light. Either bake lightmaps in Blender from the exported glTF, or accept ambient + a directional light.
5. Extract the map's `.vpost_c` and apply its **LUT** as a post pass — VRF exports `.vpost` + LUT `.raw`
   **[VRF]**. This is a cheap, high-impact win because CS2's grade is a big part of its identity.
6. Accept that the 3D skybox is missing ([#967](https://github.com/ValveResourceFormat/ValveResourceFormat/issues/967)) — see §4.2.

**Option 2 — Drive `ValveResourceFormat.Renderer` directly. (Recommended if fidelity outranks browser
delivery.)** You keep real materials, lightmaps, light probes, fog and particles, at the cost of a
.NET desktop/local-server app and no browser build. Feed it parsed ticks yourself.

**Option 3 — Procedural/approximate map.** Keep the grey model (or a CC0-authored map) and invest in
lighting, fog, skybox and post to evoke CS2. Lowest fidelity, zero legal exposure, works today with no
game files — this is the repo's current "route C".

**Option 4 — Radar-image 2.5D.** Use the 2D radar PNGs ([cs2-map-icons](https://github.com/MurkyYT/cs2-map-icons))
with per-player markers. This is what literally every shipping web viewer does (§3.4). Not 3D, but
*exactly* the representation most AI/analysis work actually consumes.

## 3.4 Projects that have attempted a game-accurate web replay

**Honest finding: nobody has rendered a web 3D scene from real extracted CS2 map geometry.** Everything
practical is 2D radar:

| Project | Stack | Fidelity | License |
|---|---|---|---|
| [zenojunior/cs2d](https://github.com/zenojunior/cs2d) | Vue 3 + Rust/WASM parser, radar PNGs from cs2-map-icons, 100 % offline | 2D radar | — |
| [sparkoo/csgo-2d-demo-viewer](https://github.com/sparkoo/csgo-2d-demo-viewer) | Go/WASM parser + Preact, live at 2d.sparko.cz | 2D radar | — |
| [Igor-Losev/deadem](https://github.com/Igor-Losev/deadem) | **MIT** TypeScript; parses *and plays back* CS2/Dota2/Deadlock demos in Node/browsers, web UI at deadem.com | **Replay state only — no 3D world.** Best available JS replay-state source to reuse | MIT |
| [jackgraver/cs2c](https://github.com/jackgraver/cs2c), [Sylixa/cs2demoview](https://github.com/Sylixa/cs2demoview), [shuji-koike/dem](https://github.com/shuji-koike/dem) | various | 2D | — |
| [pnxenopoulos/awpy](https://github.com/pnxenopoulos/awpy) | Python | Tick data, nav-mesh distances, matplotlib heatmaps/GIFs. Not 3D, not web | MIT |
| [reka-ai/cs2-dem-renderer](https://github.com/reka-ai/cs2-dem-renderer) | **Launches real CS2** with a server plugin, pipes TGA → ffmpeg | **Real engine pixels.** Its browser "viewer" is synced video + 2D minimap, not 3D | — |
| FACEIT 2D Replay, [MemorIN](https://chromewebstore.google.com/detail/memorin-cs2-demo-viewer/gfnedekjjjpencchajbhmdbapcmgcdhm) | browser extensions | 2D | — |

**Implication:** the 3D-with-real-assets approach is unclaimed territory. That is both an opportunity and
a warning — there is no reference implementation to copy, and the reason is probably that the fidelity
ceiling is lower and the asset-legal path is awkward.

**Note `reka-ai/cs2-dem-renderer`:** it solves the fidelity problem honestly by *not* reimplementing the
engine — it records the engine. If the goal is synthetic ground-truth observations, **this is the
highest-fidelity option available and should be the reference baseline**, even though it is
non-interactive and Windows/CS2-install-dependent.

---

# Part 4 — Fidelity gap analysis

Setup: Three.js, real VRF-extracted geometry/materials/textures, our own smoke/fire shaders, demo-driven
positions. Everything below is what still will **not** match.

## 4.1 Complete gap table

| # | Dimension | Real CS2 | Three.js + extracted assets | Expected match |
|---|---|---|---|---|
| 1 | **Map geometry / silhouette** | — | Identical mesh | **95–99 %** **[J]** |
| 2 | **Materials** | Complex layered shaders, detail textures, tints, blends | glTF PBR; **layers collapse to one** (#1161), foliage vertex-colour shaders don't translate (#1180), cloth sheen has no equivalent (#576) | **60–85 %**, worse on blend-heavy surfaces **[J]** |
| 3 | **Static lighting / GI** | Baked **lightmaps** + light probes + env maps | **Lightmaps NOT exported**; only `light_environment` becomes a light; point/spot lights not exported at all | **30–60 %** **[J]** — the single biggest loss |
| 4 | **Skybox** | 2D sky material **+ separate 3D skybox map** | 3D skybox explicitly **not bundled** ([#967](https://github.com/ValveResourceFormat/ValveResourceFormat/issues/967)); horizon geometry missing | **20–50 %** **[J]** — very visible on Mirage |
| 5 | **Fog** | Gradient/volumetric fog entities | Not in glTF export (VRF's own viewer *does* render fog) | 0 % in glTF path |
| 6 | **Particles** | `.vpcf` systems, many operators | **glTF has no particle export at all**; must hand-write | 0 % unless reimplemented |
| 7 | **Smoke** | Voxel volume + raymarched volumetric emitter | Own raymarch over decoded voxels (if parser work done) — different sim + noise + no engine LOD/upsample tuning | **70–85 %** **[J]** |
| 8 | **Fire** | 16 cells + normal-oriented points + 4 particle snapshot layers + scorch decals | Cells exact from demo; particle layers + decals hand-made | **60–80 %** **[J]** |
| 9 | **Decals / overlays** | Runtime bullet holes + authored overlays | Overlays back-projected only for `.vmap`; **not in the glTF export list**; bullet decals runtime-only | **0–40 %** **[J]** |
| 10 | **Characters (static)** | Real agent models | Same meshes; 8 weights/vertex, skeleton intact | **90–95 %** **[J]** |
| 11 | **Characters (animation)** | Animgraph 2 state machine, IK, constraints, additive layers | **Graphs are never executed** — clips only, no state machine/IK/constraints; blend spaces collapse to first animation; vertical root motion zeroed | **40–70 %** **[J]** — the biggest *gameplay-visible* gap |
| 12 | **Anti-aliasing / resolution** | MSAA + engine upscaling options, tuned resolve | WebGL `antialias: true` (typically 4× MSAA) or post AA; different edge + temporal character | **60–80 %** **[J]** |
| 13 | **Post-processing** | `.vpost` LUTs, tonemap, bloom, DoF, motion blur | LUT **is** extractable (`.vpost` + `.raw`) and appliable in Three.js — good win; bloom/DoF/motion blur must be written | **50–75 %** with LUT, less without **[J]** |
| 14 | **Shadows** | Cascaded, tuned for the map | Your own CSM; only `light_environment` exists as a light source | **40–70 %** **[J]** |
| 15 | **Interpolation / tick** | 64-tick server + client interp, smoothed | Demo is 64 tick; you must replicate interpolation or players jitter | Achievable, easy to get wrong |
| 16 | **Viewmodel / first-person** | Separate `*_pm_arm.vmdl` arms, decoupled by AG2 | `*_pm_arm` carries no animgraph reference (per `NOTES.md`); Valve ships arms separately under `agents/models/shared/arms/glove_*` | Low, and a separate port |
| 17 | **Cloth / soft body** | Cloth solver | **No solver — cloth bones rigidly pinned** | ~0 % **[VRF]** |
| 18 | **Hitboxes** | Precise per-bone volumes | **Not exported**; you must reconstruct from skeleton | Manual |

## 4.2 The three losses that hurt most

1. **Baked lighting (lightmaps) is gone [VRF].** This is the dominant one. CS2's maps are lit by
   bake-time GI; real-time Three.js lighting cannot reproduce bounce, so interiors read flat and wrong.
   The only real fixes are (a) baking your own lightmaps in Blender from the exported glTF, or (b) using
   `ValveResourceFormat.Renderer`, which reads the real lightmaps and light probes.
2. **The animation state machine is not executed [VRF].** Characters will be *posed*, but the motion
   language — blending, IK foot planting, twist constraints, additive lean — is absent. Since
   `NOTES.md` already documents that Valve's agents carry 16 `CTiltTwistConstraint` lists and ported
   models have none, this is a known, unfixed deformation source in this very repo.
3. **The 3D skybox is not exported [VRF].** On maps with prominent 3D skyboxes the horizon will simply
   be wrong, and it is very noticeable.

## 4.3 Quantified bottom line

| Scenario | Estimated similarity to engine output |
|---|---|
| Static wide shot, real geometry + textures, no lighting bake | **~70–80 %** **[J]** |
| Static wide shot, plus own lightmap bake + CS2 LUT post | **~80–90 %** **[J]** |
| Close-up on blend-heavy surfaces | **~50–65 %** **[J]** |
| **Motion**: players, utility, effects on screen | **~50–65 %** **[J]** |
| **AI-relevant geometry** (positions, LOS, smoke occupancy, fire cells) | **~95–99 %** — *if* the parser work in §1.3 is done |

**These are calibrated judgements, not measurements** — but the direction is not in doubt: the
*geometric and gameplay* channel can be made near-exact while the *pixel* channel plateaus well short of
the engine. For synthetic AI observation generation that trade is usually the right one, provided you
are explicit that pixel-domain domain-gap exists and validate any vision model against real engine
captures (`reka-ai/cs2-dem-renderer` output is the obvious ground truth).

## 4.4 What this means for the smoke/fire work specifically

Because gameplay-relevant smoke is the *volume*, not the *pixels*, the highest-value work is:

1. **Decode `m_VoxelFrameData` and `m_firePositions` from demos** (§1.3). This makes smoke occlusion and
   fire footprints *exact*, which is what an AI agent consumes. Everything about rendering is secondary
   to this.
2. **Render the smoke as a raymarch over the decoded voxel volume** — not as sprites, and not as a
   144-unit sphere. Because the underlying data is the engine's own field, the visual result can be much
   better than the "70–85 %" guess once the density mapping is tuned.
3. **Model fire from `m_firePositions` + `m_BurnNormal` + `m_nInfernoType`**, using 7.0 s / 5.5 s
   lifetimes and 42-unit spawn spacing as the tuning constants, rather than inventing a grid.
4. **Do not join grenades by entity id** — it is reused (§1.3).

## 4.5 Reference projects for the fidelity verdict

No project has produced a game-accurate web replay, so the honest references are the adjacent ones:
[reka-ai/cs2-dem-renderer](https://github.com/reka-ai/cs2-dem-renderer) (real engine pixels, via a live
CS2 + plugin pipeline — the fidelity ceiling), [ValveResourceFormat.Renderer](https://www.nuget.org/packages/ValveResourceFormat.Renderer/)
(the best fidelity achievable *without* the engine), [Igor-Losev/deadem](https://github.com/Igor-Losev/deadem)
(reusable browser replay-state plumbing), and the radar-based viewers in §3.4 for the current state of the
art in what actually ships on the web.

---

# Part 5 — Recommendations, in priority order

1. **Prove the parser path first.** Spend one focused experiment on `demoinfocs-golang` (or
   `cs2-demoparser`) reading `CInferno.m_firePositions` and `CSmokeGrenadeProjectile.m_VoxelFrameData`
   from `0823_inferno.dem`. This is the highest-value unknown in the whole project and is decoupled from
   rendering and from having game files. **If it works, the AI-geometry goal is essentially solved.**
   If it fails, fall back to patching the Rust `demoparser`.
2. **Decide the renderer before writing shaders:** Three.js (browser, lower fidelity) vs.
   `ValveResourceFormat.Renderer` (desktop/.NET, keeps lightmaps + real materials + particles). This
   decision changes everything downstream and should be made against the actual goal (AI observation
   generation vs. realistic visual simulation), not by default.
3. **Get the game files.** The pipeline is tool-complete and asset-blocked. `docs/PLAN.md` notes ~17 GB
   free vs. a 20–30 GB need on this machine; alternatives are an external drive, another machine, or
   SteamCMD.
4. **Update `Source2Viewer-CLI`** past 19.2.6339 before scripting (missing `--game` and
   `--gltf_compose_additive`), and script with `.gltf` not `.glb` for maps.
5. **Extract the map's `.vpost` LUT** early — it is cheap and disproportionately improves perceived
   fidelity.
6. **Do not use public GLB repos as a shortcut.** They are CS:GO-era, character-only, celebration-only,
   and their MIT licence does not cover Valve's assets.
7. **Never host extracted Valve geometry.** Ship the extraction step (§2.6).

---

## Appendix — Primary sources

**Valve machine-readable schemas & data** (all [V])
- [`CSmokeGrenadeProjectile.h`](https://github.com/SteamTracking/GameTracking-CS2/blob/master/DumpSource2/schemas/server/CSmokeGrenadeProjectile.h) · [`C_SmokeGrenadeProjectile.h`](https://github.com/SteamTracking/GameTracking-CS2/blob/master/DumpSource2/schemas/client/C_SmokeGrenadeProjectile.h)
- [`CInferno.h`](https://github.com/SteamTracking/GameTracking-CS2/blob/master/DumpSource2/schemas/server/CInferno.h) · client `C_Inferno.h` · [schema pages](https://s2v.app/SchemaExplorer/cs2/server/CInferno)
- [`C_OP_RenderVolumetricEmitter.h`](https://github.com/SteamTracking/GameTracking-CS2/blob/master/DumpSource2/schemas/particles/C_OP_RenderVolumetricEmitter.h) · [`ParticleVolumetricSmokeType_t.h`](https://github.com/SteamTracking/GameTracking-CS2/blob/master/DumpSource2/schemas/particles/ParticleVolumetricSmokeType_t.h)
- [`DumpSource2/convars.txt`](https://github.com/SteamTracking/GameTracking-CS2/blob/master/DumpSource2/convars.txt) — every convar default above
- [GameTracking-CS2 repo](https://github.com/SteamTracking/GameTracking-CS2) · [Source 2 Schema Explorer](https://s2v.app/SchemaExplorer/cs2)

**VRF / Source 2 Viewer** (all [VRF])
- [Command-line utility](https://s2v.app/ValveResourceFormat/guides/command-line.html) · [Exporting maps](https://s2v.app/ValveResourceFormat/guides/exporting-maps.html) · [Exporting models](https://s2v.app/ValveResourceFormat/guides/exporting-models.html) · [**Format Support and Export Fidelity**](https://s2v.app/ValveResourceFormat/guides/format-support.html)
- [Repo](https://github.com/ValveResourceFormat/ValveResourceFormat) · [`CLI/Decompiler.cs`](https://github.com/ValveResourceFormat/ValveResourceFormat/blob/master/CLI/Decompiler.cs) · [`ValveResourceFormat.Renderer` on NuGet](https://www.nuget.org/packages/ValveResourceFormat.Renderer/) · [.dem replay issue #1122](https://github.com/ValveResourceFormat/ValveResourceFormat/issues/1122) · [3D skybox #967](https://github.com/ValveResourceFormat/ValveResourceFormat/issues/967) · [blend layers #1161](https://github.com/ValveResourceFormat/ValveResourceFormat/issues/1161) · [cloth #653](https://github.com/ValveResourceFormat/ValveResourceFormat/issues/653) · [cloth sheen #576](https://github.com/ValveResourceFormat/ValveResourceFormat/issues/576) · [foliage #1180](https://github.com/ValveResourceFormat/ValveResourceFormat/issues/1180) · [LOD #535](https://github.com/ValveResourceFormat/ValveResourceFormat/issues/535)

**Community / secondary** (all [C])
- [CS:GO `DT_Inferno` netprops (gamesense-lua)](https://gamesensical.gitbook.io/docs/developers/netprops/other/cinferno) — the `m_fireXDelta/Y/Z[64]` layout
- [How long does smoke last in CS2 (swap.gg)](https://swap.gg/blog/how-long-does-smoke-last-in-cs2) — 18 s, volumetric behaviour, HE/bullet interactions
- [CS2 smoke in Unreal Engine 5 (More VFX Academy)](https://morevfxacademy.com/counter-strike-2-smoke-in-unreal-engine-5/) · [Acerola raymarch/SDF reconstruction](https://www.bilibili.com/video/BV1Web86YEif/)
- [SourceIO (Blender)](https://github.com/REDxEYE/SourceIO) · [cs2-map-icons](https://github.com/MurkyYT/cs2-map-icons) · [cs2.agents GLB repo](https://github.com/rabbytesoftware/cs2.agents) · [counter-strike-in-browser](https://github.com/StarKnightt/counter-strike-in-browser)
- Demo viewers: [cs2d](https://github.com/zenojunior/cs2d) · [csgo-2d-demo-viewer](https://github.com/sparkoo/csgo-2d-demo-viewer) · [deadem](https://github.com/Igor-Losev/deadem) · [awpy](https://github.com/pnxenopoulos/awpy) · [cs2-dem-renderer](https://github.com/reka-ai/cs2-dem-renderer) · [cs2c](https://github.com/jackgraver/cs2c) · [cs2demoview](https://github.com/Sylixa/cs2demoview)
- Parsers: [demoinfocs-golang entity/property system](https://deepwiki.com/markus-wa/demoinfocs-golang/3.3-entity-management) · [demoparser](https://github.com/LaihoE/demoparser) · [cs2-demoparser](https://github.com/yickling/cs2-demoparser)

**Legal** — [Steam Subscriber Agreement](https://store.steampowered.com/subscriber_agreement/) · [Valve Video Policy](https://store.steampowered.com/video_policy) · [Steam Online Conduct](https://store.steampowered.com/online_conduct/Steam?l=english)

**In-repo evidence ([M])** — `notes/scratch/inferno_probe.py`, `smoke_grenade.py`, `tickrate.py` and their
`*_result.json`; `docs/PLAN.md`; `NOTES.md` (AG2 skeleton: 74 bones rooted at `root_motion`; no
`CTiltTwistConstraint`s on ported models).
