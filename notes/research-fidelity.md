# CS2 Fidelity Reference — Smokes, Fire, Player Models, Units, Tooling

**Purpose:** the precise visual/gameplay parameters needed to reproduce Counter-Strike 2 (Source 2) smoke grenades, molotov/incendiary fire and player models in a web-based 3D replay viewer **faithfully, not approximately**.

**Compiled:** ~September 2026. CS2 build era: patch notes mined through 2026-09-10; depot manifest `pak01_dir.txt` (135,686 entries) and `convars.txt` as currently published by SteamTracking/GameTracking-CS2.

**Evidence tiers used throughout:**
| Tier | Meaning |
|---|---|
| **[V]** | Valve primary — shipped convar/schema/FGD dumps, depot file manifest, official patch notes, Valve wiki |
| **[M]** | Measured — from a real CS2 demo or reproduced from Valve data by arithmetic |
| **[C]** | Community/wiki — third-party, widely reproduced, not Valve-authored |
| **[I]** | Inferred — my interpretation of a name/value; explicitly flagged |

**Primary machine-readable sources**
- Valve CS2 depot manifest (full VPK file list): <https://raw.githubusercontent.com/SteamTracking/GameTracking-CS2/master/game/csgo/pak01_dir.txt>
- Valve CS2 convar/command dump: <https://raw.githubusercontent.com/SteamTracking/GameTracking-CS2/master/DumpSource2/convars.txt>
- Valve CS2 entity schemas: <https://github.com/SteamTracking/GameTracking-CS2/tree/master/DumpSource2/schemas>
- Valve CS2 FGDs: <https://raw.githubusercontent.com/SteamTracking/GameTracking-CS2/master/game/csgo/csgo.fgd> · <https://raw.githubusercontent.com/SteamTracking/GameTracking-CS2/master/game/csgo/postprocessing.fgd> · <https://raw.githubusercontent.com/SteamTracking/GameTracking-CS2/master/game/core/lights_base.fgd>
- Full CS2 cvar list w/ defaults (community mirror): <https://raw.githubusercontent.com/SuGolYolLom/CS2-Cvars-Cmds/main/cs2/cvarlist.md>
- Official CS2 patch notes: Steam news API (`ISteamNews/GetNewsForApp`, appid 730) — quoted per item below.

---

## 1. Units, scale and coordinate system

### 1.1 Hammer unit → metres

| Scale | Ratio | 1 HU | Source |
|---|---|---|---|
| Maps, architecture, prop models | 1 foot = **16 HU** | **1.905 cm = 0.01905 m** | Valve/wiki **[V]**, [Team Fortress Wiki – Hammer unit](https://wiki.teamfortress.com/wiki/Hammer_unit) |
| Skyboxes (1/16 scale of maps) | 1 foot = **1 HU** | 0.3048 m | same |
| “Human character models and certain other models” | 1 foot = **12 HU** | **2.54 cm** | same |

**This two-scale split is the source of the classic confusion, and it matters:**
- Valve’s *map* scale is 16 HU/ft → a 72-HU player is 1.372 m (4 ft 6 in) — obviously not the intent.
- Valve authors *character models* at 12 HU/ft → a 72-HU player is exactly **6 ft = 1.8288 m**, and eye height 64 HU = 1.626 m, which is anatomically right.
- The practical reading: **the player is deliberately oversized relative to the architecture** (72 HU in a world where 6 ft = 96 HU). That is why doorways are 128 HU (2.44 m at map scale).

**Recommendation for the viewer:** store and simulate everything in **Hammer units, 1:1**, exactly as the demo gives them. Convert to metres only for a scale bar / UI, using **1 HU = 0.01905 m** (the map scale). Do **not** try to “fix” the player/architecture ratio — it is part of the game’s look. glTF is metres/Y-up, so the exporter/importer will rotate Z-up→Y-up and scale; if you scale, pick one factor and apply it uniformly.

### 1.2 Conversion table (all values used in this report)

| HU | metres | HU | metres |
|---:|---:|---:|---:|
| 16 | 0.305 | 110 | 2.096 |
| 26 | 0.495 | 128 | 2.438 |
| 32 | 0.610 | 144 | 2.743 |
| 42 | 0.800 | 150 | 2.858 |
| 54 | 1.029 | 156 | 2.972 |
| 61 | 1.162 | 176 | 3.353 |
| 64 | 1.219 | 256 | 4.877 |
| 68 | 1.295 | 288 | 5.486 |
| 72 | 1.372 | 298 | 5.677 |
| 88 | 1.676 | 4096 | 78.03 |

### 1.3 Coordinate system and forward axis — **resolved [V]**

- World axes: **X = forward/East (yaw 0), Y = left/North, Z = up**. Source is right-handed at the world level (X × Y = Z).
- **Source 2 is explicitly “Z-up, X-forward”** — stated by Valve in the Animgraph-2 skeleton document, `MPropertyDescription` on `CNmSkeletonDocument.m_gameplayRelevantBones`: *“The set of bones that need to be converted at import to match the S2 coordinate system (Z-up, X-forward)”* — <https://raw.githubusercontent.com/SteamTracking/GameTracking-CS2/master/DumpSource2/schemas/animdoclib/CNmSkeletonDocument.h>. Corroborated by Valve’s Source 2 Blender export guide (“Forward / Up = X Forward and Z Up”).
- QAngle: X = pitch (+down), **Y = yaw (+left)**, Z = roll; degrees. Z clamped to ±50 for player viewangles — [Valve wiki QAngle](https://developer.valvesoftware.com/wiki/QAngle).
- **yaw → forward**, pitch = 0: `forward = (cos yaw, sin yaw, 0)`. yaw 0 → +X, 90 → +Y, 180 → −X, 270 → −Y.
- With pitch *p* (positive = down): `forward = (cos p · cos y, cos p · sin y, −sin p)`; `right = (sin y, −cos y, 0)`; `up = (sin p · cos y, sin p · sin y, cos p)`.
- **Handedness trap:** the world triple (X,Y,Z) is right-handed, but the entity triple (forward, right, up) satisfies forward × right = **−up**, i.e. left-handed. Net effect: **+Y is the player’s LEFT**, “right” is −Y at yaw 0. Models are authored facing their own +X. If your renderer assumes −Z forward, apply a −90° yaw correction and verify against a known model — [Valve wiki Coordinates](https://developer.valvesoftware.com/wiki/Coordinates), [AngleVectors](https://developer.valvesoftware.com/wiki/AngleVectors()).

### 1.4 Tickrate

- Official/competitive servers run **128 tick** (awpy `DEFAULT_SERVER_TICKRATE = 128`). **Dematch/MM demos can be 64-tick** — the probe demo in this workspace (`0823_inferno.dem`) measured `s_per_tick = 0.015625` → exactly **64 tick** **[M]**. Always derive tickrate from the demo (`game_time` prop regression) rather than assuming.
- Seconds ↔ ticks: `s = ticks / tickrate`. At 128: 1 tick = 7.8125 ms; at 64: 15.625 ms.

---

## 2. Smoke grenade (CS2)

### 2.1 Entity / networking — what a demo actually contains [V]

`C_SmokeGrenadeProjectile` (client schema), <https://raw.githubusercontent.com/SteamTracking/GameTracking-CS2/master/DumpSource2/schemas/client/C_SmokeGrenadeProjectile.h>:

```cpp
class C_SmokeGrenadeProjectile : public C_BaseCSGrenadeProjectile {
    int32  m_nSmokeEffectTickBegin;
    bool   m_bDidSmokeEffect;
    int32  m_nRandomSeed;
    Vector m_vSmokeColor;                       // observed [180,129,50] in a real demo
    VectorWS m_vSmokeDetonationPos;
    C_NetworkUtlVectorBase<uint8> m_VoxelFrameData;   // <-- the volumetric data, streamed
    int32  m_nVoxelFrameDataSize;
    int32  m_nVoxelUpdate;
    bool   m_bSmokeVolumeDataReceived;
    bool   m_bSmokeEffectSpawned;
};
```

**Key consequence for a replay viewer:** CS2’s smoke is a **volumetric voxel field streamed over the network** (`m_VoxelFrameData` is a networked `uint8` vector, with an explicit `m_bSmokeVolumeDataReceived` flag). The demo therefore *in principle* carries the real smoke volume — but in practice the parsers in wide use do not expose it usefully (see §5.4). Related events available per demo: `smokegrenade_detonate`, `smokegrenade_expired`.

Gameplay-visible smoke radius for bots is a cvar: **`bot_max_visible_smoke_length = 200` HU** — “Bots will see players through smoke clouds up to this length” **[V]**. This is the *penetration depth* at which the AI treats smoke as opaque, not the cloud radius.

### 2.2 Size and shape

| Quantity | Value | Source |
|---|---|---|
| CS:GO smoke diameter | **≈256 HU** (≈4.88 m) | [Valve CS:GO Mapper’s Reference](https://web.archive.org/web/20250101150843/https://developer.valvesoftware.com/wiki/Counter-Strike:_Global_Offensive/Mapper%27s_Reference) **[V]** |
| CS:GO smoke height | **≈128 HU** (≈2.44 m) | same |
| Corridor fully covered by one smoke | **≤200 HU** wide with no gaps; ≥256 HU cannot be covered | same **[V]** |
| CS:GO smoke diameter (wiki figure) | **288 HU = 144 HU radius** | [CS wiki – Smoke Grenade](https://counterstrike.fandom.com/wiki/Smoke_Grenade) **[C]** |
| CS2 `cl_smoke_torus_ring_radius` | **61** HU | Valve `convars.txt` **[V]** |
| CS2 `cl_smoke_torus_ring_subradius` | **88** HU | Valve `convars.txt` **[V]** |
| CS2 `cl_smoke_origin_height` | **68** HU above ground | Valve `convars.txt` **[V]** |
| CS2 `cl_smoke_edge_feather` | **21** HU | Valve `convars.txt` **[V]** |
| CS2 `cl_smoke_lower_speed` | 1 | Valve `convars.txt` **[V]** |
| CS2 `cl_smoke_volume_growth` | 1 | Valve `convars.txt` **[V]** |
| `r_csgo_smoke_avoid_flat` | true | Valve `convars.txt` **[V]** |

**Interpretation [I] (flagged — the cvar *names* are Valve’s, the geometry reading is mine):** the names describe a **torus of sub-volumes** — centres on a ring of radius **61 HU**, each sub-volume of radius **88 HU**, with the torus centre line **68 HU above ground**. Union extent: horizontal radius ≈ 61 + 88 = **149 HU (diameter ≈298 HU ≈ 5.68 m)**, vertical ±88 about z=68 → top ≈156 HU (2.97 m), bottom clipped at the ground. That lands squarely on the 288-HU / 144-HU-radius community figure and is a reasonable description of what CS2 smoke looks like: **a broad, slightly flattened, ground-hugging lens, not a sphere**. It is independently corroborated by the CS:GO smoke constant **`CONSTANT_UNITS_SMOKEGRENADERADIUS = 166 HU`**, the radius the game used for all smoke↔fire interaction maths **[V]** — i.e. the engine’s own working smoke radius sits between 149 and 166 HU, not at 288.

**How to render it faithfully without the voxel data:**
1. Start from a **rounded, ground-clipped volume** centred **68 HU above the impact point**, horizontal radius ≈ **130–149 HU**, vertical half-extent ≈ **88 HU**, top at ≈156 HU.
2. Clip against world geometry and let it “flow” upward/along openings (CS2 smoke fills rooms and stairwells rather than clipping through walls).
3. Apply **21 HU edge feathering** so the silhouette is soft, not a hard ball.
4. Treat the volume as **opaque for LOS** out to the same radius; the gameplay-visible threshold for AI is 200 HU of penetration.

### 2.3 Timing: fuse, bloom, duration

| Event | Value | Source |
|---|---|---|
| Fuse starts **after the grenade lands**, not on throw | “instantly release a cloud … after a very short fuse time (the fuse time only starts after the grenade lands, so it will not explode midair)” | [CS wiki – Smoke Grenade](https://counterstrike.fandom.com/wiki/Smoke_Grenade) **[C]** |
| Minimum fuse duration | added 2023-11-02: “Smoke grenades now have a minimum fuse duration to prevent smokes from prematurely detonating if stuck in crevices” | [Release Notes 11/2/2023](https://steamstore-a.akamaihd.net/news/externalpost/steam_community_announcements/5288974951208087929) **[V]** |
| Bloom / full expand | no cvar exists; **sub-second to ~1 s**. Community timeline: “begins after the smoke lands and fully deploys, which takes about a second” | [Swap.gg – How long does smoke last](https://swap.gg/blog/how-long-does-smoke-last-in-cs2) **[C]** |
| Expand algorithm fixes | 2023-06-08 “Fixed smoke expansion algorithm when detonating very close to a wall”; 2023-08-15 “Minor adjustments to smoke cloud shape”; 2023-08-31 “Continued iterating on smoke shape” | Valve patch notes **[V]** |
| Coverage duration | **18 s** from detonation (CS:GO and CS2 alike) | CS wiki **[C]**; Swap.gg **[C]** |
| Radar-block duration | **15 s** (CS:GO; shorter than the visual cloud) | CS wiki **[C]** |
| Tail-off | final **2–3 s** the cloud visibly thins at the edges | Swap.gg **[C]** |
| Parser fallback duration in the wild | awpy `DEFAULT_SMOKE_DURATION_IN_SECS = 20.0` — used only when the `smokegrenade_expired` event is missing | [awpy constants.py](https://github.com/pnxenopoulos/awpy/blob/main/awpy/constants.py) **[C]** |

**Important negative finding [V]:** there is **no cvar for smoke radius or smoke duration in CS2** — a full-prefix search of Valve’s current `convars.txt` finds only client-side geometry/appearance cvars. Both values are hardcoded. The complete set of exposed smoke tunables is:

| Convar | Default | Notes |
|---|---|---|
| `cl_smoke_torus_ring_radius` | 61 | geometry |
| `cl_smoke_torus_ring_subradius` | 88 | geometry |
| `cl_smoke_origin_height` | 68 | geometry |
| `cl_smoke_edge_feather` | 21 | edge softness |
| `cl_smoke_lower_speed` | 1 | settle rate |
| `cl_smoke_volume_growth` | 1 | growth rate |
| `cl_smoke_volumeprop` | true | |
| `cl_smoke_player_particle_effect` | false | |
| `smoke_use_noise_texture` | true | volumetric noise |
| `smoke_volume_lod_ratio_change` | 0.6 | LOD |
| `smoke_param1` / `2` / `3` / `4` / `5` | 6.26 / 8.27 / 0.13 / 0 / 0 | undocumented raymarch/density shaping params **[I]** |
| `smoke_grenade_t_color` / `smoke_grenade_ct_color` | `[180,129,50]` / `[75,127,155]` | team tint |
| `sv_smoke_volume_blind_start` | 0.2 | silhouette/blind threshold |
| `bot_max_visible_smoke_length` | 200 | AI LOS penetration |
| `r_csgo_smoke_shadow` / `_fullres_pass` / `_fullres_enhance` / `_avoid_flat` / `_clip_sniper` / `_upscale_discard_pixels_behind` | true / true / false / true / true / false | render pipeline |
| `r_csgo_smoke_overlay_min_dt` | 0.0156863 | = 4/255 dither |
| `r_csgo_effects_bloom_when_smoked` | false | |

**On the “18 s vs 20 s” discrepancy:** the widely repeated figure is 18 s of usable cover; awpy’s fallback is 20 s (i.e. detonation → fully gone). A local probe of `0823_inferno.dem` (64-tick) measured a median smoke detonate→expire delta of **1412 ticks = 22.06 s** and called it “smoke_lifetime”; the same probe’s pairing logic is demonstrably unreliable (it also reported a 124,703-tick maximum and `position_match: false`), so **treat 1412 ticks as unverified**. The safe, defensible numbers are: **18 s of full cover, ~20 s to fully dissipate, 15 s of radar block**.

### 2.4 Bullets, HE and other interactions (relevant to rendering) [V]

All of these are official patch-note text, which is the strongest available evidence:

| Date | Change |
|---|---|
| 2023-03-24 | “Smoke bullet holes are now fixed relative to the firstperson camera.” · “Fixed smoke cloud volume lighting.” |
| 2023-03-30 | “HE grenades no longer affect smokes through walls.” |
| 2023-06-06 | **[VOLUMETRIC SMOKE]** “Improved lighting/rendering when smoke overlaps multiple distinctly lit regions.” · “**HE grenade effect times on smoke has been reduced.**” · “**Shotguns now create larger holes in smoke volumes.**” |
| 2023-11-02 | “Decoy grenades interact aesthetically with smoke clouds.” |
| 2024-02-06 | “**Smokes now cast shadows.** Rendering and animation have been improved.” |
| 2024-02-08 | “Adjusted visibility of player silhouettes in smoke.” · “Fixed a bug where enemies were visible on the radar when viewed through a combination of smoke and translucent surfaces such as glass.” |
| 2024-02-28 | “Fixed smoke shadow rendering issues and improvements to perf.” |
| 2024-10-28 | “HE grenade explosions within smoke now clear the accumulated smoke screen effect on players inside that smoke within range of the HE grenade explosion.” |
| 2025-05-08/09 | Feedback-sound and “fire visible through smoke” fixes. |
| 2026-07-21 | “The bomb explosion now interacts with smoke grenade clouds and molotov/incendiary fire.” |

**For the renderer:** bullet-punch holes and HE-clearing are **visible and gameplay-relevant**, so a “faithful” viewer should implement at least:
- **Bullet holes**: a small radial carve-out at the bullet’s entry point on the smoke surface, scaling with calibre (shotguns > rifles), healing over roughly 1–2 s. Community figure: holes “close within a second or two” ([Swap.gg](https://swap.gg/blog/how-long-does-smoke-last-in-cs2)) **[C]**.
- **HE blast**: a large temporary void (community: **2–3 s** before it re-expands) **[C]**, plus clearing the *screen-space* smoke overlay for players inside.
- **Smoke ↔ fire**: a smoke grenade landing on fire **extinguishes it** and prevents fire-device detonation. The CS:GO source gives the exact rules — extinguishing triggers on `smokegrenade_detonate` **only if the smoke lands within `SmokeGrenadeRadius × 4 = 664 HU` of the inferno’s ignition point**; if the smoke cloud overlaps the ignition point, **all** flames are extinguished, otherwise only the flames inside the cloud are (and if ≥ ⅓ of them were, the rest are wiped too). `IsFirePosInSmokeCloud()` also **denies new ignition and denies spread into smoke**. The cloud used for this test is an anisotropic ellipsoid, not a sphere: `CONSTANT_UNITS_SMOKEGRENADERADIUS = 166`, with `_InfernoAffectingZ = 120`, `_XY_topedge = 100`, `_XY_equator = 150`, `_XY_bottomedge = 128`; the vertical fire-to-smoke reach is `2 × InfernoFire_HalfWidth + 4 = 64 HU`. A bouncing smoke that projects onto a live flame with clear LOS within **60 HU** is teleported to it and detonated immediately. **[V, CS:GO source]**
- **Silhouettes**: since Feb 2024, players inside smoke are faintly visible as silhouettes (shader `player_visibility` + `player_visibility_stencil_proxy`, see §2.5). The threshold convar is **`sv_smoke_volume_blind_start = 0.2`**.
- **Team tint**: CS2 tints the freshly-deployed smoke by the thrower’s team — **`smoke_grenade_t_color = [180, 129, 50]`** (warm brown) and **`smoke_grenade_ct_color = [75, 127, 155]`** (cool blue) **[V]**. A live demo’s `m_vSmokeColor` measured exactly `[180, 129, 50]` **[M]**, confirming the T value. The tint fades to neutral grey as the cloud ages, which players use as a visual expiry cue **[C]**.
- **Shadows**: smoke casts shadows since Feb 2024 (`r_csgo_smoke_shadow = true`).

### 2.5 How CS2 actually renders it [V evidence, I interpretation]

**It is a true volumetric ray-marched effect, not billboards.** The CS2 shader manifest contains:

```
shaders/vfx/smoke_volume.ini
shaders/vfx/smoke_volume_cs.ini          <- CS-specific variant
shaders/vfx/smoke_volume_depth.ini       <- writes depth for water reflections / compositing
shaders/vfx/smoke_volume_lpv.ini         <- lighting from light probe volumes
shaders/vfx/smoke_volume_mask.ini
shaders/vfx/overlay_smoke.ini            <- the screen-space "you are inside smoke" overlay
shaders/vfx/write_smoke_depth_water_reflection.ini
shaders/vfx/mboit_mixed_combine.ini      <- Moment-Based Order-Independent Transparency
shaders/vfx/mboit_mixed_upsample.ini
shaders/vfx/mboitfinal.ini
shaders/vfx/deferred_particle_shadows.ini
shaders/vfx/player_visibility.ini
shaders/vfx/player_visibility_stencil_proxy.ini
shaders/vfx/inferno.ini                  <- fire
shaders/vfx/luminaire.ini
```
Source: `game/csgo/shaders_pc_dir.txt` in [GameTracking-CS2](https://github.com/SteamTracking/GameTracking-CS2). **[V]**

Reading **[I]**: the smoke is rasterised as a **volume into a low-res buffer, then upsampled** (`r_csgo_smoke_fullres_pass = true` “Does a full res pass to cover holes and artifacts in smoke low res”, `r_csgo_smoke_upscale_discard_pixels_behind = false`, `r_csgo_smoke_overlay_min_dt = 0.0156863` = 4/255), lit partly from **light probe volumes** (`smoke_volume_lpv`), composited with **MBOIT** (order-independent transparency, so overlapping smoke and particles blend correctly), and then combined with the volumetric lighting/bloom chain. Client render cvars: `r_csgo_smoke_shadow true`, `r_csgo_smoke_fullres_pass true`, `r_csgo_smoke_fullres_enhance false`, `r_csgo_smoke_avoid_flat true`, `r_csgo_smoke_clip_sniper true`, `r_csgo_smoke_upscale_discard_pixels_behind false`.

**The “volumetric depth” trick is real and it is why the shape follows geometry:** `m_VoxelFrameData` is a networked voxel grid; the client raymarches it. There is an explicit `r_csgo_smoke_upscale_discard_pixels_behind` cvar that discards smoke pixels *behind solid depth* — i.e. the volume is depth-tested against the world.

**Known particle/material assets [V]** (`pak01_dir.txt`):

```
particles/explosions_fx/explosion_smokegrenade_voxel.vpcf_c        <- the voxel smoke effect
particles/explosions_fx/explosion_smokegrenade_adaptive.vpcf_c
particles/explosions_fx/explosion_smokegrenade_simulated.vpcf_c
particles/explosions_fx/explosion_smokegrenade.vpcf_c / _init / _fallback
particles/explosions_fx/explosion_smokegrenade_ct.vpcf_c / _t.vpcf_c
particles/explosions_fx/explosion_smokegrenade_distort.vpcf_c / _refract.vpcf_c
particles/explosions_fx/explosion_smoke_screen_smokegrenade_new.vpcf_c
particles/explosions_fx/smokegrenade/weapon_smokegrenade_pointcloud.vsnap_c      <- particle "snapshot"
particles/explosions_fx/smokegrenade/weapon_smokegrenade_pointcloud_col.vsnap_c
particles/explosions_fx/smokegrenade/weapon_smokegrenade_adaptive.vsnap_c
particles/explosions_fx/smokegrenade/smokegrendae.vsnap_c                        (Valve's spelling)
particles/characters/smokegrenade_body_fx.vpcf_c / _temp / _trails
materials/particle/particle_smokegrenade.vmat_c     (+ particle_smokegrenade1.vmat_c)
materials/particle/particle_smokegrenade.vtex_c     (+ _color, _trans variants)
materials/particle/particle_smoke_swirl.vmat_c
materials/particle/smokestack.vmat_c
weapons/models/grenade/smokegrenade/weapon_smokegrenade.vmdl_c
```
Note the **`.vsnap_c` “particle snapshot”** format: this is the Source 2 mechanism where a particle system is *baked into a snapshot* so it can be replayed/networked cheaply — exactly what a demo-streamed smoke volume needs. There is no `sv_smoke_*` or `mp_smoke_*` cvar; the smoke cannot be re-tuned server-side.

**Practical rendering recipe for the web viewer** (approximation ladder):
1. **Cheapest credible:** one raymarched or shell-rendered ellipsoid, radius ≈140 HU, centre 68 HU up, ground-clipped, 21 HU feathered edge, with a noise-distorted boundary. Opaque for LOS. Add bullet holes as alpha carves and HE voids as a shrinking sphere.
2. **Good:** 8–16 metaball sub-volumes seeded on a 61-HU ring + centre, each r≈88 HU, unioned and clipped to geometry, raymarched at half-res with an upsample pass and 4/255 dither — this reproduces CS2’s actual construction and its soft, swirling edge.
3. **Best:** consume `m_VoxelFrameData` if your parser can decode it (the field exists and is networked), grow the grid where it meets openings, raymarch.

---

## 3. Molotov / incendiary grenade (CS2)

### 3.1 `C_Inferno` — entity data layout [V]

**CS2** (current), <https://raw.githubusercontent.com/SteamTracking/GameTracking-CS2/master/DumpSource2/schemas/client/C_Inferno.h>:

```cpp
class C_Inferno : public C_BaseModelEntity {
    ParticleIndex_t m_nfxFireDamageEffect;
    CStrongHandle<InfoForResourceTypeIParticleSnapshot> m_hInfernoPointsSnapshot;
    CStrongHandle<InfoForResourceTypeIParticleSnapshot> m_hInfernoFillerPointsSnapshot;
    CStrongHandle<InfoForResourceTypeIParticleSnapshot> m_hInfernoOutlinePointsSnapshot;
    CStrongHandle<InfoForResourceTypeIParticleSnapshot> m_hInfernoClimbingOutlinePointsSnapshot;
    CStrongHandle<InfoForResourceTypeIParticleSnapshot> m_hInfernoDecalsSnapshot;
    VectorWS m_firePositions[64];          // absolute world positions of fire cells
    VectorWS m_fireParentPositions[64];
    bool     m_bFireIsBurning[64];
    Vector   m_BurnNormal[64];
    int32    m_fireCount;
    int32    m_nInfernoType;
    float32  m_nFireLifetime;
    bool     m_bInPostEffectTime;
    int32    m_lastFireCount;
    int32    m_nFireEffectTickBegin;
    int32    m_drawableCount;
    bool     m_blosCheck;
    int32    m_nlosperiod;
    float32  m_maxFireHalfWidth;
    float32  m_maxFireHeight;
    VectorWS m_minBounds;
    VectorWS m_maxBounds;
    float32  m_flLastGrassBurnThink;
};
```

**CS2 server class** (authoritative for gameplay logic), <https://raw.githubusercontent.com/SteamTracking/GameTracking-CS2/master/DumpSource2/schemas/server/CInferno.h>:

```cpp
class CInferno : public CBaseModelEntity {
    VectorWS[64] m_firePositions;            VectorWS[64] m_fireParentPositions;
    bool[64]     m_bFireIsBurning;           Vector[64]   m_BurnNormal;
    int32        m_fireCount;                int32        m_nInfernoType;
    int32        m_nFireEffectTickBegin;     float32      m_nFireLifetime;
    bool         m_bInPostEffectTime;        bool         m_bWasCreatedInSmoke;
    Extent       m_extent;                                // server-side AABB
    CountdownTimer m_damageTimer;            CountdownTimer m_damageRampTimer;
    Vector       m_splashVelocity;           Vector       m_InitialSplashVelocity;
    VectorWS     m_startPos;                 VectorWS     m_vecOriginalSpawnLocation;
    IntervalTimer m_activeTimer;
    int32        m_fireSpawnOffset;          int32        m_nMaxFlames;
    int32        m_nSpreadCount;
    CountdownTimer m_BookkeepingTimer;       CountdownTimer m_NextSpreadTimer;
    uint16       m_nSourceItemDefIndex;
};
```
`m_damageTimer` / `m_damageRampTimer` are the damage clock and the ramp clock; `m_nMaxFlames` is the per-inferno cell cap (from `inferno_max_flames`); `m_nSpreadCount` + `m_NextSpreadTimer` are the live spread counter and next-spread clock; `m_extent` is the server-side AABB (the client exposes `m_minBounds`/`m_maxBounds` instead).

**CS:GO — only six fields were networked.** The entire CS:GO `DT_Inferno` block was:

| Field | Type | Bits / bytes | Network offset |
|---|---|---|---|
| `m_fireXDelta[64]` | int | 15 bits each | 924 |
| `m_fireYDelta[64]` | int | 15 bits each | 1180 |
| `m_fireZDelta[64]` | int | 15 bits each | 1436 |
| `m_bFireIsBurning[64]` | bool | 1 bit on the wire (1-byte table stride) | 1692 |
| `m_fireCount` | int | 7 bits | 2524 |
| `m_nFireEffectTickBegin` | int | 32 bits | 2532 |

Everything else (`m_fireParentPositions`, `m_nInfernoType`, `m_nFireLifetime`, `m_maxFireHalfWidth`, `m_maxFireHeight`, `m_bInPostEffectTime`, `m_lastFireCount`, `m_drawableCount`, `m_blosCheck`, `m_nlosperiod`, `m_flLastGrassBurnThink`) is **CS2-only**. Deltas were **integer-truncated offsets from the inferno origin** — `m_fireXDelta.Set(i, (int)vecDelta.x)` where `vecDelta = firePos - GetAbsOrigin()` — so a CS:GO-era parser must add the inferno origin; CS2 hands you `m_firePositions` in world space directly. Arrays are size **64** in both, but that 64 is only the array/network capacity: the effective cap is `inferno_max_flames` (16), enforced server-side (`if (m_fireCount >= MIN(MAX_INFERNO_FIRES /*64*/, m_nMaxFlames)) break;`).

**Actionable for a viewer [V→M]:** you can reconstruct the exact fire footprint per tick from a demo by reading `m_firePositions[0..m_fireCount-1]` and `m_bFireIsBurning[]`, plus `m_maxFireHalfWidth` / `m_maxFireHeight` for the cell extents and `m_minBounds` / `m_maxBounds` for the overall AABB. The particle-snapshot handles map 1:1 onto the assets below.

### 3.2 Complete CS2 fire parameter set (Valve convar defaults) [V]

From Valve’s `convars.txt` / the full cvar dump — these are the authoritative numbers:

| Convar | Default | Meaning (Valve’s own description where given) |
|---|---|---|
| `inferno_damage` | **40** | Damage per second |
| `inferno_damage_ct` | **40** | Damage per second from CT inferno |
| `inferno_damage_timer` | **0.2** | **How long between times for the inferno to deal damage** (i.e. 5 Hz) |
| `inferno_flame_lifetime` | **7** | Average lifetime of each flame in seconds (**molotov**) |
| `inferno_flame_lifetime_incendiary` | **5.5** | Average lifetime of each flame in seconds (**incgrenade**) |
| `inferno_max_range` | **150** | Maximum distance flames can spread from their initial ignition point (**molotov**) |
| `inferno_max_range_ct` | **110** | Same, for an **incendiary** |
| `inferno_max_flames` | **16** | Maximum number of flames that can be created (the effective cell cap; the arrays hold 64) |
| `inferno_spread_speed_mult` | **1** | Spread-speed multiplier, molotov |
| `inferno_spread_speed_mult_ct` | **10** | Spread-speed multiplier, **incendiary** — this is the 2025-07-16 “fire spreads more rapidly” change |
| `inferno_flame_spacing` | **42** | Minimum distance between separate flame spawns (HU) |
| `inferno_initial_spawn_interval` | **0.02** | Time between spawning flames for first fire (s) |
| `inferno_per_flame_spawn_duration` | **3** | Duration each new flame will attempt to spawn new flames (s) |
| `inferno_spawn_angle` | **45** | Angular change from parent (degrees) |
| `inferno_child_spawn_max_depth` | **4** | Max child-generation depth |
| `inferno_child_spawn_interval_multiplier` | **0.1** | Amount spawn interval increases for each child |
| `inferno_max_child_spawn_interval` | **0.5** | Largest time interval for child flame spawning (s) |
| `inferno_forward_reduction_factor` | **0.9** | Forward bias damping |
| `inferno_surface_offset` | **15** | Offset of flame above the surface (HU) |
| `inferno_velocity_factor` | **0.003** | |
| `inferno_velocity_factor_ct` | **0.003** | |
| `inferno_ct_experiment` | true | Enables the CT-incendiary experiment (the per-side `_ct` parameter set) |
| `inferno_velocity_decay_factor` | **0.2** | |
| `inferno_velocity_normal_factor` | **0** | |
| `inferno_friendly_fire_duration` | **6** | For this long, FF is credited back to the thrower |
| `inferno_smoke_volume_density` | **0.03** | Density of the smoke volume the fire contributes |
| `inferno_dlights` | **30** | Min FPS at which molotov dlights will be created |
| `inferno_dlight_spacing` | **7200** | Inferno dlights are at least this far apart (HU) |
| `inferno_fire` | **2** | |
| `inferno_batched_rays` | true | |
| `weapon_molotov_maxdetonateslope` | **30** | Maximum angle of slope on which the molotov will detonate (degrees) |
| `molotov_throw_detonate_time` | **2** | Air-burst fuse (s) |
| `cl_inferno_bodyburn` | true | Client-side body-burn effect |
| `r_csgo_render_inferno_decals` | true | Render inferno scorch decals (CS2’s replacement for CS:GO’s `inferno_scorch_decals`) |

**Cvars that do NOT exist** (verified by grepping the full `inferno_` prefix in Valve’s current dump): `inferno_scorch_decals` (present in CS:GO as `1`, **removed in CS2** — decals are now gated by `r_csgo_render_inferno_decals true`), `inferno_spread`, and `inferno_max_child_spawn_distance`. Do not carry these over from CS:GO configs.

**Fire Sale implemented as per-side parameters.** The 2024-05-23 change was not a hardcoded tweak — CS2 carries a full parallel `_ct` parameter set: `inferno_damage_ct 40`, `inferno_flame_lifetime_incendiary 5.5` (vs 7), `inferno_max_range_ct 110` (vs 150), `inferno_velocity_factor_ct 0.003`, and (added 2025-07-16) `inferno_spread_speed_mult_ct 10` (vs 1), all under `inferno_ct_experiment true`. Reading the patch notes carefully: under the **“Incendiary Grenade”** sub-heading sit “Reduced duration” and “Reduced area covered by spreading flames” — **incendiary only**. The single line that applies to both weapons is “Height of the ‘pillar’ of flame for both Molotov and Incendiary grenade now decreases over time.”

### 3.3 Footprint, spread and timing — measured cross-check [M]

The cvar defaults are **confirmed against a real demo** (local probe of `0823_inferno.dem`, 64-tick, from this workspace):

| Thing | Cvar | Measured in demo |
|---|---|---|
| Molotov flame lifetime | 7 s | median **449–450 ticks @64 = 7.016–7.031 s** (450/64 = 7.03125 exactly) |
| Incendiary flame lifetime | 5.5 s | median **352 ticks @64 = 5.500 s** (exactly 5.5) |
| Damage interval | 0.2 s | (not directly measured) |

That two-peak distribution (352 for incendiaries, 449/450 for molotovs) is a clean empirical validation of `inferno_flame_lifetime_incendiary = 5.5` vs `inferno_flame_lifetime = 7`. awpy independently hardcodes `DEFAULT_INFERNO_DURATION_IN_SECS = 7.03125` and `7.03125 × 128 = 900` ticks — the same 7.03125 s **at 128 tick** **[C]**.

**Footprint geometry:**
- **Molotov spread radius = 150 HU (2.86 m)** → diameter 300 HU (5.72 m).
- **Incendiary spread radius = 110 HU (2.10 m)** → diameter 220 HU (4.19 m). This is the CS2 “Fire Sale” reduction; the CS wiki states plainly: *“In Counter-Strike 2, Incendiary Grenades have a smaller radius than Molotovs since the ‘Fire Sale’ update”* — [CS wiki – Incendiary Grenade](https://counterstrike.fandom.com/wiki/Incendiary_Grenade) **[C]**.
- **Fire cells**: up to **16** (`inferno_max_flames`) although the arrays are size 64. Cell **spacing 42 HU**; child spawns recurse up to depth 4 with interval ×0.1 per level, capped at 0.5 s; the initial burst spawns a cell every **0.02 s** so the footprint is essentially fully laid out within ~0.3 s of ignition, then the edges keep creeping for up to the per-flame 3 s spawn window.
- **Does it grow over time?** Yes, but fast: the initial ring is laid at 20 ms intervals, and further spread happens through child flames for up to `inferno_per_flame_spawn_duration = 3 s`, bounded by `inferno_max_range`. The **2024-05-23 “Fire Sale”** update additionally made “**Height of the ‘pillar’ of flame for both Molotov and Incendiary grenade now decreases over time**” **[V]**, and the **2025-07-16** update made the **incendiary spread more rapidly** (“incendiary grenade fire spreads more rapidly”, to “occupy space rapidly to counter rushing Ts”) **[V]**.
- **Spread algorithm internals [V, CS:GO source]:** the bookkeeping/spread pass runs every **0.1 s** (separate from the 0.2 s damage pass). Each spread step for a flame with remaining spawn budget picks a direction away from its parent, deviating by `±inferno_spawn_angle` (**45°**), with a step length of **random(50, 75) HU**, clamped so distance from the ignition point never exceeds `inferno_max_range` (150 / 110 HU). It then traces **down from `pos + 50 HU` with `maxDrop = 200 HU`** (it must hit ground to place a cell) and requires a clear LOS trace at **flame height + 30 HU** — so **walls and doors block spread, but players and chickens do not**. Up to **4 retries** per attempt, at most **`inferno_max_trace_per_tick = 16`** traces per tick, and every failed trace multiplies the splash velocity by `inferno_forward_reduction_factor` (**0.9**). Children get `interval = min(0.1 × parent interval, 0.5 s)` and inherit the parent’s 3 s spawn budget; the tree stops at `depth ≥ 4`. This is what makes molotov spread creep toward openings and stall against walls.

**Detonation:**
- Shatters on a walkable surface whose slope is **≤30°** from horizontal (`weapon_molotov_maxdetonateslope = 30`); otherwise it **bounces**.
- **Air-burst after 2 s of flight** (`molotov_throw_detonate_time = 2`) with no surface contact — *except* if it explodes **within 128 HU above a surface**, in which case the flames fall to the ground and burn **[C]**, [CS wiki – Molotov](https://counterstrike.fandom.com/wiki/Molotov).
- 2026-01-27 patch: grenades that bounce off an enemy player get a **one-time fuse extension** to prevent air-bursting.

**Damage model** **[V from the CS:GO source leak; consistent with the CS wiki and community measurement]:**
- **Per-tick:** the damage think interval is `inferno_damage_timer = 0.2 s`; the base damage is `inferno_damage (40) × 0.2 = 8` per tick.
- **Ramp:** the value is scaled by `m_damageRampTimer.GetElapsedRatio()` over the **first 2.0 s measured from ignition** (not from the time the victim entered the fire) — hence the observed “1 → 8 per hit” climb, saturating at **8 damage/tick = 40 DPS**.
- **Armour does not reduce fire damage at all**: the burn damage type is excluded from armour (`bDamageTypeAppliesToArmor` excludes `DMG_BURN`), i.e. **100 % armour penetration**.
- **Stacking is subtler than “overlapping flames stack” [V]:** each entity is damaged **at most once per 0.2 s per inferno**, because the hit test runs per inferno entity, not per flame cell. So overlapping *cells of the same inferno* do **not** stack; two **separate** inferno entities (e.g. a molotov plus an incendiary, or two grenades) each run their own timer and **do** stack.
- **Hit test [V]:** the inferno collects candidates with `UTIL_EntitiesInBox(..., 256, m_extent.lo, m_extent.hi, 0)`, then requires `IsTouching(entity, flameRadius = 2 × InfernoFire_HalfWidth = 60 HU, checkLOS)`. For players the reference point is raised by **30 HU** and a LOS trace (`INFERNO_MASK_DAMAGE`) must be clear — which is why you can stand just behind a thin lip and take nothing.
- Time-to-kill: **3.6 s** if you stand in it from ignition (40 damage accumulate over the 2 s ramp, then 8 per 0.2 s), or **~2.6 s** if you enter after the ramp has completed. Community measurements say “3.5–4 s” **[C]**. A molotov therefore kills a full-HP player within its 7 s life with time to spare — which is why a molotov denies a 5 s defuse.
- **All flames expire together [V]:** `CreateFire` sets each new flame’s lifetime to `flameLifetime − elapsed`, so the footprint keeps growing but the whole inferno ends at ~7 s from ignition (`inferno_expire`).
- Scorch decals were applied in CS:GO via `UTIL_DecalTrace` on a **100-HU downward trace** using the decal name `MolotovScorch`.
**CS:GO → CS2 parameter drift** (both from Valve data): `inferno_surface_offset` **20 → 15 HU**, and `inferno_dlight_spacing` **200 → 7200 HU**. Everything else in the §3.2 table is unchanged between the games unless marked otherwise.

**On CS:GO column confidence (methodology caveat worth knowing):** several of these cvars (`inferno_initial_spawn_interval`, `inferno_child_spawn_interval_multiplier`, `inferno_forward_reduction_factor`, `inferno_max_child_spawn_interval`, `inferno_velocity_decay_factor`, `inferno_velocity_factor`, `inferno_max_trace_per_tick`) are `FCVAR_GAMEDLL | FCVAR_CHEAT` and **not replicated**. A **client-context** cvarlist dump therefore reports `0` for them; only **server-context** dumps show the real defaults. The CS:GO values used here are confirmed by the leaked source *and* three independent server-side dumps — Mapeadores (`cvarlist.txt`, build 1.37.9.9), [deziau/csgo-list-of-cvars](https://raw.githubusercontent.com/deziau/csgo-list-of-cvars/master/cvars_all.log) and [ArmynC/ArminC-CSGO-Cvars](https://raw.githubusercontent.com/ArmynC/ArminC-CSGO-Cvars/master/cvarlist_all.log) — all agreeing exactly. If you ever cross-check cvars, **take the server-context dump for non-replicated server cvars.** The same trap applies to community CS2 sites: totalcsgo lists `inferno_surface_offset 20` and `inferno_smoke_volume_density 0.05`, both wrong for CS2 (correct: **15** and **0.03**).

### 3.4 How CS2 renders the fire [V evidence, I interpretation]

- **Shader:** `shaders/vfx/inferno.ini`, with `deferred_particle_shadows.ini` for shadows.
- **The fire is a “particle snapshot” system, not free-running particles.** `C_Inferno` holds five `IParticleSnapshot` handles, and the depot has exactly matching snapshot assets:

| Entity handle | Asset |
|---|---|
| `m_hInfernoPointsSnapshot` | `particles/inferno_fx/fire_core.vsnap_c` |
| `m_hInfernoFillerPointsSnapshot` | `particles/inferno_fx/fire_filler.vsnap_c` |
| `m_hInfernoOutlinePointsSnapshot` | `particles/inferno_fx/fire_edge.vsnap_c` |
| `m_hInfernoClimbingOutlinePointsSnapshot` | `particles/inferno_fx/incendiary_groundfire_climbingoutline.vpcf_c` (name matches exactly) |
| `m_hInfernoDecalsSnapshot` | scorch decals — `materials/decals/molotovscorch.vmat_c`, `materials/decals/scorch/scorch1..3.vmat_c` |

This is **the** most actionable finding in this section: the fire is composed of **three layers — a core, a filler and an edge/outline — instantiated at each burning cell**, with a separate climbing-outline variant for fire that runs up walls, and the cells are the 16 `m_firePositions` entries. A faithful-looking viewer reproduces fire by placing a **core sprite + filler sprite + bright edge sprite at each cell**, not by simulating fluid.

**What a `.vsnap` is [V]:** resource type string `"vsnap"`; written by the particle operator `C_OP_ModelSurfaceSnapshotGenerator` (“snapshot control point number”, “input model”, “Recalculation Rate”, “U/V Spacing”, “Surface Offset”, “Set Normal/Up/Gravity/UV”), read by `C_INIT_InitFromCPSnapshot` (“snapshot control point number”, “snapshot subset”, “field to read/write”, “Snapshot increment amount”, “Manual Snapshot Index”), with dev convar `particle_snapshot_allow_combined_models false`. The documented analogue elsewhere in CS2 is `C_BaseCSGrenadeProjectile`, which holds `m_nSnapshotTrajectoryEffectIndex` + `m_hSnapshotTrajectoryParticleSnapshot` + `Vector[] m_arrTrajectoryTrailPoints` — i.e. **the entity simulates a point list and holds a vsnap the particle system reads**. That is exactly the role of the `C_Inferno` snapshot handles, and it means a viewer can reproduce the fire by generating the same point lists rather than by authoring particles. There is a **fourth** fire-relevant vsnap in the depot: `particles/weapons/models/grenade/molotov/weapon_molotov/molotov_ragfx.vsnap_c`.

Note also that CS2 renders these with `C_OP_RenderSprites` (screen-aligned billboards with per-particle radius and `m_flMinSize`/`m_flMaxSize` clamps). **There is no fixed HU flame-size constant in the CS2 renderer** — which is precisely why no CS2 flame-size number is published, and why the CS:GO constants below are the only hard numbers available.

- Ground-glue: `inferno_surface_offset = 15 HU` lifts each flame off the surface. There are also `*_groundfire_projected` and `*_groundfire_remnant` variants for the residual burn/scorch.
- Other fire assets (**157 files** ship under `particles/inferno_fx/` in CS2): `particles/inferno_fx/molotov_groundfire*.vpcf_c` (main, main_center, main_fancy, main_glow, child_base, child_embers, child_glow, endcap, endcap_ground, endcap_groundsmoke, endcap_wall, outline, outlineset, projected, remnant, scortch, snapshot_child_base, streaks, fallback, lighting); 12 `molotov_child_flame*` variants; `particles/inferno_fx/incendiary_groundfire*.vpcf_c` (same family plus `_filler_napalm`, `_filler_replicator`, `_fillsparks`, `_climbing*`, `_outline`); `molotov_bodyburn*` / `incendiary_child_flame01a/03a`; `molotov_explosion*`; `particles/inferno_fx/extinguish_fire*.vpcf_c` (the smoke-extinguish effect); `particles/critters/chicken/chicken_roasted_firensteam.vpcf_c` (yes, really). CS2 also ships `/materials/sprites/white.vmat` and `materials/dev/inferno.vmat`, both precached by the Inferno modules.
- **Flame sprite textures:** `materials/particle/fire_onsurface/fire_onsurface.vtex_c` (2.07 MB) and `fire_onsurface_2.vtex_c` (11.5 MB, the highest-resolution fire texture in the game), `fire_small_sim/fire_small_sim_{a,b}_v02*` (up to 9 MB), `fire_gas/fire_gas_batch_{a,b}{,_top}{,_mv,_v2}`, `flames/{flames, flame_omni, flame_wisp, flame_dissipate, beam_flame}.vtex_c`, `fancy_fire_particle_001/002`, `fire_particle_2/3`, `fire_burst`, `fire_fast_side`, `fire_explosion_1`, `fire_climbing` (the wall-climb fire), `fire_burning_character`.
- **Flame geometry (CS:GO source constants — the closest thing to a published number):** per-inferno `m_maxFireHalfWidth = 30.0 HU`, `m_maxFireHeight = 80.0 HU`; server constants `InfernoFire_HalfWidth = 30.0`, `InfernoFire_FullHeight = 80.0`. Each flame gets a random `m_maxSize` in **70–90 HU**, and the drawn quad is **half-width = size/3, height = size** — i.e. a flame cell is a roughly **20-HU half-width (40 HU wide) × 70–90 HU tall** billboard. So the realistic working figure is **~80 HU (2.0 m) tall flames standing on ~60-HU-diameter ground patches**, which sits inside the mapper-folklore “64–100 HU” range. **[V for CS:GO; CS2 highly likely]**
- **In CS2, read the real extents from the entity instead:** `m_maxFireHalfWidth` and `m_maxFireHeight` are client-side netprops on `C_Inferno` (updated live, since the 2024-05-23 patch makes the flame pillar’s height decrease over time), and `m_extent` is the server-side AABB. Number of flames per inferno is `m_fireCount` (≤ `m_nMaxFlames`, i.e. ≤ 16), and the fire is **one cell = one core + one filler + one edge instance**, not a fixed sprite count.
- **Flame cell lifecycle — the fire “bloom” timing [V, CS:GO source]:** exactly **one sprite per burning cell** (`SynchronizeDrawables()` walks the `m_fireCount` networked entries and matches each to a Drawable by position within a 12 HU tolerance). A new cell’s size grows from 0 to its `m_maxSize` at **`m_maxSize / 2` per second ⇒ 2.0 s to reach full size**, then shrinks back to 0 over another **2.0 s** once the server stops marking it burning (state machine `STARTING → BURNING → GOING_OUT → FIRE_OUT`). Each flame also gets a random `m_framerate` of **0.04–0.06** and a 50 % chance of horizontal mirroring. **For a faithful viewer: grow each flame over ~2 s, hold, then shrink over ~2 s — do not pop flames in and out.**
- **Lighting:** fire emits dynamic lights. In CS2, `inferno_dlights = 30` (a *minimum-FPS gate*, not a count) and `inferno_dlight_spacing = 7200` HU minimum separation (CS:GO used **200** HU). CS:GO’s elight parameters, for reference: one light per *other* flame, origin = flame position **+64 HU in z**, colour **rgb(254, 100, 10)**, radius random **60–120 HU**, `exponent = 5`. The fire also contributes a smoke volume (`inferno_smoke_volume_density = 0.03`) that the volumetric smoke system composites. CS:GO’s particle path (selected by `inferno_fire = 2`) created **one particle system per inferno, named `molotov_groundfire`, with one control point per flame** — the same architecture CS2 uses via the `.vsnap` point lists.

### 3.5 CS2 fire patch history (behaviour you may need to match per-era demo) [V]

| Date | Change |
|---|---|
| 2023-09-06 | Fixed fire area being small in corners; fixed grenades exploding at the thrower’s feet |
| 2023-10-02 | Fixed a fire ring in the air; improved extinguish performance |
| 2024-03-13 | Adjusted molotov fire so a player who could see through it could also be seen |
| **2024-05-23 (“Fire Sale”, also 2024-05-24 post)** | “Height of the ‘pillar’ of flame for both Molotov and Incendiary grenade now decreases over time.” · Incendiary: “Adjusted the explosion and flame visual treatment · **Reduced duration** · **Reduced area covered by spreading flames** · Reduced price $600 → $500”. This is the source of `inferno_flame_lifetime_incendiary = 5.5` and `inferno_max_range_ct = 110`. |
| 2024-11-14 | Performance optimisations for molotov/incendiary effects |
| **2025-07-16 (Season Two)** | “Incendiary Grenade — incendiary grenade fire spreads more rapidly” |
| 2026-01-27 | Bounce off an enemy player → one-time fuse extension |
| 2026-07-21 | Bomb explosion now interacts with smoke clouds and molotov/incendiary fire |

**Practical implication:** if you replay 2023–early-2024 demos, the incendiary was *larger and longer* than the current values. Use the demo’s build date to pick the parameter set.

---

## 4. Player models

### 4.1 Collision hull, view offsets, eye height [V for CS:GO source; high confidence for CS2]

CS:GO shipped source (`g_CSViewVectors`, <https://raw.githubusercontent.com/perilouswithadollarsign/cstrike15_src/master/game/shared/cstrike15/cs_gamerules.cpp>):

```cpp
Vector( 0, 0, 64 ),    // VEC_VIEW        eye position
Vector(-16,-16, 0 ),   // VEC_HULL_MIN
Vector( 16, 16, 72 ),  // VEC_HULL_MAX
Vector(-16,-16, 0 ),   // VEC_DUCK_HULL_MIN
Vector( 16, 16, 54 ),  // VEC_DUCK_HULL_MAX
Vector( 0, 0, 46 ),    // VEC_DUCK_VIEW   duck eye position
```

| Quantity | Value |
|---|---|
| Standing hull | (−16,−16,0) → (16,16,**72**) — 32×32×72 HU (0.61×0.61×1.372 m) |
| Standing eye height | **64.093811 HU** (mapper-ref measured; 64.0 nominal) |
| Crouching hull | (−16,−16,0) → (16,16,**54**) — 32×32×54 HU (1.029 m) |
| Crouching eye height | **46.076218 HU** (46.0 nominal) |
| Standing feet level | 0.031250 HU |
| Crouch transition | `TIME_TO_DUCK_MSECS = 200` for CS (vs 400 for other Source games) |
| Observer hull | 20×20×20 HU (±10) |
| Dead view height | 14 HU |
| **Origin** | **at the feet** (hull min z = 0); eye = origin + view offset |

Source: [Valve CS:GO Mapper’s Reference (archived)](https://web.archive.org/web/20250101150843/https://developer.valvesoftware.com/wiki/Counter-Strike:_Global_Offensive/Mapper%27s_Reference) **[V]**.

**CS2 uses the same dimensions.** The Valve wiki *Dimensions* page states for Source 2 games: *“For Counter-Strike 2 dimensions, see Counter-Strike: Global Offensive/Mapper’s Reference (same dimensions apply)”* — [archived](http://web.archive.org/web/20241228110533/https://developer.valvesoftware.com/wiki/Dimensions) **[V]**. Independent corroboration from Valve’s own shipped CS2 nav data, `game/csgo/pak01_dir/scripts/nav_hulls.vdata`: `m_agentRadius = 16.0` (⇒ 32×32 footprint), `m_agentShortHeightEnabled = true`, `m_agentMaxClimb = 16.0`, `m_agentMaxJumpUpDist = 68.0`, `m_agentMaxJumpDownDist = 157.0` **[V]**. ⚠️ There is no CS2 *source* dump (the game is closed-source), so “CS2 = CS:GO hull” is *very high confidence by equivalence*, not a byte-level proof. **Note also that generic Source SDK values are wrong for CS** (default Source duck hull is 36, duck view 28) — do not use those.

**Procedural fallback for the viewer (recommended MVP):**
- Standing: box or capsule **32 × 32 × 72 HU**, origin at the centre of the bottom face, eye marker at **z = +64**.
- Crouched: **32 × 32 × 54 HU**, eye at **z = +46**; animate the transition as a **200 ms** lerp of height 72→54 and eye 64→46 (18 HU of view drop).
- Note the crouch footprint is **32×32 in both stances** (the engine requires it); only the height changes.

### 4.2 Hitboxes and hitgroups [V for structure/IDs; [C] for the numbers]

**CS2 hitgroup IDs are shipped in Valve’s schema** (`DumpSource2/schemas/client/HitGroup_t.h`) — use this, not plugin bindings:

```
HITGROUP_INVALID = -1, GENERIC = 0, HEAD = 1, CHEST = 2, STOMACH = 3,
LEFTARM = 4, RIGHTARM = 5, LEFTLEG = 6, RIGHTLEG = 7,
NECK = 8, UNUSED = 9, GEAR = 10, SPECIAL = 11, COUNT = 12
```
⚠️ **Citation trap:** CounterStrikeSharp’s generated `EHitGroup` uses `Gear = 8` / `Miss = 9`, contradicting Valve’s CS2 schema (`NECK = 8`, `GEAR = 10`). Cite `HitGroup_t.h`.

**CS2 hitboxes are per-model, stored inside the `.vmdl`.** The compiled structure (`DumpSource2/schemas/modellib/CHitBox.h`):

```cpp
CUtlString m_name; CUtlString m_sSurfaceProperty; CUtlString m_sBoneName;
Vector m_vMinBounds; Vector m_vMaxBounds;
float32 m_flShapeRadius;   // capsule/sphere radius
int32   m_nGroupId;        // hitgroup id
uint8   m_nShapeType;      // box / capsule / sphere
bool    m_bTranslationOnly;
```
The presence of `m_flShapeRadius` and `m_nShapeType` is **structural proof that CS2 hitboxes are not box-only** — corroborating the community claim that CS2 moved to **capsules** (CS:GO used boxes for feet/hands). Source 2 Viewer already exports `Box`/`Capsule`/`Sphere`.

**Damage multipliers** (CS:GO-verified from `CCSPlayer::TraceAttack`; CS2 server is closed-source but every community CS2 source agrees) **[C, CS:GO-verified]**:

| Hitgroup | Multiplier |
|---|---|
| HEAD (1) | **×4** (halved by heavy armour; then `mp_damage_scale_*_head`) |
| CHEST (2) | ×1.0 |
| STOMACH (3) | **×1.25** |
| ARMS (4,5) | ×1.0 |
| LEGS (6,7) | **×0.75** |
| GENERIC/GEAR | ×1.0 / no damage |

**Head hitbox equalisation:** Valve’s only official note is the 2023-10-09 patch, `[ANIMATION]` — *“Fixed several hitbox alignment bugs”* (<https://store.steampowered.com/news/app/730/view/3747615443061321346>) **[V]**. The stronger claim — that CS2 **equalised head hitbox volume across all agents** (increasing it **4–20 %** for shorter/rounder agents) and moved to capsules — comes from Ansimist’s reverse engineering reported by csgo.com/ru.csgo.com **[C]**. In CS:GO, head hitbox volume differed by up to **17 %** between agent models.

**No public numeric hitbox bounds in HU exist.** They must be read from a decompiled `agents/models/**/*.vmdl_c` in Source 2 Viewer (hitbox sets are attached via `CRenderMesh.m_hitboxsets`; ValveResourceFormat models them as `ModelData.Hitbox` / `HitboxSet` / `HitboxShape`). This is ~1 CLI command if you have a CS2 install.

### 4.3 Model paths — **the brief’s premise is wrong: `models/players/` does not exist in CS2** [V]

Verified against Valve’s shipped CS2 depot manifest (135,686 entries):
- `grep -c 'models/players'` → **0**
- `grep -c '.mdl ' / .vvd / .vtx` → **0** (no Source 1 formats at all)
- `.vmdl_c` entries → **12,967**
- the only `/players/` path is UI art (`panorama/images/econ/tournaments/players`)

**Correct CS2 scheme:** `agents/models/<family>/<family>[_variantX].vmdl` (shipped compiled as `.vmdl_c`).

**11 families, 80 player body models + 12 glove/arms models:**

| Team | Family | Base (non-variant) model | Variants |
|---|---|---|---|
| CT | `ctm_sas` | `agents/models/ctm_sas/ctm_sas.vmdl` ← **default CT** | variantf, variantg |
| CT | `ctm_fbi` | `agents/models/ctm_fbi/ctm_fbi.vmdl` | a–h (8) |
| CT | `ctm_st6` | — (variants only) | e, g, i, j, k, l, m, n (8) |
| CT | `ctm_swat` | — | e–k (7) |
| CT | `ctm_gendarmerie` | — | a–e (5) |
| CT | `ctm_diver` | — | a, b, c (3) |
| T | `tm_phoenix` | `agents/models/tm_phoenix/tm_phoenix.vmdl` ← **default T** | a–d, f, g, h, i |
| T | `tm_professional` | — | varf, varf1–f5, varg, varh, vari, varj (10) |
| T | `tm_balkan` | — | f–l (7) |
| T | `tm_jungle_raider` | — | a, b, b2, c, d, e, f, f2 (8) |
| T | `tm_leet` | — | a–k |
| shared | `agents/models/shared/arms/` | — | 12 gloves: `glove_bloodhound`, `glove_brokenfang`, `glove_cloth_collision`, `glove_fingerless`, `glove_fullfinger`, `glove_handwrap`, `glove_hardknuckle`, `glove_hydra`, `glove_motorcycle`, `glove_slick`, `glove_specialist`, `glove_sporty` |

**Only three families ship a base (non-variant) model:** `ctm_sas` (default CT, used by all CT map-based defaults), `tm_phoenix` (default T, used by all T map-based defaults) and `ctm_fbi` (exists but is unused as a map default). `ctm_st6`, `ctm_swat`, `ctm_gendarmerie`, `ctm_diver`, `tm_balkan`, `tm_jungle_raider`, `tm_leet` and `tm_professional` ship **only** as `_variantX` files. If you need one representative model per side, use `ctm_sas` and `tm_phoenix`.

**Do not use these names — they exist only as econ item names, not as model files:** `ctm_gign`, `ctm_gsg9`, `ctm_idf`, `tm_anarchist`, `tm_pirate`, `tm_separatist`. In `items_game.txt` they are aliases that all resolve to just three files (`ctm_sas.vmdl`, `tm_phoenix.vmdl`, `ctm_fbi.vmdl`) **[V]**. Likewise the per-map `t_models`/`ct_models` blocks and the `ct_arms`/`t_arms` keys in `gamemodes.txt` are **inherited CS:GO keyvalues with no CS2 files behind them** — CS2 ships exactly one shared arms model, `weapons/models/shared/arms/weapon_arms.vmdl`, plus the 12 `glove_*` models **[V]**. The only `models/player/` survivors in the whole CS2 depot are two contact-shadow meshes: `models/player/contactshadow/contactshadow_leftfoot.vmdl` and `…_rightfoot.vmdl`.

**Second, parallel tree:** `characters/models/<family>/<file>.vmdl_c` exists with the same names **plus** extras (`ctm_heavy`, `tm_phoenix_heavy`, `tm_jumpsuit_varianta..c`, `shared/observer/cs_observer`, `shared/spawn/{ctm,tm}_spawnpoint`, `shared/particles/character_fxmodel*`). The `characters/` copies are **orders of magnitude smaller** (`characters/models/ctm_sas/ctm_sas.vmdl_c` = **4,791 bytes** vs `agents/models/ctm_sas/ctm_sas.vmdl_c` = **563,655 bytes**), so `characters/` is a thin document/legacy layer. **Load from `agents/models/`** **[V/I]**.

Weapon world models live at `weapons/models/<weapon>/weapon_*.vmdl` (e.g. `weapons/models/ak47/weapon_rif_ak47.vmdl`); equipment at `models/weapons/w_eq_armor.vmdl`.

**Agent ↔ path ↔ market-name mapping:** the community JSON API regenerated from `items_game.txt` is authoritative and machine-readable — <https://raw.githubusercontent.com/ByMykel/CSGO-API/main/public/api/en/agents.json> (each entry carries `model_player`).

### 4.4 Are the models skeletal? Yes — Animgraph 2, one shared skeleton [V]

| Extension | Count in CS2 depot | Meaning |
|---|---|---|
| `.vnmclip_c` | **2,355** | animation clips (Animgraph 2) |
| `.vnmgraph_c` | **231** | animation graphs |
| `.vnmskel_c` | **68** | skeletons |
| `.vanim_c` / `.vagrp_c` / `.vseq_c` / `.mdl` / `.vvd` / `.vtx` | **0** | Source 1 animation formats are gone |

**One shared skeleton for every human:** `animation/skeletons/characters/worldmodel.vnmskel_c` (all third-person players), `viewmodel.vnmskel_c` (first-person arms), `chicken.vnmskel_c`. That shared skeleton is the *structural reason* every agent has identical hitboxes — hitboxes bind to bones.

**Canonical skeleton:** `worldmodel.vnmskel`, **74 bones**, root `root_motion` → `pelvis`:
```
root_motion → pelvis
pelvis → spine_0 → spine_1 → spine_2 → spine_3 → neck_0 → head_0
spine_3 → clavicle_L/R → arm_upper_L/R → arm_lower_L/R → hand_L/R
pelvis → leg_upper_L/R → leg_lower_L/R → ankle_L/R → ball_L/R
root_motion → wpnPivot → wpn (wpnHand_L/R, wpnTip, wpnEnd, wpnAimIntent under it)
hand_L/R → attachHand_L/R ; ankle_L/R → attachFoot_L/R ; root_motion → attachWorld
+ 5-finger rigs: finger_{thumb,index,middle,ring,pinky}_{meta,0,1,2}_{L,R}
```
Note: there is **no `spine_4`** (chain is `spine_0`→`spine_3`→`neck_0`), and **uppercase `_L`/`_R` is mandatory** — lowercase rigs do not bind. (Bone list from an in-game-verified community porter, <https://github.com/SergeantCS2/CS2-AG2-Model-Porter>.)

**CS2 animation clip names are real and weapon-suffixed.** The CS:GO `idle` / `walk` / `run` / `crouch_walk` / `death` names in the brief **do not exist in CS2**. The actual clips live at `animation/anims/world/<knife|pistol|rifle|grenade|equipment|shared>/` and look like:

- Locomotion: `idle_knife`, `idle_crouch_knife`, `walk_n_knife` … `walk_nw_knife` (8-way, plus `_bck`/`_fwd`/`_mir`), `run_n_knife` …, `crouch_n_knife` …, `jump_stand_knife`, `jump_n_knife`, `inair_stand_knife`, `inair_e_knife`, `turn_left_90_knife`, `draw_knife`, `draw_crouch_knife`
- Shared (96 clips): `breathing`, `flashed`, `flashed_crouch`, `death_chest_a/b`, `death_gut_a/b`, `death_rknee_a/b`, `death_rshoulder`, `death_lowviolence`, `death_poses_body`, `death_poses_head`, `death_poses_crouch_body/head`, `flinch_head` (+`_left/_right/_rear/_knife/_pistol`), `flinch_chest*`, `flinch_stomach*`, `flinch_arm_left/right*`, `flinch_leg_left/right*`, **`flinch_molotov*`**, `jump_additive`, `jump_additive_start`, `jump_additive_land`, `jump_additive_crouch_start/land`
- C4: `equipment/c4/planting`, `planting_crouch`, `idle_c4`, `idle_crouch_c4`, `turn_*_c4`
- Defuse: `defuse/defuse_enter_{rifle,pistol,knife}`, `defuse_loop_*`, `defuse_crouch_enter_*`, `defuse_crouch_loop_*`
- Grenades: `grenade/_default_grenade/{idle, idle_crouch, draw, draw_crouch, pullpin_crouch, crouch_throw_near, crouch_throw_far}_grenade`
- Reloads: e.g. `pistol/pistol_glock/reload_glock`, `reload_empty_glock`, `reload_crouch_glock`
- Ladder: `ladder_facing_{fast,slow,crouch}_knife` + rotations; planted/defusing leg sets `planted_e2w_knife`, `planted_start_n`, …
- UI: `animation/anims/ui_anims/...` + `animation/graphs/ui/uimodel_celebration.vnmgraph+celebrate_<agent>` / `+defeat_<agent>`

Graphs: `animation/graphs/worldmodel/worldmodel_gun.vnmgraph_c` + ~30 per-weapon combos (`+ak47`, `+awp`, `+deagle`, …), `worldmodel_grenade.vnmgraph_c` + 6 combos. There is **no separate third-person locomotion graph file** in the manifest.

⚠️ **For a CS:GO-era reference only** (do not present as CS2): CS:GO’s sequence contract was `#define DEFAULT_IDLE_NAME "idle_upper_"`, `DEFAULT_CROUCH_IDLE_NAME "crouch_idle_upper_"`, `DEFAULT_CROUCH_WALK_NAME "crouch_walk_upper_"`, `DEFAULT_WALK_NAME "walk_upper_"`, `DEFAULT_RUN_NAME "run_upper_"`, plus `deploy_%s`, `reload_<suffix>`, `%s_light_l/r/heavy` — from `cs_playeranimstate.cpp`. So `idle_all_01`, `run_all_01`, `death_01`, `ragdoll` were never CS sequence names either; they are HL2/Source-SDK folklore.

### 4.5 Agent heights

Visual silhouettes differ (bulky backpacks/helmets vs slim), but the **collision hull and hitbox set are identical across all agents** — one shared skeleton, one hitbox layout, and (per community reverse-engineering) head hitbox volume equalised in the CS2 Limited Test. Shots that clip a backpack or helmet edge do no damage. **[C]** — no Valve note with numbers exists.

---

## 5. Existing tools: how third parties render CS2 replays

### 5.1 Bottom line

**No open-source project renders a CS2 replay in 3D with real extracted player models and volumetric smoke.** That combination does not exist publicly as of Sept 2026. The pieces exist separately (asset export works; parsers work; one project renders real collision geometry in the browser) but nobody has joined them.

### 5.2 The projects that matter

| Project | URL | 3D? | Assets | Smoke / fire | Players | Licence |
|---|---|---|---|---|---|---|
| **programmer1o1/demoscope** | <https://github.com/programmer1o1/demoscope> | **Yes**, real 3D (Rust + WASM → self-contained HTML) | **Real CS2 map geometry decoded from the map `.vpk`** — VPK v2 → Source 2 resource → KV3 → `world_physics.vmdl_c` **collision mesh only**. `de_nuke`: 5,988 hulls + 11 meshes = 115,738 verts / 191,628 tris. Textured render meshes explicitly out of scope. | Smokes not rendered. Molotov not rendered as fire (but resolves thrower via `CInferno.m_hOwnerEntity`). | **Simple coloured boxes**, death-hiding via `m_lifeState`, held weapon, WASD input overlay, kill/teleport arcs | **NO LICENCE FILE** — all rights reserved. Read-only reference; do not copy code. |
| **CS2.CAM** | <https://cs2.cam> · <https://cs2.cam/en/wiki/2d-viewer-basics> | **Yes** — shipped paid “3D Replay Mode”, toggles 2D↔3D preserving filters/playback/drawings | **Unverified** — wiki never mentions map geometry, `.vpk` or Source 2 assets; may be extruded radar rather than real meshes | Documented in **2D**: “Smokes react to HE grenades just like in CS2, showing the extinguish effect”; “Molotovs & Incendiaries — display the actual spread pattern from the game” | HUD + per-tick positions/view angles (`setpos …; setang …` copy). No models mentioned. | Closed-source commercial SaaS; 3D is paid-tier |
| **Skinshotter** | <https://skinshotter.com> · [Babylon.js writeup](https://forum.babylonjs.com/t/skinshotter-inspecting-cs2-skins-in-browser-with-babylon-js-webgpu-source-2-shaders-ported-to-wgsl/63763) | Yes, high fidelity, **WebGPU** | **Real CS2 assets + real Source 2 shading**: paint-finish shaders reverse-engineered from compiled game code and reimplemented in **WGSL**; **real lighting data exported from actual maps**; Source 2 lighting model reimplemented rather than swapped for stock PBR | None (skin inspector) | **Real agent models and gloves** load and render | Closed-source; author has offered licensing contact — a licensing conversation is viable |
| **awpy** | <https://github.com/pnxenopoulos/awpy> | **No — 2D radar only** (matplotlib) | Official radar PNGs + `.nav` meshes | `dem.smokes` / `dem.infernos` dataframes; drawn as 2D circles; `plot.map_control(method="raycast"/"vision"/"reachability")` uses active smokes/molotovs for LOS | 2D markers with yaw, HP/armour | **MIT** |
| **reka-ai/cs2-dem-renderer** | <https://github.com/reka-ai/cs2-dem-renderer> | **No renderer** — it **launches real CS2** with a version-matched C++ server plugin, replays recorded inputs, and captures CS2’s own frames to HEVC MP4 | The real game’s assets, by construction | Whatever CS2 renders — pixel-perfect | Real models | No licence file — source-available, unclear |
| **akiver/cs-demo-manager** | <https://github.com/akiver/cs-demo-manager> | No — 2D viewer (Electron) | Radar images with lower-radar layering for multi-level maps; drawing layer; voice-chat extraction | 2D only | 2D dots | **MIT** (~2,000★) — also the reference implementation for the CS2 server-plugin replay technique |
| **Igor-Losev/deadem** | <https://github.com/Igor-Losev/deadem> | No | Pure JS/TS Source 2 protobuf+snappy parser and player (≈21.5k ticks/s) | none | none | **MIT** |
| **sparkoo/csgo-2d-demo-viewer** | <https://github.com/sparkoo/csgo-2d-demo-viewer> | No | Go parser → **WASM**, protobuf → Preact | 2D | 2D | MIT |
| **khanghugo/kdr** | <https://github.com/khanghugo/kdr> | Yes, 3D (GoldSrc) | GoldSrc BSP | — | — | **GPL-3.0** — licence-clean modern Rust 3D map+demo viewer |
| **oscarbrooks/CSGO_DemoInspector** | <https://github.com/oscarbrooks/CSGO_DemoInspector> | Yes, 3D (C#, 2018, Source 1) | Source 1 | — | — | No licence; stale |
| **HungryProton/godot_volumetric_smoke** | <https://github.com/HungryProton/godot_volumetric_smoke> | Yes | — | “loosely emulate the volumetric smoke for CS2”; author warns: not a voxel grid, subject to **leaks**, not optimised, **not a drop-in solution** | — | MIT |

**Corrections to common claims:**
- **Leetify has no publicly evidenced 3D replay.** Full blog index plus targeted searches found no 3D feature; their pipeline is server-side demo parsing + **video** highlights. Treat any “Leetify 3D replay” claim as unsupported unless proven with a logged-in screenshot.
- **Scope.gg is 2D, not 3D.** Their own copy says “watch the **2D replay**” and “**2D-demo viewer**”; the 0.25 changelog “new **3D effects** in Replay” means 3D-styled VFX inside the 2D replay, not a 3D scene. ([blog.scope.gg/scopegg-cs2-en](https://blog.scope.gg/scopegg-cs2-en/), [update 0.25](https://blog.scope.gg/scopegg-update-025-en/))

### 5.3 Real CS2 assets → glTF: the pipeline that works [V/C]

**ValveResourceFormat / Source 2 Viewer** — <https://github.com/ValveResourceFormat/ValveResourceFormat> · <https://s2v.app> · [format support matrix](https://s2v.app/ValveResourceFormat/guides/format-support.html):
- `vmdl` (model) → **glTF/GLB**, `vmap`/`vwrld` (map) → glTF/GLB; also `vmesh`, `vnmclip`, `vnmskel`, `vphys`, `nav`.
- Exports geometry, up to 8 bone weights/vertex, morph targets, skeletal/additive/morph **animations**, PBR-approximated materials with ORM repacking, and collision as a separate `<name>_physics` file.
- Every exported material carries the **original shader name + parameters in a `vmat` extras block** — this is the hook for recovering Source 2 semantics instead of accepting PBR flattening (exactly what Skinshotter exploited).
- **Documented losses you must plan around:** only the highest LOD; layered blend materials export one layer only; cloth sheen has no glTF equivalent; no **hitboxes**; **no baked lighting (lightmaps/light probes) exported**; for maps only `light_environment` exports (point/spot lights don’t); 3D skybox not bundled; clutter ignored; >2 GB exports must use `.gltf` (chunked) not `.glb`.
- CLI animation export must use the **weapon-suffixed CS2 clip names** (`idle_knife, run_n_knife, …`), not `idle`/`run`.
- Known issue: **“Invalid skinning bone index”** when decompiling CS2 character models — <https://github.com/ValveResourceFormat/ValveResourceFormat/issues/845>.

**Alternative / supplementary:** `REDxEYE/SourceIO` (Blender addon) for offline import; `MurkyYT/cs2-map-icons` for radars **and `overview` data** (the world→radar-pixel transform: `pos_x/pos_y/scale`) — needed for any 2D minimap or 2D/3D registration.

**Licence warning:** extracted Valve assets remain Valve copyright. `demoscope` and `cs2-map-icons` have **no licence**; Skinshotter is licensable on request. Licence-clean reference points: `kdr` (GPL-3.0), `cs-demo-manager` (MIT), `deadem`/`Rupas1k/source2-demo` (MIT/Apache-2.0).

### 5.4 Smoke in third-party tools — the honest state

Demos carry smoke **events** (`smokegrenade_detonate`, `smokegrenade_expired`) plus the entity origin and — in CS2 — the networked `m_VoxelFrameData` field. **No public project consumes the voxel data.** In practice the field comes through as zeros in `demoparser2`’s `parse_grenades(extra=[...])` output (measured locally on `0823_inferno.dem`: `m_VoxelFrameData`, `m_nVoxelFrameDataSize`, `m_nVoxelUpdate` all present but zero-valued across 187,833 rows, while `m_vSmokeColor` = `[180,129,50]` and `m_vSmokeDetonationPos` populate correctly) **[M]**. Conclusion: **treat volumetric smoke as something you re-simulate or approximate; you cannot currently read CS2’s voxel grid out of a demo with off-the-shelf tooling.**

### 5.5 Where to build from

1. **Parser (licence-clean):** `LaihoE/demoparser` (demoparser2, Rust/Python — what awpy uses), `markus-wa/demoinfocs-golang`, `Rupas1k/source2-demo` (Apache-2.0, Rust), `Igor-Losev/deadem` (MIT, JS/WASM). `demoscope` is **no-licence — reference only**.
2. **Map geometry, two routes:** *fast/grey* — decode `world_physics.vmdl_c` from the map VPK (collision mesh; geometrically exact, untextured, and gives you raycast/LOS collision for free); *faithful* — Source 2 Viewer `vmap` → glTF for real render meshes + materials, then relight (no lightmaps exported) and feed a custom material layer from the `vmat` extras block.
3. **Players:** VRF `vmdl` → glTF (skeleton + animations export). Nobody has done this in a replay yet — it is the gap you would be filling. MVP is correctly-dimensioned capsules at the HU values above.
4. **Smoke:** don’t promise volumetric fidelity from demo data. Model extinguish (`hegrenade_detonate` near an active smoke) and it will *read* as correct to players even if the volume is approximate.
5. **Molotov:** model the **spread footprint from `C_Inferno.m_firePositions`**, not particles. The gameplay-truthful part is which ground denies movement; flame particles are cosmetic.
6. **“Faithful” = shaders, not assets.** Skinshotter is both proof and warning: fidelity came from reverse-engineering Valve’s finish shaders and reimplementing the Source 2 lighting model + real per-map lighting data in WGSL — **not** from loading a GLB.

---

## 6. Camera, lighting and the “CS2 look”

### 6.1 FOV — **vertical 73.7398°** is the number you actually want

Valve’s shipped defaults **[V]**:
```
default_fov 90        (clientdll, cheat)     <- the world FOV
fov_cs_debug 0        (clientdll, cheat)     "Sets the view fov if cheats are on."
fov_desired 75        (clientdll, archive, userinfo)  <- legacy, does not affect CS2 world FOV
viewmodel_fov 60      (clientdll, archive, userinfo, per_user; range [60, 68])
viewmodel_offset_x/y/z   1 / 1 / -1
viewmodel_presetpos 1 (Desktop)
```

Source specifies FOV **horizontally on a 4:3 rectangle**, expanding horizontally for wider aspects (**Hor+**), not cropping vertically — [Valve wiki Field of View](https://web.archive.org/web/20241206094720/https://developer.valvesoftware.com/wiki/Field_of_View). The 90° value is the 4:3 horizontal reference, which corresponds to a **fixed vertical FOV of 73.7398°** = `2·atan(tan(45°)·3/4)`.

| Aspect | Horizontal | **Vertical** | Diagonal |
|---|---|---|---|
| 5:4 | 86.30° | **73.7398°** | 100.42° |
| **4:3** | **90.00°** | **73.7398°** | 102.68° |
| **16:10** | 100.39° | **73.7398°** | 109.51° |
| **16:9** | **106.26°** | **73.7398°** | 113.66° |
| 21:9 | 120.51° | **73.7398°** | 124.58° |
| 32:9 | 138.89° | **73.7398°** | 140.30° |

The derivation is validated by an independent community measurement that reports **73.74° vertical / 106.26° horizontal / 113.66° diagonal** for CS2 at 16:9 — <https://steamcommunity.com/app/714010/discussions/0/4289187621809424581/> **[C]**.

**The single most common bug in web CS2 viewers:** three.js/Babylon `PerspectiveCamera.fov` is **vertical**, so set **`fov = 73.7398°`** and let aspect derive horizontal. Do not hard-code 90 as a vertical FOV, and do not hard-code 106.26 as a horizontal unless you also force the aspect.

**Spectator / demo FOV:** CS2 exposes **no** `spec_fov`, `fov_tv`, `cl_fov` or `sv_fov` — a full grep of Valve’s convar and command dumps finds none, and the ~195-entry demo/GOTV convar group has no FOV entry **[V]**. The spectator camera therefore uses the normal 90°/4:3 path ⇒ **vertical 73.74°, 106.26° horizontal at 16:9**. (Confidence: high but *by absence* — no source states a spectator FOV number outright.) What does exist: `spec_chasedistance 96` (range 16–296), `cl_obs_interp_enable true`, `spec_autodirector true`, `mp_forcecamera 1`.

**Viewmodel:** default **60**, clamped **[60, 68]** in current CS2 (the old CS:GO floor of 54 no longer applies). Rendered in a **separate pass** with its own projection — changing it does not change world FOV. At 60 horizontal-on-4:3 that is 46.83° vertical / 75.18° horizontal at 16:9.

### 6.2 How CS2 maps are lit [V]

From Valve’s CS2 Lighting page ([archived](https://web.archive.org/web/20251116085233/https://developer.valvesoftware.com/wiki/Counter-Strike_2_Workshop_Tools/Level_Design/Lighting)):
- **Baked lightmaps** on static geometry (GPU-raytraced bakes; release settings Resolution 8192 / Quality Final).
- **Light probe volumes + cubemaps** (`env_combined_light_probe_volume` = cubemap box + light probe volume) for dynamic objects and reflections. Voxel sizes **12 / 24 / **48 (default)** / 108 HU**; box mins/maxs default `-72 -72 -72` / `72 72 72`.
- **Indirect/bounce lighting is baked** into lightmaps + probes; artists use `light_rect` to fake bounces (especially on Inferno).
- **No volumetric fog** — “Counter-Strike 2 does not use the volumetric fog system utilized by Half-Life: Alyx”.
- **CS2 excludes several post entities from Hammer** (`csgo.fgd`): `env_tonemap_controller`, `trigger_tonemap`, `env_volumetric_fog_controller`, `env_volumetric_fog_volume`, `fog_volume`, `color_correction`, `color_correction_volume`, `postprocess_controller`. **There is no `env_cs_tonemap` anywhere.** Exposure is controlled **only** by `post_processing_volume`. (The engine schema still contains `CTonemapController2`, but CS2 does not expose it.)
- `light_environment` defaults (`game/core/lights_base.fgd`): `color 255 255 255`, **`brightness 1.0`**, `castshadows 1`, `angulardiameter 1.0` (shadow softness), `numcascades 3`, `nearclipplane 1`, `cascadecrossfade 0.1`, `cascadedistancefade 0.05`, `bouncescale 1.0`, `skytexture ""` (points at an `env_sky`), `skyintensity 1.0`, `skytexturescale 1.0`, `skybouncescale 1.0`, `skyambientbounce 0 0 0`, **`sunlightminbrightness 32`** (clamps the sun disc out of the IBL so the sun isn’t double-counted), `minroughness 0`, `ambient_occlusion 0`, `range 512`, `directlight 3` (Stationary), `baked_light_indexing 1`.

### 6.3 Tone mapping and exposure — the authoritative numbers [V]

Engine default tone-mapping layer (`CToneMappingLayer.h`, <https://raw.githubusercontent.com/SteamTracking/GameTracking-CS2/master/DumpSource2/schemas/resourcecompiler/CToneMappingLayer.h>) — these are exactly the **Hable / Uncharted-2** constants:

```
m_flExposureBias          0.0
m_flShoulderStrength      0.15
m_flLinearStrength        0.50
m_flLinearAngle           0.10
m_flToeStrength           0.20
m_flToeNum                0.02
m_flToeDenom              0.30
m_flWhitePoint            4.0
m_flLuminanceSource       0.0
m_flMinShadowLum 0.0   m_flMaxShadowLum 0.5
m_flMinHighlightLum 2.0   m_flMaxHighlightLum 8.0
```

`post_processing_volume` CS2 defaults (`game/csgo/postprocessing.fgd`): `minexposure 1.0`, `maxexposure 1.0`, `enableexposure 1`, `exposurespeedup 1`, `exposurespeeddown 1`, `fadetime 1.0` s. **Because min = max = 1.0, CS2’s default is an effectively locked exposure of 1.0** — but every official map overrides it. (Generic Source 2/HL:A defaults are `minexposure 0.25` / `maxexposure 8` — do **not** use those for CS2.)

**Per-map measured values**, decompiled from shipped `.vpost` files by source2.wiki ([example presets](https://www.source2.wiki/EngineTools/PostProcessingEditor/example_presets)) **[C, decompiled]**:

| Map | `.vpost` (in `lighting/postprocessing/`) | Min exposure | Max exposure | Tonemap | Bloom mode | Bloom strength | Bloom threshold | Bloom fade | Skybox bloom |
|---|---|---|---|---|---|---|---|---|---|
| de_train | `de_train_prefab/de_train_postprocess.vpost` | 0.8 | 1.2 | Hable | Screen | 0.30 | 1.03 | 1.03 | 0.75 |
| de_inferno | `de_inferno_prefab/de_inferno_prefab.vpost` | 0.8 | 1.1 | Hable | Additive | 0.23 | 0.55 | 0.91 | 0.18 |
| de_ancient | `de_ancient_prefab/de_ancient_postprocess_v1.vpost` | 1.0 | 1.7 | Hable | Additive | 0.16 | 1.14 | 1.02 | 1.00 |

All three use the **Hable (U2/HLVR)** preset curve — Valve states this is their choice for newer maps: *“Valve’s post processing for newer maps is the ‘Hable (U2/HLVR)’ Preset for tonemapping.”* Bloom blur weights are **0.20** at every mip (1/2 → 1/32) and tints are `255 255 255` on all three maps. Runtime order: **tone mapping + bloom first, then all other effects**; for stacked layers, the bottom-most wins.

**Everything except tonemap + bloom is baked into a single colour LUT at build time**, so the per-layer grading (Levels, Curves, Color Balance, Hue/Sat, Vibrance, Local Contrast…) is **not recoverable** — only the LUT is. The depot confirms this per map: `lighting/postprocessing/de_inferno_prefab/inferno_s2_lookup.bmp` (98 KB LUT), and `.vpost_c` files for every competitive map (`de_dust2_prefab`, `de_mirage_prefab`, `de_nuke_prefab`, `de_vertigo_prefab`, `de_overpass_prefab`, `de_anubis_prefab`, `de_train_prefab`, `de_cache_prefab`, `de_ancient_prefab`) **[V]**.

### 6.4 Per-map sun and sky (the actual "look" knobs) [V, archived Valve wiki]

From the CS2 Sky List ([archived](https://web.archive.org/web/20260107153925/https://developer.valvesoftware.com/wiki/Counter-Strike_2_Workshop_Tools/CS2_Sky_List)); colours 0–255 sRGB, sun angle `pitch yaw roll`:

| Map | `env_sky` material | Sun angle | Sun colour | Sun brightness | Indirect mult. | Sky colour | Sky intensity | IBL scale | Sky indirect mult. |
|---|---|---|---|---|---|---|---|---|---|
| de_inferno | `s2_de_inferno_sky01` | 63 218 0 | 255 247 235 | **3.2** | 1.75 | 136 199 255 | 0.875 | 1 | 1.5 |
| de_mirage | `sky_de_mirage` | 60 318 0 | 255 223 179 | **3.0** | 1.3 | 130 205 255 | 1 | 1 | 1.5 |
| de_dust2 | `sky_de_dust2` | 50 43 0 | 255 222 189 | **2.5** | 1 | 211 226 248 | 0.96 | 1.2 | 1.7 |
| de_nuke | `sky_de_nuke` | 60 25 0 | 255 252 239 | **2.4** | 1.6 | 67 122 182 | 2.2 | 1.1 | 2.6 |
| de_ancient | `sky_hr_aztec_02_lighting` | 49 42 0 | 241 234 209 | **4.0** | 1.5 | 127 179 200 | 2.5 | 1.5 | 1.25 |
| de_vertigo | `sky_de_vertigo` | 61 53 0 | 255 243 218 | **3.25** | 1.75 | 146 173 203 | 1.65 | 1 | 1.5 |
| de_anubis | `sky_de_annubis` *(sic, Valve’s typo)* | 61 75 0 | 255 207 156 | **6.35** | 1.4 | 255 255 255 | 1.76 | 1 | 1.33 |
| de_overpass | `sky_de_overpass_01` | 51 69 0 | 248 212 171 | **2.8** | 1 | 85 124 173 | 1 | 1.4 | 2.3 |
| cs_office | `sky_cs_office` | 45 345 0 | 219 230 241 | 0.0 | 1 | 85 124 173 | 1 | 1 | 1 |
| cs_italy | `cs_italy_s2_skybox_2` | 50 194 0 | 252 194 159 | 4.0 | 1 | 45 174 255 | 2.5 | 1 | 1.5 |

`env_sky` is referenced from `light_environment` as the **Sky IBL Source**; values are produced by “Estimate Lighting from HDR Skybox”. The depot confirms the naming convention and that these are per-map HDR cubemaps (`materials/skybox/sky_cs15_daylight01_hdr_cube_pfm_*.vtex_c`, `materials/skybox/jungle_cube_pfm_*.vtex_c`, `materials/de_inferno/skybox/inferno_sky_01_exr_*.vtex_c` at 6.5 MB, `materials/de_cache/sky/de_cache_sky_001_exr_*.vtex_c` at 7.5 MB) **[V]**. **de_train’s sky is not in the wiki table** (a `de_train_skybox.vpk` prefab exists in the depot) — treat as unverified.

### 6.5 Global light / shadows / fog [V]

Complete `cl_globallight_*` set (Valve dump; all developmentonly, no descriptions shipped):
```
cl_globallight_world_top_height        4096       cl_globallight_world_bottom_height 0
cl_globallight_expansion               200        cl_globallight_shadow_mode         2
cl_globallight_depth_bias              -999       cl_globallight_slope_scale_depth_bias -999
cl_globallight_orig_calc_frustum       true       cl_globallight_use_optimized_calc_frustum true
cl_globallight_use_alt_focus_region    false      cl_globallight_use_shaadow_near_offset true  [sic]
```
Reading **[I]**: the sun’s single cascaded-shadow focus volume spans world z = **0 → 4096 HU (0 → 78 m)**, expanded **200 HU (5.1 m)** around the region of interest. `-999` is an “unset” sentinel; real depth biases come from `mat_depthbias_shadowmap 0.0005` and `mat_slopescaledepthbias_shadowmap 4`. `cl_globallight_shadow_mode 2`’s enum meaning is **undocumented — do not assume**. CSM coverage: `r_csgo_csm_max_visible_distance 7500` HU (190 m), `r_csgo_csm_pushback_distance 7000`, `r_csgo_viewmodel_csm_pushback_distance 1500`. **The CSM frustum FOV is 75°, independent of render FOV** (`csm_fov_override` entity default `75`: “This value will be used to override the FOV for cascading shadow maps only. Actual camera FOV remains unchanged.”).

**Fog** (CS2-native entities, `csgo.fgd`): `env_cubemap_fog` — `cubemapfogstartdistance 0`, **`cubemapfogenddistance 4000`**, `cubemapfogfalloffexponent 2.0`, `cubemapfogmaxopacity 1.0`, `cubemapheightfog 1`, `cubemapfogheightstart 0`, `cubemapfogheightend 4000`, `cubemapfogheightexponent 2.0`, `cubemapfoglodbiase 0.5`. `env_gradient_fog` — `fogcolor 255 255 255`, `fogstrength 1.0`, `fogmaxopacity 1.0`, `fogstart 0`, **`fogend 4000`**, `fogfalloffexponent 2.0`, `heightfog 1`, `fogstartheight 0`, `fogendheight 4000`, `fogverticalexponent 2.0`, `fadetime 1.0`. Switches: `r_csgo_enable_cubemap_fog true`, `r_csgo_enable_gradient_fog true`, `r_csgo_enable_volume_fog false`. **Distance fog 0 → 4000 HU (0 → 101.6 m), exponent 2.0, height fog 0 → 4000 with vertical exponent 2.0** is the practical default.

### 6.6 Gamma and render defaults [V]

- **There is no `mat_monitorgamma` in CS2** (zero matches in the full dump). The gamma convar is **`r_fullscreen_gamma 2.2`, range [1, 4]** — this is what the in-game Brightness slider drives. Output transfer = **standard sRGB / gamma 2.2**.
- **There is no in-game contrast or saturation slider** — CS2 “vibrance” is a GPU-driver setting, not part of the game’s look. Leave it neutral.
- Look-defining defaults: **`r_ssao true`, `r_ssao_radius 30` HU (0.76 m), `r_ssao_strength 1.2`, `r_ssao_bias 0.5`, `r_ssao_blur true`** — CS2 ships with fairly strong, tight, blurred AO (not the usual “off” default). `r_directional_lightmaps true` + `mat_colorcorrection true` mean lightmaps carry directional shading and the per-map LUT is applied. `r_csgo_multiscattering true`, `r_csgo_microshadowing true`, `r_csgo_cubemap_normalization true`, `r_csgo_enable_high_precision_lighting true`, `r_csgo_depth_prepass true`, `mat_max_lighting_complexity 8`.
- Near plane is unusually far: **`fov_cs_near_z 6.5` HU (≈0.124 m)** — CS2 pushes the near plane out for depth precision. Viewmodel pass has its own planes: `r_csgo_viewmodel_near_plane 1`, `r_csgo_viewmodel_far_plane 100`, its own envmap bias 0.85 and probe clamp 16.
- `r_csgo_render_post_film_grain 0`, `r_csgo_render_post_fxaa 0`, `r_csgo_render_post_local_contrast true`, `r_csgo_render_post_bloom 1`, `r_csgo_render_post_bloom_strength -1` (sentinel ⇒ use the `.vpost` value), `r_csgo_render_post_colorcorrection 0` (per-entity colour correction off in CS2).

### 6.7 Minimal “CS2 look” preset for the viewer

```
FOV                vertical 73.7398°  (⇒ 90° @4:3, 100.39° @16:10, 106.26° @16:9)
near plane         6.5 HU (0.124 m)
tone curve         Hable: A .15  B .50  C .10  D .20  E .02  F .30  W 4.0
exposure           fixed 1.0; optional per-map override: inferno .94, train .98, ancient 1.30
bloom              per-map (inferno: additive, strength .23, threshold .55, fade .91, sky .18)
AO                 on, radius 30 HU, strength 1.2, bias .5, blurred
sun                per-map brightness/colour/angle (table 6.4); softness angulardiameter 1.0
ambient            prefiltered sky cubemap × skyintensity × IBL scale, tinted by Sky colour,
                   ground bounce = skyambientbounce, IBL sun-disc clamp 32
fog                0 → 4000 HU, exponent 2.0; height fog 0 → 4000, vertical exponent 2.0
shadows            3 cascades, 75° CSM frustum, ~7500 HU range, crossfade .1, distance fade .05
output             tone map → sRGB gamma 2.2
viewmodel (opt.)   separate pass, FOV 60 (clamp 60–68), offsets (1,1,-1)
```

---

## 7. Actionable number sheet

### 7.1 Units and coordinates
| Thing | Value |
|---|---|
| Hammer unit (map scale) | 1 HU = 0.75 in = **1.905 cm** (16 HU/ft) |
| Hammer unit (character authoring) | 1 HU = 1 in = 2.54 cm (12 HU/ft) |
| Player origin | at the **feet** |
| Forward axis | **+X**; yaw 0 = +X; `forward = (cos yaw, sin yaw, 0)` |
| Up / left | **+Z up**, **+Y is LEFT** |
| Default tickrate | 128 official; **always derive from the demo** |

### 7.2 Smoke
| Thing | Value |
|---|---|
| Shape | ground-clipped lens/torus, **not** a sphere |
| Torus ring radius | 61 HU |
| Sub-volume radius | 88 HU |
| Origin height | **68 HU** above impact |
| Horizontal radius | ≈149 HU (diameter ≈298 HU ≈ 5.68 m) |
| Top of cloud | ≈156 HU (2.97 m) |
| Edge feather | 21 HU |
| CS:GO reference | diameter ≈256 HU / height ≈128 HU; wiki says 288 HU diameter (144 radius) |
| Fuse | starts **after landing**; minimum-fuse rule since 2023-11-02 |
| Bloom | sub-second, ≈1 s to full size |
| Duration | **18 s** cover, ≈20 s to fully gone, **15 s** radar block |
| Bot LOS penetration | 200 HU |
| Bullet holes | heal in ≈1–2 s; shotguns make larger holes |
| HE in smoke | void ≈2–3 s, then re-expands; clears screen effect |
| Shadows | yes (since 2024-02-06) |
| Team tint | T `[180,129,50]`, CT `[75,127,155]`, fading to grey as it ages |
| Silhouette threshold | `sv_smoke_volume_blind_start = 0.2` |
| Volumetric extras | `smoke_use_noise_texture true`, `smoke_volume_lod_ratio_change 0.6`, `smoke_param1..5 = 6.26 / 8.27 / 0.13 / 0 / 0` |
| Shaders | `smoke_volume{,_cs,_depth,_lpv,_mask}`, `overlay_smoke`, MBOIT |
| Key particles | `explosion_smokegrenade_voxel.vpcf_c`, `weapon_smokegrenade_pointcloud.vsnap_c`, `smokegrendae.vsnap_c` |
| Material | `materials/particle/particle_smokegrenade.vmat_c` |
| Voxel data in demos | field exists (`m_VoxelFrameData`) but reads as zeros via demoparser2 |

### 7.3 Molotov / incendiary
| Thing | Molotov | Incendiary |
|---|---|---|
| Spread radius | **150 HU** (2.86 m) | **110 HU** (2.10 m) |
| Flame lifetime | **7 s** (900 ticks @128) | **5.5 s** (352 ticks @64 — measured) |
| Damage | 40 DPS, tick every **0.2 s** (⇒ 8 dmg/tick) | same |
| Ramp | trunc(8 × min(t/2 s, 1)) per tick, t = seconds **since ignition**; armour irrelevant (`DMG_BURN`) | same |
| Stacking | cells of the **same** inferno do **not** stack (one hit/0.2 s/entity); **separate** infernos do | same |
| Time to kill (100 HP) | **3.6 s** from ignition, ~2.6 s if entering after the ramp | same |
| Flame grow / shrink | **2.0 s** to full size, **2.0 s** to fade; one sprite per cell | same |
| Max cells | **16** (`inferno_max_flames`), arrays are size 64 | same |
| Cell spacing | **42 HU** | same |
| Initial burst | one cell every **0.02 s** | same, but spreads faster since 2025-07-16 |
| Child spread | depth 4, interval ×0.1/level, cap 0.5 s, window 3 s | same |
| Spawn angle | 45° from parent | same |
| Surface offset | 15 HU | same |
| Detonation slope | ≤30° shatters, else bounces | same |
| Air-burst | 2 s of flight; **<128 HU above a surface ⇒ flames drop to ground** | same |
| Pillar height | decreases over time (since 2024-05-23) | same |
| Flame cell size | half-width **30 HU**, height **80 HU**; sprite `m_maxSize` random **70–90 HU**, quad half-width = size/3 | same |
| Spread-speed multiplier | **1** | **10** (`inferno_spread_speed_mult_ct`, since 2025-07-16) |
| Rendering | `particles/inferno_fx/fire_{core,filler,edge}.vsnap_c` at each cell + `shaders/vfx/inferno` | same |
| Scorch | `materials/decals/molotovscorch.vmat_c`, `materials/decals/scorch/scorch1..3` | same |

### 7.4 Player
| Thing | Value |
|---|---|
| Standing hull | 32 × 32 × **72** HU, origin at feet |
| Standing eye | **64.0938** HU (64.0 nominal) |
| Crouch hull | 32 × 32 × **54** HU |
| Crouch eye | **46.0762** HU (46.0 nominal) |
| Duck time | 200 ms |
| Observer hull | 20 × 20 × 20 HU |
| Hitgroups | HEAD 1, CHEST 2, STOMACH 3, LEFTARM 4, RIGHTARM 5, LEFTLEG 6, RIGHTLEG 7, NECK 8, GEAR 10 |
| Multipliers | head ×4, stomach ×1.25, chest/arms ×1.0, legs ×0.75 |
| Hitbox shape | capsules (`m_flShapeRadius` + `m_nShapeType` in `CHitBox`) |
| Model root | `agents/models/<family>/<file>.vmdl` (**not** `models/players/`) |
| Default CT / T | `agents/models/ctm_sas/ctm_sas.vmdl` / `agents/models/tm_phoenix/tm_phoenix.vmdl` |
| Families | ctm_sas, ctm_fbi, ctm_st6, ctm_swat, ctm_gendarmerie, ctm_diver, tm_phoenix, tm_professional, tm_balkan, tm_jungle_raider, tm_leet (+ 12 gloves) |
| Skeleton | `worldmodel.vnmskel`, **74 bones**, root `root_motion` → `pelvis` |
| Animation format | Animgraph 2: `.vnmclip` / `.vnmgraph` / `.vnmskel` (no Source 1 formats) |
| Clip names | weapon-suffixed: `idle_knife`, `run_n_knife`, `crouch_n_knife`, `jump_stand_knife`, `death_chest_a`, `flinch_head`, `equipment/c4/planting`, `defuse/defuse_loop_*` |

### 7.5 Camera / look
| Thing | Value |
|---|---|
| World FOV | 90° horizontal @4:3 ⇒ **73.7398° vertical**, 106.26° horizontal @16:9 |
| Spectator FOV | same (no cvar exists) |
| Viewmodel FOV | 60 (range 60–68), separate pass |
| Exposure | locked 1.0 by default; inferno 0.8–1.1, train 0.8–1.2, ancient 1.0–1.7 |
| Tonemap | Hable (U2/HLVR): .15/.50/.10/.20/.02/.30, white 4.0 |
| AO | radius 30 HU, strength 1.2, bias 0.5, blurred |
| Fog | 0 → 4000 HU, exponent 2.0 |
| Gamma | 2.2 (`r_fullscreen_gamma`) |
| CSM | 3 cascades, 75° shadow frustum, 7500 HU range |

---

## 8. Confidence, gaps and caveats

**Verified from Valve primary artefacts [V]:** all `inferno_*` / `cl_smoke_*` / `smoke_*` / FOV / lighting convar defaults (re-verified against Valve’s *current* `convars.txt`, which is newer than the Nov-2024 community mirror — several cvars exist only in the newer dump); the `C_Inferno` and `C_SmokeGrenadeProjectile` client **and server** schema layouts; the §4 model path scheme and full family list; every particle/material/shader path quoted; the `.vpost`/LUT file layout; `light_environment` and `env_sky` defaults; `HitGroup_t.h`; `CHitBox.h`; the Z-up/X-forward statement; the CS:GO hull constants; all patch-note quotes.

**Verified from the CS:GO source leak [V]:** the `CInferno` damage tick/ramp code (`inferno_damage × inferno_damage_timer`, ramped by `m_damageRampTimer` over 2.0 s from ignition), the burn-damage armour exclusion, the flame geometry constants (`InfernoFire_HalfWidth 30.0`, `InfernoFire_FullHeight 80.0`, `m_maxSize` 70–90 HU, quad half-width = size/3), the `MolotovScorch` decal over a 100-HU trace, the `MIN(64, m_nMaxFlames)` cell cap, the int-truncated `m_fireXDelta` offsets, and the six-field CS:GO `DT_Inferno` network block. Source: <https://github.com/SwagSoftware/Kisak-Strike> (`game/server/cstrike15/Effects/inferno.cpp`, `game/client/cstrike15/Effects/clientinferno.cpp`, `game/server/cstrike15/cs_player.cpp`).

**Measured locally [M]:** demo tickrate 64 for `0823_inferno.dem`; molotov lifetime 449–450 ticks (7.016–7.031 s) and incendiary 352 ticks (5.500 s) — a clean independent confirmation of `inferno_flame_lifetime` / `inferno_flame_lifetime_incendiary`; `m_VoxelFrameData` / `m_nVoxelFrameDataSize` / `m_nVoxelUpdate` present but zero-valued via `demoparser2`; `m_vSmokeColor = [180,129,50]` in a real demo. A separately reported smoke detonate→expire figure of 1412 ticks (22.06 s) came from pairing logic with `position_match: false` and an implausible 124,703-tick maximum — **do not rely on it**.

**Interpretations I am flagging as inferred [I]:** the torus geometry reading of `cl_smoke_torus_ring_radius`/`subradius`/`origin_height`; the MBOIT/upsample reading of the smoke render pipeline; the reading of `cl_globallight_world_top_height`/`expansion`; the claim that `characters/models/` is a stub layer.

**Open questions / not established:**
1. **Exact CS2 hitbox bounds in HU.** Must decompile a shipped agent `.vmdl_c` with Source 2 Viewer (`CHitBox.m_vMinBounds/m_vMaxBounds/m_flShapeRadius/m_nGroupId`). ~1 CLI command if a CS2 install is available.
2. **Smoke bloom duration** has no cvar and no Valve number — only “≈1 s” from community coverage.
3. **CS2 smoke duration** is hardcoded; the defensible figure is 18 s of cover / ~20 s total. A clean per-demo measurement (detonate event → expired event, same entity, with position matching) would settle it.
4. **Flame sprite height in CS2 specifically** — the CS:GO source constants (half-width 30 HU, full height 80 HU, per-flame `m_maxSize` 70–90 HU, 2 s grow / 2 s shrink) are solid, and CS2 exposes client-side `m_maxFireHalfWidth` / `m_maxFireHeight`, but no one has published the *live CS2* values and the CS2 renderer (`C_OP_RenderSprites`) has no fixed HU size constant at all. Read them straight off the entity in a demo.
5. **`cl_globallight_shadow_mode` enum** — undocumented.
6. **Per-map colour LUT contents** — baked, inside VPKs, not published. Only Train/Inferno/Ancient exposure+bloom are public. Do not assume Mirage/Dust2/Nuke/Vertigo/Overpass/Anubis match.
7. **`fov_desired 75`** exists in the dump but does not appear to affect CS2’s world FOV (legacy).
8. **CS2 damage multipliers** are CS:GO-verified only; CS2 server is closed-source.
9. **CS2.CAM’s 3D pipeline** (`cs2.cam`) is undocumented — whether it uses real extracted map meshes or projected radar geometry is unresolved, as is its engine.
10. **Agent head-hitbox equalisation (+4–20 %, capsules)** is community reverse-engineering (Ansimist) reported by outlets; Valve’s only official note is “Fixed several hitbox alignment bugs” (2023-10-09).
11. **Licensing:** extracted Valve assets remain Valve copyright. `demoscope` and `cs2-map-icons` have no licence. Budget legal review before shipping extracted maps/models publicly.

---

## Appendix — reproducing the primary-source mining

The strongest numbers in this report came from mining Valve’s shipped CS2 artefacts rather than from articles. Anyone can re-run these; each is a single fetch.

```bash
# 1. Full CS2 VPK file manifest (135,686 entries: every model/particle/material/shader path)
curl -sL -o pak01_dir.txt \
  https://raw.githubusercontent.com/SteamTracking/GameTracking-CS2/master/game/csgo/pak01_dir.txt

# 2. Every inference/smoke/model question, answered by grep:
grep -iE "^materials/particle/" pak01_dir.txt | grep -iE "fire|smoke|flame"
grep -E "vsnap_c" pak01_dir.txt | grep -iE "fire|smoke|inferno"
grep -E "^particles/inferno_fx/" pak01_dir.txt
grep -E "^agents/models/.*\.vmdl_c" pak01_dir.txt
grep -E "^lighting/" pak01_dir.txt          # per-map .vpost + colour LUTs
grep -E "^shaders/" shaders_pc_dir.txt | grep -E "smoke|inferno|mboit|player_visibility"

# 3. Authoritative convar defaults — use VALVE's dump, not a community mirror.
#    (Community mirrors lag: the Nov-2024 list this report started from is missing
#     inferno_spread_speed_mult_ct, inferno_velocity_factor_ct and the smoke_* cvars.)
curl -sL -o vconvars.txt \
  https://raw.githubusercontent.com/SteamTracking/GameTracking-CS2/master/DumpSource2/convars.txt
grep -nE "^inferno_" vconvars.txt
grep -oE "^[a-z0-9_]*smoke[a-z0-9_]*" vconvars.txt | sort -u
grep -inE "^default_fov|^viewmodel_fov|^r_ssao|^cl_globallight" vconvars.txt

# Compare against the community dump only to spot *deleted* cvars:
curl -sL -o cvars.md \
  https://raw.githubusercontent.com/SuGolYolLom/CS2-Cvars-Cmds/main/cs2/cvarlist.md
grep -iE "^\| ?inferno" cvars.md

# 4. Entity field layouts — client AND server
B=https://raw.githubusercontent.com/SteamTracking/GameTracking-CS2/master/DumpSource2/schemas
curl -sL $B/client/C_Inferno.h                  # client view (render extents, snapshots)
curl -sL $B/server/CInferno.h                   # server logic (timers, m_nMaxFlames, m_nSpreadCount)
curl -sL $B/client/C_SmokeGrenadeProjectile.h
curl -sL $B/client/HitGroup_t.h
curl -sL $B/modellib/CHitBox.h
# Byte-offset schema explorer: https://s2v.app/SchemaExplorer/cs2/client/C_Inferno

# 5. CS:GO (Source 1) equivalents, for cross-era work
#    CS:GO source leak (ConVar defaults, damage code, flame geometry):
#      https://github.com/SwagSoftware/Kisak-Strike
#      game/server/cstrike15/Effects/inferno.cpp  ·  game/client/cstrike15/Effects/clientinferno.cpp
#    CS:GO final-build netprop + cvar dumps (DT_Inferno bit offsets):
#      https://github.com/Mapeadores/CSGO-Dumps   (netprops.txt, cvarlist.txt)
```

**Official patch notes** were mined from Steam’s public news API (appid 730, paginated by `enddate`); every date/quote in §2.4 and §3.5 comes from there:

```bash
curl -s "https://api.steampowered.com/ISteamNews/GetNewsForApp/v2/?appid=730&count=100&maxlength=0&format=json&enddate=<unix>"
# then grep the `contents` field for smoke|molotov|incendiar|inferno
```

**Demo-side measurements** quoted in §2.3/§3.3 came from `demoparser2` against `0823_inferno.dem` (64-tick) — see `notes/scratch/{tickrate,smoke_grenade,inferno_probe}.py` in this repo. The molotov/incendiary lifetime cross-check (449–450 ticks ≈ 7.031 s vs 352 ticks = 5.500 s) is the cleanest confirmation in this document: it validates `inferno_flame_lifetime = 7` and `inferno_flame_lifetime_incendiary = 5.5` independently of the cvar list.

