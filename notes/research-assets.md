# CS2 real-asset acquisition — research report

Scope: how to obtain genuine Counter-Strike 2 game assets (player models, grenade models, smoke/flame
textures, map materials) for a Three.js demo-replay viewer, on **macOS (Apple Silicon, arm64) with only
~17 GB free**, with CS2 not installed. Tool available:
`/Users/wanjungu/Downloads/Source2Viewer-CLI/Source2Viewer-CLI` (**v19.2.6339**, Mach-O arm64).

Date of research: 2026-09-16. All sizes below are from live Steam data pulled during this research
(steamcmd `app_info_print`), not from memory.

---

## 0. Bottom line (read this first)

| Question | Answer |
| --- | --- |
| CS2 dedicated-server SteamCMD app id | **730** — client and dedicated server were **merged into one app**. There is **no separate DS app id**. `2347770` is a **depot** id (the shared game-content depot), not an app. |
| Anonymous download? | **Yes in principle** — Valve's own wiki says "login anonymously, or using your account". Anonymous login successfully fetched the full app-730 depot/manifest data in my test. |
| Practical blocker on this machine | **Disk space.** CS2 needs ~**59.5 GB download / ~71.1 GB staged on disk**. With 17 GB free, steamcmd fails immediately with `Error! App '730' state is 0x202 after update job.` — the community-confirmed "out of disk space" signal. |
| Does the DS depot contain rendering assets? | **Yes — it is the client install.** No stripped down DS depot exists. Depot 2347770 (62.7 GB) contains `game/csgo/pak01_dir.vpk` + its numbered data chunks, and therefore `agents/models/`, `materials/`, `particles/`. |
| Native macOS install possible? | **No.** CS2's macOS depot is a 9,276-byte stub. You must force `@sSteamCmdForcePlatformType linux` (or `windows`) in steamcmd. |
| Best route with 17 GB free | **DepotDownloader selective download** (`-manifest-only` + `-filelist`): fetch `pak01_dir.vpk` (7.6 MB) + only the 24–120 MB `pak01_NNN.vpk` chunks you need. Secondary: public GLB repos (CS:GO-era models only). |

---

## 1. SteamCMD route

### 1.1 App ids

- **App 730 = "Counter-Strike 2"** — client *and* dedicated server.
  Valve Developer Community: *"The CS2 dedicated server and game client have been merged together
  (appid 730). Previously the CSGO client (appid 730) and dedicated server (appid 740) were separate."*
  ([VDC, archived](http://web.archive.org/web/20260911193551/https://developer.valvesoftware.com/wiki/Counter-Strike_2/Dedicated_Servers))
- Install instructions on that same page: `force_install_dir …` → **`login anonymous`** ("You need to login
  into Steam anonymously, or using your account") → `app_update 730 validate`. Stated storage requirement:
  **65 GB**, download "approximately 60GB".
- LinuxGSM's CS2 server config agrees: `appid="730"`, `systemdir="${serverfiles}/game/csgo"`
  ([raw config](https://raw.githubusercontent.com/GameServerManagers/LinuxGSM/master/lgsm/config-default/config-lgsm/cs2server/_default.cfg))
- ZAP-Hosting's Linux guide also uses `+login anonymous +app_update 730 validate`
  ([docs source](https://raw.githubusercontent.com/zaphosting/docs/master/docs/dedicated-linux-cs2.md)).
- **`2347770` is not an app.** SteamDB has no app with that id ("No app was found matching this AppID /
  did you mean depot 2347770?"). `steamcmd +app_info_print 2347770` returns an **empty** appinfo body
  (`"2347770" { }`), and `+app_update 2347770` fails with `Invalid platform`. It is the *depot* id.

### 1.2 Sizes (live from Steam, `steamcmd +login anonymous +app_info_print 730`)

Public branch, values in bytes (`size` = staged on disk, `download` = compressed transfer):

| Depot | Platform | size (on disk) | download | what it is |
| --- | --- | ---: | ---: | --- |
| **2347770** | *(all)* | **62,735,072,496 (62.74 GB)** | **53,900,780,944 (53.90 GB)** | shared game content (`game/csgo/pak01_*`, maps, etc.) |
| 2347771 | windows | 7,712,252,582 (7.71 GB) | 4,995,013,808 (5.00 GB) | Windows binaries |
| 2347773 | linux | 7,221,977,502 (7.22 GB) | 4,604,239,936 (4.60 GB) | Linux binaries |
| 2347772 | macos | **9,276 (~9 KB)** | 1,696 | macOS stub — **no macOS CS2 content** |
| 2347774 | *(all)* | 1,142,056,124 (1.14 GB) | 949,604,816 | extra content |
| 2347777 | *(all)* | 88,125,520 (88 MB) | 26,889,664 | small content depot |
| 2347779 | *(all)* | 2,170,752,148 (2.17 GB) | 582,148,112 | `dlcappid 2279721` = **CS2 Workshop Tools DLC** (optional) |
| 731 / 732 | system-defined | 8 bytes public | 64 | branch holders; branch `csgo_legacy` = **32.72 GB / 13.64 GB download**; branch `demo_viewer` = 4.85 GB (download 0) |

Install totals confirmed by steamcmd's own generated `appmanifest_730.acf` on the aborted installs:

| Forced platform | BytesToDownload | BytesToStage (disk) |
| --- | ---: | ---: |
| linux | 59,454,625,760 (**59.45 GB**) | 71,099,106,130 (**71.10 GB**) |
| windows | 59,845,399,632 (**59.85 GB**) | 71,589,381,210 (**71.59 GB**) |
| macos | (n/a — CS2 dropped macOS) | 9 KB stub |

Official macOS Steam client install is therefore impossible today; a `steamcmd` Linux/Windows-forced
download is the only Steam path. Note: the macOS `steamcmd` binary is `x86_64` and needs Rosetta 2
(verified: `file steamcmd` → `Mach-O 64-bit executable x86_64`; it ran fine here).

### 1.3 Is it anonymous, and does it contain the rendering assets?

**Anonymous: yes, per Valve and by observation of the download machinery.** My own test (macOS steamcmd,
`+@sSteamCmdForcePlatformType linux +force_install_dir /tmp/cs2ds730 +login anonymous +app_update 730 validate`)
connected anonymously, retrieved config/user info, computed the full depot manifest
(`BytesToDownload 59,454,625,760`) and then failed:

```
Update state (0x3) reconfiguring, progress: 0.00 (0 / 0)
Update state (0x0) unknown,        progress: 0.00 (0 / 0)
Error! App '730' state is 0x202 after update job.
```

`0x202` = `UpdateRequired | UpdatePaused` with `UpdateResult 12`, and it is the well-known
**out-of-disk-space** result, not a licence denial:

- CubeCoders support thread with the identical error: AMP logged `SteamCMD last reported exit state was
  OutOfDiskSpace`; resolution was that CS2 needed ~60 GB and the install volume didn't have it. The
  maintainer's summary: *"CS2 requires something like 60GB, since there is no separate server appid and so
  the whole game needs to be installed."* ([thread](https://discourse.cubecoders.com/t/new-counter-strike-2-server-error-app-730-state-is-0x202-after-update-job/32256))
- Steam community: "This is a no space left exception."
  ([thread](https://steamcommunity.com/discussions/forum/14/3464975249567851250/?l=english))
- Control experiment on the same machine/account: `+login anonymous +app_update 90 validate` (Half-Life DS,
  966 MB) downloaded normally (`Update state (0x61) downloading, progress: 37.4 …`). So anonymous downloads
  *do* work from this environment; app 730 failed only because 15 GB < 71 GB.
- `+app_update 730` with **no** platform override fails with `Invalid platform` (macOS has no depots),
  not with a licence error — consistent with the above.

**Uncertainty (stated honestly):** I could not complete a real download to 100%, so "anonymous install
succeeds on a machine with enough disk" rests on Valve's wiki + community practice, not on my own
end-to-end run. What is *directly* verified is that anonymous login resolves app 730's depots and their
sizes, and that the failure mode here is disk space.

**Does the depot contain the full client content (not stripped)?** Yes, with strong, converging evidence:

1. App 730 *is* both client and DS; there is no separate DS app to be stripped (VDC; CubeCoders).
2. joedwards32/CS2 (the DS Docker image referenced by the VDC wiki) issue #24:
   *"the game files used by the client are also downloaded when running a CS2 server, rather than just
   files dedicated to that purpose"*, and it lists **depot 2347770 = "Common asset files (including
   maps)"**, 2347771 = Windows binaries, 2347773 = Linux binaries.
   ([issue](https://github.com/joedwards32/CS2/issues/24))
3. Depot 2347770 is 62.74 GB, cross-platform (`osarch 64`, no `oslist` restriction), and SteamDB's indexed
   file list for that depot includes `game/csgo/pak01_dir.vpk` and per-map VPKs such as
   `game/csgo/maps/de_poseidon.vpk`, `game/csgo/maps/cs_alpine.vpk`, `game/csgo/maps/ui/inspect_item.vpk`
   (steamdb.info/depot/2347770/ — SteamDB blocks automated fetches, so this is from indexed snippets).
4. `pak01_dir.vpk` is exactly where CS2's renderable assets live — VRF's own docs list CS2 paths
   `agents/models/` (player/agent models), `weapons/models/`, `models/`, `materials/` all inside
   `pak01_dir.vpk` ([exporting-models guide](https://s2v.app/ValveResourceFormat/guides/exporting-models.html)),
   and Source2Wiki's VPK page uses CS2 paths like `particles/explosions_fx/explosion_smoke.vpcf_c`.
5. Nothing in the depot list indicates a stripped/“server-only” variant: the DS is the full ~71 GB install,
   matching the wiki's "65 GB storage / ~60 GB download".

Not verified locally (no game installed): an actual VPK listing proving `agents/models/…vmdl_c` entries
exist in the current `pak01_dir.vpk`. The command to prove it in one step is in §4(a).

---

## 2. App 730 (the CS2 client) without owning the game?

- **App 730 is a free-to-play app, and Valve's own DS instructions say anonymous login is sufficient.**
  There is no licence purchase needed to *download* it. ✔ (VDC wiki, ZAP-Hosting, LinuxGSM all use
  `login anonymous` + `app_update 730`.)
- **But anonymous cannot be conflated with "small download":** you get the whole 59.5 GB / 71.1 GB install.
  There is no "client-only" or "server-only" variant to choose.
- Anonymous `app_info_print 730` returned the **complete** depot manifest (sizes, GIDs, branches) — evidence
  that anonymous sessions can see and fetch this app's content metadata.
- Size: see the table in §1.2 — 59.45 GB download / 71.10 GB on disk for a Linux-forced install
  (59.85 / 71.59 GB for Windows). The macOS depot is a 9 KB stub.

---

## 3. Public asset sources (GitHub / other)

| Source | What you get | Notes / licence |
| --- | --- | --- |
| [rabbytesoftware/cs2.agents](https://github.com/rabbytesoftware/cs2.agents) | **CS:GO agent (player) models converted to GLB**; `agents/` (models) + `animations/` (celebrate_* anims as separate GLBs). Repo ~212 MB, last push 2023-09. | Repo is **MIT** (`LICENSE`, © 2023 Rabbyte Software) — but that covers the repo's own work; the **models themselves are Valve's**. Explicitly CS:GO-era, author says CS2 re-ports are pending. Stated to be web-optimised/compressed. *No textures guaranteed — README doesn't promise materials.* |
| [AiLexie/cs2-stripped-models](https://github.com/AiLexie/cs2-stripped-models) | ~37 MB of **CS2 GLB meshes** ("mesh data only, no textures/animations") made for external chams. | Licence not stated (README fetch 404 on `main`; content confirmed only via repo description). |
| [CTG-Furien/CS2-Bambook-Assets](https://github.com/CTG-Furien/CS2-Bambook-Assets) | GitHub **Releases**: "viewer-compatible models — 65 GLBs (191,151,004 bytes)" plus "Source 2 models — 55 weapon/knife exports from installed CS2 build 2000908, with all 758 GLB/PNG files referenced by those exports". | Weapons/knives/gloves only (no player models). No licence file stated. Explicitly says base-model availability ≠ every skin renderable. |
| [LielXD/CS2-WeaponPaints-Website](https://github.com/LielXD/CS2-WeaponPaints-Website) | Upstream of those 65 viewer-compatible weapon GLBs (UV layout/orientation already suited to a web viewer). | Website code licence is separate from the Valve-derived assets. |
| [goribby/CS2-MultiRig](https://github.com/goribby/CS2-MultiRig) | Blender 4.0+ IK rig containing **every CS2 glove and arms (viewmodel) model** + glove skins with procedural shaders (repo listing ~602 MB, but the `.blend` is distributed via Gumroad). | No licence declared; distribution is free (Gumroad). Blender-native, would need glTF export yourself. |
| [EasterLee/easter_prefabs](https://github.com/EasterLee/easter_prefabs) | Ready-made **CS2 particle prefabs (`.vpcf`)** — recommended from the VDC CS2 Particles page. | The `.vpcf` text references real CS2 sprite textures; useful for learning the exact texture paths. |
| Steam Workshop via steamcmd (`+workshop_download_item 730 <id>`) | Community map/model VPKs, anonymously downloadable. | Community-authored content (their own licences), not Valve's CS2 player/grenade assets. |

**What does *not* appear to exist publicly:** a reputable repo shipping **CS2 player/agent models as
GLB/glTF with skeletons, animations, materials and textures**. The closest is `cs2.agents`, and it is
**CS:GO-era**, converted, with no texture guarantee. Anything else that shows up in searches is either
mesh-only (chams), weapon-only, or an unaudited "asset dump" of unclear provenance/licence.

**Smoke / fire particle textures:** no dedicated public CS2 particle-sprite repo was found. The reliable
path is to extract them yourself from `pak01_dir.vpk`:

1. `--vpk_list` filter `particles/` with `-e vpcf_c` to find smoke/molotov particle systems.
2. Decompile a `.vpcf_c` to text and read the **texture resource path** out of its renderer blocks
   (`m_hTexture` / `texture`); CS2 sprite textures typically live under `materials/particle/`.
3. Extract that `.vtex_c` to PNG (§4d).

Known-good example path shape (from Source2Wiki's CS2 examples): `particles/explosions_fx/explosion_smoke.vpcf_c`.
Exact smoke-grenade and molotov `.vpcf` filenames were **not** verified — treat any guessed name as
unconfirmed and enumerate instead.

---

## 4. Source2Viewer-CLI: verified command lines

Verified against the local binary's own `--help` (`Source2Viewer-CLI --version` →
`19.2.6339+c72208352f5bf62f1482447ed166c548f303f8fa`, macOS arm64) and the VRF docs
([command-line guide](https://s2v.app/ValveResourceFormat/guides/command-line.html),
[models](https://s2v.app/ValveResourceFormat/guides/exporting-models.html),
[maps](https://s2v.app/ValveResourceFormat/guides/exporting-maps.html),
[textures](https://s2v.app/ValveResourceFormat/guides/exporting-textures.html),
[format support](https://s2v.app/ValveResourceFormat/guides/format-support.html)).

⚠️ **Version gotcha, verified by running it:** the docs list a `--game <path to gameinfo.gi>` option, but
**this build rejects it**: `Argument '--game' is not recognized.` Do not use `--game`. Dependency
resolution instead relies on the input living inside a game folder (or being read *from* the game's VPK),
which is why the commands below point the CLI at the real `pak01_dir.vpk` rather than at loose
extracted files.

Useful flags (all confirmed present in `--help`):
`-i/--input`, `-o/--output`, `-d/--vpk_decompile`, `--vpk_list/-l`, `--vpk_dir`, `-f/--vpk_filepath`
(comma-separated), `-e/--vpk_extensions`, `--threads`, `--vpk_verify`, `--vpk_cache`,
`--gltf_export_format {gltf|glb}`, `--gltf_export_animations`, `--gltf_animation_list`,
`--gltf_mesh_list`, `--gltf_export_materials`, `--gltf_textures_adapt`, `--gltf_export_extras`,
`--texture_decode_flags {none|auto|ForceLDR}`, `-a/--all`, `-b/--block`, `--stats*`.
`-o` semantics (from the docs): *"Treated as a folder when it is an existing folder, ends with a path
separator, or has no file extension; otherwise it names the file to write, which requires the input to be
a single file or `--vpk_filepath` to match exactly one file."*

Set `CS2=/path/to/installed/cs2/game/csgo` (the folder holding `pak01_dir.vpk`).

### (a) List player models inside pak01_dir.vpk

```bash
BIN=/Users/wanjungu/Downloads/Source2Viewer-CLI/Source2Viewer-CLI
VPK="$CS2/pak01_dir.vpk"

# raw listing filtered by path + extension
"$BIN" -i "$VPK" --vpk_list -f "agents/models/,models/players/,models/player/" -e "vmdl_c"

# same, with per-entry metadata (archive index / offset / length) — use this to learn which
# pak01_NNN.vpk chunk holds each file
"$BIN" -i "$VPK" --vpk_dir -f "agents/models/" -e "vmdl_c"
```

Notes: CS2's player/agent models live under `agents/models/` per VRF's docs; `models/` holds props,
vehicles, foliage, and *some* legacy `models/player/` paths — enumerate both rather than assuming.
`--vpk_list` applies the `-e`/`-f` filters, so this is cheap on a 7.6 MB dir file.

### (b) Export one player model to GLB with animations + materials + textures

```bash
"$BIN" -i "$VPK" \
  --vpk_filepath "agents/models/t_phoenix/t_phoenix.vmdl_c" \
  -o /path/out/t_phoenix.glb \
  -d \
  --gltf_export_format glb \
  --gltf_export_animations \
  --gltf_export_materials \
  --gltf_textures_adapt
```

- `-o` as a **file name** is only legal because `--vpk_filepath` matches exactly one entry.
- Replacing `--vpk_filepath` with `-f` and `-o` with a **folder** (trailing slash) exports *all* matches
  (batch mode), which is the practical way to export a whole agent folder.
- Restrict animations with `--gltf_animation_list "idle,walk,run"` (docs say it requires
  `--gltf_export_animations`).
- Prefer **`gltf` over `glb`** for anything large: GLB is capped at 2 GB by spec (VRF docs, and echoed in
  the format-support page). A `.gltf` writes separate `.bin`+texture files.
- Expect lossy-but-useful output: glTF gets one LOD only, PBR-flattened materials, baked morph weights,
  no hitboxes; the original shader name/params survive in a `vmat` extras block.
- Reading the model *out of the VPK* (rather than extracting the `.vmdl_c` to a scratch folder first) is
  what makes the material/texture dependencies resolve — there is no `--game` flag in this build.

### (c) Export a map (`.vmap_c` / world mesh) to glTF

Official maps are **self-contained per-map VPKs**, not inside pak01:

```bash
"$BIN" -i "$CS2/maps/de_dust2.vpk" -o /path/out/de_dust2/ -d          # extract everything
"$BIN" -i "$CS2/maps/de_dust2.vpk" \
   --vpk_filepath "maps/de_dust2.vmap_c" \
   -o /path/out/de_dust2.gltf -d \
   --gltf_export_format gltf --gltf_export_materials
```

- Per VRF docs, inside a map VPK the data is at `maps/<name>.vmap_c` plus a `maps/<name>/` dependency
  folder of world nodes/entities. `Source2Wiki` measured **`maps/de_dust2.vpk` ≈ 262 MB** — the map VPK
  itself is small enough to be practical, but its **materials/textures still come from pak01's chunks**,
  so a full textured map export is not possible from the map VPK alone.
- `.vwrld_c` also exports to glTF if you need only world geometry.
- Map glTF export excludes baked lighting/lightmaps/probes and non-`light_environment` lights; props are
  included. Some `_vanity` map variants are truncated menu backdrops — VRF explicitly warns to use the
  non-`_vanity` file.
- Alternative: decompile to `.vmap` for Hammer (needs the CS2 Workshop Tools DLC, app 2279721).

### (d) Extract a `.vtex_c` texture to PNG

```bash
"$BIN" -i "$VPK" --vpk_filepath "materials/particle/<name>.vtex_c" -o /path/out/<name>.png -d
# batch:
"$BIN" -i "$VPK" -o /path/out/ -d -e "vtex_c" -f "materials/particle/"
# HDR sources (skyboxes/probes):
"$BIN" -i "$VPK" --vpk_filepath "<...>.vtex_c" -o /path/out/<name>.exr -d
```

- PNG for LDR, EXR for HDR (VRF auto-selects by format; `--texture_decode_flags` accepts
  `none|auto|ForceLDR`). File extraction always uses the **largest mip**; cubemaps/arrays/volumes export
  per face/slice; textures that store a whole JPEG/PNG/WebP are copied byte-exact.

### (e) Print / decompile a `.vpcf` particle file

```bash
# text dump of every block to stdout
"$BIN" -i "$VPK" --vpk_filepath "particles/explosions_fx/explosion_smoke.vpcf_c" --all
# decompile to .vpcf source text (writes <name>.vpcf into the folder)
"$BIN" -i "$VPK" --vpk_filepath "particles/..." -o /path/out/ -d
# or from an already-extracted loose file:
"$BIN" -i /path/particles/foo.vpcf_c -o /path/out/foo.vpcf -d
```

Per VRF's format-support matrix, `vpcf` is **fully supported** and "round-trip exactly: extraction
re-emits the original stored KV3 tree". The GUI can also simulate most particle functions; `--stats_particles`
collects operators/renderers/emitters/initializers if you want to survey what CS2 uses.

**Other formats worth knowing for this project:** `vmat` → `.vmat` + unpacked texture inputs
(`--gltf_export_materials` handles them automatically for models); `vents` (entity lump, e.g. demo/match
entity data) → glTF/text; `vdata` for CS2 bomb damage; **`.dem` demo parsing is explicitly out of scope for
VRF** ([#1122](https://github.com/ValveResourceFormat/ValveResourceFormat/issues/1122)) — the demo itself
must be parsed by another library.

---

## 5. Working within 17 GB: concrete options

### Option A (recommended): selective download with DepotDownloader

`DepotDownloader` (SteamRE) supports exactly what is needed — **`-filelist`** ("each file path on its own
line", `regex:` prefix allowed) and **`-manifest-only`** ("downloads a human readable manifest for any
depots that would be downloaded"). Install on macOS: `brew tap steamre/tools && brew install depotdownloader`
(or a release binary; needs .NET 8).

Why this works, in numbers (from Source2Wiki's VPK page, updated 2026-08-31):

> *"a current CS2 install splits **48.6 GB of content across 495 numbered archives**, indexed by a
> **7.6 MB directory file**"* … *"Shipped chunks stay far below the limit; CS2's, for example, range from
> **24 to 120 MB**."* — and CS2's `pak01_dir.vpk` is exactly **7,586,006 bytes**.

So the index is tiny and each data chunk is small. Procedure:

```bash
# 1) get the manifest (file list + sizes) without downloading content
DepotDownloader -app 730 -depot 2347770 -manifest-only -os linux -dir ./cs2_manifest

# 2) download ONLY the index
printf 'game/csgo/pak01_dir.vpk\n' > files.txt
DepotDownloader -app 730 -depot 2347770 -os linux -dir ./cs2 -filelist files.txt

# 3) use the index to find which chunk holds each wanted file
Source2Viewer-CLI -i ./cs2/game/csgo/pak01_dir.vpk --vpk_list -f "agents/models/,materials/,particles/" \
                  -e "vmdl_c,vtex_c,vpcf_c"
Source2Viewer-CLI -i ./cs2/game/csgo/pak01_dir.vpk --vpk_dir  -f "agents/models/" -e "vmdl_c"
#    (VPK entries carry archiveIndex/offset/length; map them to pak01_NNN.vpk)

# 4) download just those chunks (24–120 MB each) + any single-map VPK
cat > chunks.txt <<'EOF'
regex:game/csgo/pak01_0(0[0-9]|1[0-9]|2[0-9])\.vpk
game/csgo/maps/de_dust2.vpk
EOF
DepotDownloader -app 730 -depot 2347770 -os linux -dir ./cs2 -filelist chunks.txt
```

Keep `pak01_dir.vpk` and the downloaded `pak01_NNN.vpk` siblings in the same directory so VRF can resolve
them. Budget: index 7.6 MB + a few dozen chunks (~1–4 GB) + one map VPK (262 MB) comfortably fits 17 GB.
**Uncertainty:** I could not run DepotDownloader here (not installed in this sandbox), and whether
`-manifest-only` works anonymously for depot 2347770 was not tested. The `--vpk_dir` claim that per-entry
archive indices are printed is inferred from the tool's own description ("list of files … and information
about them"); verify with one run.

### Option B: full anonymous install on a drive with ≥ 80 GB free

```bash
curl -LO https://steamcdn-a.akamaihd.net/client/installer/steamcmd_osx.tar.gz && tar xzf steamcmd_osx.tar.gz
./steamcmd.sh +@sSteamCmdForcePlatformType linux \
              +force_install_dir /Volumes/External/cs2ds \
              +login anonymous +app_update 730 validate +quit
```
Then extract with Source2Viewer-CLI. This is the highest-fidelity route (real CS2 assets, correct
materials) but needs ~75 GB free for staging + the exports.

### Option C: public GLB repos (no Valve download at all)

`cs2.agents` (~212 MB repo, CS:GO-era agents as GLB, MIT-labelled), plus the weapon-only CS2 GLB sets in
`CS2-Bambook-Assets` / `CS2-WeaponPaints-Website`. Fits the disk budget immediately, but you will not get
CS2 player models, CS2 grenades, or CS2 particle sprites — expect visible fidelity/era mismatch versus the
demo you are replaying.

### Option D: CS:GO legacy assets

App 730's `csgo_legacy` beta branch (depot 731/732) is **32.72 GB on disk / 13.64 GB download** — still
over the 17 GB staging budget, and it is Source-1 CS:GO content (`.mdl`/`.vmt`/`.vtf`), needing a
different conversion pipeline. Only attractive if you can free disk.

*(Curiosity, unverified: depots 731/732 also carry a branch literally named **`demo_viewer`** — 4.85 GB,
0 bytes of download because its files are identical to `public`. Possibly the CS2 demo-playback viewer
build. I did not confirm what it contains; flagging it only as a lead.)*

---

## 6. Licensing / legal reality check

- Valve assets (models, textures, particles, maps) are **Valve's copyrighted content**, licensed to you for
  use through Steam under the [Steam Subscriber Agreement](https://store.steampowered.com/subscriber_agreement/),
  not redistributed. Extracting them for **personal, local use** (including a personal research/rendering
  project) is the widely accepted community practice and is what every modding tool in this report is
  built for; **publishing or shipping the extracted Valve assets** (in a repo, on a website, in a released
  build) is where it becomes infringement.
- Practical consequences for this project:
  - Keeping extracted CS2 assets in a **local, git-ignored cache** outside the repo is the normal approach.
    Do not commit them; do not serve them from a public URL.
  - Public repos that redistribute Valve models (the ones in §3) rely on Valve's general tolerance of
    modding/porting, not on a licence grant. Their **repo licences (e.g. MIT) cover the repo's own code,
    not the Valve-derived models** — treat any such model's licence as "unclear". Valve has issued
    DMCA takedowns for large asset-bearing mods before (e.g. the Halo-3-in-CS2 mod coverage from Ars
    Technica), so provenance matters if you ever publish.
  - Workshop items have their own author licences and are the cleanest redistribution vector if you ship
    content.
  - Source code/format tooling (VRF, ValveResourceFormat, DepotDownloader) is separately licensed
    (VRF is MIT) and is fine to depend on.

---

## 7. Explicit uncertainties

1. Anonymous end-to-end install of app 730 was **not** completed here (disk-bound); the claim rests on
   Valve's wiki + community reports + the fact that anonymous got the full manifest and that an unrelated
   anonymous app downloaded fine.
2. Depot 2347770's file list was **not** read directly (SteamDB blocks bots); the presence of
   `game/csgo/pak01_dir.vpk` and per-map VPKs comes from indexed SteamDB snippets plus the DS-is-the-client
   evidence chain.
3. Exact CS2 paths for smoke-grenade / molotov `.vpcf` files and their sprite `.vtex_c` names were **not**
   verified — enumerate with `--vpk_list` and read the texture path out of the decompiled `.vpcf`.
4. `DepotDownloader -manifest-only` behaviour against CS2 depots, and whether `--vpk_dir` prints per-entry
   archive indices, were not executed here.
5. `cs2.agents` is CS:GO-era; whether its GLBs carry textures/materials is not documented in its README.

---

## Appendix: raw evidence captured during this research

`steamcmd +login anonymous +app_info_print 730` (2026-09-16) — depot table, bytes:

```
depot   2347770 os=-        size=62735072496  dl=53900780944
depot   2347771 os=windows  size=7712252582   dl=4995013808
depot   2347772 os=macos    size=9276         dl=1696
depot   2347773 os=linux    size=7221977502   dl=4604239936
depot   2347774 os=-        size=1142056124   dl=949604816
depot   2347777 os=-        size=88125520     dl=26889664
depot   2347779 os=-        size=2170752148   dl=582148112   (dlcappid 2279721)
depot   2347770 branch csgo_legacy (depot 731): size=32722411363 dl=13644292112
depot   731     branch demo_viewer:            size=4844677826  dl=0
```

`appmanifest_730.acf` after the aborted linux-forced anonymous install:

```
"name"            "Counter-Strike 2"
"StateFlags"      "514"           # 0x202 = UpdateRequired|UpdatePaused
"UpdateResult"    "12"
"BytesToDownload" "59454625760"   # 59.45 GB
"BytesToStage"    "71099106130"   # 71.10 GB
"TargetBuildID"   "25218825"
"InstalledDepots" { }             # nothing installed
```

Control run (`+login anonymous +app_update 90 validate`, forced linux) reached
`Update state (0x61) downloading, progress: 68.16 (658891931 / 966663354)` — proving anonymous downloads
work from this machine, and isolating app 730's failure to disk space.
