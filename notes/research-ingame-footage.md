# Real in-engine footage from CS2 `.dem` files — feasibility research

**Date of research:** 2026-09-15
**Target machine:** macOS, Apple Silicon (Steam installed, CS2 **not** installed, ~15 GiB free on the data volume — `df -h` reports 460 GiB total / 15 GiB available / 97 % used)
**Question:** Is there a route to produce *actual rendered in-game footage* (real Source 2 engine rendering) automatically from a `.dem`, as an alternative to a web-based procedural 3D reconstruction?

---

## 0. Bottom line up front

| Route | Real in-engine 3D? | Automated? | Works on this Mac? | Verdict |
|---|---|---|---|---|
| **HLAE + CS2 (Windows)** | ✅ Yes, genuine Source 2 frames | ✅ scriptable (`mirv_cmd`, `mirv_campath`, `mirv_streams`) | ❌ Windows-only; needs Windows VM/Wine (unverified) | The only serious route — **but not on this machine** |
| **CS2 built-in `startmovie` / camera_path** | ✅ Yes (engine) | ⚠️ partly; `startmovie` is dev-only/locked | ❌ same as above | Not a turnkey renderer |
| **CS2 "clip"/highlight render task** | ✅ Yes (engine, internal) | ❌ no public API | ❌ | Not usable for arbitrary demos |
| **Steam Game Recording / OBS + scripted playback** | ✅ Yes (captures the engine) | ✅ (that is the point of tools like `cs2pov-player`) | ❌ needs CS2 running | Cheapest "real footage" path, still Windows+CS2 |
| **2D radar renderers (open source)** | ❌ No, 2D only | ✅ | ✅ (pure CPU/Python) | Works today; not "in-game footage" |
| **Cloud render service with a real API** | ✅ Yes (in-engine POV clips) | ✅ REST API + MCP | ✅ (remote) | **[DEMO-SLAP](https://demo-slap.net/)** ≈ €0.02–0.06 per 10-s clip; **[CLUTCHKINGS.gg](https://www.clutchkings.gg/)** free. First thing to try |
| **Cloud Windows GPU + HLAE** | ✅ Yes, free-camera 3D | ✅ | ✅ (remote) | Only path to *custom-camera* Source 2 footage; ~$0.71/hr AWS T4; unproven unattended |
| **CS:GO legacy app + `startmovie`** | ✅ Yes, genuine Source 1 frames | ⚠️ scriptable via `playdemo`/`startmovie` | ⚠️ Windows/Linux; macOS is a long shot | **Only if your demos are CS:GO-era** (~34 GB) — see §6.5 |
| **2D radar services / viewers** | ❌ 2D only | ⚠️ Leetify API only | ✅ | No 2D-radar-as-MP4 API exists anywhere |
| **Commission a CS2 movie maker** | ✅ Yes | ❌ manual | ✅ (they render) | $15–60 typical; ~8/12 gigs render the demo themselves |

**Recommendation, in priority order:**

1. **Try the cloud render APIs first — they are cheap, real in-engine, and need no local install.** [DEMO-SLAP](https://demo-slap.net/) exposes a genuine REST API (`/analyze/url` → `/render-custom` with tick range + `povSteamId` → signed `.mp4`) at roughly €0.02–0.06 per 10-second clip with 17 free credits/day; [CLUTCHKINGS.gg](https://www.clutchkings.gg/) renders free (1080p120, TrueView/X-Ray/HUD) with no API. Caveat: Demo-Slap is a two-person side project, so do not build a product on it.
2. **Check what your `.dem` files actually are.** CS2 cannot play CS:GO demos and vice versa. If they are CS:GO-era, the free legacy app (ID 4465480) plus CS:GO's normal `startmovie` gives genuine frames for ~34 GB on Windows/Linux (§6.5) — the cheapest self-hosted real-footage route by far.
3. **Do not attempt local CS2 in-engine rendering on this Mac.** No macOS CS2 build exists, HLAE is Windows-only DLL injection, and CS2 needs ~72 GB installed (Valve states 85 GB) versus 15 GiB free here.
4. **For custom-camera Source 2 cinematics** (HLAE `mirv_campath`/`mirv_streams`), the only route is a rented Windows GPU instance (~$0.71/hr on AWS `g4dn.xlarge`) or a Windows PC with ~150 GB free — and note nobody has publicly documented CS2 + HLAE running unattended in the cloud.
5. **Otherwise** keep the web reconstruction, add open-source **2D radar** rendering (fully automated, CPU-only, free, and there is no competing open-source 3D alternative to duplicate), and commission an editor for a few hero clips.

⚠️ **Legal:** the [Valve Video Policy](https://store.steampowered.com/video_policy) states you "can't sell or license your videos to others for a payment of any kind", and the SSA licenses Steam content for "personal, non-commercial use". Fine for a research project; a constraint if cs-net-v5 ever monetises footage.

---

## 1. HLAE (Half-Life Advanced Effects) and CS2

HLAE is a Windows tool that injects a hook DLL (`AfxHookSource2.dll`) into the game process. For CS2 the hook is called **AfxHookSource2**.

- Repo: <https://github.com/advancedfx/advancedfx>
- Manual/commands: <https://github.com/advancedfx/advancedfx/wiki/AfxHookSource2> and <https://doc.hlae.site/commands/AfxHookSource2/introduction>
- Command list: <https://github.com/advancedfx/advancedfx/wiki/Source2:Commands>

### 1.1 Which version added CS2 support

From the GitHub Releases API (all 414 releases enumerated), the first release containing a CS2 hook is:

- **HLAE 2.146.1 (2023-07-14)** — "Added **AfxHookSource2 0.0.1** … First version. Can not do much, all it currently does is unlock hidden / protected cvars and cmds in CS2 when used with custom loader", targeting `...\Counter-Strike Global Offensive\game\bin\win64\cs2.exe` with `-steam -insecure +sv_lan 1 -console -novid -window -w 1920 -h 1080`.
  <https://github.com/advancedfx/advancedfx/releases>
- Follow-ups in 2023 built the feature set: 2.147.0 (2023-07-24) added `mirv_campath`; 2.149.0.1 shared `mirv_skip`; 2.152.1 (2023-12-05) added `mirv_campath draw`.
- **Latest release as of 2026-09-15: HLAE v2.192.2 (2026-09-12)**, shipping AfxHookSource2 0.41.2 (includes alpha-matte `mirv_streams`, `mirv_fix forceClInterpRatio`, ProRes/ProRes-alpha FFmpeg presets).

So CS2 support started in the **CS2 beta era (July 2023)** and has been continuously maintained since, with a release cadence of roughly every 1–3 weeks as CS2 updates land.

### 1.2 Requirements

| Requirement | Detail | Source |
|---|---|---|
| **OS** | **Windows x64 only.** HLAE injects a DLL into `cs2.exe` (`game\bin\win64`). The installer is a Windows `.exe`/`.msi`. | wiki `AfxHookSource2` |
| **macOS** | ❌ Not supported. FAQ: *"While we currently don't plan to support Linux natively, some games may work through Wine or Proton. However, we cannot guarantee that every game will work."* There is **no** mention of macOS at all — the FAQ's only portability statement is Wine/Proton on Linux, and even that is explicitly not guaranteed. | <https://github.com/advancedfx/advancedfx/wiki/FAQ> |
| **CS2 install** | Required, genuine Steam version. HLAE is *not* supported with cracked/NOSTEAM copies. | FAQ |
| **`-insecure`** | Yes. FAQ: *"HLAE will force `-insecure` when launching a [Source/Source 2 engine] game, which prevents you from joining VAC-protected servers. As long as you only watch demos with HLAE, you will not get banned."* | FAQ |
| **Maturity of `-insecure`** | Historically shaky in early CS2. HLAE 2.149.2 (2023-10-05) carried: *"CS2 HLAE users: Joining GOTV or VAC protected servers will get you VAC banned, despite using -insecure"*, and temporarily renamed the DLL to `AfxHookSource2_IF_YOU_USE_THIS_TO_JOIN_GOTV_OR_VAC_SERVERS_YOU_WILL_GET_VAC_BANNED.dll`. 2.151.0 (2023-10-18) renamed it back, saying Valve "seems to have addressed the issue" but *"we are not sure"*. | GitHub releases |
| **FFmpeg** | Optional but strongly recommended (bundled by the installer) for direct encoding; without it you get huge uncompressed image sequences. | `Source2:mirv_streams` |
| **GPU for campath drawing** | DirectX 11, Pixel Shader 5.0+, Vertex Shader 5.0+ | wiki `Source2:mirv_campath` |
| **Version fragility** | HLAE must be updated after CS2 updates; release notes repeatedly say "Adjusted to CS2 update (1.41.x)". If you don't update the game, older HLAE keeps working. | release notes; FAQ |

### 1.3 What HLAE can actually do for CS2 demo playback

Confirmed AfxHookSource2 command set (wiki `Source2:Commands`, September 2026):

| Command | Purpose |
|---|---|
| `mirv_streams` | **Recording with FFmpeg.** Screen recording (`mirv_streams record screen enabled 1`), arbitrary fps (`mirv_streams record fps 300`), custom output path, presets (`afxFfmpeg`, `afxFfmpegLosslessBest/Fast`, `afxFfmpegHuffyuv`, `afxFfmpegRaw`, `afxFfmpegYuv420p`, `afxSampler30`, `afxFfmpegProres`, `afxFfmpegProresAlpha`), TGA image sequences by default, **multiple simultaneous streams** (normal / depth / matte), **OpenEXR depth**, **before-Panorama capture** (`capture beforeUi`) for HUD separation, **alpha matte** streams, audio via `record startMovieWav`. |
| `mirv_campath` | **Smooth camera paths** with keyframes, interpolation, FOV per keyframe, visual drawing (`mirv_campath draw enabled 1`). Note the CS2 page only documents *system requirements*; full docs live on the Source 1 page. |
| `mirv_camio` | Camera import/export with FOV (Source 1 docs apply). |
| `mirv_input` | Override the game camera and control it freely, including when the demo is paused. |
| `mirv_cmd` | Execute commands at a specific demo time/tick (sub-tick supported since CS2 0.4.1, Sept 2023) — **this is the automation primitive**. |
| `mirv_skip` | Adjust demo time with more precision. |
| `mirv_fov`, `mirv_sky`, `mirv_viewmodel`, `mirv_glow`, `mirv_noflash`, `mirv_colors`, `mirv_panorama`, `mirv_deathmsg`, `mirv_replace_name`, `mirv_endofmatch`, `mirv_fix`, `mirv_listentities`, `mirv_cvar_unhide_all`, `mirv_cvar_unlock_sv_cheats`, `mirv_loadlibrary`, `mirv_reshade` | Camera/visual/HUD control; `mirv_reshade` + [ReShade advancedfx](https://github.com/advancedfx/ReShade_advancedfx) for post-processing inside the capture. |
| `mirv_script_*` (experimental JS, "mirv-script") | `mirv_script_load`, `mirv_script_exec`, plus bundled snippets: `mirv_script_aim` (mimics old `mirv_aim`, aims at entities/points), `mirv_script_point` (campath keyframes → `mirv_input`), **`mirv_script_spec_lock`** (locks `spec_player` on one player), `mirv_script_view`, `mirv_script_voice`. Also `mirv.fs`, `mirv.getDemoFilePath`, `mirv.getGameDirectory` (HLAE 0.40.0, 2026-07-10). |

### 1.4 Concrete limitations (CS2 vs CS:GO)

1. **Windows-only.** No native macOS or Linux build; the only portability story is Wine/Proton on Linux and it is explicitly not guaranteed. On Apple Silicon you would have to run the Windows `cs2.exe` (via CrossOver or a Windows VM) *and* get a Windows DLL injected into it — see §6.
2. **No `mirv_calcs` for CS2.** The wiki says there are [no plans](https://github.com/advancedfx/advancedfx/issues/873) to bring `mirv_calcs` (weapon-mounted cameras, arbitrary bone-follow cameras, advanced calculations) to CS2. Workarounds listed are Source 2 Filmmaker, a bespoke `mirv-script`, or a third-party tool ([paper's HLAE Observer Tool](https://github.com/papesgit/hot)).
3. **Much smaller command surface than CS:GO.** CS:GO's AfxHookSource has ~60 commands (`mirv_aim`, `mirv_calcs`, `mirv_vpanel`, `mirv_snd_filter`, `mirv_streams` advanced matte, `mirv_pov`, `mirv_guides`, …). AfxHookSource2 has ~25, and several are thin wrappers ("See Source:… page").
4. **Only one Source 2 game is supported: CS2.** No Deadlock/Dota/HL:A support. (`Source2:mirv_streams`: *"Counter-Strike 2 is the only Source 2 game that is supported."*)
5. **Frequent breakage.** Every CS2 update can break the hook until a new HLAE ships; release notes are full of "Adjusted to CS2 update". Conversely, updates have also broken *demos*: HLAE 2.149.1.1 notes "Adjusted to CS2 update. PS: they broke old demos again."
6. **VAC/hosting risk.** `-insecure` blocks VAC servers by design; the 2023 CS2 window showed that even `-insecure` did not reliably prevent bans when joining live CSTV/game servers. Demo-only use is the documented safe path.
7. **Recording cost is real-time-ish and heavy.** `mirv_streams record fps 300` on a 40-minute match generates tens of thousands of frames; high resolution, high fps, multiple streams and FFmpeg lossless presets all increase time (HLAE FAQ "Can I speed up the recording process?"). There is no "render faster than real time" mode.
8. **Partial matte/HUD separation.** CS2 HUD can be removed with `mirv_streams edit n capture beforeUi`, but the alpha-HUD method is documented as *"not perfect and has some flaws"*.

---

## 2. CS2 built-in movie-making commands

### 2.1 Evidence base

The cleanest hard evidence is a **`cvarlist` dump from a real CS2 client**: <https://github.com/SuGolYolLom/CS2-Cvars-Cmds> (`cs2/cvarlist.md`, `steam.inf` → `PatchVersion=1.40.5.1`, `VersionDate=Nov 15 2024`, `appID=730`). It lists 5,104 entries with their exact flags. Flag meanings: `FCVAR_RELEASE` = normal, visible, usable; `FCVAR_DEVELOPMENTONLY` = hidden and gated; `FCVAR_CHEAT` = needs `sv_cheats 1`; `FCVAR_DONTRECORD` = not written into demos.

Cross-checks: <https://csbepro.com/cs2-console-commands/demo/> (195 demo/GOTV entries from the live game, updated), <https://totalcsgo.com/commands/playdemo>, <https://totalcsgo.com/commands/startmovie>, and <https://github.com/saul/cvar-unhide-s2>.

### 2.2 Command-by-command status in CS2

| Command | Present in CS2? | Flags in CS2 | Notes |
|---|---|---|---|
| `playdemo` | ✅ | `FCVAR_RELEASE` — "Play a recorded demo file (.dem)" | Works. `playdemo replays/x`; `.dem` extension optional. totalcsgo labels it "CS2 & CS:GO Compatible". |
| `demo_gototick` | ✅ | `FCVAR_RELEASE` — "Skips to a tick in demo." | Works. |
| `demo_goto` | ✅ | `FCVAR_RELEASE` | `demo_goto [ticks] [relative] [pause]`. |
| `demo_pause` / `demo_resume` / `demo_togglepause` | ✅ | `FCVAR_RELEASE` | Works. |
| `demo_timescale` | ✅ | `FCVAR_RELEASE` — "Sets demo replay speed." | **The CS2 replacement for `host_timescale`-style demo slow-mo.** Values 0.25/0.5/1/2/4/8. |
| `demo_marktick` / `demo_gotomark` | ✅ | `FCVAR_RELEASE` | Bookmarks. |
| `demoui` / `demo_info` / `listdemo` / `demolist` | ✅ | `FCVAR_RELEASE` | |
| `record` | ✅ | `FCVAR_RELEASE` (demo recording) | Local POV recording, not movie output. |
| `spec_player` | ✅ | `FCVAR_CLIENTDLL FCVAR_CLIENTCMD_CAN_EXECUTE` — "Spectate a player by name or slot" | Works. |
| `spec_goto` | ✅ | `FCVAR_CLIENTDLL FCVAR_CLIENTCMD_CAN_EXECUTE` — `spec_goto x y z pitch yaw` | Free-camera positioning. |
| `spec_mode`, `spec_next`, `spec_prev`, `spec_pos` | ✅ | client / (`spec_pos` is cheat) | |
| `spec_lock_to_accountid` | ⚠️ present but **`FCVAR_DEVELOPMENTONLY`** | hidden/gated | Needs unlocking. HLAE ships `mirv_script_spec_lock` as a workaround. |
| `spec_show_xray` | ✅ | `FCVAR_CLIENTDLL FCVAR_ARCHIVE FCVAR_RELEASE`, default **1** | |
| `cl_draw_only_deathnotices` | ✅ | present | HUD cleanup. |
| `sv_cheats` | ✅ | `FCVAR_NOTIFY FCVAR_REPLICATED FCVAR_RELEASE`, default false | As CSbePRO puts it: *"Works only with sv_cheats 1, so on your own server or in offline practice. On a matchmaking server it does nothing."* During local demo playback the client runs the recorded session locally, so `sv_cheats 1` is the usual way to enable camera/visual commands — but note that HLAE ships a dedicated `mirv_cvar_unlock_sv_cheats` command to *"Unlock client-side sv_cheats protection as much as possible"*, implying CS2 does not leave `sv_cheats` fully open even locally. |
| **`host_framerate`** | ✅ | **`FCVAR_RELEASE`** — "Set to lock per-frame time elapse." | Registered, unrestricted, and this is the classic engine-lock trick. Caveat: in CS2 the documented, demo-aware speed control is `demo_timescale`; the practical effect of `host_framerate` on Source 2 demo playback is not spelled out in any Valve or HLAE documentation I found. |
| `host_timescale` | ✅ | `FCVAR_REPLICATED FCVAR_CHEAT` | Present, cheat-gated. |
| **`startmovie`** | ⚠️ **present but `FCVAR_DEVELOPMENTONLY FCVAR_DONTRECORD`** — "Start recording movie frames." | hidden/gated | See §2.3. |
| **`endmovie`** | ⚠️ present but `FCVAR_DEVELOPMENTONLY FCVAR_DONTRECORD` — "Stop recording movie frames." | hidden/gated | |
| `movie_fixwave` | ✅ | `FCVAR_DEVELOPMENTONLY` — "Fixup corrupted .wav file if engine crashed during startmovie/endmovie, etc." | Direct evidence that `startmovie`/`endmovie` are *real* CS2 engine code paths that write `.wav` (+frames). |
| `ui_render_task_generate_clips` | ✅ | `FCVAR_CLIENTDLL FCVAR_RELEASE`, default **false** | CS2's internal clip renderer. See §3. |
| `ui_render_task_file` | ✅ | default `rendertask` | Output basename for the clip renderer. |
| `ui_render_task_fps` | ✅ | default **60** | Frame rate for the clip renderer. |
| `ui_render_task_clips_label` | ✅ | default `dealt_damage` | |
| `demo_movie_write_intervals` | ✅ | `FCVAR_HIDDEN FCVAR_RELEASE`, default false — "Write highlight interval metadata along with movie files when recording." | Confirms CS2 records "movie files" alongside highlight interval metadata. |
| `demo_allow_game_mismatch` | ✅ | `FCVAR_DEVELOPMENTONLY` — "Allow playback of demo even if game directories are not matched [may crash or fail to load]." | The version-lock escape hatch is dev-gated. |
| `camera_path_add` / `camera_path_clear_all` / `camera_path_delete` / `camera_path_demo` / `camera_path_edit_mode` / `camera_path_load` / `camera_path_save` | ✅ | require `sv_cheats` (per CSbePRO) | **CS2 has its own built-in camera-path system** — no HLAE needed for a basic scripted camera move. |
| `tv_demo_starttick` | ✅ | `FCVAR_DEVELOPMENTONLY`, default 0 | |
| `timedemo` / `timedemoquit` | ✅ | `FCVAR_RELEASE` | Benchmark playback, not capture. |
| `save_screenshot` / `screenshot` | ✅ | present | Single frames only. |
| `cl_demo_predict` (TrueView) | ✅ (added 2025-11-04) | 0/1/2 — disabled by default on client-version mismatch, `2` = always | See §2.5. |

totalcsgo explicitly labels `startmovie` and `endmovie` as **"CS:GO Command"** (i.e. not exposed as a CS2 command in their DB), while `playdemo` is labelled "CS2 & CS:GO Compatible" — consistent with the `FCVAR_DEVELOPMENTONLY` finding: the code is in the engine but hidden from normal CS2 use.

### 2.3 Does CS2 support `startmovie` and image sequences?

**Mechanically yes; practically it is locked and not the recommended path.**

- `startmovie` / `endmovie` exist in the CS2 binary with descriptions "Start/Stop recording movie frames", and `movie_fixwave` exists specifically to repair `.wav` files "if engine crashed during startmovie/endmovie" — i.e. the CS2 engine can write frames + a WAV audio track.
- They are flagged **`FCVAR_DEVELOPMENTONLY`** (and `FCVAR_DONTRECORD`), so they are hidden/gated in a stock client. The two community ways to unlock them are:
  - **HLAE**: `mirv_cvar_unhide_all` (AfxHookSource2 command: "Make hidden console commands and variables accessible") and `mirv_cvar_unlock_sv_cheats` ("Unlocks client-side `sv_cheats` protection as much as possible") — needs `-insecure`.
  - **cvar-unhide-s2**: a Metamod-style addon that *"reveal[s] all console variables and commands that are marked as hidden or development-only in Counter-Strike 2"* via `cvar_unhide`. Its README states: *"You must add `-insecure` to Counter-Strike's launch options for this plugin to load."* <https://github.com/saul/cvar-unhide-s2>
- Practical consequence: **on a stock CS2 install you cannot just type `startmovie x tga`.** And even unlocked, the movie-making community treats CS2's `startmovie` as inferior to HLAE's `mirv_streams`, which is why `mirv_streams` exists and is the documented recording path. (Note: I could not run a live CS2 to verify the exact runtime behaviour of an unlocked `startmovie`; the flags and the `movie_fixwave` description are the direct evidence.)

### 2.4 Is cheating / insecure mode required?

- **`-insecure`**: required for any third-party hook/injection (HLAE, cvar-unhide-s2). Not required to merely watch a demo. HLAE forces it automatically.
- **`sv_cheats 1`**: required for many useful demo/camera commands — CSbePRO marks `host_timescale`, `camera_path_*`, `noclip`, `phys_highlight_*`, `r_refraction_clip_plane_adjust`, `spec_pos` etc. as requiring it. On a local demo this is achievable because you are the local server; on a live matchmaking server `sv_cheats` "does nothing".
- **`-dev`**: not documented as a requirement, but development-only commands are the category that `-dev` historically unhides; the community route is the two hooks above rather than `-dev`.

### 2.5 Demo availability and restrictions (POV vs GOTV, version lock)

- **Valve matchmaking demos are GOTV demos.** The in-game flow is *Play → Watch → Your Matches → (match) → **Download GOTV Demo***, which lands as a `.dem` (sometimes zipped) that you place in `...\game\csgo\replays\`. (<https://www.switchbladegaming.com/cs2/demo-console-commands/>) There is no user-facing POV-demo download for matchmaking; POV demos come from third-party platforms (FACEIT/ESEA), from your own `record`, or from tournament archives.
- **Demos are version-locked.** Playing an older `.dem` after a CS2 update yields *"demo file is incompatible with this game version"* — a widely reported CS2 issue (<https://steamcommunity.com/app/730/discussions/0/801220136404146923/>, <https://www.hotspawn.com/counter-strike/guide/cs2-demo-is-incompatible-with-this-game-version-error-fix>). The dev-only `demo_allow_game_mismatch` cvar exists but is documented as "may crash or fail to load".
- **TrueView (`cl_demo_predict`)** — added in the 2025-11-04 update — reconstructs the observed player's original experience by re-running client-side prediction, and shows Damage Prediction effects. Valve explicitly notes: *"TrueView demo playback might not work properly on old demos and is disabled by default if the playback client version does not match the client version when the demo was recorded. You can set `cl_demo_predict 2` to override."* It also does **not** reconstruct precise rendered-frame timings, so recoil/muzzle flash/blood/ragdoll can be 1–2 frames late. (<https://consolepcgaming.com/counter-strike-2-adds-trueview-demo-playback-and-fixes-surfing-and-demo-bugs/>)
- **CS2 cannot play CS:GO-era demos at all.** They require the CS:GO client — which is again installable as the standalone `csgo legacy` app (Steam app 4465480, relisted 2026-03-04) or via the hidden `csgo_legacy` beta branch inside CS2. See §6.5; this distinction determines which of your `.dem` files is even usable by which engine.
- **Demo chat is now encrypted** (update 2026-09-09/10): *"Chat is now encrypted in demo files using the same rules as CS:GO: Team chat is no longer readable when playing back demo files. Global chat is only readable in demo files when played back via the in-game watch UI."* (<https://store.steampowered.com/news/app/730/view/692020124159312044>, <https://consolepcgaming.com/counter-strike-2-update-reworks-custom-cameras-and-encrypts-demo-chat/>) This matters if a downstream pipeline wanted chat text; it does not affect video.
- **`spec_player` / camera control restrictions**: `spec_player` and `spec_goto` are normal client commands (`FCVAR_CLIENTCMD_CAN_EXECUTE`); `spec_lock_to_accountid` and `spec_lock_to_current_player` are dev-only. Camera paths via `camera_path_*` need `sv_cheats`; HLAE's campath/input do not depend on `sv_cheats` to the same degree.

**In practice**: `playdemo` + `demo_gototick` + `demo_timescale` + `spec_player` + `spec_goto`/`camera_path_*` + `host_framerate` **all work in CS2** and are enough to script a deterministic camera-and-tick playback session. `startmovie`/`endmovie` exist but are gated; `spec_lock_to_accountid` is gated.

---

## 3. CS2 "Clip capture" / Highlights / Save clip

What exists inside CS2 (all from the cvarlist evidence, cross-referenced with press coverage):

- **`ui_render_task_generate_clips`** (default `false`), **`ui_render_task_file`** (default `rendertask`), **`ui_render_task_fps`** (default `60`), **`ui_render_task_clips_label`** (default `dealt_damage`). These were discovered in CS2 files in May 2024 by ThourC alongside strings like `"CLIPS (start frame, end frame, event ticks)"`, and were interpreted as groundwork for Steam clip sharing. (<https://cs-config.com/allnews/vozmozhno-skoro-v-cs2-mozhno-budet-delitsja-vnutriigrovymi-rolikami>) This is an **internal render-task system**: it renders clips from demo data at a fixed fps, in-engine.
- **Highlight system**: `demo_highlight_seconds_before` (default 6), `demo_highlight_seconds_after` (default 2), `demo_highlight_fade_duration`, `sv_highlight_distance`, `sv_highlight_duration`, `stats_highlight_interval`, plus **`csgo_fatdemo_enable` / `csgo_fatdemo_output`** (fatdemo = the highlight container format).
- **`demo_movie_write_intervals`** — "Write highlight interval metadata along with movie files when recording."

**Can any of this be automated to render a demo to video?** No, not via a supported/scriptable interface:

- There is no documented console command, config key, or API to point CS2's clip renderer at an arbitrary `.dem` and get an MP4. The `ui_render_task_*` cvars have no descriptions, are not user-facing, and are driven by the UI/Steam integration — not by a CLI.
- The **user-facing "clips/highlights" flow is manual**: watch highlights in the game UI (or Steam overlay), and rely on an external recorder.
- **Steam Game Recording** (Steam client feature, works with any game incl. CS2) continuously records gameplay in the background and can create/export clips and a timeline — <https://help.steampowered.com/en/faqs/view/23B7-49AD-4A28-9590>, <https://store.steampowered.com/gamerecording> — but it records *the game you are playing*, i.e. it is **screen/video capture of a demo playback session**, not a demo renderer. It is however a legitimate automation component: script the demo playback, let Steam Game Recording or OBS capture it.
- Third-party auto-clippers exist: **Medal.tv** markets a CS2 auto-clip integration ("Auto-clip Aces, Clutches & Knife Kills", <https://medal.tv/developer/cs2>).

**Verdict:** Clip capture/Highlights is *not* an automation route to demo→video. It is an in-game manual feature that proves Valve can render clips in-engine, but exposes no automation surface.

---

## 4. Open-source / free tools that render CS2/CS:GO demos to video

### 4.1 TL;DR of the tooling landscape

**There is no open-source tool that renders CS2 demo 3D geometry to video *without the actual CS2 client*.** Every true 3D / "real in-game footage" output goes through the real engine (`startmovie`, or HLAE `mirv_streams`, or OBS capture of a scripted playback). A maintainer of one highlight tool states it bluntly: *"Rendering video from a demo without the game is impossible."* ([Rovniy/cs2-highlights-maker](https://github.com/Rovniy/cs2-highlights-maker))

The only out-of-engine 3D replay product found is **closed source** (§4.4). All open-source activity that does *not* need the game is **2D radar**, and most of those are interactive viewers with **no video export** — you screen-capture them or pipe frames to ffmpeg yourself.

**However, you may not need to build this at all:** a handful of **hosted services render your uploaded `.dem` in-engine in the cloud** and hand back an MP4 — most usefully [DEMO-SLAP](https://demo-slap.net/) (real REST API + MCP, ~€0.02–0.06 per 10-second clip) and [CLUTCHKINGS.gg](https://www.clutchkings.gg/) (free, 1080p120, no API). Those produce **POV-style clips**, not free-camera cinematics. See §5.1 — this is the first thing worth testing.

### 4.2 The best current open-source in-engine pipelines

| Tool | Stars / license / activity | Mechanism | Platform |
|---|---|---|---|
| **[CS Demo Manager](https://github.com/akiver/cs-demo-manager)** + [video docs](https://cs-demo-manager.com/docs/guides/video) | 2,005★, MIT, pushed 2026-09-14 | **The reference implementation.** Drives CS2 per-tick through console commands with tick-range "sequences", per-player auto-sequences, camera-focus events, per-sequence CFG of game commands. Recording engine = CS2's own `startmovie` (raw TGA+WAV) **or** HLAE `mirv_streams` (direct via FFmpeg). Emits `.mp4`/`.avi` + `.wav`. | **Windows: CS2 ✅ / HLAE ✅. Linux: CS2 ✅ (startmovie+FFmpeg), HLAE ❌. macOS: CS:GO ✅, CS2 ❌, HLAE ❌** |
| **[noih/cs-demodesk](https://github.com/noih/cs-demodesk)** | 1★, **AGPL-3.0**, pushed 2026-09-15 | Rust/Tauri app. Launches a **separate CS2 process via HLAE with `-insecure`**, drives it over the game's own **netcon console**, uses a separate `USRLOCALCSGO` folder so game files are never modified. Highlight MP4s (H.264/H.265, CPU or NVIDIA), HUD toggles, size limits, 2D replay. Bundles HLAE, FFmpeg, Source 2 Viewer CLI (radar extraction), vendored `demoparser`. | **Windows only** (MS Store + portable EXE) |
| **[reka-ai/cs2-dem-renderer](https://github.com/reka-ai/cs2-dem-renderer)** (Reka "CS2-10k" dataset) | 25★, MIT, pushed 2026-06-25 | 2-pass parse → JSON action sequence → **launches CS2 via Steam with the demo loaded**; a CSDM server plugin executes the actions at exact ticks; TGA frames piped to ffmpeg (VAAPI HEVC) in real time; emits one **`.mp4` per player per round** plus parquet metadata (mouse deltas, position, yaw/pitch, WASD). POV-only, built for ML datasets rather than free-camera cinematics. | **Linux-first**; server plugin is `libserver.so`; "can work on Windows with minimal modifications". **Plugin pinned to game 1.41.6.5** |
| **[DrEAmSs59/CS2-insight-agent](https://github.com/DrEAmSs59/CS2-insight-agent)** | 869★, **PolyForm Noncommercial**, pushed 2026-09-11 | One-stop suite: demo library with directory watching (5E/完美/官匹/FACEIT), highlight/fail/clutch detection, round timelines, 2D replay, heatmaps, then **batch-launches CS2 replay and drives OBS Studio** to record; multi-track editor; optional LLM commentary. *"0 injection, 0 hook, 0 reverse-engineering of the game process, local playback, low risk."* | **Windows only** (Tauri + WebView2). Skin-rewriting core is closed source; commercial use prohibited |
| **[Rovniy/cs2-highlights-maker](https://github.com/Rovniy/cs2-highlights-maker)** | 3★, **no license**, 2026-06-12 | `demoparser2` → highlight scoring → **CS2 + HLAE `AfxHookSource2` injection** → CSDM server plugin executes a `<demo>.json` action file at exact ticks → `mirv_streams` pipes frames to FFmpeg. Auto-downloads HLAE/FFmpeg; `highlights doctor` env check. | Windows, CS2 + Steam, Python ≥3.11 |
| **[eSheikh/cs2-demo-highlighter](https://github.com/eSheikh/cs2-demo-highlighter)** | 1★, MIT, 2026-07-06 | Go CLI that **generates HLAE `.cfg` scripts** using `mirv_streams` (deliberately *without* `startmovie`) for kill-in-smoke / blinded / wallbang / noscope / headshot / multikill / clutch; POV lock via `spec_player`; jumps via `demo_pause` → `demo_gototick` → `demo_resume`. | Needs HLAE to actually record → **Windows** |
| **[grvig/auto-highlights](https://github.com/grvig/auto-highlights)** | 0★, no license, 2026-08-17 | 4 layers: awpy engine → clip plan; **CS2 + OBS WebSocket recorder**; ffmpeg stitcher; GUI. Pluggable `Recorder` interface with a planned HLAE backend. | Windows, Python 3.11+ |
| **[Co1Swet/MulNX_CS2](https://github.com/Co1Swet/MulNX_CS2)** | 118★, AGPL-3.0, pushed 2026-09-15 | x64 asm/C++ reverse-engineering + runtime hooking platform for broadcast production: hardware-level control, **cinematic camera work**, remote networking, real-time GUI. | Windows x64 only |
| **OBS-driver scripts** | [Keanoski/CS2-Highlight-Automator-for-OBS](https://github.com/Keanoski/CS2-Highlight-Automator-for-OBS) (MIT), [nickdevroye/cs2pov-player](https://github.com/nickdevroye/cs2pov-player) (MIT) | Python GUI: parse `.dem` for players/rounds, **skip to round and follow a player's POV**, hotkey round skip, `disconnect` after last round, **OBS WebSocket** start/stop. Config points at `...\game\csgo\` + OBS host/port/password. | Windows + CS2 + OBS |

**Important operational note:** `startmovie` output lands in the game install folder as uncompressed TGA + WAV, so CS Demo Manager warns you need **tens of GB free** (reportedly >40 GB) — on a machine with 15 GiB free this fails immediately. This is exactly why HLAE's `mirv_streams` (streaming frames straight into FFmpeg) is preferred.

### 4.3 2D radar tooling (fully cross-platform, works on this Mac today)

| Tool | Stars / license / activity | Output | Notes |
|---|---|---|---|
| **[awpy](https://github.com/pnxenopoulos/awpy)** | 612★, MIT, pushed 2026-09-10 | `awpy.plot.plot()` / `heatmap()` → Matplotlib figures; **`awpy.plot.gif(map_name, frames_data, output_filename, duration, lower_points_frac)` → animated GIF** (multi-level maps like Nuke). No built-in MP4 muxer | Pure Python/CPU. **Runs on macOS/Apple Silicon.** Easiest on-ramp |
| **CS Demo Manager 2D viewer** ([docs](https://cs-demo-manager.com/docs/guides/2d-viewer)) | 2,005★, MIT, 2026-09-14 | Interactive radar playback with lower-level radar, telestrator pen/eraser, **voice-comms audio sync** | **No export button** → screen-capture only. macOS builds exist |
| **[sparkoo/csgo-2d-demo-viewer](https://github.com/sparkoo/csgo-2d-demo-viewer)** ([live](https://2d.sparko.cz)) | 76★, MIT, 2026-09-10 | Go parser → **WASM**, Preact front-end, parses the binary demo **client-side**; FACEIT extensions | Any browser. No video export |
| **[zenojunior/cs2d](https://github.com/zenojunior/cs2d)** ([app](https://cs2d.app)) | 20★, **no license file**, 2026-07-02 | Vue 3 + Rust `source2-demo` → WASM; offline; heatmaps, grenade trajectories, flash effectiveness, economy, telestrator | Browser; exports `.cs2dv` project, **not video** |
| **[drjackild/cs2-demo-opener](https://github.com/drjackild/cs2-demo-opener)** | 13★, 2026-08-03 | Tauri + Rust + Preact; in-app 2D replay (multi-layer radar, grenades, kill feed) **plus** generates `voice_demo.cfg` and launches CS2 for isolated voice/POV | Desktop; launches CS2 (Windows-centric) |
| **[Linus4/csgoverview](https://github.com/Linus4/csgoverview)** | 195★, MIT, **last push 2024-05-11 (dormant)**, CS:GO only | Native Go + SDL2 2D viewer | Win/Linux/**macOS** (Mac build wiki); screen-capture only |
| **[unicbm/demotracer](https://github.com/unicbm/demotracer)** | 65★, AGPL-3.0, 2026-09-14 | Converts a demo into a **bot-replay bundle** played back through bots on a matched local server | **Windows x64 only**; in-game, but not a video encoder |
| Browser viewers (no game needed) | [2d.sparko.cz](https://2d.sparko.cz), [cs2d.app](https://cs2d.app), [deadem.com](https://deadem.com) | Interactive 2D | **Work on Apple Silicon today** |

Parser foundations (no rendering): **[demoinfocs-golang](https://github.com/markus-wa/demoinfocs-golang)** (1,057★, MIT, pushed 2026-09-11 — the engine behind CS Demo Manager's 2D/video, csgoverview and sparkoo), **[LaihoE/demoparser](https://github.com/LaihoE/demoparser)** (718★, MIT, pushed 2026-09-03 — Rust core with Python + JS/WASM bindings; the `demoparser2` package used by most 2025–26 highlight scripts), **[akiver/cs-demo-analyzer](https://github.com/akiver/cs-demo-analyzer)** (122★, MIT, 2026-09-09), **[Igor-Losev/deadem](https://github.com/Igor-Losev/deadem)** (MIT, 2026-09-12), **[Rupas1k/source2-demo](https://github.com/Rupas1k/source2-demo)** (Apache-2.0).

Radar imagery comes from CS Demo Manager's bundled assets, [MurkyYT/cs2-map-icons](https://github.com/MurkyYT/cs2-map-icons), or extraction from the game VPK with [ValveResourceFormat](https://github.com/ValveResourceFormat/ValveResourceFormat) / [invakid404/cs2-radar-extractor](https://github.com/invakid404/cs2-radar-extractor).

### 4.4 Closed-source 3D / commercial renderers

- **Skybox3D** — [Steam app 1549660](https://store.steampowered.com/app/1549660/Skybox3D/) (Skybox Technologies, 2021, free with a [skybox.gg](https://skybox.gg) account). "View maps from anywhere and any angle", Glassify, player path tracking, any-player POV. Store lists **Windows, macOS and Debian Linux**. **This is genuine out-of-engine 3D reconstruction** (its own renderer, not Source 2 in-engine) — but it is closed source, **CS:GO-era with no CS2 claim on the store page**, and interactive-only (no stated video export). This is the closest thing that exists to "3D without the game".
- **Demo-Slap** — commercial SaaS with a **real public REST API + MCP connector** that renders **in-engine** CS2 clips and fragmovies to a signed CDN `.mp4` (options include `hideUi`, `showXRay`, `showTracers`, per-player POV, 720p/1080p). ~€0.02–0.06 per 10-second clip, 17 free credits/day. Only public code is an HTTP-client skill; **the renderer itself is closed**, it is a two-person Swiss side project ("the rendering rigs run at our place"), so treat it as prototype-grade rather than durable infrastructure. Full endpoint list in §5.1.
- **CLUTCHKINGS.gg** — free cloud CS2/CS:GO demo→clip renderer (TrueView, X-Ray, HUD, radar, killfeed; H.264 1080p60/120, downloadable). No documented API. **Scope.gg Clips** is similar (server-side recording, first-person + opponent POV, downloadable 1080p60, no API).
- **cs2.cam** — closed commercial web platform with a Recording section (Record Round / Record POV / Death Analysis / Team Recording / OBS Integration) plus a desktop app. Orchestrator + OBS; all rights reserved.
- **xplay.gg** — closed commercial platform; publishes replay/console guides and recording-tool listicles, but **no open renderer and no public repo**.

### 4.5 Explicitly checked and *not* found

- No project named `cs2-radar-renderer` (only DMA-cheat radars match the string), no `csgo-dem2video`, no `s2dc`, no `cs2-demo-renderer` beyond reka-ai's dataset tool, and **`demotovideo` is an empty repo** (0★, no README).
- **No WebGL/Three.js/Babylon out-of-engine 3D CS2 demo reconstructor exists in public.** Browser demo tools (sparkoo, cs2d, deadem) are strictly 2D. So the user's web-based procedural 3D reconstruction is, as far as this survey can tell, **not duplicating any existing open-source project** — there is no off-the-shelf alternative to adopt.
- **Source 2 Viewer / ValveResourceFormat** (2,424★, MIT, pushed 2026-09-15) is an asset browser/decompiler for maps, models and materials — **not** a demo renderer, though it is the standard way to extract VCLIP/VPK assets such as radar images.
- **`Flowtter/crispy`** (MIT, 2025) is an ML *video* highlight detector over already-recorded frames; it does not read `.dem`.

### 4.6 Abandoned / legacy (for completeness)

CS:GO-era OBS/demo automation — `sourav-kanta/csgodemo_to_video` (2023), `JackLairm/csgo-demo-to-mp4` (2019), `zek/replay-master` (GPL-3.0, last push 2024-04-14), `jackal-u/csgo_competitive_dataset_tool` (2024) — **all abandoned and CS:GO-only**. Parsers: `saul/demofile` (**archived** 2025-08-31, CS:GO-era), `markus-wa/cs-demo-minifier` (last push 2023-02-25, CS:GO only), `StatsHelix/demoinfo` (last push 2021-09-22, dead), `Linus4/csgoverview` (dormant since 2024-05-11).

**But note:** CS:GO tooling is *less* dead than it looks, because the **CS:GO client itself came back in March 2026** as the standalone `csgo legacy` app (app 4465480) — and CS Demo Manager still supports CS:GO video generation **including on macOS**. See §6.5. If your demos are CS:GO-era, this legacy tooling is live infrastructure, not archaeology. Conversely, every "CS:GO demo → video" script you find online is unusable for CS2 demos.

### 4.7 Fragility and licensing warnings

- **Every in-engine approach couples to two versioned things:** a CS2-version-matched server plugin (CSDM's `server.dll`, reka-ai's `libserver.so`) and an HLAE release that survives the latest CS2 patch. CS Demo Manager and CS DemoDesk both document needing updates after CS2 patches; reka-ai pins per-game-version branches.
- **Licences are uneven.** HLAE is MIT-ish (repo metadata) but is *"technically a hack"* with a VAC warning. CS2-Insight-Agent is **source-available but non-commercial** (PolyForm NC 1.0.0) with a closed skin component. `cs2d` ships **no LICENSE file**. `Rovniy/cs2-highlights-maker`, `grvig/auto-highlights` and most 0–3★ highlight scripts have **no license at all** (default all-rights-reserved → not safely reusable in a project).

---

## 5. Cloud / remote options

**Correction to the first pass of this report:** genuine **in-engine** cloud rendering of CS2 demos *does* exist and is partly automatable. The accurate split is: **2D radar/stats services (many) · in-engine POV clip renderers (a few, one with a real API) · full-match free-camera cinematic renders (nobody)**.

### 5.1 Services that will render a `.dem` you give them

| Service | What you get | In-engine 3D or 2D? | Public API | Price |
|---|---|---|---|---|
| **[DEMO-SLAP](https://demo-slap.net/)** | **Cloud demo → MP4 clips.** OpenAPI 3.0 v2.5.0 at [api-doc.demo-slap.net/api-documentation.yaml](https://api-doc.demo-slap.net/api-documentation.yaml): `/analyze/url`, `/analyze/{jobId}/data`, `/render/{jobId}`, `/render-custom/{jobId}` (tick/duration/`povSteamId`), `/render-fragmovie/{jobId}`, `*-credits-calc`. Auth via `x-api-key`; returns a signed CDN `.mp4`. Render options `hideUi`, `showXRay`, `showTracers`, `hideKillfeed`, `hideWatermark`, 720p/1080p, per-player **POV**; analyze `source` enum includes `rungame-url`. MCP connector at `https://mcp.demo-slap.net/mcp`. | **In-engine** (POV-style) | ✅ **Yes** — real REST API + MCP | **≈ €0.02–0.06 per 10-second clip** (0.1 cr/s base, ×1.5 for 1080p, ×1.5 to drop watermark, 1-credit floor); free 17 credits/day; bundles €2.99/100cr … €14.99/800cr |
| **[CLUTCHKINGS.gg](https://www.clutchkings.gg/)** | "We record and host your CSGO and CS2 video clips **on our cloud servers**" — TrueView, X-Ray, HUD, radar, crosshair and viewmodel controls; H.264 **1080p60/120**, downloadable | **In-engine** (POV) | ❌ None documented | **Free** (100 normal-priority clips/week if Steam profile is public and you have ≥25 MM wins; else 1/week; 1 GB storage); €5/mo supporter → 1440p + logo removal |
| **Scope.gg Clips** | "Everything is being recorded on SCOPE.GG servers"; first-person **and opponent POV**, downloadable 1080p60 | **In-engine** (POV) | ❌ None found | Free tier (Steam MM unlimited, 1 clip/7 days); paid Lens/Prematch tiers |
| **[CS2.CAM](https://cs2.cam/)** | 2D viewer, heatmaps, grenade mode, "3D Replay Mode", recording queue | ⚠️ 2D + a **reconstruction** for "3D Replay Mode"; recording is **local** | ❌ | Free / Individual $20pm ($12.50 annual) / Team $79.99 / $119.99 |
| **Commercial/Fiverr route** | Human-edited render of your `.dem` | **In-engine** | ❌ manual | See §5.4 |

⚠️ **Demo-Slap is a two-person Swiss side project** — its own About page says "the rendering rigs run at our place". Treat it as an excellent prototype-grade API, not a durable production dependency. ⚠️ **CS2.CAM's marketing contradicts its own docs:** the homepage implies "cloud recording / no PC needed", but its recording docs say it launches *your locally installed* CS2 Workshop Tools plus portable OBS with `-insecure` on your own machine; only **Team Recording** streams cameras to their servers. Do not plan on CS2.CAM cloud rendering.

### 5.2 Services that only analyse (no video you can download)

| Service | Output | Video? | API |
|---|---|---|---|
| **[Leetify](https://leetify.com/)** | Stats, web 2D replay, and auto **highlight clips**; also **hosts POV demos for download** ([Demos are back!](https://leetify.com/blog/demos-are-back/)) | Highlight clips only, **no full-match MP4** | ✅ **Yes** — [api-public-docs.cs-prod.leetify.com](https://api-public-docs.cs-prod.leetify.com/), key at [leetify.com/app/developer](https://leetify.com/app/developer) (pricing behind login) |
| **csstats.gg** (formerly csgostats.gg) | Stats: scoreboard, rounds, duels, heatmaps | ❌ | ❌ (403 on /api) |
| **[Skybox.gg](https://skybox.gg/pricing/)** (EDGE / Broadcast) | Stats + **2D Replayer**; Broadcast does "2D & 3D match visualisations", "Skybox Engine", "Licensed by Valve Corporation" | ❌ self-serve render (bespoke/enterprise) | ❌ (Broadcast: "native use via APIs", partner-only) |
| **[Refrag](https://refrag.gg/)** | 2D demo viewer + in-game restrat playback | ❌ | ❌ | 
| **Noesis** [noesis.gg](https://www.noesis.gg/) | 2D online demo viewer | ❌ | ❌ | 
| **CS2Lens** [cs2lens.com](https://cs2lens.com/) | 2D replay + GIF export | GIF only | ❌ | 
| **SCL** [scl.gg](https://scl.gg/) | Scrim servers + free 2D demo viewer | ❌ | ❌ | 
| **OP.GG CS2** [op.gg/cs2/2d-replay](https://op.gg/cs2/2d-replay) | Browser 2D replay | ❌ | ❌ | 
| **Blitz.gg** [blitz.gg/cs2](https://blitz.gg/cs2) | Stats + in-game overlay | ❌ | ❌ | 
| **[CS Demo Manager](https://github.com/akiver/cs-demo-manager)** | **True in-engine render** — but **local** | ✅ mp4/mkv/avi + TGA/WAV, CLI `csdm video` | Open source (MIT); local Windows/Linux only |

**Corrections to common misconceptions found during this research:**
- **`xplay.gg` is not a demo→video tool** — it is a CS2 practice-server network (retake/DM/duels/surf; StarLadder Budapest Major 2025 partner). Its replay blog posts are console-command guides, not a renderer. There is **no** xplay 3D replay.
- **Fragster is esports news media**, not a replay tool.
- **"Strata"** could not be identified as any CS2 demo→video product. Treat it as non-existent.
- **Skybox's "3D" is a proprietary broadcast engine**, licensed by Valve, sold enterprise-only through partners (e.g. Virtex Stadium) — not a self-serve renderer. Its consumer app [Skybox3D](https://store.steampowered.com/app/1549660/Skybox3D/) is CS:GO-era, interactive-only, with no video-export claim.
- **CS2.CAM "3D Replay Mode"** is a browser feature with no documentation describing Source 2 output — almost certainly a WebGL-style reconstruction, not engine rendering.

### 5.3 Renting a Windows GPU box and running HLAE yourself

Technically viable, but **nobody has publicly documented CS2 + HLAE running unattended on a rented cloud box** — this remains unverified, and it is the main engineering risk in this route.

**Windows GPU pricing (on-demand, us-east-1 / East US, Sept 2026):**

| Provider / instance | Price | Usable for CS2+HLAE? |
|---|---|---|
| AWS `g4dn.xlarge` (T4 16 GB) | **$0.71/hr** (Windows) | ✅ Yes — AWS documents an NVIDIA **gaming/vGPU driver** path for G4dn/G5/G6 on Windows |
| AWS `g4ad.xlarge` (Radeon Pro V520) | $0.5625/hr | Likely (AMD driver path less documented) |
| AWS `g6.xlarge` (L4 22 GB) | **$0.9888/hr**; spot **~$0.27/hr** | ✅ Yes |
| AWS `g5.xlarge` (A10G) | $1.19/hr | ✅ Yes |
| Azure `NV6ads_A10_v5` | **$0.730/hr**; **spot $0.135/hr** (30-second eviction notice) | ✅ Yes |
| Azure `NC4as_T4_v3` | $0.710/hr | ✅ Yes |
| ⚠️ Azure NVv4 series | — | **Retires 30 Sep 2026** |
| **Paperspace / DigitalOcean** | — | ❌ **Windows templates retired for new users after 1 Jul 2024** → Linux only |
| **RunPod** | $0.27–0.99/hr | ❌ "Windows is not supported" |
| **Vast.ai** | cheap | ❌ "We do not support Windows" |
| **Shadow PC** | $29.99/mo Neo Lite (RTX 4060-class, 120 h/mo, **4-hour sessions**), $43.99/mo Power Lite | ⚠️ Real Windows desktop, but ToS is "private and **non-commercial**", forbids using it "for business services" or "as a server", plus auto-shutdown |
| **GeForce NOW** | $99.99–$199.99/yr (100 h/mo cap) | ❌ Locked streaming service: no desktop, no arbitrary software; ToS bars downloading unlisted software; cannot ingest your local `.dem` or run HLAE |
| **Steam Remote Play** | free | ❌ Streams *from* a PC you own — not cloud compute |

**Practical mechanics and gotchas:** install the NVIDIA gaming/vGPU driver (GRID ≥ 17 drops Server 2019 support); capture with `mirv_streams` (`record screen enabled 1`, `record fps`, `afxFfmpeg` preset → direct `.mp4`); snapshot the ~60 GB CS2 install to avoid re-downloading per session. **The real gotcha is the headless display:** on AWS Windows, NICE DCV supports console sessions only, and AWS ships a "NICE DCV Virtual Display Driver for EC2" — budget time for that. Whether CS2 itself tolerates **Windows Server** 2022/2025 is unverified. HLAE is Windows-only (issue #1122) — which is exactly why Reka's Linux pipeline drives the demo through a server plugin instead of HLAE.

### 5.4 Legal and ToS constraints (this matters for a research project)

- **There is no "no VMs" clause in the Steam Subscriber Agreement.** Full-text search of the [SSA](https://store.steampowered.com/subscriber_agreement/) (rev. 10 Sep 2026) finds no VM/hypervisor/cloud rule. What *does* matter:
  - **§2.A** — Steam content is licensed for "personal, **non-commercial** use".
  - **§1.C** — you "must not use your Account to enable a violation of this Agreement by others, such as through their **commercial use** of Steam Content and Services" (i.e. a paid service funnelling many customers' demos through one Steam account).
  - **§4.B** — you "will not tamper with the execution of Steam or Content and Services unless otherwise authorized by Valve" — HLAE is a DLL injector.
  - **§4.C** — automation restrictions.
- **VAC:** Valve's [VAC disconnect FAQ](https://help.steampowered.com/en/faqs/view/22C0-03D0-AE4B-04E8) lists "**DLL Injectors**" and "**Hypervisors**" among software to disable; the stated consequence is *disconnection from VAC-secured servers*, not a ban. HLAE's own wiki still warns in bold: "**Don't join any VAC protected servers, that might get you VAC banned**" — trivially avoided with `-insecure +sv_lan 1` and demo-only use.
- **Valve Video Policy** ([store.steampowered.com/video_policy](https://store.steampowered.com/video_policy)): "**Use of our content in videos must be non-commercial… you can't charge users to view or access your videos. You also can't sell or license your videos to others for a payment of any kind.**" YouTube-partner monetisation is explicitly allowed. **This directly constrains both paid render APIs and paid commissions** — relevant if cs-net-v5 ever monetises generated footage.
- **Freelance market (Fiverr, read Sept 2026):** floor $5 (single-kill compilation); typical **$15–60** basic; **$60–300** multi-minute cinematic; ceiling **$1,595** for 3 minutes (Neekzu, Top Rated). Delivery 1–10 days, ~2 weeks at the top. **About 8 of 12 gigs render the demo themselves** (one lists "CSTV demo file… Programs: Adobe After Effects, **HLAE**") so you often only need to supply the `.dem`. No bulk/team discounts; per-minute pricing is flat. Automated competition structurally undercuts them (Demo-Slap ≈ €0.02–0.06 per 10-second clip, ClutchKings free).
- Also useful and **MIT-licensed**: [deadem / `@deademx/cs2`](https://www.npmjs.com/package/@deademx/cs2) — a Source 2 demo parser **plus a replay playback engine for Node and the browser** ([deadem.com](https://deadem.com)). Directly relevant if cs-net-v5 wants a JS replay path.

*Confidence: the pricing/API details above come from the providers' own pages and API specs, relayed from a dedicated research pass; several of these sites are Cloudflare-protected, so treat exact prices as indicative. AWS/Azure list prices are from the providers' own pricing pages/API.*

---

## 6. macOS / Apple Silicon feasibility — honest assessment

### 6.1 Does CS2 run natively on Apple Silicon?

**No. There is no macOS build of CS2 at all.**

- Steam's own app-details API for app 730 reports `platforms: {windows: true, mac: false, linux: true}` and `OS: Windows 10` / `OS: Ubuntu 20.04` requirements. <https://store.steampowered.com/api/appdetails?appids=730&l=english>
  - ⚠️ The same API still returns a `mac_requirements` block ("MacOS X 10.11… 15 GB available space") — **this is stale CS:GO-era data**, not a supported platform. The `mac: false` flag and Valve's public statements are authoritative.
- CS:GO had a macOS client for years; it disappeared when CS2 replaced it in September 2023. Valve's stated reason was the tiny Mac player share. (<https://www.videogameschronicle.com/news/valve-has-dropped-counter-strike-2-support-for-mac-and-older-pcs/>, <https://www.gamereactor.eu/valve-drops-mac-support-for-counter-strike-2-1314203/>, <https://www.ilounge.com/articles/can-the-new-macs-run-counter-strike-2-the-honest-answer>, <https://macpaw.com/how-to/play-cs2-on-mac>)
- The depot manifests confirm it: macOS depot **2347772** exists but is a ~9 KB stub, and the macOS depot ID 733 contains 8-byte placeholder manifests for every version. There is no Mac payload.
- **An independent corroboration:** CS Demo Manager's own [video-generation compatibility matrix](https://cs-demo-manager.com/docs/guides/video) marks **macOS: CS:GO ✅ / CS2 ❌ / HLAE ❌**, and **Linux: CS:GO ✅ / CS2 ✅ (via `startmovie` + FFmpeg) / HLAE ❌**.
- **Non-native routes on Apple Silicon:**
  - **CrossOver / Whisky / PortingKit (Wine + DXVK/D3DMetal)**: works. AppleGamingWiki rates CrossOver **"Perfect"** with DXVK (reported ~50 FPS at 1080p on an M1 MacBook Air 8 GB, CrossOver 23.6); with D3DMetal performance degrades after a few minutes. Audio needs a sample-rate workaround (Audio-MIDI Setup ≥ 88 kHz). Wine/PortingKit + GPTK also rated "Perfect" with `-nojoy`. <https://www.applegamingwiki.com/wiki/Counter-Strike_2>
  - **Parallels (Windows 11 ARM + x86 emulation)**: rated **"Playable" at ~23 FPS** on the same wiki — two translation layers (x86→ARM inside the VM, plus Windows-on-ARM).
  - **Faceit/ESEA-style kernel anti-cheat**: unavailable on macOS entirely.
- Caveat: the AppleGamingWiki entries date from late 2023; CS2 has changed substantially since (and grown ~40 % in install size). Expect "it runs, imperfectly, unsupported".

### 6.2 Would HLAE work on macOS?

**Not natively, and only speculatively under Wine.** HLAE is a Windows `.exe` + installer that injects `AfxHookSource2.dll` into `cs2.exe` under `game\bin\win64`. The HLAE FAQ says only that some *games* may work through **Wine or Proton on Linux**, and *"we cannot guarantee that every game will work"* — it never claims macOS support. Wine/CrossOver *can* in principle host DLL injection, so HLAE-under-CrossOver is not physically impossible, but it is **unverified, undocumented, and unsupported** by the HLAE project. The HLAE FAQ's only portability guidance is Linux/Wine with no guarantee; the practical, honest answer for Apple Silicon is: **assume it does not work** unless you test it.

The realistic "real footage on this Mac" stack would be:
`Apple Silicon → CrossOver or a Windows 11 ARM VM → Windows CS2 (x86 translation) → HLAE DLL injection → FFmpeg → frames`
— five layers of unsupported translation, for a pipeline whose upstream project explicitly supports only Windows.

### 6.3 Disk space

**CS2 install footprint (app 730), computed from live SteamCMD depot manifests** (`https://api.steamcmd.net/v1/info/730`, `public` branch, read 2026-09-15):

| Depot | Platform | Uncompressed (on disk) | Download |
|---|---|---|---|
| 2347770 | shared content | **62.74 GB** | 53.90 GB |
| 2347771 | windows | **7.71 GB** | 5.00 GB |
| 2347774 | shared | **1.14 GB** | 0.95 GB |
| 2347777 | shared | 0.09 GB | 0.03 GB |
| **Windows client total (base)** | | **≈ 71.6 GB** | **≈ 59.9 GB** |
| 2347779 | optional DLC 2279721 (high-res textures) | +2.17 GB | +0.58 GB |
| 2347773 | linux | 7.22 GB | 4.60 GB |
| 2347772 | macos | ~9 KB (placeholder stub only — no Mac build) | — |

Cross-checks:
- **Valve's official requirement: "Storage: 85 GB available space"** for both Windows and Linux (<https://store.steampowered.com/api/appdetails?appids=730>). The 85 GB figure is the install + unpack/update headroom.
- Third-party 2026 guides disagree: one claims ~85 GB installed with up to 120 GB needed during install (<https://wiki.csgoluck.com/cs2-file-size/>), another claims 35–50 GB installed / 30–45 GB download and notes Steam's own 85 GB requirement (<https://pulse.tradeon.space/ru/blog/skol-ko-vesit-cs2-razmer-igry-i-skol-ko-mesta-zanimaet-na-komp-yutere>). The depot-manifest math (~72 GB installed, ~60 GB download) sits between them and is the most verifiable.
- The install grows with Workshop content (`steamapps\workshop\content\730`, 2–3 GB per detailed map), shader caches and logs.
- **This machine has 15 GiB free** (460 GiB volume, 97 % used). CS2 alone does not fit, before counting a Windows VM (~30+ GB), OBS, or capture output. **The disk constraint alone rules out every local route.**
- Capture output is also large: `startmovie` writes uncompressed TGA + WAV into the game folder, and CS Demo Manager explicitly warns you need tens of GB free (reportedly >40 GB) for that path; an uncompressed 1920×1080 TGA frame is ~6 MB, so a 60-second clip at 300 fps ≈ 18,000 frames ≈ 100+ GB if you keep the sequence. This is precisely why HLAE's `mirv_streams` (frames piped straight to FFmpeg, `afxSampler30`/`afxFfmpegYuv420p` presets) exists.

### 6.4 Is the automated-capture route realistic for a hobby/research project?

**On this Mac: no.**
- 15 GiB free vs ~72 GB (85 GB stated) for CS2 — blocked before anything else.
- No macOS build; the only host would be Wine or a Windows VM, both unsupported for CS2 *and* for HLAE.
- The pipeline's fragile parts (HLAE ↔ CS2 update coupling, version-locked demos, VAC/`-insecure` handling) are worse, not better, under translation layers.
- Even in a perfect Windows environment, capture is **real-time**: a 40-minute match is ≥40 minutes of GPU time per pass, usually more with multiple streams or high-fps sampling, plus transcoding. It does not scale to "many `.dem` files" cheaply.

**What is realistic (in the order worth trying):**
1. **Hosted in-engine cloud render, via API** — [DEMO-SLAP](https://demo-slap.net/) (`/analyze/url` → `/render-custom` with tick range + `povSteamId` → signed `.mp4`, ~€0.02–0.06 per 10-second clip, 17 free credits/day) or free [CLUTCHKINGS.gg](https://www.clutchkings.gg/) (no API). Zero local install, genuine engine output, but **POV clips rather than free-camera cinematics**, and Demo-Slap is a two-person side project — prototype-grade, not infrastructure.
2. **If the demos are CS:GO-era:** install the free CS:GO legacy app (ID 4465480) and use its normal `startmovie` — ~34 GB, no HLAE, scriptable, real frames (§6.5). Cheapest *self-hosted* route by a wide margin.
3. **Cloud Windows GPU instance + HLAE** — the only route to **custom-camera** Source 2 cinematics (`mirv_campath`/`mirv_streams`). ~$0.71/hr on AWS `g4dn.xlarge` (spot ~$0.27–0.37/hr); snapshot the ~60 GB install to avoid re-downloading. Caveats: **nobody has publicly documented this running unattended**, headless display needs the NICE DCV virtual display driver, and Windows Server compatibility with CS2 is unverified.
4. **A borrowed/owned Windows PC with ~150 GB free** — same pipeline, far less pain; still real-time capture.
5. **Stay 2D/open-source**: a demo parser (`demoparser`/`awpy`/`demoinfocs-golang`) plus a radar renderer gives automated, scalable, cross-platform, CPU-only, free video. Loses photoreal 3D — but note there is **no open-source alternative to duplicate**, so this is genuinely additive to the web reconstruction rather than a step down from something that exists.
6. **Commission a movie maker** ($15–60 typical, ~8/12 gigs render the demo themselves) for one-off hero footage.
7. **Hybrid**: 2D radar for bulk/automated coverage + a few API-rendered or commissioned 3D clips for hero moments, composited over the existing web 3D reconstruction.

### 6.5 Important branch: if your `.dem` files are **CS:GO-era**

This branch matters because **CS2 cannot play CS:GO demos and CS:GO cannot play CS2 demos** — the provenance of your `.dem` files decides everything. It is also the one place where the story is materially better than "you need a Windows box with 150 GB".

**What happened**
- Valve quietly relisted CS:GO as its own, *unlisted* Steam app — **new App ID 4465480** (the old 730 now belongs to CS2). It is hidden from store search and reachable only by direct link; the page displays *"at the publisher's request, this product is not displayed in the Steam store or search results"*. It is free to install, and its public branch was rebuilt **2026-02-26**.
- It is **the same `csgo_legacy` build** that was previously only reachable via the hidden beta branch inside CS2 — extracted into its own app, with exactly one fix: the launch problem caused by Steam's move to 64-bit-only clients. The `csgo_legacy` branch also still exists on app 730 (buildid 12426195, 2023-10-13).
- **It is offline-only.** Valve did not restore official matchmaking: bot practice, training courses and Workshop maps only. Playtime and achievements do **not** carry over.
- Official servers being down does **not** matter for demo playback — `playdemo` is local, and "older demo and tooling compatibility" is precisely what the legacy build is for.
- **Consequence: CS:GO-era rendering tools are not orphaned.** CS:GO is Source 1, frozen at the Oct-2023 build, so HLAE-for-CS:GO and CS Demo Manager's CS:GO mode remain viable offline.

**Platform reality — corrected (my first pass over-claimed here):**
On app 4465480 Valve's store metadata reports **Windows and Linux**, with `mac: false` and no `mac_requirements` block — and an independent research pass reached the same conclusion. **Plan for Windows or Linux, not macOS.** The app's *runtime* config does contain macOS artefacts, which is what initially looked promising:

| Signal for app 4465480 | Value |
|---|---|
| `config.launch` entries | `csgo.exe`/windows, `csgo.sh`/**macos**, `csgo.sh`/linux |
| `extended.validoslist` | `windows,macos` |
| `extended.primarycache_macos` | `733` (a real macOS content depot) |
| macOS depot 733 | 0.277 GB (versus CS2's ~9 KB stub) |
| shared depots 731 + 737 + 738 | 32.72 + 0.42 + 0.92 GB |
| **Windows/Linux installed total** | **~34.3 GB** (~14.6 GB download) |

The likeliest explanation is **stale metadata retained from CS:GO's Mac era while the platform flag stays off**. So treat a macOS install as a long shot — possibly forceable with Steam's `download_depot 4465480 733` console command, entirely unsupported and untested here. Worth a two-minute look since Steam is already installed, but **do not design around it.** CS:GO is also a 64-bit Intel binary with no Apple Silicon port, so even if it installed it would run under Rosetta 2 ([Does It Mac](https://doesitmac.com/app/counter-strike-global-offensive): *"Runs via Rosetta 2 translation"*).

**The render path (no HLAE required)**
1. Install CS:GO legacy (app 4465480 direct link, or the `csgo_legacy` beta branch under CS2) on **Windows or Linux**: ~34 GB on disk, ~14.6 GB download.
2. Put the `.dem` in the CS:GO `csgo/` or `csgo/replays/` folder and `playdemo replays/<name>`.
3. Script camera/tick control with normal CS:GO commands: `demo_gototick`, `demo_timescale`, `demo_pause`/`demo_resume`, `spec_player`, `spec_mode`, `spec_goto`, `host_framerate`.
4. Capture with **CS:GO's own `startmovie <name>`** — in CS:GO this is a plain, user-accessible command (totalcsgo's CS:GO entry: *"sv_cheats Required: No"*) writing a TGA (or AVI/raw/JPG) frame sequence plus `.wav` into the game folder. Mux with FFmpeg.
5. **[CS Demo Manager](https://github.com/akiver/cs-demo-manager)** supports exactly this — its compatibility matrix is **Windows CS:GO ✅ / Linux CS:GO ✅ / macOS CS:GO ✅ but CS2 ❌** — and it generates the tick-range sequences and CFG blocks for you. HLAE is unnecessary on this path; on Windows it would only add smooth camera paths and multi-stream/matte capture.

**Known caveats on the `startmovie` path** (from Valve's own OS X/Linux tracker):
- [issue #1397 — "startmovie command no longer records sound"](https://github.com/ValveSoftware/csgo-osx-linux/issues/1397)
- [issue #1534 — "Startmovie audio ignores host_framerate setting"](https://github.com/ValveSoftware/csgo-osx-linux/issues/1534)

Expect to handle audio separately and to find `host_framerate` slow-motion capture imperfect. CS:GO demos are also version-locked to the recording build, and only the `csgo_legacy` build is available — very old demos may still fail.

**Action item:** identify the provenance of your `.dem` files (`demo_info`/`listdemo` in-game, or a parser such as `demoparser2`/`demoinfocs-golang`, which report the demo's game/build). That single fact decides whether this section is your cheapest route or irrelevant.

### 6.6 Disk-space comparison of the candidate paths

| Path | Download | Installed | Fits in 15 GiB free? |
|---|---|---|---|
| CS2 (app 730) + HLAE | ~60 GB | ~72 GB (+2.2 GB optional HD textures) | ❌ No — needs ~40–60 GB freed |
| CS:GO legacy (app 4465480), Windows/Linux | ~14.6 GB | ~34.3 GB | ❌ Not quite; ~20–25 GB of freeing would make it work |
| 2D radar pipeline (parsers only) | < 1 GB | < 1 GB | ✅ Yes, today |

---

## 7. Method, confidence, and what would change the answer

**How this was researched (2026-09-15):** HLAE's own wiki/FAQ and the complete GitHub Releases API history (414 releases enumerated, to date the first CS2 hook); a real CS2 client `cvarlist` dump with per-command flags (<https://github.com/SuGolYolLom/CS2-Cvars-Cmds>, PatchVersion 1.40.5.1, Nov 2024) cross-checked against live command references (CSbePRO, totalcsgo) and the cvar-unhide-s2 project; **Steam's own APIs** for platform support and store requirements (`store.steampowered.com/api/appdetails?appids=730`) and **SteamCMD depot manifests** for install size (`api.steamcmd.net/v1/info/730`); the official Steam news feed for CS2 patch notes; plus a broad GitHub/web survey of tools and services.

**Confidence levels:**
- *High*: no macOS build of CS2; HLAE is Windows-only and forces `-insecure`; HLAE added CS2 support in 2.146.1; `startmovie`/`endmovie` exist in CS2 but are `FCVAR_DEVELOPMENTONLY`; `playdemo`/`demo_gototick`/`demo_timescale`/`spec_player`/`host_framerate` are normal CS2 commands; CS2 install ≈72 GB (85 GB stated); no open-source out-of-engine 3D CS2 demo renderer exists; CS:GO returned as a standalone app (ID 4465480) on 2026-03-04 and **cannot play CS2 demos**.
- *Medium*: whether Steam offers a macOS install of CS:GO legacy — the app's *runtime* metadata contains macOS artefacts (`csgo.sh` macOS launch entry, `validoslist: windows,macos`, `primarycache_macos: 733`, a 277 MB macOS depot) but the *store* metadata reports Windows+Linux with `mac: false`, confirmed by two independent passes, so macOS is most likely unsupported (stale metadata); the exact runtime behaviour of `startmovie` once unlocked in a current CS2 build (the code path and `movie_fixwave` strongly imply TGA+WAV, but this was not executed on a live install); whether HLAE survives under CrossOver/Wine on Apple Silicon (untested, undocumented) or unattended on a rented cloud Windows box (no public write-up found); exact service pricing (several pages are login- or Cloudflare-gated).
- *Lower*: third-party install-size guides (35–85 GB range) disagree with each other; I report the depot-manifest math as the most verifiable figure and Valve's 85 GB as the official requirement. Service-by-service API/pricing claims in §5.

**What would change the recommendation:**
- **Your `.dem` files turning out to be CS:GO-era** → §6.5 becomes the cheapest *self-hosted* real-footage route (~34 GB, no HLAE, no CS2). This and the demo provenance question are the highest-value things to check first.
- **The cloud render APIs working for your demos** → §5.1 collapses the whole local-hardware problem; test Demo-Slap's free daily credits and ClutchKings before buying or renting anything.
- Confirming that the CS:GO legacy macOS install is actually offered by Steam on this machine → would make §6.5 work locally, but treat this as unlikely given two independent `mac: false` reads.
- A native CS2 macOS build (no indication Valve is planning one) → the whole CS2 calculus changes.
- Confirmed HLAE-under-CrossOver operation → CS2 capture becomes *possible* on this Mac, but the ~72 GB disk requirement still blocks it here.
- A published open-source out-of-engine 3D CS2 demo renderer (does not exist today; your web reconstruction appears to be a novel implementation of exactly this idea) → would make the "real footage" question moot.
- Enough free disk (~150 GB+ for CS2, ~50 GB for CS:GO legacy) plus a Windows machine → the HLAE + CS Demo Manager stack becomes straightforward, just slow.

---

## 8. Quick reference — key sources

- HLAE AfxHookSource2 (features, launching, command line): <https://github.com/advancedfx/advancedfx/wiki/AfxHookSource2>
- HLAE CS2 command list: <https://github.com/advancedfx/advancedfx/wiki/Source2:Commands>
- HLAE FAQ (Linux/Wine, VAC, `-insecure`): <https://github.com/advancedfx/advancedfx/wiki/FAQ>
- HLAE `mirv_streams` (CS2 recording/FFmpeg/streams): <https://github.com/advancedfx/advancedfx/wiki/Source2:mirv_streams>
- HLAE releases (history, first CS2 hook = 2.146.1): <https://github.com/advancedfx/advancedfx/releases>
- CS2 cvarlist dump (flags + descriptions): <https://github.com/SuGolYolLom/CS2-Cvars-Cmds>
- cvar-unhide-s2 (hidden/dev-only unlocking, needs `-insecure`): <https://github.com/saul/cvar-unhide-s2>
- CS2 demo/GOTV command reference: <https://csbepro.com/cs2-console-commands/demo/>
- totalcsgo command pages: <https://totalcsgo.com/commands/playdemo>, <https://totalcsgo.com/commands/startmovie>
- Steam app-details API (platforms, 85 GB): <https://store.steampowered.com/api/appdetails?appids=730>
- SteamCMD depot manifests (install size math): <https://api.steamcmd.net/v1/info/730>
- Official CS2 patch notes feed: <https://api.steampowered.com/ISteamNews/GetNewsForApp/v2/?appid=730>
- Official CS2 update notes (Sept 2026, custom cameras + demo chat): <https://store.steampowered.com/news/app/730/view/692020124159312044>
- TrueView demo playback: <https://consolepcgaming.com/counter-strike-2-adds-trueview-demo-playback-and-fixes-surfing-and-demo-bugs/>
- Sept 2026 update coverage: <https://consolepcgaming.com/counter-strike-2-update-reworks-custom-cameras-and-encrypts-demo-chat/>
- Steam Game Recording: <https://help.steampowered.com/en/faqs/view/23B7-49AD-4A28-9590>, <https://store.steampowered.com/gamerecording>
- CS2 on Apple Silicon: <https://www.applegamingwiki.com/wiki/Counter-Strike_2>, <https://www.ilounge.com/articles/can-the-new-macs-run-counter-strike-2-the-honest-answer>, Valve dropped Mac support: <https://www.videogameschronicle.com/news/valve-has-dropped-counter-strike-2-support-for-mac-and-older-pcs/>, <https://www.gamereactor.eu/valve-drops-mac-support-for-counter-strike-2-1314203/>
- **The practical open-source in-engine stack:** CS Demo Manager (<https://github.com/akiver/cs-demo-manager>, <https://cs-demo-manager.com/docs/guides/video>), CS DemoDesk (<https://github.com/noih/cs-demodesk>), reka-ai dataset renderer (<https://github.com/reka-ai/cs2-dem-renderer>), HLAE CFG generators (<https://github.com/eSheikh/cs2-demo-highlighter>, <https://github.com/Rovniy/cs2-highlights-maker>), OBS drivers (<https://github.com/nickdevroye/cs2pov-player>, <https://github.com/DrEAmSs59/CS2-insight-agent>)
- **2D radar stack (works on this Mac):** <https://github.com/pnxenopoulos/awpy>, <https://github.com/LaihoE/demoparser>, <https://github.com/markus-wa/demoinfocs-golang>, <https://github.com/akiver/cs-demo-analyzer>, <https://github.com/MurkyYT/cs2-map-icons>, <https://github.com/ValveResourceFormat/ValveResourceFormat>
- **Cloud render APIs (in-engine, hosted):** Demo-Slap <https://demo-slap.net/> + OpenAPI spec <https://api-doc.demo-slap.net/api-documentation.yaml>, ClutchKings <https://www.clutchkings.gg/faq>, Scope.gg Clips <https://scope.gg/guides/scope_clips_en/>, Leetify public API <https://api-public-docs.cs-prod.leetify.com/>
- **Steam/Valve legal:** Steam Subscriber Agreement <https://store.steampowered.com/subscriber_agreement/>, Valve Video Policy <https://store.steampowered.com/video_policy>, VAC FAQ <https://help.steampowered.com/en/faqs/view/22C0-03D0-AE4B-04E8>
- **Cloud GPU pricing:** AWS on-demand <https://aws.amazon.com/ec2/pricing/on-demand/>, Azure Retail Prices API <https://prices.azure.com/api/retail/prices>, Shadow PC ToS <https://statics.shadow.tech/terms-of-use/tc-US-b2c.pdf>, GeForce NOW ToS <https://www.nvidia.com/en-us/geforce-now/terms-of-use/>
- **JS/browser replay engine:** deadem / `@deademx/cs2` (MIT) <https://www.npmjs.com/package/@deademx/cs2>, <https://deadem.com>
- **Closed 3D / commercial:** Skybox3D <https://store.steampowered.com/app/1549660/Skybox3D/>, Demo-Slap <https://api-doc.demo-slap.net/>, cs2.cam <https://cs2.cam/en/wiki/introduction>, xplay.gg <https://xplay.gg/blog/how-to-control-replay-in-cs2-replay-commands/>, Leetify POV demos <https://leetify.com/blog/cs2-pov-demos/>
- **CS:GO legacy returns (macOS branch):** app 4465480 metadata via `https://api.steamcmd.net/v1/info/4465480`; relist coverage <https://vgtimes.com/gaming-news/149594-old-school-roll-call-csgo-returns-to-steam-now-installable-separately-from-cs2.html>, <https://www.fragster.com/csgo-on-steam-in-2026-how-to-install-the-legacy-build-and-why-comeback-talk-is-trending-again/>, <https://m.ithome.com/html/925792.htm>, <https://csgo.5eplay.com/article/260304uzo7yq>, <https://zap-hosting.com/en/blog/2026/03/cs-go-is-back-in-the-steam-store-servers-available-again/>; CS:GO on Apple Silicon via Rosetta 2: <https://doesitmac.com/app/counter-strike-global-offensive>; `startmovie` on macOS/Linux quirks: <https://github.com/ValveSoftware/csgo-osx-linux/issues/1397>, <https://github.com/ValveSoftware/csgo-osx-linux/issues/1534>
