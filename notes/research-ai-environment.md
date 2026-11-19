# Counter-Strike 2 as an AI Training Environment

**Research date:** 2026-09-15
**Scope:** Can CS2 serve as a simulator/environment for RL, imitation learning, or agent research? What is actually feasible in 2026?
**Target machine (unless stated otherwise):** Apple M1 Pro, 16 GB RAM, macOS 26.6, ~15 GB free disk.

> **Citation policy.** Everything below is sourced from public material retrieved on 2026-09-15. Where I could not verify a claim, it is explicitly marked **[UNVERIFIED]** or **[INFERRED]**. Valve's official wiki (`developer.valvesoftware.com`) is behind an Anubis anti-scraping proof-of-work wall and could not be fetched directly; Valve-derived facts below come from mirrors, community docs, and third-party sources, which is noted where it matters.

---

## 0. TL;DR — the honest one-paragraph answer

CS2 is a **good source of human trajectory data** and a **poor real-time RL environment**.

- **Headless server-side state control: yes, genuinely works.** A CS2 dedicated server runs fine on Linux in Docker with Metamod:Source + CounterStrikeSharp, and a C# plugin can read full per-tick game state, spawn bots, force buys, teleport entities, and POST state to your Python process. This is the strongest, most real part of the story.
- **Pixel observations from a live, agent-controlled game: effectively closed.** The dedicated server has no renderer at all; the CS2 *client* requires a GPU, a real Steam login, a desktop session, and is **Windows/Linux only — there is no macOS build of CS2 in any form**. And there is no supported way to feed actions into the client from a plugin while capturing frames, because the client isn't plugin-extensible the way the server is.
- **The part that actually scales is offline/demo-based.** CS2 `.dem` files are fully parseable, deterministic, and available in enormous public corpora (OpenCS2: ~11 TB / ~5,000 h; Reka CS2-10k: ~10,000 h; EgoCS-400K: ~10,000 h). You can build imitation-learning and world-model pipelines on a Mac **without ever running the game**.
- **On this specific machine** (M1 Pro, 16 GB, 15 GB free): the viable path is **demo parsing + small-scale BC/world-model experiments on cached 128×128 frames**. Everything involving running CS2 itself, or training anything at useful scale, needs Linux + NVIDIA GPU + **hundreds of GB to multiple TB** of disk.

---

## 1. Headless / server-side environment

### 1.1 Can a CS2 dedicated server run headless on Linux?

**Yes, routinely, and this is well-trodden ground.** A CS2 dedicated server (`srcds`/`cs2`) has no renderer, no window, no GPU requirement. It is a pure simulation + network process. This is the single most important architectural fact in this whole report:

> **The dedicated server simulates the game; the client renders it. They are different binaries with different capabilities. Almost every frustration in this space comes from people assuming the server can produce pixels or the client can be automated like a server.**

There appears to **no longer be a separate CS2 dedicated-server AppID** the way CS:GO had `740`. CS2 servers are installed from the CS2 app (`730`) content depots via SteamCMD. Server registration requires a Game Server Login Token (GSLT) generated at `steamcommunity.com/dev/managegameservers` for AppID `730`. ([OneUptime Portainer guide, 2026-03](https://raw.githubusercontent.com/OneUptime/blog/refs/heads/master/posts/2026-03-20-deploy-counter-strike-2-server-portainer/README.md), [Sparked Host KB, updated 2026-02](https://help.sparkedhost.com/en/article/how-to-disable-valve-anti-cheat-on-a-counter-strike-2-server-lcsej1/))

#### Docker images: the concrete options

| Image | Maintainer | Notes | Typical use |
|---|---|---|---|
| [`joedwards32/cs2`](https://github.com/joedwards32/CS2) | joedwards32 | **The de-facto standard.** Actively maintained; the CM2 image is now a mirror of this one. Environment-variable driven (`CS2_SERVERNAME`, `CS2_PORT`, `CS2_RCONPW`, `CS2_MAXPLAYERS`, `CS2_GAMETYPE`, `CS2_GAMEMODE`, `CS2_STARTMAP`, `CS2_BOT_QUOTA`, `CS2_BOT_DIFFICULTY`, `CS2_HOST_WORKSHOP_COLLECTION`, `SRCDS_TOKEN`). | Recommended default |
| [`cm2network/cs2`](https://hub.docker.com/r/cm2network/cs2) | CM2Walki | Historically the most widely referenced; [the GitHub repo now states it is a mirror of joedwards32/CS2](https://github.com/CM2Walki/CS2). Docs on Docker Hub are thin. | Legacy name recognition; functionally same as above |
| [`opencs2/cs2-retakes-server`](https://github.com/opencs2/cs2-retakes-server) | opencs2 | Opinionated retakes-mode server image | Specific gamemode |
| [`ccns/cs2-server`](https://hub.docker.com/r/ccns/cs2-server), [`teriyakigod/cs2-server`](https://hub.docker.com/r/teriyakigod/cs2-server) | community | Smaller/less documented | Avoid unless you need something specific |
| [`joedwards32/silica`](https://hub.docker.com/r/joedwards32/silica) | joedwards32 | Related image in the same family | — |

The canonical example stack (verbatim from the [OneUptime guide](https://raw.githubusercontent.com/OneUptime/blog/refs/heads/master/posts/2026-03-20-deploy-counter-strike-2-server-portainer/README.md)):

```yaml
services:
  cs2:
    image: joedwards32/cs2:latest
    environment:
      - SRCDS_TOKEN=your_game_server_login_token
      - CS2_SERVERNAME=My CS2 Server
      - CS2_RCONPW=rcon_password_here
      - CS2_PORT=27015
      - CS2_MAXPLAYERS=10
      - CS2_GAMETYPE=0            # 0=Classic, 1=Deathmatch, 4=Custom
      - CS2_GAMEMODE=1            # 0=Casual, 1=Competitive, 2=Wingman
      - CS2_STARTMAP=de_dust2
      - CS2_BOT_DIFFICULTY=0
      - CS2_BOT_QUOTA=0
    volumes: [cs2-data:/home/steam/cs2-dedicated]
    ports: ["27015:27015/tcp", "27015:27015/udp", "27020:27020/udp"]
volumes: { cs2-data: }
```

#### Sizes and requirements

| Item | Figure | Source |
|---|---|---|
| CS2 **client** free disk requirement | **85 GB** | [VGTimes CS2 system requirements, updated 2026-09-07](https://vgtimes.com/games/counter-strike-2/system-requirements/) |
| CS2 **client** min RAM / GPU | 8 GB RAM, GTX 950 / R7 260 (1 GB), DX11, **Windows 10** | same |
| CS2 **dedicated server** disk | ~30–40 GB typical installed content **[INFERRED from depot content; not a single authoritative figure I could verify]** | — |
| CS2 **dedicated server** RAM | 2–4 GB baseline; ~1 GB per active bot/player under load **[INFERRED — community rule of thumb, no authoritative source found]** | — |
| **OpenCS2 rendered dataset** | 11 TB (10.6 TB bucket); ~5,000 h; 1280×720 @ 32 fps; keyboard+mouse+world state; stereo audio; all 10 POVs | [blanchon OpenCS2 collection](https://huggingface.co/collections/blanchon/opencs2), [HF post 2026-05-13](https://huggingface.co/posts/blanchon/841713477830104) |
| **Reka CS2-10k** | 10,000+ h, 600k+ player-round videos, 720p @ 48 fps; HF sample subset = 748 rows / **25.9 GB**; CC BY-NC 4.0 | [Reka blog](https://reka.ai/news/cs2-10k-a-large-scale-egocentric-counter-strike-2-dataset), [runtimewire analysis 2026-06-25](https://runtimewire.com/article/reka-cs2-10k-counter-strike-world-models) |
| **EgoCS-400K** | 400k+ videos, 10,000+ h, 40,000+ rounds, 1,000+ matches, 13 maps, ~90 s avg; CC BY 4.0 | [arXiv:2606.18180](https://arxiv.org/html/2606.18180v1) |

> **Disk reality check for this machine:** even the *client* alone (85 GB) is **5.7× the 15 GB free** on this volume. A dedicated server plus CS2 content is roughly 2× the free space. Neither the game nor a full dataset fits here without external storage.

#### Apple Silicon / macOS reality

This is a hard blocker and should be stated plainly:

1. **There is no macOS build of CS2.** Valve shipped CS:GO for macOS but explicitly did not ship CS2 for Mac, stating Mac owners were "less than 1 percent of active CS:GO players" ([Ars Technica, 2023-10-10](https://arstechnica.com/gaming/2023/10/theres-no-mac-version-of-counter-strike-2-because-there-are-no-mac-players/)). CS:GO's legacy branch (`csgo_legacy`) remains installable under CS2 on Windows/Linux, but **also not on macOS**.
2. **The dedicated server is x86-64 and ships for Linux/Windows.** There is no native ARM64 or macOS depot.
3. **An ARM64 CS2 server does exist as a community hack**, using **FEX-EMU** to run the x86-64 Linux server on ARM ([Kacper Ochnik, 2023-10-19](https://blog.kacperochnik.eu/2023/10/19/cs2-server-docker-arm64.html)). This is a 2023 experiment, x86-to-ARM binary translation, and **[UNVERIFIED for 2026 CS2 builds]**. In principle modern Apple Silicon Docker (Docker Desktop / OrbStack) with **Rosetta 2 x86-64 emulation** could run the Linux server container, but I found **no confirmed 2026 report** of a working, performant CS2 dedicated server under Rosetta on Apple Silicon. Treat as *worth a 30-minute experiment*, not as a plan.

### 1.2 Can bots be controlled programmatically?

**Partly — and the boundary matters a lot.**

#### The stack

1. **[Metamod:Source](https://www.sourcemm.net/)** — the Source 2 server plugin loader. Needed to load anything custom.
2. **[CounterStrikeSharp](https://github.com/roflmuffin/CounterStrikeSharp)** — a **.NET 8** scripting layer on top of a Metamod plugin. This is the mainstream, actively maintained way to write CS2 server plugins. C#, not Python. ([README](https://raw.githubusercontent.com/roflmuffin/CounterStrikeSharp/main/README.md))
3. **[CS2Fixes](https://github.com/Source2ZE/CS2Fixes)**, **swiftlys2**, **modsharp**, **plugify** — alternative/additional server-mod frameworks; CounterStrikeSharp is the one with the largest ecosystem.
4. **No official Python binding exists.** There is no Valve-sanctioned Python API for CS2. Python has to talk to the server over a socket/HTTP bridge that *you* write in a plugin.
5. **[cs2_vibesignatures](https://github.com/hzqst/cs2_vibesignatures)** — an LLM/IDA-assisted tool that regenerates CS2 signatures and offsets across game updates, feeding CounterStrikeSharp, CS2Fixes, CS2FOW, plugify, modsharp etc. **This is the maintenance lifeboat**: CS2 updates break binary signatures, and this project automates re-deriving them per game build.

#### What a plugin can actually do

CounterStrikeSharp's own "what works" list is the authoritative capability statement ([README](https://raw.githubusercontent.com/roflmuffin/CounterStrikeSharp/main/README.md)):

- Console/server/chat commands (`css_mycommand`, `!mycommand`)
- **Game event handlers and firing** (`player_death`, `weapon_fire`, …) with get/set of string/bool/int/float and complex values (ehandles, pawns, player controllers)
- **OnTick** and tick-based repeating timers
- Client listeners (connect/disconnect/put-in-server), `OnMapStart`
- Server info (map, game time)
- **Schema System access** — read/write networked entity fields: "current weapon, money, location etc."

Concretely, this means a plugin **can**:
- **Enumerate all players and all entities**, read position/velocity/view angles/health/armor/money/active weapon/ammo and hundreds more `m_*` fields each tick.
- **Teleport / set velocity / set view angles** — the documented community recipe is reading and writing pawn position and calling the teleport path ([CounterStrikeSharp Discussion #481](https://github.com/roflmuffin/CounterStrikeSharp/discussions/481), [Player Management docs](https://deepwiki.com/roflmuffin/CounterStrikeSharp/5.4-player-management)).
- **Spawn and manage bots** via `bot_add`, `bot_kick`, `bot_quota`, `bot_difficulty`, `bot_stop`, `bot_mimic` console commands ([Cybrancee KB: freeze/mimic bots](https://cybrancee.com/learn/knowledge-base/how-to-freeze-bots-or-make-them-mimic-your-movement-on-your-cs2-server/); bot quota/difficulty plugins like [asapverneri/CS2-BotQuotaManager](https://github.com/asapverneri/CS2-BotQuotaManager)).
- **Force buys, give weapons, set money, restart rounds, control freeze time** via convars and entity writes.
- **Hook and rewrite events** — e.g. suppress damage, change kill attribution.
- **Stream state out** over HTTP/WebSocket/UDP/shared memory to an external trainer. (You write this yourself; see §1.3.)

A plugin **cannot** (the important negatives):
- **Replace the bot brain.** CS2's bot AI is the legacy closed-source `CCSBot`/navmesh system. There is no exposed API to hand a bot a policy. `bot_difficulty 0–3` and `bot_mimic` are the only supported levers. Even *spawning* bots on a map with no navmesh fails on a null `g_pNavMesh` check that has to be binary-patched ([hzqst CS2 VibeSignatures skill catalogue](https://skills.rest/skill/find-ccsbotmanager-addbot-botnavignore)) — clear evidence the bot system is closed and only reachable via reverse engineering.
- **Step the simulation deterministically.** See §1.5.
- **Render anything.** No renderer in the server binary.

**Realistic interpretation:** you can build a *state-space* RL environment where **you** control a player pawn via entity writes, and the opponents are stock bots or a scripted mix. You cannot make the stock bots "learn" or expose them as learning agents. An open-source plugin ecosystem exists around *improving* bots ([CS2-Bot-Improver](https://forum.cshub.ro/index.php?/topic/133-plugin-cs2-bot-improver-counterstrikesharp/)) but these are heuristics, not learned policies.

### 1.3 Getting per-tick state out of a live server

Three options, and they are **not** equivalent:

#### Option A — RCON (built in, weakest)
The server has an RCON port. You can send `status`, `mp_restartgame 1`, etc. ([OneUptime guide](https://raw.githubusercontent.com/OneUptime/blog/refs/heads/master/posts/2026-03-20-deploy-counter-strike-2-server-portainer/README.md)). RCON gives you **text output, request/response, no per-tick stream, no entity coordinates**. Fine for orchestration ("restart the match"), useless as an RL observation channel.

#### Option B — Game State Integration (GSI) — **client-side only**
GSI makes the game POST JSON snapshots to an HTTP endpoint you configure via a `gamestate_integration_*.cfg` file. The payload schema and rich event set are well documented and well-tooled:
- [**cs2-gsi** (Rust)](https://github.com/ccc007ccc/cs2-gsi) — typed `GameState` model, ~45 derived events (`PlayerDied`, `BombPlanted`, `RoundPhaseUpdated`, `KillFeed`, …), auto-writes the cfg, Steam library discovery on Win/Linux/**macOS**. Explicitly notes Windows 10/11 is the primary target "because that's where CS2 lives".
- [**antonpup/CounterStrike2GSI** (C#)](https://github.com/antonpup/CounterStrike2GSI) — the upstream library cs2-gsi reimplements.
- [**kierandavies06/cs2gsi**](https://github.com/kierandavies06/cs2gsi) — GSI relay + web dashboard.
- [Valve's GSI reference](https://developer.valvesoftware.com/wiki/Counter-Strike_Global_Offensive_Game_State_Integration) (CS:GO-era doc, still the schema reference).

**Critical constraints:**
- GSI is emitted by the **game client**, not the dedicated server. A bare headless server has no GSI. (There is a long-standing bug report that GSI misbehaves in server contexts: [ValveSoftware/csgo-osx-linux#4071 "CS2 Server Bug: Game State Integration (GSI) Not Working"](https://github.com/ValveSoftware/csgo-osx-linux/issues/4071).)
- It is **throttled**, not per-tick. The cfg has `buffer`, `throttle`, `heartbeat` fields (defaults on the order of `0.1` s and `10.0` s) and a `data` block toggling which sections are sent (`allplayers_id`, `allplayers_match_stats`, `allgrenades`, `allplayers_position`, …). You are looking at **~10 Hz at best**, and only for the fields you enable.
- It is a **snapshot of the client's view**, so it inherits client-side culling/visibility caveats.

**Verdict:** GSI is excellent for reward signals and match-level bookkeeping (kills, deaths, round phase, bomb state, health, money). It is **not** a per-tick observation channel.

#### Option C — custom CounterStrikeSharp plugin + your own transport (the real answer)
Hook `OnTick` in a C# plugin, serialize the entity state you care about, and push it over a transport you choose:

| Transport | Rate | Notes |
|---|---|---|
| HTTP POST | ~64 Hz feasible | Simple; per-request overhead; fine for a single env |
| WebSocket | ~64 Hz | Best ergonomics; backpressure handling needed |
| UDP socket | 64+ Hz | Lowest latency; you handle dropped packets |
| Shared memory / local file ring buffer | 64+ Hz | Fastest; Linux-only, same-host only |
| **In-process .NET ↔ Python via `pythonnet`** | native | Run the policy *inside* the plugin process. Eliminates IPC entirely. **[UNVERIFIED]** — no public project does exactly this that I found, but it's mechanically straightforward with .NET 8 + PythonNET. |

**Caveat no one mentions:** a C# plugin doing per-tick JSON serialization of 10 players × hundreds of fields will itself consume significant CPU. Budget for binary serialization and field whitelisting. At 64 tick with 10 pawns this is genuinely non-trivial.

**No maintained off-the-shelf "RL bridge" plugin exists.** I searched specifically for one and found none. **This is a genuine gap in the ecosystem and probably the single highest-value thing you could build.**

### 1.4 Tickrate and time control

| Lever | Scope | Requires | Effect |
|---|---|---|---|
| `host_timescale <mult>` | Server | `sv_cheats 1` | 0.5 = half speed, 2 = double speed ([CS2Flow](https://cs2flow.com/commands/host_timescale)) |
| `host_framerate <n>` | Server | `sv_cheats` | "Frame rate" of the host; **lower = time passes faster, higher = slower**; `0` disables ([CS2Flow](https://cs2flow.com/commands/host_framerate)) |
| `demo_timescale <mult>` | Client, demo playback | — | 20 = 20× playback ([Total CS](https://totalcsgo.com/commands/categories/demo-and-overwatch)) |
| `demo_gototick <tick>` / `demo_goto <delta> 1` | Client, demo playback | — | **Seek to an exact tick**; negative delta rewinds. 64 ticks/s ([Total CS](https://totalcsgo.com/commands/categories/demo-and-overwatch)) |
| `demo_pause` / `demo_resume` / `demo_togglepause` | Client, demo playback | — | Freeze playback |
| `-tickrate` | Server launch | — | **CS2's relationship to explicit tickrate is genuinely murky in 2026.** CS2 replaced fixed-tick input with **subtick**; community servers can still be configured with different tickrates, and there is ongoing player debate about 64 vs 128 tick ([varidata comparison](https://cn.varidata.com/blog/18412.html), [5EPlay 2026-01-06 article](https://csgo.5eplay.com/article/260106v6jwlh)). **[UNVERIFIED — I could not confirm a current authoritative `-tickrate` spec from Valve.]** |
| `demo_recordcommands 1` | Client | — | Records console commands into a `.dem` — useful for building action replays |

#### Can you "step the game"?

**No — not in the Gym sense.** There is no API to advance exactly N ticks and return the resulting observation synchronously while a live match runs. `host_timescale` changes *wall-clock rate*, not *step granularity*; the server still free-runs. Your plugin's `OnTick` fires asynchronously relative to your policy's control loop. This is the fundamental mismatch: **CS2 is a real-time game, not a turn-based steppable simulator.**

**The one place you *can* step:** **demo playback**. `demo_gototick` + `demo_pause` + `startmovie` gives you a **deterministic, seekable, replayable, arbitrarily-slowable** view of recorded state. This is exactly what CS Demo Manager and Reka's renderer exploit. But it is **read-only** — your actions do not affect the world. It's a *dataset generator*, not an *environment*.

**Practical workaround for RL:** slow the server to `host_timescale 0.25–0.5`, run the policy at a matched effective rate, and accept that (a) throughput is poor, (b) episodes are long in wall-clock, (c) there is unavoidable jitter. For state-based RL this is workable but painful; for pixel-based RL it multiplies an already-severe bottleneck.

---

## 2. Client-side rendering for pixel observations

### 2.1 Can CS2 render headlessly on a server?

**No. Full stop.** The dedicated server binary contains no renderer — no D3D/Vulkan device, no scene graph evaluation for output, no framebuffer. This is not a configuration problem; the code path does not exist. Source 2's dedicated server is compiled for simulation + networking only.

People sometimes try `-nod3d9ex`, `-nojoy`, `-windowed`, xvfb, or `DISPLAY=` tricks. These do nothing, because there is nothing to draw. **XVFB does not give the CS2 dedicated server a renderer.**

### 2.2 What people actually do to get pixels

Five real patterns, ordered roughly by how many projects use them:

#### Pattern 1 — VM/GPU instance running the real client (most robust)
Run CS2 in a Windows or Linux VM on a GPU box, drive it, capture the window. This is what the earliest and most-cited CS:GO work did.

- **[TeaPearce/Counter-Strike_Behavioural_Cloning](https://github.com/TeaPearce/Counter-Strike_Behavioural_Cloning)** (IEEE CoG 2022, **Best Paper**) — the canonical example. Windows-only. Clicks into the game with `ctypes`-based simulated mouse/keyboard, grabs frames with `win32gui`/`win32ui` + OpenCV, reads internal state from **process memory via hazedumper offsets** plus **GSI** for scoreboard metadata. Agent runs at **16 fps**.
- **[microsoft/imitation_learning_in_modern_video_games](https://github.com/microsoft/imitation_learning_in_modern_video_games)** (AAMAS 2025 ALA workshop; paper [arXiv:2312.02312](https://arxiv.org/abs/2312.02312)) — Microsoft's follow-up. **Explicitly: "*Rollouts with this script are only supported in Windows*"** for CS:GO, while training is Linux. Uses GSI for meta-information, `csgo_legacy` branch on Steam, and a custom aim-assist workshop map for evaluation.

**Bottlenecks:** one GPU per game instance, no parallelism, real-time-only, fragile to game patches, and simulated input is exactly what anti-cheat looks for.

#### Pattern 2 — client rendering *demos* (the 2026 production method)
Do not play the game. **Replay demos in the client and capture the output.** Because playback is deterministic and seekable, you get clean, label-aligned data at scale.

- **[reka-ai/cs2-dem-renderer](https://github.com/reka-ai/cs2-dem-renderer)** (June 2026) — "**Built for Linux** but can work on Windows with minimal modifications." Pipeline:
  1. **Parse** (two-pass, Go + `demoinfocs-golang`): per-player spawn/death tick intervals, per-frame button inputs.
  2. **Render**: build a JSON action sequence → **launch CS2 via Steam with the demo loaded** → a **C++ server plugin** executes the action sequence while the client renders first-person.
  3. **Encode**: stream raw **TGA frames** from CS2's movie output directory to **ffmpeg (VAAPI HEVC)**.
  4. **Parquet**: per-round metadata — `actions` (W/A/S/D/J/V/C/R/`[`), `mouse_x_delta`, `mouse_y_delta`, position XYZ, yaw, pitch.
  - **Requirements: Go 1.21+, CMake, C++ compiler, ffmpeg with VAAPI, and CS2 installed via Steam.** Server plugin breaks on CS2 updates (main branch pinned to build `1.41.6.5`; version branches provided).
- **[EgoCS-400K](https://arxiv.org/html/2606.18180v1)** (June 2026, CityU HK) — "render the selected interval from the target player's first-person viewpoint using **CS Demo Manager and the Counter-Strike client**. Since CS2 demos run at 64 ticks per second, videos are rendered at 32 FPS." Parses with **DemoParser2**, exports per-tick `ticks.csv` (position, pitch/yaw, mouse-delta proxies, velocity, movement buttons, fire/reload/use, ground/duck state, active weapon, ammo, scoped/reload/inspect state), builds atomic action spans, then DP-segments clips into 2.0–6.5 s windows protecting action boundaries.

**Bottlenecks:** needs a **GPU**, a **real Steam account with CS2 installed (85 GB)**, ffmpeg, and — critically — this is *offline batch* work. Rendering 10,000 hours is a cluster job, not a laptop job.

#### Pattern 3 — capture during human play (cheap, high quality, slow)
Record your own gameplay with OBS/`startmovie`, sync with GSI, build a dataset. This is how the smaller expert datasets were made (e.g. the 24 GB expert dust2 set + 6 GB aim-map set in TeaPearce). **Quality is much higher than scraped data**; volume is limited by your patience.

#### Pattern 4 — HLAE / camera tooling (machinima, not AI)
**[HLAE (Half-Life Advanced Effects)](https://github.com/advancedfx/advancedfx)** is the standard tool for advanced CS camera work, free-camera, and high-fps capture. It is a **CS:GO-era tool**; CS2 support has been the subject of ongoing work and I could **not verify mature CS2 support** as of 2026-09 **[UNVERIFIED]**. HLAE is aimed at film-making, not RL, and does not give you an action interface.

#### Pattern 5 — synthetic re-rendering in a custom engine
**Covered in §4.3.**

### 2.3 Anti-cheat, Trusted Mode, `-insecure`

This is where well-meaning projects die, so be precise:

**Client side — Trusted Mode.**
- CS2 launches in **Trusted Mode** by default. Trusted Mode is a VAC-adjacent posture that blocks third-party code injecting into the game process. Valve's framing: trusted mode gives players a protected environment free of injected code; violating it subjects you to a **VAC ban** ([Valve's Trusted Mode FAQ](https://help.steampowered.com/en/faqs/view/09A0-4879-4353-EF95), [Steam Support KB 8835-KLXV-0292](https://support.steampowered.com/kb_article.php?ref=8835-KLXV-0292)).
- **`-untrusted` is the legacy flag; the current one is `-allow_third_party_software`** ([Steam community discussion](https://steamcommunity.com/app/730/discussions/0/3109142047892713744/)). Launching untrusted means you **cannot play on official/Valve servers** and your **Trust Factor drops**. Community servers with `sv_lan`/`-insecure` setups are the intended venue.
- **Screen capture is NOT blocked in Trusted Mode.** OBS, Discord overlay, and RTSS-style overlays work in trusted mode — this is directly stated in the community thread above and matches how streamers operate. So *read-only pixel capture* is safe.
- **Simulated mouse/keyboard input and process-memory reads are the risky part.** TeaPearce's own README warns: "*it's possible that Valve may detect the usage of some of these scripts in game (for example simulated mouse movements and RAM parsing) which in turn might lead to suspicion of cheating.*"

**Server side — `-insecure`.**
- A CS2 dedicated server started with `-insecure` **disables VAC** on that server. This is the documented, supported way to run mods/plugins: hosting panels expose an "Enable VAC" toggle ([Sparked Host KB, updated 2026-02-06](https://help.sparkedhost.com/en/article/how-to-disable-valve-anti-cheat-on-a-counter-strike-2-server-lcsej1/)). Turning VAC off "is sometimes done for private testing environments, custom mods, or controlled community gameplay where VAC-secured restrictions may interfere with certain modifications."
- **Metamod:Source + CounterStrikeSharp require an insecure/untrusted posture.** This is fine and normal — **for a private, local, `sv_lan`-ish research server.** It is *not* fine to point an automated agent at public matchmaking.

**Bottom line for AI work:**
| Activity | Risk | Verdict |
|---|---|---|
| Private Linux dedicated server, `-insecure`, Metamod + CSS, bots only, no Steam client attached | None (not connected to Valve infra) | **Safe. Do this.** |
| Client rendering demos on Linux, offline, not joining servers | Low | **Safe. This is what Reka/EgoCS do.** |
| Client in `-allow_third_party_software` + automated input on a private server | Low–moderate; Trust Factor hit | Acceptable for research; use a burner account |
| Automated input / memory reads on **official matchmaking** | **High — VAC ban / Overwatch** | **Do not.** |
| Distributing anything that looks like an aimbot | Bannable + ToS-violating | **Do not.** |

**The cleanest architecture by far:** *never automate the client's input at all.* Use the **server plugin** for control (legitimate, in-process, no injection into a protected client) and the **client only for offline demo rendering** (no input automation needed, the plugin drives playback). That combination stays entirely inside the lines.

### 2.4 The specific reason pixel RL is closed

Putting it together: to do pixel-based RL you need a tight loop of *observe frame → choose action → apply action → advance*. In CS2:

```
observe frame  ──> only the CLIENT can produce frames (needs GPU, Steam, desktop)
choose action  ──> your policy
apply action   ──> server plugin can write entity state; CLIENT input automation is
                   the risky/unsupported path
advance        ──> server free-runs; no step() exists
```

There is **no single process that can both render and be programmatically stepped with a stable action interface.** Every published approach picks one side and pays for it. **This is the core structural finding of this report.**

---

## 3. Existing projects and research

### 3.1 RL environments / Gym wrappers

| Project | Architecture | Status | Notes |
|---|---|---|---|
| **gym-csgo** (`pip install gym-csgo`) | Gym wrapper | **[UNVERIFIED — effectively abandoned]** | Listed at version 0.0.4 on [Snyk](https://security.snyk.io/package/pip/gym-csgo/0.0.4). No active development found. CS:GO-era. |
| **TeaPearce/Counter-Strike_Behavioural_Cloning** | Windows client + screen capture + RAM offsets | **Unmaintained, explicitly** — "There's no plan to actively maintain this repo" | [GitHub](https://github.com/TeaPearce/Counter-Strike_Behavioural_Cloning). CS:GO only (tested 1.37.7.0–1.38.0.1). Still the most complete reference implementation. |
| **htglvl/CS2_Imitation_Bot** | — | **Not what it claims.** Despite "Auto CS2 bot using reinforcement learning," [its README is a verbatim copy of the CS:GO BC repo](https://raw.githubusercontent.com/htglvl/CS2_Imitation_Bot/main/README.md). | **Do not cite as CS2 work.** |
| **kiskaserver/ExternalCS2** | External CUDA aimbot | External process, reads screen | [GitHub](https://github.com/kiskaserver/ExternalCS2). "For research and educational use only" — but this is an aimbot; using it against real players is a ban. |
| **AimBot_Unity_MLAgents** | Unity ML-Agents | Toy | Not CS2 — a Unity reimplementation. Illustrates the alternative: build your own sim. |

> **There is no maintained, general-purpose CS2 RL environment.** This is a real gap, not an oversight in my search. The reason is §2.4.

### 3.2 Imitation learning

| Project / paper | Year | Architecture | Data |
|---|---|---|---|
| **Counter-Strike Deathmatch with Large-Scale Behavioural Cloning** — Pearce & Zhu, IEEE CoG 2022 **Best Paper**; [arXiv:2104.04258](https://arxiv.org/abs/2104.04258); [code](https://github.com/TeaPearce/Counter-Strike_Behavioural_Cloning) | 2021–22 | Spectator scraping + **inverse dynamics model** to infer actions from metadata; HDF5; LSTM policy; runs at 16 fps in-game | **700 GB** scraped dust2 deathmatch + 24 GB expert dust2 + 6 GB aim map. Action space: keys onehot, L/R click onehot, mouse_x/y onehot. |
| **Visual Encoders for Data-Efficient Imitation Learning in Modern Video Games** — Schäfer et al., AAMAS 2025 ALA workshop; [arXiv:2312.02312](https://arxiv.org/abs/2312.02312); [code](https://github.com/microsoft/imitation_learning_in_modern_video_games) | 2023–25 | BC on frozen pretrained encoders (**DINOv2, CLIP, FocalNet, SD 2.1 VAE**) + MLP/LSTM/GPT policy. Reports kills-per-minute. | CS:GO + MineRL. **Notable negative result: "we were unable to train successful agents with the GPT transformer architecture."** |
| **EgoCS-400K** — Guo et al., [arXiv:2606.18180](https://arxiv.org/html/2606.18180v1) | 2026-06 | Replay-grounded dataset: parse demos → render FPV → DP-segment → VLM captions; aimed at world models / VLA | 400k+ videos, 10,000+ h |
| **D2E: Scaling Vision-Action Pretraining on Desktop Data** — ICLR 2026, [arXiv:2510.05684](https://arxiv.org/abs/2510.05684) | 2025-10 → 2026-03 | **OWA Toolkit** (152× compression) + **Generalist-IDM** (zero-shot inverse dynamics across unseen games via timestamp-based event prediction) + **VAPT** transfer to robotics. 1B params → 96.6% LIBERO, 83.3% CANVAS | 1.3K+ h; includes a **Counter-Strike 2** subset in [D2E-480p](https://huggingface.co/datasets/open-world-agents/D2E-480p) |

**Architecture summary of the IL field:** nobody does live RL on CS2 at scale. Everyone either (a) scrapes/captures human play and does behavioural cloning, or (b) parses demos offline. **The inverse-dynamics trick from TeaPearce — recover the action from the resulting state change rather than logging it — is the key reusable technique**, and D2E's Generalist-IDM is its 2026 generalization to arbitrary games.

### 3.3 Datasets (the actually-usable assets)

| Dataset | Size | Content | License | Link |
|---|---|---|---|---|
| **OpenCS2** | **11 TB** (10.6 TB bucket), ~5,000 h | Rendered pro CS2 matches; **every round from all 10 player POVs**; frame-aligned; **keyboard + mouse + world state (position, velocity, weapon)**; HD stereo audio; 1280×720 @ 32 fps | — | [collection](https://huggingface.co/collections/blanchon/opencs2), [dataset](https://huggingface.co/datasets/blanchon/opencs2_dataset), [WDS](https://huggingface.co/datasets/blanchon/opencs2_dataset_wds), [viewer Space](https://huggingface.co/spaces/blanchon/opencs2-dataset-viewer) |
| **Reka CS2-10k** | 10,000+ h (sample 25.9 GB / 748 rows) | 720p @ 48 fps, per-player-round; parquet with map, round, team, frame count, FOV, active movement/action keys, mouse-move proxies, world position, yaw/pitch | **CC BY-NC 4.0** | [HF](https://huggingface.co/datasets/RekaAI/CS2-10k), [blog](https://reka.ai/news/cs2-10k-a-large-scale-egocentric-counter-strike-2-dataset), [renderer](https://github.com/reka-ai/cs2-dem-renderer) |
| **EgoCS-400K** | 10,000+ h, 400k+ videos | Per-tick `ticks.csv`, atomic action spans, protected chains, DP segments, VLM captions | **CC BY 4.0** | [arXiv](https://arxiv.org/html/2606.18180v1) |
| **HLTV CS2 Demos Dataset** (`blanchon/cs2_dataset_demo`) | metadata Parquet = **tens of MB**; `.dem` files as loose blobs | One row per `.dem`: map, patch version, rounds, per-player KAST/ADR/rating, **every kill with weapon/headshot/tick/distance**, round winners+reasons | research use | [HF](https://huggingface.co/datasets/blanchon/cs2_dataset_demo) |
| **D2E-480p / Counter-Strike_2** | small (<1K) | OWA-standardized vision-action | — | [HF](https://huggingface.co/datasets/open-world-agents/D2E-480p) |
| **TeaPearce datasets** | 700 GB + 24 GB + 6 GB + … | CS:GO deathmatch HDF5 with screenshots + onehot actions | research only, **no commercial** | [repo](https://github.com/TeaPearce/Counter-Strike_Behavioural_Cloning) |

> **The `blanchon/cs2_dataset_demo` card deserves special attention for this machine.** It is queryable **entirely over the network** with DuckDB via `hf://` paths — you can run SQL over the metadata parquet *without downloading the demos*, and fetch individual `.dem` files on demand. Metadata is "tens of MB for the whole dataset." That is the ideal pattern when you have 15 GB of disk.

```sql
-- no download, runs against the Hub
SELECT match_id, map_name, patch_version, score_ct, score_t, rounds_played
FROM 'hf://datasets/blanchon/cs2_dataset_demo/data/*.parquet'
WHERE patch_version = '14141' AND map_name = 'de_overpass';
```

### 3.4 World models on CS2 — the 2026 frontier

This is where the most interesting current work is.

**[blanchon/cs2_vae](https://huggingface.co/buckets/blanchon/cs2_vae_backup/tree/machine_1/README.md)** (synced 2026-05-18) — the most concrete public CS2 world-model project I found. A **discrete-tokenization VAE over CS:GO `de_dust2` frames feeding an AR-LM world model targeting 60 s @ 32 fps action-conditional rollouts.** Reported headline results:

| run | tokens/frame | K | val NLL @10k | nats/frame | sample LPIPS | ceiling |
|---|---|---|---|---|---|---|
| FSQ-81 (VAE only) | 3600 | — | — | — | — | PSNR 30.40 |
| v200a-on-v300a-temporal-K8 | 880 | 8 | 1.165 | 1025 | 0.509 | 0.451 |
| v200a-on-v300b-temporal-K8 | 220 | 8 | 1.155 | 254 | 0.506 | 0.488 |
| **v200a-on-v300b-temporal-K16** | **220** | **16** | **0.909** | **200** | 0.520 | 0.489 |

Storage footprint of that project: **runs/ ~22 GB, checkpoints_legacy/ ~48 GB, logs/ ~7 GB** = **~77 GB** for one person's VAE/world-model line. It also reports a **failed action-conditioning attempt**: "v200e action-conditioning corruption test showed pose shuffle Δ=0.000 → frame-boundary effect, **not pose conditioning**." **That is an important negative result**: naive action conditioning on CS frames can appear to work while actually learning nothing about pose.

**EgoCS-400K** explicitly targets this: "action-conditioned future prediction, state- and event-aware scene rollout, controllable egocentric video simulation" — i.e. exactly the world-model use case, with the supervision needed to *check* whether conditioning is real.

### 3.5 LLM/VLM agent benchmarks in CS

**I found no dedicated, community-recognized "CS2 as an LLM-agent benchmark."** What exists is adjacent:

- **[Cultivating Gaming Sense for Yourself: Making VLMs Gaming Experts](https://aclanthology.org/2025.acl-long.643/)** — Lu et al., **ACL 2025** (pp. 13132–13152). The most directly relevant VLM-gaming paper. Its thesis is directly applicable to CS2: **using a VLM as a direct per-action controller is too slow for FPS** ("relying on isolated VLM reasoning for each action makes it impossible to handle tasks requiring high reactivity (e.g., FPS shooting)"). Instead the VLM acts as a **developer** that writes specialized real-time execution modules. Reports first fluent gameplay across ACT/FPS/Flappy Bird. **Anyone proposing "LLM plays CS2 by looking at frames" should read this first — the paper is largely an argument that this does not work.**
- **GameSense** — the framework introduced in that paper.
- **[OmniGameArena](https://www.chatpaper.ai/ru/paper/40b24def-3898-4868-9923-2d1fa7500ada)** — a UE5-based unified benchmark for VLM game agents. Multi-game, not CS-specific, but the right comparison class.
- **[CS2-Ai-Chatter](https://github.com/drippycatcs/CS2-Ai-Chatter)** — reads match chat and replies via Ollama/OpenRouter, web dashboard. An LLM **social** agent, not a gameplay agent. Fun, and a legitimate CS2+LLM integration.
- **[CSTrader](https://www.emergentmind.com/papers/2606.31461)** — language-grounded trading testbed **using CS2 skins as the asset market**. Relevant if you care about "CS2 as a domain for LLM agents" in the economic sense rather than gameplay.

**Honest read:** CS2 is a *plausible* LLM-agent benchmark substrate (rich state, natural language callouts, multi-agent coordination, partial observability), but **nobody has built it**. The blocker is §2.4 again — you cannot cheaply give an LLM agent a closed perception→action loop in CS2.

### 3.6 Low-level / reverse-engineering tooling

- **[hzqst/cs2_vibesignatures](https://github.com/hzqst/cs2_vibesignatures)** — LLM+IDA-assisted generation of CS2 signatures/offsets; maintains gamedata for CounterStrikeSharp, CS2Fixes, CS2FOW, swiftlys2, plugify, cs2kz-metamod, modsharp, CS2Surf/Timer. Versioned per game build (e.g. `14156`). **This is what keeps server plugins alive across CS2 patches.**
- **[CS2Fixes](https://github.com/Source2ZE/CS2Fixes)** — source of much early CS2 reverse-engineering.
- **[demoinfocs-golang](https://github.com/markus-wa/demoinfocs-golang)** (Go), **[demoparser2](https://github.com/LaihoE/demoparser)** (Rust → Python/JS/WASM), **[awpy](https://github.com/pnxenopoulos/awpy)** (Python), **[demofile-net](https://github.com/saul/demofile-net)** (C#), **[clarity](https://github.com/skadistats/clarity)** (Java).

---

## 4. Demo-based (offline) approaches

### 4.1 The core insight

A CS2 `.dem` is **not a video**. It is a complete, deterministic record of a match: per-tick entity state, view angles, button bitmask, game events, and round structure. This means a demo gives you something almost no other game dataset does — **the exact state trajectory and the exact inputs that produced it, from a human expert, for free, at scale.**

That is why, in 2026, **every serious CS2+AI effort is demo-based**. It sidesteps the entire §2.4 problem.

### 4.2 Parsers — pick one and go

| Parser | Language | CS2 support | Notes |
|---|---|---|---|
| **[awpy](https://github.com/pnxenopoulos/awpy)** | **Python 3.11+** | **CS2 `PBDEMS2` only; "The older CS:GO format is not supported."** | **Best choice if you want to start today.** Ships as a **pre-built wheel with a Rust parser inside — no compiler required.** Polars DataFrames. `demo.ticks()` gives one row per player per tick; `demo.snapshots()` gives position/angles/health/armor; `demo.kills`, `demo.rounds`, `demo.grenades`, `demo.shots`, `demo.stats` (KAST/ADR); `VisibilityChecker` downloads collision meshes for line-of-sight. CLI included. Also `awpy[plot]` for radar plots. |
| **[demoparser2](https://github.com/LaihoE/demoparser)** | Python / JS / WASM (Rust core) | CS2 | **Query-style API** (`parse_ticks`, `parse_event`) instead of streaming hooks. **Massive field catalogue** — the README lists ~150 player fields (`X/Y/Z`, `pitch`, `yaw`, `velocity`, `duck_amount`, `is_scoped`, `shots_fired`, `aim_punch_angle`, `flash_duration`, `in_buy_zone`, …), button bitmask decomposition (`FORWARD/LEFT/RIGHT/BACK/FIRE/RIGHTCLICK/RELOAD/INSPECT/USE/ZOOM/WALK`), game-state fields, weapon fields, and **`usercommands`** (`usercmd_mouse_dx`, `usercmd_mouse_dy`, `usercmd_viewangle_x/y/z`, `usercmd_buttonstate_1..3`, `usercmd_forward_move`, `usercmd_subtick_moves`). Benchmark: 50 demos / **4.6 GB in 6.14 s on a Ryzen 5900X (~749 MB/s)**; ~328 MB/s on a ThinkPad. |
| **[demoinfocs-golang](https://github.com/markus-wa/demoinfocs-golang)** | Go | CS2 | Used by reka-ai/cs2-dem-renderer |
| **[demofile-net](https://github.com/saul/demofile-net)** | C# | CS2 | Pairs naturally with CounterStrikeSharp |
| **[clarity](https://github.com/skadistats/clarity)** | Java | — | Legacy |

> **`usercommands` is the crown jewel.** `usercmd_mouse_dx/dy` and `usercmd_buttonstate_*` give you the **actual per-tick human input**, not a reconstruction. This is what makes genuine behavioural cloning possible — you don't need TeaPearce's inverse-dynamics inference step anymore for CS2 demos.

### 4.3 State of the art

Three mature pipelines now exist, all published in 2026:

1. **Reka CS2-10k** → [cs2-dem-renderer](https://github.com/reka-ai/cs2-dem-renderer). Go parser + C++ server plugin + Steam-launched CS2 client + ffmpeg VAAPI → `<uuid>.mp4` + `<uuid>.parquet` per player-round. Handles the whole thing including a **browser-based viewer** (File System Access API) with synchronized video + action playback, mouse-delta charts, and XY minimap.
2. **EgoCS-400K** → CS Demo Manager + CS client rendering, then DemoParser2 parsing, then rule-based atomic-action extraction, then DP segmentation with protected action chains, then **prior-guided VLM captioning** (structured priors constrain the caption; the video supplies appearance). Notably it renders **32 fps from 64-tick demos** so every frame maps to exactly 2 ticks.
3. **OpenCS2** → 11 TB of the above, done at scale, with a WebDataset variant for high-throughput training.

**Two 2026 lessons worth internalizing:**
- **Render at a tick-aligned fps.** 64-tick demos → 32 fps (EgoCS) or 48 fps from pro demos (Reka). Never resample arbitrarily; you destroy the frame↔tick mapping that makes the dataset useful for action-conditioned modelling.
- **Filter aggressively.** EgoCS explicitly discards spectator-view recordings, failed captures, missing/inconsistent player views, and clips whose duration doesn't match the parsed tick interval.

### 4.4 Synthetic re-rendering in a custom engine (Unity/Unreal/Three.js)

**Does anyone do this for AI training on CS2? From what I can find: no, not for CS2.**

What exists:
- **Partial**: [EgoCS-400K's viewer](https://arxiv.org/html/2606.18180v1), [Reka's viewer](https://huggingface.co/spaces/RekaAI/CS2-10k-viewer), and [awpy's radar plotting + `VisibilityChecker`](https://awpy.readthedocs.io/en/stable/getting-started.html) all reconstruct *2D map-space* views (radar/minimap) from parsed state. That is synthetic re-rendering of a **low-dimensional** observation, and it works well. Radar-image observations are a genuinely practical, cheap observation space on this machine.
- **Full 3D**: nobody has published a CS2 map-geometry + agent-animation renderer feeding an AI pipeline. awpy *does* download per-map **collision meshes** and radar assets from [awpy-data](https://github.com/pnxenopoulos/awpy-data), so the geometry pipeline is partially available — but map *visual* assets (materials, props, lighting) are not distributed in a renderer-ready form, and redistributing Valve's art assets would be legally fraught.

**Sim-to-real implications if you did build one:**
- **Pros:** unlimited data, perfect labels, arbitrary viewpoints, controllable lighting, no anti-cheat, no GPU-desktop requirement, and it runs on a Mac. This is exactly why DeepMind/OpenAI built bespoke sims instead of instrumenting commercial games.
- **The gap is large and specific.** A custom renderer approximates CS2's visuals, but RL/IL policies are notoriously sensitive to *exactly* the distribution they trained on. Key mismatches: Source 2's deferred renderer and lighting, material/PBR response, anti-aliasing and post-processing, viewmodel (hands/weapon) rendering, smoke/molotov volumetric effects, decals, the specific HUD and crosshair, and the fact that CS2's **subtick input** means the visual result of an action isn't a clean function of the tick. A policy trained on a Three.js approximation will very likely not transfer to the real game.
- **The honest framing:** a custom renderer is a **world-model / representation-learning research tool**, not a route to a competitive CS2 agent. If your goal is *"study whether a learned model can predict CS2 dynamics,"* synthetic re-rendering is great. If your goal is *"my agent plays CS2,"* it is a detour.
- **The cheaper, more honest middle path:** render from **parsed demo state only** at low dimensionality (radar image + scalar features), or use the **real rendered datasets** (OpenCS2 / CS2-10k / EgoCS) and skip synthesis entirely.

---

## 5. Honest assessment for this machine

### 5.1 The hardware baseline

| Resource | Value | Implication |
|---|---|---|
| CPU | Apple **M1 Pro** | Fine for parsing, dataloading, small models |
| RAM | **16 GB** | Comfortable for parsing; **tight** for training — unified memory is shared between CPU and GPU |
| Disk free | **~15 GB** | **The binding constraint.** No CS2 client (85 GB). No CS2 server (~30–40 GB). No full dataset (25.9 GB minimum for just the CS2-10k sample; 11 TB for OpenCS2). |
| OS | macOS 26.6 | **No CS2 in any form.** No CS2 dedicated server. No CUDA. MPS backend only. |
| GPU | M1 Pro (16-core, ~5.2 TFLOPS FP32) | ~1/8 to 1/15 of an RTX 4090 for CNN training; **no bf16/tf32, limited FlashAttention maturity via MPS** |

### 5.2 Per-task verdict

#### A. RL from pixels — **not realistic on this machine. Not realistic as a plan, full stop.**

- Cannot run the renderer-carrying client (no macOS build; 85 GB > 15 GB free; no NVIDIA).
- Cannot run the dedicated server natively (x86-64 only; ARM emulation unverified).
- Even with a Linux+GPU box: no `step()` API, no parallelism, per-instance GPU cost, and the observation pipeline is real-time-bound. Sample efficiency would be catastrophic and throughput would be a few thousand frames/hour rather than the millions/hour you need.
- **Verdict: ✗ on this machine. ✗ as a first project anywhere. Only viable on a multi-GPU Linux cluster, and even then it is a research bet, not an engineering task.**

#### B. RL from state — **possible in principle, not on this machine; genuinely viable on a Linux box.**

- Requires the dedicated server, which requires Linux x86-64 (or an unverified ARM emulation path).
- The plugin-bridge to a Python trainer does not exist off the shelf — **you would build it. This is real, valuable, publishable work.**
- Once built: 64 tick, ~10 agents, entity-observation space (positions/angles/health/weapon/ammo/money + raycasts), discrete+continuous action head writing to the pawn each tick. This is a *legitimate* RL environment.
- **Caveats to accept going in:** no deterministic stepping; `host_timescale` only; stock bots are unmodifiable opponents; episode throughput will be low; every CS2 patch risks breaking your plugin (mitigated by cs2_vibesignatures).
- **Verdict: ✗ on macOS. ✓ on a Linux box with a few GB RAM and ~40 GB disk. This is the most *interesting* thing you could build, but it cannot start here.**

#### C. Imitation learning from demos — **realistic on this machine, at reduced scale. This is the recommendation.**

Why it fits:
- **No game required.** `pip install awpy` gives you a compiler-free Rust parser on Python 3.11+.
- **Disk can be sidestepped.** Use DuckDB `hf://` to query `blanchon/cs2_dataset_demo` metadata over the network (tens of MB). Download individual `.dem` files on demand (a single `.dem` is typically tens to a few hundred MB) rather than the whole corpus.
- **Compute is tractable.** A small CNN + LSTM behaviour-cloning policy on **128×128 or 96×96 grayscale/4:3-cropped frames**, with frame stacking, is trainable on an M1 Pro in hours-to-days. Microsoft's `pixelbc` code already supports `image_height`/`image_width` config, frame stacking, downsampling, and **pre-computed frozen embeddings from DINOv2/CLIP** — precomputing embeddings and training only the policy head is *exactly* the right move for 16 GB of unified memory.
- **Labels are free.** `demoparser2`'s `usercmd_*` columns give you real per-tick human input. No inverse-dynamics inference needed.

What you must give up:
- No 700 GB scraped corpus locally. Work with **10–100 demos**, not 10,000.
- No full-resolution training. Downsample hard.
- Metrics will be modest. Aim for "the policy moves and aims plausibly," not "kills-per-minute beats a bot."

**Concrete free-disk budget on 15 GB:**

| Item | Size |
|---|---|
| Python env (awpy/demoparser2, torch, duckdb) | ~4–6 GB |
| 20–40 `.dem` files | ~4–8 GB |
| Extracted frames for ~1,000 clips @ 128×128 JPEG | ~0.5–1 GB |
| Checkpoints + logs | ~1–2 GB |
| **Total** | **~10–17 GB — tight but workable if you delete `.dem` files after extracting frames.** Use an external drive for the raw demos. |

**Verdict: ✓ on this machine, with ~40 demos and 128×128 observations. Scale-up needs Linux + GPU + 1–2 TB.**

#### D. Training a world model / observation generator — **partially realistic; the ambitious version is not.**

Three tiers:

| Tier | What | Feasible here? |
|---|---|---|
| **D1. Radar/2D state renderer** | Parse demos → rasterize player positions onto awpy's radar images (`awpy[plot]` already does frames/heatmaps/GIFs; `VisibilityChecker` gives real line-of-sight from collision meshes). Train a small model to predict next radar frame from action. | **✓ YES.** Cheap, uses no game assets, runs entirely on parsed state. Excellent first world-model project. |
| **D2. Small VAE / tokenizer on downsampled frames** | The cs2_vae direction: train a discrete-tokenization VAE on 64×64 or 128×128 crops. | **~ Marginal.** A *small* VAE is trainable on an M1 Pro, but you need frames — and frames come from the 25.9 GB Reka sample or from rendering, neither of which fits. If you extract a modest frame bank (say 200k frames @ 64×64) and stream it, it's plausible. **Note the cs2_vae author's own footprint: ~77 GB across runs + checkpoints + logs for this line of work.** |
| **D3. Action-conditioned AR-LM world model at 32 fps / 60 s rollouts** | The actual cs2_vae target; Genie/GameNGen-class interactive video world models. | **✗ NO.** This is multi-GPU, multi-TB, weeks-of-training territory. **And note the published negative result: cs2_vae's action-conditioning attempt showed pose-shuffle Δ=0.000 — the model was exploiting frame-boundary effects, not learning action conditioning.** Expect this failure mode. |

**Verdict: D1 ✓, D2 ~, D3 ✗. D1 is the sweet spot and is genuinely under-explored.**

### 5.3 Recommendation matrix

Rating scale: **✓✓** = do it now on this machine · **✓** = doable with care here · **~** = possible but cramped/limited · **✗** = not on this machine · **✗✗** = not viable anywhere without a cluster

| Goal | On M1 Pro / 16 GB / 15 GB free (macOS) | Minimum viable setup elsewhere | Recommended first step |
|---|:---:|---|---|
| **Run a CS2 dedicated server headless** | **✗✗** (no macOS/x86 build; ~40 GB) | Linux x86-64, 4 GB RAM, 40 GB disk, Docker + `joedwards32/cs2` | Try Docker + Rosetta once to confirm; otherwise use a €5/mo Linux VPS |
| **Metamod + CounterStrikeSharp bot/state plugin** | **✗** (same blocker) | Same Linux box; `-insecure`, `sv_cheats 1` | Build a `css_obs` plugin that logs per-tick state to a file; validate offline |
| **Custom plugin → Python RL bridge** | **✗** to develop; can *read* a remote bridge | Linux server + Python trainer (can be split across hosts) | Write the WebSocket bridge. **This is the highest-value missing piece in the ecosystem.** |
| **State-space RL (own pawn vs. bots)** | **✗** | Linux + GPU optional (state RL is cheap); low `host_timescale` | Build the bridge first; then a PPO/SAC baseline on a single map |
| **Pixel RL (live game)** | **✗✗** | Multi-GPU Linux cluster + Windows/Linux GPU clients; 85 GB/instance | **Don't.** Read [`Cultivating Gaming Sense`](https://aclanthology.org/2025.acl-long.643/) first — it argues VLM-direct-control FPS is the wrong architecture anyway |
| **Behavioural cloning from demos (state-only, MLP/transformer)** | **✓✓** | Nothing extra — pure CPU is nearly enough | `pip install awpy demoparser2`; parse 5 demos; fit next-action from state. **Start here.** |
| **Behavioural cloning from demos (pixels, 128×128)** | **✓** (small scale) | Linux + one 12 GB GPU for serious runs; 1–2 TB disk | Parse → extract 128×128 frames → frozen DINOv2 embeddings (`pixelbc`) → LSTM head |
| **Imitation learning at paper scale (700 GB)** | **✗✗** | Linux + A100/H100, multi-TB | Use OpenCS2 streaming + WebDataset; rent compute |
| **World model D1 — radar/2D state renderer** | **✓✓** | Nothing extra | Parse demos → `awpy` radar frames → small next-frame predictor |
| **World model D2 — small frame VAE/tokenizer** | **~** | 1× 24 GB GPU, 1 TB disk | Stream 64×64 crops from `blanchon/cs2-frames` / Reka sample |
| **World model D3 — action-conditioned video AR-LM** | **✗✗** | Multi-GPU, multi-TB, months | **Don't start here.** Contribute to `cs2_vae` or use EgoCS-400K |
| **LLM/VLM agent benchmark in CS2** | **✗** (needs the env) | Linux server + inference GPU | Build the *state* benchmark (text observations via the plugin bridge) — nobody has |
| **Dataset tooling / parsers / analysis (no game)** | **✓✓** | Nothing extra | DuckDB over `hf://` metadata; `awpy`; `demoparser2` |
| **Custom synthetic 3D renderer (Unity/UE/Three.js)** | **~** (Three.js + parsed state, yes) | GPU + legal review of Valve assets | Only as a world-model research tool; **not** a route to a playing agent |

### 5.4 The staged plan I would actually recommend

**Stage 0 — today, on this machine, zero game required (~1 day)**
1. `pip install awpy demoparser2 duckdb polars` (prebuilt wheels, no compiler).
2. Query `blanchon/cs2_dataset_demo` metadata over `hf://` — pick ~20 pro matches on 2–3 maps.
3. Download ~20 individual `.dem` files **to an external drive**, parse with `demo.ticks([...])` and `demoparser2.parse_ticks(["X","Y","pitch","yaw","usercmd_mouse_dx","usercmd_mouse_dy","usercmd_buttonstate_1", ...])`.
4. Confirm you can reconstruct a per-tick `(state, action)` stream. **This step alone validates the whole thesis.**

**Stage 1 — offline IL on state (~1 week)**
5. Build a PyTorch `Dataset` yielding `(state_t, action_t)` windows. Fit a small transformer/MLP to predict `(mouse_dx, mouse_dy, buttons)`.
6. Evaluate by held-out action NLL and by *rollout plausibility* on held-out rounds. Delete `.dem` files once features are extracted, to reclaim disk.

**Stage 2 — pixel IL at 128×128 (~2–4 weeks)**
7. Extract frames for ~1,000 clips at 128×128 (keep the frame↔tick mapping at 32 fps from 64-tick demos).
8. Precompute frozen DINOv2/CLIP embeddings in batches (watch RAM; reduce batch size on failure — this is documented in `pixelbc`).
9. Train an LSTM policy head on embeddings. Report kills-per-minute or aim-error, honestly.

**Stage 3 — either (a) radar world model on this machine, or (b) go Linux for real RL**
- **(a)** Rasterize parsed state onto `awpy` radar images; train a next-frame predictor conditioned on action. This is a genuine, publishable, Mac-friendly world-model contribution.
- **(b)** Rent a Linux GPU box. Stand up `joedwards32/cs2` + Metamod + CounterStrikeSharp. Build the tick→WebSocket→Python bridge. Then, and only then, attempt state-space RL.

**Do not** begin by trying to make CS2 run on this Mac, and **do not** begin with pixels. Both are dead ends that will consume weeks.

---

## 6. Appendix

### 6.1 Key repos and links (one place)

**Server / plugins**
- https://github.com/joedwards32/CS2 — CS2 dedicated server Docker image (recommended)
- https://hub.docker.com/r/cm2network/cs2 — the commonly-referenced mirror
- https://github.com/roflmuffin/CounterStrikeSharp — C#/.NET 8 server plugin framework
- https://docs.cssharp.dev/ — CounterStrikeSharp docs
- https://github.com/Source2ZE/CS2Fixes — CS2 server fixes / RE reference
- https://github.com/hzqst/cs2_vibesignatures — automated signature/offset regeneration (keeps plugins alive)
- https://www.sourcemm.net/ — Metamod:Source

**Demos / parsing**
- https://github.com/pnxenopoulos/awpy — Python, CS2-only, prebuilt wheel, Polars DataFrames
- https://awpy.readthedocs.io/en/stable/getting-started.html — awpy guide
- https://github.com/LaihoE/demoparser — Rust core, Python/JS/WASM, huge field list incl. `usercmd_*`
- https://github.com/markus-wa/demoinfocs-golang — Go
- https://github.com/saul/demofile-net — C#
- https://github.com/pnxenopoulos/awpy-data — collision meshes + radar assets

**Rendering**
- https://github.com/reka-ai/cs2-dem-renderer — demo → mp4 + parquet, Linux
- https://github.com/akiver/cs-demo-manager — CS Demo Manager
- https://github.com/advancedfx/advancedfx — HLAE (CS2 support unverified)

**Datasets**
- https://huggingface.co/collections/blanchon/opencs2 — 11 TB / 5,000 h
- https://huggingface.co/datasets/RekaAI/CS2-10k — 10,000 h, CC BY-NC 4.0
- https://huggingface.co/datasets/blanchon/cs2_dataset_demo — HLTV demo + metadata parquet
- https://huggingface.co/datasets/open-world-agents/D2E-480p — D2E (ICLR 2026) incl. CS2

**Research / models**
- https://github.com/TeaPearce/Counter-Strike_Behavioural_Cloning — CS:GO BC, IEEE CoG 2022 Best Paper
- https://github.com/microsoft/imitation_learning_in_modern_video_games — visual encoders for BC (AAMAS 2025)
- https://arxiv.org/abs/2312.02312 — Visual Encoders for Data-Efficient IL
- https://arxiv.org/html/2606.18180v1 — EgoCS-400K
- https://arxiv.org/abs/2510.05684 — D2E (ICLR 2026)
- https://aclanthology.org/2025.acl-long.643/ — Cultivating Gaming Sense (VLM gaming, ACL 2025)
- https://huggingface.co/buckets/blanchon/cs2_vae_backup — cs2_vae world-model research log

**GSI**
- https://github.com/ccc007ccc/cs2-gsi — Rust GSI listener, 45 events, macOS Steam discovery
- https://github.com/antonpup/CounterStrike2GSI — C# upstream
- https://github.com/kierandavies06/cs2gsi — GSI relay + dashboard

### 6.2 Command cheat-sheet

```bash
# --- Server (Linux, Docker) ---
# docker run with SRCDS_TOKEN, CS2_RCONPW, CS2_MAXPLAYERS, CS2_BOT_QUOTA, ...
# launch flags for a research server:
#   -insecure            # disable VAC (required for mods)
#   -tickrate 128        # [UNVERIFIED for current CS2]
# server.cfg:
sv_cheats 1              # required by host_timescale / host_framerate
host_timescale 0.5       # 0.5 = half speed, 2 = double
host_framerate 0         # 0 disables; *lower* = faster, *higher* = slower
bot_quota 10
bot_difficulty 3         # 0..3
sv_lan 1

# --- Client, demo playback (deterministic, seekable, READ-ONLY) ---
playdemo <file>.dem
demo_pause ; demo_resume ; demo_togglepause
demo_gototick 7000       # exact tick = your stepping primitive
demo_goto -64 1          # rewind 64 ticks
demo_timescale 20        # 20x playback
demoui                   # playback UI
spec_show_xray 1         # see players through walls
startmovie <name>        # dump TGA frames for ffmpeg

# --- Client launch posture ---
-allow_third_party_software   # modern flag; = untrusted mode; no official servers, Trust Factor hit
-untrusted                    # legacy flag
```

### 6.3 Known traps

1. **`developer.valvesoftware.com` is behind Anubis** (proof-of-work anti-scraping). Automated fetching fails; use mirrors/community docs.
2. **TeaPearce's repo is Windows-only for the game loop and explicitly unmaintained.** Excellent reference, not a foundation.
3. **`htglvl/CS2_Imitation_Bot` is a renamed copy of the CS:GO repo.** Not CS2 work.
4. **GSI is client-side and throttled** (~10 Hz, configurable via `throttle`/`heartbeat`). Not a tick stream.
5. **`host_timescale` needs `sv_cheats 1`.**
6. **CS2 updates break server plugin signatures.** Pin to a game build; use cs2_vibesignatures to regenerate.
7. **`bot_add` fails on navmesh-less maps** unless you patch the `g_pNavMesh` null-check in `CCSBotManager_AddBot`.
8. **Trusted Mode does not block screen capture** — only injection. Capture is safe; simulated input is the risk.
9. **No `step()` anywhere.** `host_timescale` for live matches; `demo_gototick` for offline replay. Don't build an architecture that assumes Gym semantics.
10. **Disk is the silent killer.** CS2 client = 85 GB. CS2-10k sample = 25.9 GB. OpenCS2 = 11 TB. `cs2_vae`'s own working set = ~77 GB. Plan storage before anything else.
11. **A world model can fake action-conditioning.** cs2_vae measured pose-shuffle Δ=0.000 — a corruption test that shows the model was not using pose at all. **Always run an action/pose ablation** before believing your conditioning works.
12. **VLM-as-direct-controller is the wrong architecture for FPS.** Per ACL 2025's GameSense paper, per-action VLM reasoning is fundamentally too slow for FPS; use the VLM to *write* real-time modules instead.

---

## 7. One-line answers to the original questions

1. **Headless server?** Yes on Linux x86-64 via Docker (`joedwards32/cs2`, `cm2network/cs2`). **No on macOS** — CS2 has no Mac build and the server is x86-64 only; ARM emulation (FEX-EMU) is an unverified 2023 hack.
2. **Programmatic bots?** Partly. Metamod + CounterStrikeSharp (C#) can spawn/teleport/force-buy/read everything, but **cannot replace bot AI** — it's a closed navmesh system. **No official Python binding exists.**
3. **Per-tick state out?** Yes, but **you write the bridge**: a CounterStrikeSharp `OnTick` plugin + HTTP/WebSocket/UDP to Python. RCON is orchestration-only; GSI is client-side and ~10 Hz.
4. **High tickrate / time acceleration?** `host_timescale` and `host_framerate` (both need `sv_cheats`) for live; `demo_timescale` for playback. **No deterministic stepping.**
5. **Headless rendering on the server?** **No.** The dedicated server has no renderer.
6. **How do people get pixels?** Linux/Windows GPU client in a VM/instance, or — the 2026 method — **render demos in the client and capture with ffmpeg** ([cs2-dem-renderer](https://github.com/reka-ai/cs2-dem-renderer)). HLAE CS2 support unverified.
7. **Anti-cheat a blocker?** For the **client** with automated input and memory reads: **yes**. For a **private `-insecure` server** and **offline demo rendering**: no.
8. **Existing CS2 RL environment?** **No maintained one exists.** This is a real, valuable gap.
9. **Academic work?** CS:GO BC (CoG 2022), visual encoders for BC (2025), EgoCS-400K (2026), D2E (ICLR 2026), GameSense (ACL 2025). **All demo- or capture-based; none live-RL.**
10. **Demo-based?** **This is the SOTA and the right answer.** 11 TB OpenCS2, 10,000 h CS2-10k, 400k-video EgoCS-400K, plus prebuilt Rust parsers (`awpy`, `demoparser2`) that install with no compiler on Python 3.11+.
11. **Synthetic re-rendering?** Nobody does it in 3D for CS2. 2D radar reconstructions from parsed state **do** work and are the practical middle path. Sim-to-real gap for a custom 3D renderer would be large.
12. **On this Mac?** **Demo-based IL from state ✓, pixel IL at 128×128 ✓ (small scale), radar world model ✓. Live server ✗, pixel RL ✗✗, AR-LM video world model ✗✗.** Disk (15 GB) is the binding constraint — use DuckDB over `hf://` and an external drive, and never try to install CS2.
