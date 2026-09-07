# CS2 Demo JSON Format V2

> **Version**: `cs2.demo.v2`  
> **CLI**: `python scripts/demo_to_json.py --demo match.dem --out output.json`

---

## 1. Design Philosophy

The V2 format is designed to eliminate the redundancy present in V1 while remaining
human-readable and easy to consume from Python / NumPy / PyTorch.

### V1 → V2 improvements

| Problem in V1 | Solution in V2 |
|---|---|
| 64-bit `steamid` strings repeated in every tick | Players defined once in header; referenced by index 0–9 |
| `[{player_dict}, {player_dict}, ...]` per tick | Columnar arrays: `{"x": [1,2,3], "y": [4,5,6]}` |
| `future_kills` / `future_damage` stored in EVERY tick | Events stored as a flat timeline; consumers compute "future from T" |
| Smoke/inferno active set duplicated per tick | Stored as `{ts, te, x, y, z}` intervals |
| `weapon_name` / `last_place_name` strings repeated | Lookup tables in header → integer indices in tick data |

### Estimated size reduction

For a typical 30-round match at `interval=0.25s`:

| Component | V1 (approx.) | V2 (approx.) | Reduction |
|---|---|---|---|
| Player identity | ~5 MB (steamid strings × N ticks) | ~1 KB (header only) | ~5000× |
| Player states | ~200 MB (dicts) | ~40 MB (columnar arrays) | ~5× |
| future_kills/damage | ~150 MB (repeated per tick) | ~0.2 MB (timeline) | ~750× |
| Projectiles (smoke/inferno) | ~80 MB (per-tick active set) | ~0.1 MB (intervals) | ~800× |
| **Total** | **~400–500 MB** | **~40–50 MB** | **~10×** |

---

## 2. Top-Level Schema

```json
{
  "format": "cs2.demo.v2",
  "map": "de_mirage",
  "players": [ ... ],        // length 10, index = player reference
  "weapons": { ... },        // weapon name → int index
  "places": { ... },         // place name → int index
  "rounds": [ ... ]          // one entry per round
}
```

| Field | Type | Description |
|---|---|---|
| `format` | `string` | Always `"cs2.demo.v2"` |
| `map` | `string` | Map name (e.g. `"de_mirage"`, `"de_dust2"`) |
| `players` | `array[Player]` | 10 players in canonical index order (see §2.1) |
| `weapons` | `object{string→int}` | Weapon name → integer index for tick data |
| `places` | `object{string→int}` | Place name → integer index for tick data |
| `rounds` | `array[Round]` | Per-round data (see §3) |

### 2.1 Player metadata

```json
{
  "steamid": "76561198000000001",
  "name": "s1mple"
}
```

| Field | Type | Description |
|---|---|---|
| `steamid` | `string` | SteamID64 |
| `name` | `string` | In-game name |

> **Note**: Team affiliation is **not** stored at the player level because teams
> swap sides at halftime (round 12).  Instead, each round contains a `teams`
> array (see §3) mapping each player index to `"CT"` or `"T"` for that round.

The array index of each player is used throughout tick data to reference them
(e.g. `"a": 0` in a kill event means the attacker was `players[0]`).

---

## 3. Round Schema

```json
{
  "id": 0,
  "teams": ["CT", "T", "CT", "T", "CT", "T", "T", "T", "CT", "CT"],
  "winner": "CT",
  "end_reason": "t_killed",
  "bomb_planted_time": 45.2,
  "ticks": [5000, 5016, 5032, ...],
  "round_seconds": [0.5, 1.0, 1.5, ...],
  "bomb_planted": [false, false, false, ...],
  "bomb_dropped": [false, false, true, ...],
  "bomb_position": [null, null, [100.0, 200.0, 0.0], ...],
  "players": [ ... ],
  "events": { ... }
}
```

| Field | Type | Description |
|---|---|---|
| `id` | `int` | Round number (0-indexed) |
| `teams` | `[string]` | Team for each player index (length 10): `"CT"` or `"T"`.  Changes at halftime (round 12). |
| `winner` | `string` | Winning team (`"CT"` or `"T"`) |
| `end_reason` | `string` | `"t_killed"` (all T dead), `"ct_killed"` (all CT dead), `"bomb_exploded"`, `"bomb_defused"`, `"time_ran_out"` |
| `bomb_planted_time` | `float \| null` | Round-relative seconds when bomb was planted, or `null` |
| `ticks` | `[int]` | Sampled tick numbers (length **N**) |
| `round_seconds` | `[float]` | Round-relative seconds for each tick (length **N**) |
| `bomb_planted` | `[bool]` | Whether bomb is planted at each tick |
| `bomb_dropped` | `[bool]` | Whether bomb is dropped at each tick |
| `bomb_position` | `[[float,float,float] \| null]` | Bomb **carrier** position at each tick; tracks the player holding the C4 (not the planted bomb).  `null` if no one has picked up the bomb yet.  After plant, the C4 leaves the carrier's inventory so the last known position is frozen at the planter's location. |
| `players` | `[PlayerStates]` | 10 player state arrays (see §3.1) |
| `events` | `Events` | Timeline events (see §3.2) |

### 3.1 Player states (columnar)

Each player entry contains parallel arrays, all of length **N** (same as `ticks`):

```json
{
  "x":        [100.0, 100.5, 101.0, ...],
  "y":        [200.0, 199.8, 199.5, ...],
  "z":        [0.0, 1.0, 1.5, ...],
  "yaw":      [45.0, 46.0, 47.5, ...],
  "pitch":    [-2.0, -1.5, -1.0, ...],
  "v":        [0.0, 200.0, 195.0, ...],
  "vx":       [0.0, 100.0, 98.0, ...],
  "vy":       [0.0, 173.0, 168.0, ...],
  "vz":       [0.0, 0.0, 0.0, ...],
  "hp":       [100, 100, 95, ...],
  "armor":    [100, 100, 100, ...],
  "helmet":   [true, true, true, ...],
  "defuser":  [false, false, false, ...],
  "alive":    [true, true, true, ...],
  "weapon":   [7, 7, 7, ...],
  "inventory":[[7, 4, 8], [7, 4, 8], ...],
  "flash":    [0.0, 0.0, 0.0, ...],
  "flash_alpha": [0.0, 0.0, 255.0, ...],
  "place":    [0, 0, 0, ...],
  "spotted":  [[], [5], [5, 6], ...],
  "shots":     [0, 1, 0, 0, 2, ...],
  "footsteps": [0, 0, 3, 1, 0, ...]
}
```

| Field | Type | Description |
|---|---|---|
| `x`, `y`, `z` | `[float]` | World-space position |
| `yaw`, `pitch` | `[float]` | View angles in degrees |
| `v` | `[float]` | Velocity magnitude (units/s) |
| `vx`, `vy`, `vz` | `[float]` | Velocity components (NaN → 0.0) |
| `hp` | `[int]` | Health (0–100) |
| `armor` | `[int]` | Armor value (0–100) |
| `helmet` | `[bool]` | Has helmet |
| `defuser` | `[bool]` | Has defuse kit (CT only) |
| `alive` | `[bool]` | Is alive |
| `weapon` | `[int]` | Active weapon index into `weapons` lookup |
| `inventory` | `[[int]]` | All carried items as weapon indices |
| `flash` | `[float]` | Remaining flash duration (seconds) |
| `flash_alpha` | `[float]` | Maximum flash alpha (0–255) |
| `place` | `[int]` | Location name index into `places` lookup |
| `spotted` | `[[int]]` | Enemy player indices who can see this player |
| `shots` | `[int]` | Number of times this player fired between previous sample and this sample |
| `footsteps` | `[int]` | Number of footstep sounds by this player between previous sample and this sample |

> **Note**: When a player is dead (`alive: false`), position/angle values
> reflect their death location and do not change.  Always check `alive` before
> using player state data.

### 3.2 Events

```json
{
  "kills": [
    {
      "t": 15000, "a": 0, "v": 5, "as": -1,
      "w": 7, "hs": true, "ts": false,
      "ab": false, "ai": false, "af": false,
      "dmg": 98
    }
  ],
  "damage": [
    {"t": 12000, "a": 0, "v": 5, "hp": 25, "w": 7}
  ],
  "bomb": [
    {"t": 20000, "e": "planted", "s": 45.2},
    {"t": 25000, "e": "exploded", "s": 85.0}
  ],
  "grenades": [
    {"t": 8000, "ty": "CSmokeGrenadeProjectile", "th": 3, "x": 100.0, "y": 200.0, "z": 50.0}
  ],
  "smokes": [
    {"ts": 8.5, "te": 26.5, "x": 100.0, "y": 200.0, "z": 0.0}
  ],
  "infernos": [
    {"ts": 12.0, "te": null, "x": 200.0, "y": 300.0, "z": 0.0}
  ],
  "sound": [
    {"t": 8000, "ty": "fire", "p": 0, "w": 7, "sil": false},
    {"t": 8100, "ty": "footstep", "p": 5}
  ]
}
```

#### Kills

| Field | Type | Description |
|---|---|---|
| `t` | `int` | Tick when the kill occurred |
| `a` | `int` | Attacker player index |
| `v` | `int` | Victim player index |
| `as` | `int` | Assister player index (`-1` if none) |
| `w` | `int` | Weapon index |
| `hs` | `bool` | Headshot |
| `ts` | `bool` | Through smoke |
| `ab` | `bool` | Attacker was blind (flashed) |
| `ai` | `bool` | Attacker was in air |
| `af` | `bool` | Flash assist |
| `dmg` | `int` | Damage dealt by killing blow |

#### Damage

| Field | Type | Description |
|---|---|---|
| `t` | `int` | Tick |
| `a` | `int` | Attacker index |
| `v` | `int` | Victim index |
| `hp` | `int` | HP damage dealt |
| `w` | `int` | Weapon index |

#### Bomb

| Field | Type | Description |
|---|---|---|
| `t` | `int` | Tick |
| `e` | `string` | Event type: `"planted"`, `"exploded"`, `"defused"` |
| `s` | `float` | Round-relative seconds |

#### Grenades (in-flight entities)

| Field | Type | Description |
|---|---|---|
| `t` | `int` | Tick |
| `ty` | `string` | Grenade type: `"CSmokeGrenadeProjectile"`, `"CFlashbangProjectile"`, `"CHEGrenadeProjectile"`, `"CMolotovProjectile"` (includes incendiaries), `"CDecoyProjectile"` |
| `th` | `int` | Thrower player index |
| `x`, `y`, `z` | `float` | Current position |

> **Note**: These are the raw C++ entity class names from the game engine.
> `CMolotovProjectile` covers both Molotov (T-side) and Incendiary (CT-side).
> Injury may still be active for a few seconds after the detonation event.

#### Sound

| Field | Type | Description |
|---|---|---|
| `t` | `int` | Tick when the sound event occurred |
| `ty` | `string` | `"fire"` or `"footstep"` |
| `p` | `int` | Player index who made the sound |
| `w` | `int` | Weapon index (fire only) |
| `sil` | `bool` | Whether the shot was silenced (fire only) |

#### Smokes / Infernos (active on ground)

| Field | Type | Description |
|---|---|---|
| `ts` | `float` | Round-relative start time |
| `te` | `float \| null` | Round-relative end time (`null` = still active at round end) |
| `x`, `y`, `z` | `float` | Position on ground |

---

## 4. Usage Examples

### Command line

```bash
# Basic conversion (0.5s interval)
python scripts/demo_to_json.py -d match.dem -o match.json -v

# Fine-grained for ML training (0.25s interval, compressed)
python scripts/demo_to_json.py -d match.dem -o match.json.gz -i 0.25 -z -v

# Compact (no indentation) for minimal file size
python scripts/demo_to_json.py -d match.dem -o match.json --compact
```

### Python API

```python
from demo_parser import parse_demo, save_demo_json

# Parse a demo
data = parse_demo("match.dem", interval=0.5, verbose=True)

# Access data
print(data["map"])                          # "de_mirage"
print(data["players"][0]["name"])           # "s1mple"
print(data["weapons"])                      # {"ak47": 0, "m4a1": 1, ...}
print(len(data["rounds"]))                  # 30

# Round data
round_0 = data["rounds"][0]
print(round_0["winner"])                    # "CT"
print(round_0["teams"][0])                  # "CT" (or "T" after halftime round 12)
ticks = round_0["ticks"]                    # [5000, 5016, 5032, ...]

# Player trajectories (player index 0, round 0)
p0 = round_0["players"][0]
xs = p0["x"]                                # all x positions
ys = p0["y"]
zs = p0["z"]
alive = p0["alive"]

# Events timeline
for kill in round_0["events"]["kills"]:
    print(f"Player {kill['a']} killed {kill['v']} at tick {kill['t']}")

# Smoke coverage
for smoke in round_0["events"]["smokes"]:
    print(f"Smoke from {smoke['ts']}s to {smoke['te']}s at ({smoke['x']}, {smoke['y']})")

# Save to file
save_demo_json(data, "output.json", compact=False, compress=True)
```

### Reconstruct "future from tick T"

In V1, `future_kills` told you which kills occur after each tick.  In V2,
compute it from the timeline:

```python
def future_events(tick: int, events: list[dict]) -> list[dict]:
    """Return all events that occur after `tick`."""
    return [e for e in events if e["t"] > tick]

# Find all future kills from tick 15000
future = future_events(15000, round_0["events"]["kills"])
```

### Check which smokes are active at a given time

```python
def active_smokes(round_seconds: float, smokes: list[dict]) -> list[dict]:
    """Return smokes active at a given round-relative time."""
    return [
        s for s in smokes
        if s["ts"] <= round_seconds and (s["te"] is None or s["te"] >= round_seconds)
    ]

active = active_smokes(15.0, round_0["events"]["smokes"])
```

---

## 5. V1 ↔ V2 comparison

### V1 (old format)

```json
[
  {
    "round": 1,
    "tick": 5000,
    "round_label": {"round_info": {"winner": "CT", "reason": "elimination"}},
    "map_name": "de_mirage",
    "round_seconds": 0.5,
    "is_bomb_planted": false,
    "is_bomb_dropped": false,
    "bomb_planted_time": null,
    "bomb_planted_duration": null,
    "bomb_position": [100.0, 200.0, 0.0],
    "entity_grenades": [],
    "players_info": [
      {
        "steamid": "76561198000000001",
        "name": "s1mple",
        "X": 100.0, "Y": 200.0, "Z": 0.0,
        "last_place_name": "A Site",
        "weapon_name": "ak47",
        "inventory": ["ak47", "deagle", "flashbang", "smokegrenade"],
        "inventory_as_ids": [7, 1, 43, 45],
        "pitch": -2.0, "yaw": 45.0,
        "is_alive": true, "health": 100,
        "flash_duration": 0.0, "flash_max_alpha": 0.0,
        "armor": 100, "has_helmet": true, "has_defuser": false,
        "team_num": "CT",
        "spotted_by": [],
        "velocity": 0.0,
        "velocity_X": 0.0, "velocity_Y": 0.0, "velocity_Z": 0.0
      },
      // ... 9 more players
    ],
    "projectiles": [],
    "future_kills": [
      {
        "attacker_name": "s1mple", "attacker_steamid": "76561198000000001",
        "victim_name": "NiKo", "victim_steamid": "76561198000000002",
        "weapon": "ak47", "headshot": true, "dmg_health": 98,
        "time": 15.0
      }
    ],
    "future_damage": [
      {
        "attacker_name": "s1mple", "attacker_steamid": "76561198000000001",
        "victim_name": "NiKo", "victim_steamid": "76561198000000002",
        "dmg_health": 25, "weapon": "ak47", "time": 8.0
      }
    ]
  },
  // ... next tick (repeats all above structure)
]
```

### V2 (new format)

```json
{
  "format": "cs2.demo.v2",
  "map": "de_mirage",
  "players": [
    {"steamid": "76561198000000001", "name": "s1mple"},
    {"steamid": "76561198000000002", "name": "NiKo"}
    // ... 8 more
  ],
  "weapons": {"ak47": 0, "deagle": 1, "flashbang": 2, "smokegrenade": 3},
  "places": {"A Site": 0, "B Site": 1, "Mid": 2},
  "rounds": [
    {
      "id": 0,
      "teams": ["CT", "T", "CT", "T", "CT", "T", "T", "T", "CT", "CT"],
      "winner": "CT",
      "end_reason": "t_killed",
      "bomb_planted_time": null,
      "ticks": [5000, 5016, 5032],
      "round_seconds": [0.5, 1.0, 1.5],
      "bomb_planted": [false, false, false],
      "bomb_dropped": [false, false, false],
      "bomb_position": [[100,200,0], [100,200,0], [100,200,0]],
      "players": [
        {
          "x": [100.0, 100.2, 100.5],
          "y": [200.0, 199.8, 199.5],
          "z": [0.0, 0.0, 0.0],
          "yaw": [45.0, 46.0, 47.0],
          "pitch": [-2.0, -1.5, -1.0],
          "v": [0.0, 200.0, 195.0],
          "vx": [0.0, 100.0, 98.0],
          "vy": [0.0, 173.0, 168.0],
          "vz": [0.0, 0.0, 0.0],
          "hp": [100, 100, 100],
          "armor": [100, 100, 100],
          "helmet": [true, true, true],
          "defuser": [false, false, false],
          "alive": [true, true, true],
          "weapon": [0, 0, 0],
          "inventory": [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]],
          "flash": [0.0, 0.0, 0.0],
          "flash_alpha": [0.0, 0.0, 0.0],
          "place": [0, 0, 0],
          "spotted": [[], [5], [5]],
          "shots": [0, 1, 0],
          "footsteps": [0, 0, 0]
        }
        // ... 9 more players
      ],
      "events": {
        "kills": [
          {"t": 15000, "a": 0, "v": 5, "as": -1, "w": 0, "hs": true, "ts": false,
           "ab": false, "ai": false, "af": false, "dmg": 98}
        ],
        "damage": [
          {"t": 12000, "a": 0, "v": 5, "hp": 25, "w": 0}
        ],
        "bomb": [],
        "grenades": [
          {"t": 8000, "ty": "CSmokeGrenadeProjectile", "th": 3, "x": 100.0, "y": 200.0, "z": 50.0}
        ],
        "smokes": [],
        "infernos": [],
        "sound": []
      }
    }
  ]
}
```

---

## 6. Notes

- **Tick alignment**: All per-player arrays within a round have identical length **N**
  and are aligned with `round.ticks` and `round.round_seconds`.
- **Player ordering**: The `players` array in the header and the `players` arrays
  within each round share the same index ordering.  `round.players[i]` corresponds
  to `data["players"][i]`.
- **Missing values**: NaN velocity components are replaced with `0.0`.  Dead
  players retain their last position.
- **CS2 tick rate**: The game server runs at 64 ticks/second.  The `interval`
  parameter controls the sampling density — `0.5` gives ~128 samples/minute,
  `0.25` gives ~256 samples/minute.
- **World damage**: Fall damage and other non-player sources produce damage
  events with `a = -1` (no attacker) and `w = -1` (no weapon).
- **Negative timestamps**: Smoke and inferno intervals may have negative `ts`
  values when a grenade was thrown before the round freeze_end event (e.g. in
  warmup).  Consumers should handle `ts < 0` gracefully.
- **Half-time team swap**: Team assignments change at round 12.  Always read
  `round.teams[i]` rather than caching team from a previous round.
