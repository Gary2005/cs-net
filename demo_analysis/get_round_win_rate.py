import argparse
import json
import math
import os
from copy import deepcopy
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

from data.process_demo import get_important_ticks_by_round
from demoparser2 import DemoParser
from demoparser_utils.state_extract import extract_states_by_group
from models.model3_space_only import CSModelV3


SPACE_SIZE = 31
N_PLAYERS = 10

MAP_CONFIG = {
    "maps": {
        "de_mirage": {
            "center": [-605.8900146484375, -866.8900146484375, -171.6199951171875],
        },
        "de_dust2": {
            "center": [-199.0, 977.0, 32.220001220703125],
        },
        "de_inferno": {
            "center": [481.07000732421875, 1396.47998046875, 137.91000366210938],
        },
        "de_nuke": {
            "center": [265.9599914550781, -772.5, -381.8999938964844],
        },
        "de_overpass": {
            "center": [-2027.3900146484375, -812.9000244140625, 324.95001220703125],
        },
        "de_ancient": {
            "center": [-435.5, -348.0, 43.650001525878906],
        },
        "de_anubis": {
            "center": [-77.38999938964844, 618.9000244140625, -6.800000190734863],
        },
        "de_train": {
            "center": [-118.25, -2.0, -128.52000427246094],
        },
    },
    "ranges": {
        "x": [-5000, 5000],
        "y": [-5000, 5000],
        "z": [-2000, 2000],
    },
}

MAP_NAME_TO_IDX = {map_name: idx for idx, map_name in enumerate(MAP_CONFIG["maps"].keys())}


def clip_and_scale(value, range_vals):
    min_val, max_val = range_vals
    if value < min_val:
        value = min_val
    elif value > max_val:
        value = max_val
    return value / max(abs(min_val), abs(max_val))


def load_tokenizer_cfg():
    with open("demoparser_utils/tokenizer.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_weapon_index(tokenizer_cfg):
    weapons = tokenizer_cfg.get("weapons", [])
    return {name: i for i, name in enumerate(weapons)}


def build_projectile_index(tokenizer_cfg):
    projectiles = tokenizer_cfg.get("projectiles", [])
    return {name: i for i, name in enumerate(projectiles)}


def find_checkpoint(folder: str) -> str:
    for name in sorted(os.listdir(folder)):
        if name.endswith((".pth", ".pt")):
            return os.path.join(folder, name)
    raise RuntimeError(f"No checkpoint found in {folder}")


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def find_head_yaml(head_dir: str) -> str:
    last = os.path.basename(os.path.normpath(head_dir))
    if last == "alive":
        return "config/model3_alive_space_only.yaml"
    if last == "duel":
        return "config/model3_duel_space_only.yaml"
    if last == "nxt_death":
        return "config/model3_death_space_only.yaml"
    if last == "nxt_kill":
        return "config/model3_kill_space_only.yaml"
    if last == "win_rate":
        return "config/model3_win_space_only.yaml"
    raise RuntimeError(f"No config mapping for head dir: {head_dir}")


def load_model_and_cfg(head_dir: str, device: torch.device):
    cfg_path = find_head_yaml(head_dir)
    cfg = load_yaml(cfg_path)
    model = CSModelV3(cfg).to(device)
    ckpt = find_checkpoint(head_dir)
    state = torch.load(ckpt, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    model.eval()
    return model, cfg, ckpt


def apply_temperature_scaling(logits: torch.Tensor, cfg: dict):
    calib = (cfg or {}).get("calibration", {}).get("temperature_scaling")
    if not calib:
        return logits
    temperature = float(calib.get("temperature", 1.0)) or 1.0
    scaled = logits / temperature
    bias = calib.get("bias")
    if bias is None:
        return scaled
    if isinstance(bias, list):
        b = torch.tensor(bias, dtype=scaled.dtype, device=scaled.device)
        return scaled + b.unsqueeze(0)
    return scaled + float(bias)


def process_xyz(x, y, z, map_name):
    center = MAP_CONFIG["maps"][map_name]["center"]
    x -= center[0]
    y -= center[1]
    z -= center[2]
    return [
        clip_and_scale(x, MAP_CONFIG["ranges"]["x"]),
        clip_and_scale(y, MAP_CONFIG["ranges"]["y"]),
        clip_and_scale(z, MAP_CONFIG["ranges"]["z"]),
    ]


def build_tick_features(state, weapon2idx, projectile2idx):
    map_name = state.get("map_name")
    if map_name not in MAP_NAME_TO_IDX:
        raise ValueError(f"Unknown map: {map_name}")

    mlp1_f = [[0.0, 0.0, 0.0] for _ in range(SPACE_SIZE)]
    mlp1_i = [0 for _ in range(SPACE_SIZE)]
    mlp1_mask = [False for _ in range(SPACE_SIZE)]

    mlp2_f = [[0.0 for _ in range(14)] for _ in range(SPACE_SIZE)]
    mlp2_mask = [False for _ in range(SPACE_SIZE)]

    mlp3_f = [[0.0] for _ in range(SPACE_SIZE)]
    mlp3_i = [0 for _ in range(SPACE_SIZE)]
    mlp3_mask = [False for _ in range(SPACE_SIZE)]

    mlp4_f = [[0.0 for _ in range(4)] for _ in range(SPACE_SIZE)]
    mlp4_mask = [False for _ in range(SPACE_SIZE)]

    mlp5_f = [[[0.0 for _ in range(13)] for _ in range(9)] for _ in range(SPACE_SIZE)]
    mlp5_i = [[0 for _ in range(9)] for _ in range(SPACE_SIZE)]
    mlp5_mask = [[False for _ in range(9)] for _ in range(SPACE_SIZE)]

    emb1_i = [[0 for _ in range(9)] for _ in range(SPACE_SIZE)]
    emb1_mask = [[False for _ in range(9)] for _ in range(SPACE_SIZE)]

    emb2_i = [0 for _ in range(SPACE_SIZE)]
    emb2_mask = [False for _ in range(SPACE_SIZE)]

    dead_mask = [False for _ in range(SPACE_SIZE)]
    pad_mask = [False for _ in range(SPACE_SIZE)]

    players_info = state.get("players_info", [])
    for idx, player in enumerate(players_info[:N_PLAYERS]):
        if not player.get("is_alive", False):
            dead_mask[idx] = True
            continue

        mlp1_f[idx] = process_xyz(player["X"], player["Y"], player["Z"], map_name)
        mlp1_i[idx] = MAP_NAME_TO_IDX[map_name]
        mlp1_mask[idx] = True

        mlp2_f[idx] = [
            float(player.get("armor", 0) > 0),
            float(player.get("has_helmet", False)),
            float(player.get("has_defuser", False)),
            float(player.get("flash_duration", 0) > 0),
            math.cos(math.radians(float(player.get("pitch", 0.0) or 0.0))),
            math.sin(math.radians(float(player.get("pitch", 0.0) or 0.0))),
            math.cos(math.radians(float(player.get("yaw", 0.0) or 0.0))),
            math.sin(math.radians(float(player.get("yaw", 0.0) or 0.0))),
            float(player.get("health", 0)) / 100.0,
            float(player.get("team_num") == "CT"),
            clip_and_scale(float(player.get("velocity", 0.0) or 0.0), [0, 8000]),
            clip_and_scale(float(player.get("velocity_X", 0.0) or 0.0), [-8000, 8000]),
            clip_and_scale(float(player.get("velocity_Y", 0.0) or 0.0), [-8000, 8000]),
            clip_and_scale(float(player.get("velocity_Z", 0.0) or 0.0), [-1000, 1000]),
        ]
        mlp2_mask[idx] = True

        inventory = player.get("inventory", []) or []
        inv_ids = [weapon2idx.get(w, weapon2idx.get("knife", 0)) for w in inventory]
        if len(inv_ids) > 9:
            inv_ids = inv_ids[:9]
        emb1_i[idx] = inv_ids + [0] * (9 - len(inv_ids))
        emb1_mask[idx] = [True] * len(inv_ids) + [False] * (9 - len(inv_ids))

    bomb_pos = state.get("bomb_position")
    if bomb_pos is not None:
        mlp1_f[10] = process_xyz(bomb_pos[0], bomb_pos[1], bomb_pos[2], map_name)
        mlp1_i[10] = MAP_NAME_TO_IDX[map_name]
        mlp1_mask[10] = True

        emb2_i[10] = MAP_NAME_TO_IDX[map_name]
        emb2_mask[10] = True

        planted = bool(state.get("is_bomb_planted", False))
        dropped = bool(state.get("is_bomb_dropped", False))
        planted_duration = state.get("bomb_planted_duration")
        if not planted:
            planted_duration = 0.0
        if planted_duration is None:
            planted_duration = 0.0

        mlp4_f[10] = [
            float(state.get("round_seconds", 0.0)) / 160.0,
            float(planted),
            float(dropped),
            float(planted_duration) / 40.0,
        ]
        mlp4_mask[10] = True

    projectiles = list(state.get("projectiles", []) or [])
    max_projectiles = SPACE_SIZE - 11
    if len(projectiles) > max_projectiles:
        projectiles = projectiles[:max_projectiles]

    for idx, proj in enumerate(projectiles):
        token_idx = idx + 11
        proj_type = proj.get("type")
        if proj_type not in projectile2idx:
            continue

        pos = proj.get("position") or [0.0, 0.0, 0.0]
        mlp1_f[token_idx] = process_xyz(pos[0], pos[1], pos[2], map_name)
        mlp1_i[token_idx] = MAP_NAME_TO_IDX[map_name]
        mlp1_mask[token_idx] = True

        duration = float(proj.get("duration", 0.0) or 0.0)
        mlp3_f[token_idx] = [duration / 25.0]
        mlp3_i[token_idx] = projectile2idx[proj_type]
        mlp3_mask[token_idx] = True

    num_tokens = 11 + len(projectiles)
    for i in range(num_tokens, SPACE_SIZE):
        pad_mask[i] = True

    for idx, player in enumerate(players_info[:N_PLAYERS]):
        if not player.get("is_alive", False):
            continue

        mlp5_features = []
        mlp5_index = []

        for idx2, other in enumerate(players_info[:N_PLAYERS]):
            if idx == idx2:
                continue
            if not other.get("is_alive", False):
                continue

            dx = other["X"] - player["X"]
            dy = other["Y"] - player["Y"]
            dz = other["Z"] - player["Z"]
            dist = (dx * dx + dy * dy + dz * dz) ** 0.5
            j_is_teammate = player.get("team_num") == other.get("team_num")
            j_is_enemy = not j_is_teammate

            yaw_deg = float(player.get("yaw", 0.0) or 0.0)
            pitch_deg = float(player.get("pitch", 0.0) or 0.0)
            yaw_rad = math.radians(yaw_deg)
            pitch_rad = math.radians(pitch_deg)

            xy_norm = math.hypot(dx, dy)
            if xy_norm > 0:
                fwd_x = math.cos(yaw_rad)
                fwd_y = math.sin(yaw_rad)
                dot_xy = (dx * fwd_x + dy * fwd_y) / xy_norm
                dot_xy = max(-1.0, min(1.0, dot_xy))
                d_theta_xy = math.degrees(math.acos(dot_xy))
            else:
                d_theta_xy = 0.0

            xy_plane_dist = math.hypot(dx, dy)
            if xy_plane_dist > 0:
                target_pitch = math.atan2(dz, xy_plane_dist)
            else:
                target_pitch = math.pi / 2 if dz > 0 else (-math.pi / 2 if dz < 0 else 0.0)
            d_theta_z = abs(math.degrees(pitch_rad - target_pitch))

            rel = [
                clip_and_scale(dx, MAP_CONFIG["ranges"]["x"]),
                clip_and_scale(dy, MAP_CONFIG["ranges"]["y"]),
                clip_and_scale(dz, MAP_CONFIG["ranges"]["z"]),
                math.log(clip_and_scale(dist, [0, 5000]) + 1.0),
                float(j_is_teammate),
                float(j_is_enemy),
                0.0,
                float(j_is_enemy and (player.get("steamid") in (other.get("spotted_by") or []))),
                float(j_is_enemy and (other.get("steamid") in (player.get("spotted_by") or []))),
                math.cos(math.radians(d_theta_xy)),
                math.sin(math.radians(d_theta_xy)),
                math.cos(math.radians(d_theta_z)),
                math.sin(math.radians(d_theta_z)),
            ]

            mlp5_features.append(rel)
            mlp5_index.append(idx2)

        if len(mlp5_features) > 9:
            mlp5_features = mlp5_features[:9]
            mlp5_index = mlp5_index[:9]

        mlp5_f[idx] = mlp5_features + [[0.0 for _ in range(13)] for _ in range(9 - len(mlp5_features))]
        mlp5_i[idx] = mlp5_index + [0 for _ in range(9 - len(mlp5_index))]
        mlp5_mask[idx] = [True] * len(mlp5_features) + [False] * (9 - len(mlp5_features))

    return {
        "mlp1_f": mlp1_f,
        "mlp1_i": mlp1_i,
        "mlp1_mask": mlp1_mask,
        "mlp2_f": mlp2_f,
        "mlp2_mask": mlp2_mask,
        "mlp3_f": mlp3_f,
        "mlp3_i": mlp3_i,
        "mlp3_mask": mlp3_mask,
        "mlp4_f": mlp4_f,
        "mlp4_mask": mlp4_mask,
        "mlp5_f": mlp5_f,
        "mlp5_i": mlp5_i,
        "mlp5_mask": mlp5_mask,
        "emb1_i": emb1_i,
        "emb1_mask": emb1_mask,
        "emb2_i": emb2_i,
        "emb2_mask": emb2_mask,
        "dead_mask": dead_mask,
        "pad_mask": pad_mask,
    }


def build_batch(round_states, weapon2idx, projectile2idx, device):
    samples = [build_tick_features(s, weapon2idx, projectile2idx) for s in round_states]

    def stack(key, dtype, is_bool=False):
        data = [s[key] for s in samples]
        tensor = torch.tensor(data, dtype=dtype)
        if is_bool:
            tensor = tensor.bool()
        return tensor.to(device)

    return {
        "mlp1_f": stack("mlp1_f", torch.float32),
        "mlp1_i": stack("mlp1_i", torch.long),
        "mlp1_mask": stack("mlp1_mask", torch.bool, is_bool=True),
        "mlp2_f": stack("mlp2_f", torch.float32),
        "mlp2_mask": stack("mlp2_mask", torch.bool, is_bool=True),
        "mlp3_f": stack("mlp3_f", torch.float32),
        "mlp3_i": stack("mlp3_i", torch.long),
        "mlp3_mask": stack("mlp3_mask", torch.bool, is_bool=True),
        "mlp4_f": stack("mlp4_f", torch.float32),
        "mlp4_mask": stack("mlp4_mask", torch.bool, is_bool=True),
        "mlp5_f": stack("mlp5_f", torch.float32),
        "mlp5_i": stack("mlp5_i", torch.long),
        "mlp5_mask": stack("mlp5_mask", torch.bool, is_bool=True),
        "emb1_i": stack("emb1_i", torch.long),
        "emb1_mask": stack("emb1_mask", torch.bool, is_bool=True),
        "emb2_i": stack("emb2_i", torch.long),
        "emb2_mask": stack("emb2_mask", torch.bool, is_bool=True),
        "dead_mask": stack("dead_mask", torch.bool, is_bool=True),
        "pad_mask": stack("pad_mask", torch.bool, is_bool=True),
    }


def compute_duel_matrix(model, cfg, batch, alive_mask, team_ct, team_t):
    if not team_ct or not team_t:
        return None

    with torch.no_grad():
        x = model.encode_tick(batch)
        x = model.space_tf(x, batch["pad_mask"], batch["dead_mask"])

        pairs = [(ct, t) for ct in team_ct for t in team_t]
        a_idx = torch.tensor([p[0] for p in pairs], device=x.device)
        b_idx = torch.tensor([p[1] for p in pairs], device=x.device)

        token_a = x[:, a_idx, :]
        token_b = x[:, b_idx, :]
        logits = model.head(torch.cat([token_a, token_b], dim=-1)).squeeze(-1)
        logits = apply_temperature_scaling(logits, cfg)
        probs = torch.sigmoid(logits)

        duel = torch.full((x.shape[0], N_PLAYERS, N_PLAYERS), 0.5, device=x.device)
        for i, (a, b) in enumerate(pairs):
            duel[:, a, b] = probs[:, i]
            duel[:, b, a] = 1.0 - probs[:, i]

    alive_mask = alive_mask.to(duel.device)
    for i in range(N_PLAYERS):
        for j in range(N_PLAYERS):
            invalid = (~alive_mask[:, i]) | (~alive_mask[:, j])
            if invalid.any():
                duel[invalid, i, j] = 0.5

    return duel


def run_round_inference(round_states, models, cfgs, weapon2idx, projectile2idx, device, batch_size):
    batch = build_batch(round_states, weapon2idx, projectile2idx, device)
    total = batch["mlp1_f"].shape[0]

    team_ct = [i for i, p in enumerate(round_states[0]["players_info"][:N_PLAYERS]) if p.get("team_num") == "CT"]
    team_t = [i for i, p in enumerate(round_states[0]["players_info"][:N_PLAYERS]) if p.get("team_num") == "T"]

    results = []
    for start in range(0, total, batch_size):
        end = min(total, start + batch_size)
        sub = {k: v[start:end] for k, v in batch.items()}
        B = end - start

        alive_mask = torch.zeros((B, N_PLAYERS), dtype=torch.bool, device=device)
        for i in range(B):
            players = round_states[start + i].get("players_info", [])
            for j in range(min(N_PLAYERS, len(players))):
                alive_mask[i, j] = bool(players[j].get("is_alive", False))

        with torch.no_grad():
            win_logits, _ = models["win"]({**sub, "label": torch.zeros(B, device=device)})
            win_logits = apply_temperature_scaling(win_logits, cfgs["win"])
            win_probs = torch.sigmoid(win_logits).cpu().tolist()

            if models.get("alive") is not None:
                alive_logits, _ = models["alive"]({**sub, "alive_in_the_end": torch.zeros((B, N_PLAYERS), device=device)})
                alive_logits = apply_temperature_scaling(alive_logits, cfgs["alive"])
                alive_probs = torch.sigmoid(alive_logits).cpu().tolist()
            else:
                alive_probs = None

            if models.get("kill") is not None:
                kill_logits, _ = models["kill"]({**sub, "nxt_kill": torch.zeros(B, dtype=torch.long, device=device)})
                kill_logits = apply_temperature_scaling(kill_logits, cfgs["kill"])
                kill_probs = torch.softmax(kill_logits, dim=-1).cpu().tolist()
            else:
                kill_probs = None

            if models.get("death") is not None:
                death_logits, _ = models["death"]({**sub, "nxt_death": torch.zeros(B, dtype=torch.long, device=device)})
                death_logits = apply_temperature_scaling(death_logits, cfgs["death"])
                death_probs = torch.softmax(death_logits, dim=-1).cpu().tolist()
            else:
                death_probs = None

            if models.get("duel") is not None:
                duel_probs = compute_duel_matrix(models["duel"], cfgs["duel"], sub, alive_mask, team_ct, team_t)
                duel_probs = duel_probs.cpu().tolist() if duel_probs is not None else None
            else:
                duel_probs = None

        for i in range(B):
            state = round_states[start + i]
            state["ct_win_rate"] = float(win_probs[i])
            state["alive_pred"] = (
                alive_probs[i]
                if alive_probs is not None
                else [0.5 if alive_mask[i, j].item() else 0.0 for j in range(N_PLAYERS)]
            )
            state["next_kill"] = (
                kill_probs[i] if kill_probs is not None else [1.0 / 11.0] * 11
            )
            state["next_death"] = (
                death_probs[i] if death_probs is not None else [1.0 / 11.0] * 11
            )
            state["duel"] = duel_probs[i] if duel_probs is not None else None

    return round_states


def _safe_div(a, b):
    return a / b if b else 0.0


def compute_kill_difficulty(round_result, kill_time, attacker_idx, victim_idx, window_s=5.0):
    if attacker_idx is None or victim_idx is None:
        return 0.0
    if attacker_idx >= 10 or victim_idx >= 10:
        return 0.0

    a_kill = a_death = v_kill = v_death = 0.0
    n = 0
    for state in round_result:
        sec = float(state.get("round_seconds", 0.0))
        if sec > kill_time:
            continue
        if sec < kill_time - window_s:
            continue
        nk = state.get("next_kill") or []
        nd = state.get("next_death") or []
        if attacker_idx < len(nk):
            a_kill += float(nk[attacker_idx])
        if attacker_idx < len(nd):
            a_death += float(nd[attacker_idx])
        if victim_idx < len(nk):
            v_kill += float(nk[victim_idx])
        if victim_idx < len(nd):
            v_death += float(nd[victim_idx])
        n += 1

    if n == 0:
        return 0.0
    avg_a_kill = a_kill / n
    avg_a_death = a_death / n
    avg_v_kill = v_kill / n
    avg_v_death = v_death / n
    num = avg_v_kill * avg_a_death
    den = avg_a_kill * avg_v_death
    return _safe_div(num, den)


def process_round_json(round_result):
    processed_json = {}

    name_to_idx = {}
    for idx, p in enumerate(round_result[0].get("players_info", [])):
        name = p.get("name")
        if name is not None and idx < 10:
            name_to_idx[name] = idx

    win_rate = []
    for state in round_result:
        win_rate.append({
            "round_seconds": state["round_seconds"],
            "ct_win_rate": state["ct_win_rate"],
        })
    winner = round_result[-1]["round_label"]["round_info"]["winner"]
    if winner not in ["CT", "T"]:
        return {"error": f"Invalid winner: {winner}"}
    last_second = round_result[-1]["round_seconds"]
    win_rate.append({
        "round_seconds": last_second + 0.25,
        "ct_win_rate": 1.0 if winner == "CT" else 0.0,
    })

    processed_json["win_rate"] = win_rate

    all_kills = round_result[0]["future_kills"]

    team_ct = [p["name"] for p in round_result[0]["players_info"] if p["team_num"] == "CT"]
    team_t = [p["name"] for p in round_result[0]["players_info"] if p["team_num"] == "T"]

    start_inventory = [
        {
            "player": p.get("name", "Unknown"),
            "team_num": p.get("team_num", "Unknown"),
            "inventory": p.get("inventory", []),
            "armor": p.get("armor"),
            "has_helmet": p.get("has_helmet"),
            "has_defuser": p.get("has_defuser"),
        }
        for p in round_result[0]["players_info"]
    ]

    player_data = {n: {"kill_contribution": 0, "tactical_contribution": 0}
                   for n in team_ct + team_t}

    processed_json["kills"] = []
    processed_json["player_data"] = [deepcopy(player_data)]

    for i in range(len(win_rate) - 1):
        d_win = win_rate[i + 1]["ct_win_rate"] - win_rate[i]["ct_win_rate"]
        kill_count = 0

        alive_players = [
            p["name"] for p in round_result[i]["players_info"] if p["is_alive"]
        ]

        window_kills = [
            k for k in all_kills
            if win_rate[i]["round_seconds"] <= k["time"] < win_rate[i + 1]["round_seconds"]
            and k["attacker_name"] in player_data
            and k["victim_name"] in player_data
        ]
        kill_count = len(window_kills)

        if kill_count > 0:
            for kill in window_kills:
                killer = kill["attacker_name"]
                victim = kill["victim_name"]
                team_killer = "CT" if killer in team_ct else "T"
                team_victim = "CT" if victim in team_ct else "T"

                kill_impact = 0.0
                if team_killer == "CT":
                    if team_victim == "T":
                        player_data[killer]["kill_contribution"] += d_win / kill_count
                        player_data[victim]["kill_contribution"] -= d_win / kill_count
                        kill_impact = d_win / kill_count
                    else:
                        player_data[killer]["kill_contribution"] += d_win / (kill_count * 2)
                        player_data[victim]["kill_contribution"] += d_win / (kill_count * 2)
                        kill_impact = d_win / kill_count
                        alive_t = [p for p in alive_players if p in team_t]
                        if alive_t:
                            for p in alive_t:
                                player_data[p]["tactical_contribution"] -= d_win / (kill_count * len(alive_t))
                else:
                    if team_victim == "CT":
                        player_data[killer]["kill_contribution"] -= d_win / kill_count
                        player_data[victim]["kill_contribution"] += d_win / kill_count
                        kill_impact = -d_win / kill_count
                    else:
                        player_data[killer]["kill_contribution"] -= d_win / (kill_count * 2)
                        player_data[victim]["kill_contribution"] -= d_win / (kill_count * 2)
                        kill_impact = -d_win / kill_count
                        alive_ct = [p for p in alive_players if p in team_ct]
                        if alive_ct:
                            for p in alive_ct:
                                player_data[p]["tactical_contribution"] += d_win / (kill_count * len(alive_ct))

                difficulty = compute_kill_difficulty(
                    round_result,
                    float(kill["time"]),
                    name_to_idx.get(killer),
                    name_to_idx.get(victim),
                )

                processed_json["kills"].append({
                    "killer": killer,
                    "assister": kill.get("assister_name"),
                    "victim": victim,
                    "round_seconds": kill["time"],
                    "kill_impact": kill_impact,
                    "weapon": kill["weapon"],
                    "headshot": bool(kill.get("headshot", False)),
                    "difficulty": difficulty,
                    "assistedflash": bool(kill.get("assistedflash", False)),
                    "attackerblind": bool(kill.get("attackerblind", False)),
                    "attackerinair": bool(kill.get("attackerinair", False)),
                    "thrusmoke": bool(kill.get("thrusmoke", False)),
                    "dmg_health": kill.get("dmg_health"),
                })
        else:
            alive_t = [p for p in alive_players if p in team_t]
            alive_ct = [p for p in alive_players if p in team_ct]
            if alive_ct:
                for p in alive_ct:
                    player_data[p]["tactical_contribution"] += d_win / len(alive_ct)
            if alive_t:
                for p in alive_t:
                    player_data[p]["tactical_contribution"] -= d_win / len(alive_t)

        processed_json["player_data"].append(deepcopy(player_data))

    processed_json["winner"] = winner
    processed_json["CT_players"] = team_ct
    processed_json["T_players"] = team_t
    processed_json["start_inventory"] = start_inventory

    return processed_json


def radar_player_view(players_info):
    out = []
    for p in players_info:
        out.append({
            "name": p.get("name"),
            "steamid": p.get("steamid"),
            "team_num": p.get("team_num"),
            "X": p.get("X"),
            "Y": p.get("Y"),
            "Z": p.get("Z"),
            "yaw": p.get("yaw"),
            "is_alive": bool(p.get("is_alive")),
            "health": p.get("health"),
            "weapon_name": p.get("weapon_name"),
            "last_place_name": p.get("last_place_name"),
            "inventory": p.get("inventory"),
            "armor": p.get("armor"),
            "has_helmet": p.get("has_helmet"),
            "has_defuser": p.get("has_defuser"),
            "flash_duration": p.get("flash_duration"),
            "flash_max_alpha": p.get("flash_max_alpha"),
        })
    return out


def build_round_ticks(round_states, duel_fallback):
    ticks = []
    for s in round_states:
        duel = s.get("duel")
        if duel is None:
            duel = duel_fallback

        players = s.get("players_info", [])
        alive = [bool(p.get("is_alive", False)) for p in players[:10]]
        if len(alive) < 10:
            alive.extend([False] * (10 - len(alive)))

        duel_marked = []
        for i in range(10):
            row = []
            for j in range(10):
                if (not alive[i]) or (not alive[j]):
                    row.append("/")
                else:
                    row.append(duel[i][j])
            duel_marked.append(row)

        ticks.append({
            "round_seconds": s.get("round_seconds"),
            "players_info": radar_player_view(s.get("players_info", [])),
            "ct_win_rate": s.get("ct_win_rate"),
            "alive_pred": s.get("alive_pred"),
            "next_kill": s.get("next_kill"),
            "next_death": s.get("next_death"),
            "duel": duel_marked,
            "is_bomb_planted": s.get("is_bomb_planted"),
            "is_bomb_dropped": s.get("is_bomb_dropped"),
            "bomb_planted_time": s.get("bomb_planted_time"),
            "bomb_planted_duration": s.get("bomb_planted_duration"),
            "bomb_position": s.get("bomb_position"),
            "projectiles": s.get("projectiles", []),
            "entity_grenades": s.get("entity_grenades", []),
        })
    return ticks


def resolve_head_dirs(model_root: Path):
    head_dirs = {
        "alive": model_root / "alive",
        "kill": model_root / "nxt_kill",
        "death": model_root / "nxt_death",
        "win": model_root / "win_rate",
        "duel": model_root / "duel",
    }

    if not head_dirs["win"].is_dir():
        raise RuntimeError("win_rate model dir is required under model_root")

    optional = [h for h, p in head_dirs.items() if h != "win" and not p.is_dir()]
    if optional:
        print(f"[warn] optional heads missing: {optional}")

    return {k: str(v) if v.is_dir() else None for k, v in head_dirs.items()}


def to_jsonable(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def main():
    parser = argparse.ArgumentParser(
        description="Batched multi-head inference: win rate + alive + next kill/death + duel"
    )
    parser.add_argument("--demo_path", required=True, help="Path to .dem file")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    parser.add_argument("--device", default="cuda", help="Device to run inference on")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--model_root",
        help="Root dir containing alive/ nxt_kill/ nxt_death/ win_rate/ duel/",
    )

    args = parser.parse_args()

    requested = args.device
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA not available, falling back to CPU")
        requested = "cpu"
    device = torch.device(requested)

    demo_path = Path(args.demo_path)
    output_path = Path(args.output)

    if not demo_path.exists():
        raise FileNotFoundError(f"Demo file not found: {demo_path}")
    if not args.model_root:
        raise RuntimeError("--model_root is required")

    model_root = Path(args.model_root)
    head_dirs = resolve_head_dirs(model_root)

    models = {}
    cfgs = {}
    for head, head_dir in head_dirs.items():
        if head_dir is None:
            models[head] = None
            cfgs[head] = None
            continue
        model, cfg, ckpt = load_model_and_cfg(head_dir, device)
        models[head] = model
        cfgs[head] = cfg
        print(f"Loaded {head} from {ckpt}")

    tokenizer_cfg = load_tokenizer_cfg()
    weapon2idx = build_weapon_index(tokenizer_cfg)
    projectile2idx = build_projectile_index(tokenizer_cfg)

    demo_parser = DemoParser(str(demo_path))
    ticks_by_round = get_important_ticks_by_round(demo_parser, interval=0.25)
    ticks_group = [ticks_by_round[r] for r in sorted(ticks_by_round.keys())]

    results_group = extract_states_by_group(str(demo_path), ticks_group)

    results = {}
    duel_fallback = [[0.5] * 10 for _ in range(10)]

    for idx, round_id in enumerate(tqdm(sorted(ticks_by_round.keys()), desc="rounds")):
        round_states = results_group[idx]
        if not round_states:
            results[f"error_round_{round_id}"] = "No states extracted"
            continue

        try:
            run_round_inference(
                round_states,
                models,
                cfgs,
                weapon2idx,
                projectile2idx,
                device,
                args.batch_size,
            )
        except Exception as e:
            results[f"error_round_{round_id}"] = f"{type(e).__name__}: {str(e)}"
            print(f"Error processing round {round_id}: {e}")
            continue

        round_result = process_round_json(round_states)
        if isinstance(round_result, dict) and "error" in round_result:
            results[f"error_round_{round_id}"] = round_result["error"]
            print(f"  Error in round {round_id}: {round_result['error']}")
            continue

        round_result["map_name"] = round_states[0].get("map_name")
        round_result["ticks"] = build_round_ticks(round_states, duel_fallback)
        results[str(round_id)] = round_result

    output_path.parent.mkdir(parents=True, exist_ok=True)
    results = to_jsonable(results)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f)

    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
