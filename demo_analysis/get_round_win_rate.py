"""
Batched multi-head inference for the demo_analysis pipeline.

For each round in a demo we run all five heads (win / alive / kill / death / duel)
in a single pass by reusing the shared frozen tick_encoder across models. Each
tick in the output JSON carries:

    ct_win_rate: float
    alive_pred : list[10] float
    next_kill  : list[11] float
    next_death : list[11] float
    duel       : list[10][10] float   (0.5 on the diagonal and where one player is dead)

On top of that we still emit the round-level dashboard fields the web app already
uses (win_rate curve, kills, player_data contribution, winner, inventories) and
a `ticks` array with the minimal radar view of every tick.

This mirrors the approach in ``cs2-demo-analytics/scripts/inference.py`` —
compute tick embeddings once, then re-use them for every head.
"""

import argparse
import json
import os
import sys
import traceback
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

from data.create_training_data import process_json_bytes
from data.process_demo import get_important_ticks_by_round
from demoparser2 import DemoParser
from demoparser_utils.state_extract import extract_states_by_group
from demoparser_utils.tick_tokenizer import TickTokenizer
from models.model2 import Model2


# ---------------------------------------------------------------------------
# model / fs helpers
# ---------------------------------------------------------------------------

DEFAULT_SUBDIRS = {
    "alive": "alive",
    "kill":  "nxt_kill",
    "death": "nxt_death",
    "win":   "win_rate",
    "duel":  "duel",
}

HEAD_ORDER = ("alive", "kill", "death", "win", "duel")


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def find_yaml(folder):
    for name in sorted(os.listdir(folder)):
        if name.endswith(".yaml") and "tokenizer" not in name:
            return os.path.join(folder, name)
    raise RuntimeError(f"No model yaml found in {folder}")


def find_checkpoint(folder):
    for preferred in ("latest_checkpoint.pth", "best_checkpoint.pth",
                      "latest_checkpoint.pt", "best_checkpoint.pt"):
        p = os.path.join(folder, preferred)
        if os.path.exists(p):
            return p
    for name in sorted(os.listdir(folder)):
        if name.endswith((".pth", ".pt")):
            return os.path.join(folder, name)
    raise RuntimeError(f"No checkpoint found in {folder}")


def find_tokenizer_yaml(folder):
    direct = os.path.join(folder, "tokenizer.yaml")
    if os.path.exists(direct):
        return direct
    for root, _, files in os.walk(folder):
        for name in files:
            if name == "tokenizer.yaml":
                return os.path.join(root, name)
    raise RuntimeError(f"tokenizer.yaml not found under {folder}")


def load_model(folder, device):
    cfg = load_config(find_yaml(folder))
    model = Model2(cfg).to(device)

    ckpt = find_checkpoint(folder)
    state = torch.load(ckpt, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    model.eval()
    print(f"Loaded {os.path.basename(folder)} model from {ckpt}")
    return model, cfg


def _looks_like_ckpt_dir(path):
    """True if `path` itself is a single-head checkpoint dir (has yaml + .pt/.pth)."""
    if not path or not os.path.isdir(path):
        return False
    has_yaml = any(
        n.endswith(".yaml") and "tokenizer" not in n for n in os.listdir(path)
    )
    has_ckpt = any(n.endswith((".pth", ".pt")) for n in os.listdir(path))
    return has_yaml and has_ckpt


def resolve_head_dirs(args):
    """
    Expand --model_root into per-head dirs, honoring per-head overrides.
    Missing heads are returned as None — the pipeline runs with sensible
    fallbacks for all heads except `win`, which is required.
    Legacy layout (user only has `win_rate/`) is accepted two ways:
      - `--model_root path/to/win_rate`  (auto-detected as a bare win head dir)
      - `--winrate_ckpt_dir path/to/win_rate` (explicit override)
    """
    explicit = {
        "alive": args.alive_ckpt_dir,
        "kill":  args.kill_ckpt_dir,
        "death": args.death_ckpt_dir,
        "win":   args.winrate_ckpt_dir,
        "duel":  args.duel_ckpt_dir,
    }

    root = args.model_root
    legacy_win_root = root and _looks_like_ckpt_dir(root) and not any(
        os.path.isdir(os.path.join(root, DEFAULT_SUBDIRS[h])) for h in HEAD_ORDER
    )

    dirs = {}
    for head in HEAD_ORDER:
        if explicit[head]:
            candidate = explicit[head]
        elif legacy_win_root and head == "win":
            candidate = root
        elif root and not legacy_win_root:
            candidate = os.path.join(root, DEFAULT_SUBDIRS[head])
        else:
            candidate = None
        if candidate and os.path.isdir(candidate):
            dirs[head] = candidate
        else:
            dirs[head] = None

    if dirs["win"] is None:
        raise RuntimeError(
            "win rate model dir is required. Pass --model_root <root containing win_rate/> "
            "or --winrate_ckpt_dir <dir>, or point --model_root directly at the legacy "
            "win_rate checkpoint dir."
        )

    missing = [h for h in HEAD_ORDER if dirs[h] is None]
    if missing:
        print(f"[warn] optional heads missing — running with fallbacks: {missing}")

    return dirs


def to_jsonable(v):
    if isinstance(v, dict):
        return {k: to_jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [to_jsonable(x) for x in v]
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    return v


# ---------------------------------------------------------------------------
# inference core — shared tick embeddings, batched heads
# ---------------------------------------------------------------------------

def apply_temperature_scaling(logits, config):
    calib = (config or {}).get("calibration", {}).get("temperature_scaling")
    if calib is None:
        return logits
    T = float(calib.get("temperature", 1.0)) or 1.0
    scaled = logits / T
    bias = calib.get("bias")
    if bias is None:
        return scaled
    if isinstance(bias, list):
        b = torch.tensor(bias, dtype=scaled.dtype, device=scaled.device)
        return scaled + b.unsqueeze(0)
    return scaled + float(bias)


def pad_round_tensor(tensor, seq_len, pad_token):
    if tensor.shape[1] < seq_len:
        pad = torch.full(
            (tensor.shape[0], seq_len - tensor.shape[1]),
            pad_token,
            dtype=tensor.dtype,
        )
        return torch.cat([tensor, pad], dim=1)
    return tensor[:, :seq_len]


def compute_tick_embeddings(model, token_tensor, device, batch_size):
    embs, masks = [], []
    for i in range(0, token_tensor.shape[0], batch_size):
        batch = token_tensor[i:i + batch_size].to(device)
        with torch.no_grad():
            e, m = model.get_tick_embeddings(batch)
        embs.append(e.cpu())
        masks.append(m.cpu())
    return torch.cat(embs, dim=0), torch.cat(masks, dim=0)


def build_sliding_windows(tick_emb, tick_mask, ticks_per_sample):
    total = tick_emb.shape[0]
    if total < ticks_per_sample:
        raise RuntimeError(
            f"need >= {ticks_per_sample} ticks, got {total} after padding"
        )
    n = total - ticks_per_sample + 1
    emb_win = torch.stack([tick_emb[i:i + ticks_per_sample] for i in range(n)], 0)
    msk_win = torch.stack([tick_mask[i:i + ticks_per_sample] for i in range(n)], 0)
    return emb_win, msk_win


def infer_alive_probs(model, cfg, eb, mb):
    outs = []
    for p in range(10):
        cond = torch.full((eb.size(0), 1), p, dtype=torch.long, device=eb.device)
        logit = model.get_predictions_from_tick_emb(eb, mb, cond).squeeze(-1)
        logit = apply_temperature_scaling(logit, cfg)
        outs.append(torch.sigmoid(logit))
    return torch.stack(outs, dim=1)  # (B, 10)


def infer_duel_probs(model, cfg, eb, mb, team_ct, team_t):
    B = eb.size(0)
    out = torch.full((B, 10, 10), 0.5, device=eb.device)
    for ct in team_ct:
        for t in team_t:
            cond = torch.zeros((B, 2), dtype=torch.long, device=eb.device)
            cond[:, 0] = ct
            cond[:, 1] = t
            logit = model.get_predictions_from_tick_emb(eb, mb, cond).squeeze(-1)
            logit = apply_temperature_scaling(logit, cfg)
            p = torch.sigmoid(logit)
            out[:, ct, t] = p
            out[:, t, ct] = 1.0 - p
    return out


def run_multihead(
    round_tensor,
    round_states,
    models,
    configs,
    device,
    batch_size,
    compute_duel,
):
    """
    Enrich each state dict in round_states with head predictions.

    `models` and `configs` are dicts keyed by head name (alive / kill / death / win / duel).
    Missing heads → fallback values (uniform / 0.5). The `win` head is required and
    drives `ct_win_rate`; all other heads are optional. Since every head shares the
    same frozen tick_encoder, embeddings are computed from whichever model is loaded.
    """
    win_m = models["win"]
    win_cfg = configs["win"]
    if win_m is None:
        raise RuntimeError("run_multihead: win-rate model is required")

    # Any loaded model works — the tick_encoder is a frozen copy of the pretrained embedder.
    encoder_model = next((m for m in models.values() if m is not None), None)
    if encoder_model is None:
        raise RuntimeError("no loaded models")

    ref_cfg = win_cfg
    seq_len = ref_cfg["data"]["tick_seq_len"]
    ticks_per_sample = ref_cfg["data"]["temporal_seq_len"]
    pad_token = ref_cfg["model"]["pad_token_id"]

    team_ct, team_t = [], []
    for idx, p in enumerate(round_states[0]["players_info"]):
        (team_ct if p["team_num"] == "CT" else team_t).append(idx)

    tensor = pad_round_tensor(round_tensor, seq_len, pad_token)
    front = torch.full(
        (ticks_per_sample - 1, tensor.shape[1]),
        pad_token,
        dtype=tensor.dtype,
    )
    padded = torch.cat([front, tensor], dim=0)

    tick_emb, tick_mask = compute_tick_embeddings(encoder_model, padded, device, batch_size)
    emb_win, msk_win = build_sliding_windows(tick_emb, tick_mask, ticks_per_sample)

    N = emb_win.shape[0]
    if N != len(round_states):
        raise RuntimeError(
            f"window count {N} does not match state count {len(round_states)}"
        )

    alive_m = models.get("alive")
    kill_m = models.get("kill")
    death_m = models.get("death")
    duel_m = models.get("duel")

    alive_fb = [0.5] * 10
    # Uniform over the 11 kill/death slots (10 players + 1 "no event").
    kd_fb = [1.0 / 11.0] * 11

    for start in range(0, N, batch_size):
        eb = emb_win[start:start + batch_size].to(device)
        mb = msk_win[start:start + batch_size].to(device)
        B = eb.shape[0]

        with torch.no_grad():
            if alive_m is not None:
                alive_probs = infer_alive_probs(alive_m, configs["alive"], eb, mb).cpu().numpy()
            else:
                alive_probs = None

            if kill_m is not None:
                k_logit = kill_m.get_predictions_from_tick_emb(eb, mb, None)
                k_logit = apply_temperature_scaling(k_logit, configs["kill"])
                kill_probs = torch.softmax(k_logit, dim=-1).cpu().numpy()
            else:
                kill_probs = None

            if death_m is not None:
                d_logit = death_m.get_predictions_from_tick_emb(eb, mb, None)
                d_logit = apply_temperature_scaling(d_logit, configs["death"])
                death_probs = torch.softmax(d_logit, dim=-1).cpu().numpy()
            else:
                death_probs = None

            w_logit = win_m.get_predictions_from_tick_emb(eb, mb, None).squeeze(-1)
            w_logit = apply_temperature_scaling(w_logit, win_cfg)
            win_probs = torch.sigmoid(w_logit).cpu().numpy()

            if compute_duel and duel_m is not None:
                duel_probs = infer_duel_probs(
                    duel_m, configs["duel"], eb, mb, team_ct, team_t
                ).cpu().numpy()
            else:
                duel_probs = None

        for j in range(B):
            s = round_states[start + j]
            s["ct_win_rate"] = float(win_probs[j])
            s["alive_pred"] = alive_probs[j].tolist() if alive_probs is not None else list(alive_fb)
            s["next_kill"] = kill_probs[j].tolist() if kill_probs is not None else list(kd_fb)
            s["next_death"] = death_probs[j].tolist() if death_probs is not None else list(kd_fb)
            s["duel"] = duel_probs[j].tolist() if duel_probs is not None else None

    return round_states


# ---------------------------------------------------------------------------
# contribution / dashboard assembly (unchanged logic)
# ---------------------------------------------------------------------------

def _safe_div(a, b):
    return a / b if b else 0.0


def compute_kill_difficulty(round_result, kill_time, attacker_idx, victim_idx, window_s=5.0):
    """
    Mirror of cs2-demo-analytics/visualization/compute_round_swing.py difficulty:
    Look at next_kill / next_death predictions in the [kill_time - window_s, kill_time]
    window, average them, then return (avg_v_kill * avg_a_death) / (avg_a_kill * avg_v_death).

    >1 → attacker won an "uphill" duel the model thought they would lose.
    <1 → attacker won an expected duel.
    """
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
    """Translate enriched states → round-level dashboard dict (kills, contributions)."""
    processed_json = {}

    # Map player name → index (0..9) so we can resolve kill participants for difficulty.
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
    team_t  = [p["name"] for p in round_result[0]["players_info"] if p["team_num"] == "T"]

    # Keep round-start inventories so downstream LLM analysis can reference economy/loadout context.
    start_inventory = [
        {
            "player": p.get("name", "Unknown"),
            "team_num": p.get("team_num", "Unknown"),
            "inventory": p.get("inventory", []),
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
                        # CT team kill
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
                        # T team kill
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


# ---------------------------------------------------------------------------
# radar-friendly per-tick view
# ---------------------------------------------------------------------------

def radar_player_view(players_info):
    """Slim each player dict down to fields the 2D top-down radar / panels need."""
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
        ticks.append({
            "round_seconds": s.get("round_seconds"),
            "players_info": radar_player_view(s.get("players_info", [])),
            "ct_win_rate": s.get("ct_win_rate"),
            "alive_pred": s.get("alive_pred"),
            "next_kill": s.get("next_kill"),
            "next_death": s.get("next_death"),
            "duel": duel,
            "is_bomb_planted": s.get("is_bomb_planted"),
            "is_bomb_dropped": s.get("is_bomb_dropped"),
            "bomb_position": s.get("bomb_position"),
        })
    return ticks


# ---------------------------------------------------------------------------
# round-level pipeline
# ---------------------------------------------------------------------------

def process_round_states(
    round_id,
    round_states,
    models,
    configs,
    tokenizer,
    valid_maps,
    device,
    batch_size,
    compute_duel,
    duel_fallback,
):
    try:
        json_bytes = json.dumps(to_jsonable(round_states)).encode()
        round_tensors, _, _, _, _, _, _ = process_json_bytes(
            json_bytes, tokenizer, valid_maps
        )
        if len(round_tensors) == 0:
            return {"error": "No valid tensors generated (map unsupported or time range anomalous)"}

        tensor = round_tensors[0]

        run_multihead(
            tensor,
            round_states,
            models,
            configs,
            device,
            batch_size,
            compute_duel=compute_duel,
        )

        round_dashboard = process_round_json(round_states)
        if isinstance(round_dashboard, dict) and "error" in round_dashboard:
            return round_dashboard

        round_dashboard["map_name"] = round_states[0].get("map_name")
        round_dashboard["ticks"] = build_round_ticks(round_states, duel_fallback)
        return round_dashboard

    except Exception as e:
        return {
            "error": f"{type(e).__name__}: {str(e)}",
            "traceback": traceback.format_exc(),
        }


def main():
    parser = argparse.ArgumentParser(
        description="Batched multi-head inference: win rate + alive + next kill/death + duel"
    )
    parser.add_argument("--demo_path", required=True, help="Path to .dem file")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    parser.add_argument("--device", default="cuda", help="Device to run inference on")
    parser.add_argument("--batch_size", type=int, default=32)

    # preferred: single root containing alive / nxt_kill / nxt_death / win_rate / duel
    parser.add_argument("--model_root",
                        help="Root dir containing alive/ nxt_kill/ nxt_death/ win_rate/ duel/")

    # per-head overrides (take precedence over --model_root)
    parser.add_argument("--alive_ckpt_dir")
    parser.add_argument("--kill_ckpt_dir")
    parser.add_argument("--death_ckpt_dir")
    parser.add_argument("--winrate_ckpt_dir")
    parser.add_argument("--duel_ckpt_dir")

    parser.add_argument("--skip_duel", action="store_true",
                        help="Skip duel inference (fills 0.5 matrix)")

    args = parser.parse_args()

    requested = args.device
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA not available, falling back to CPU")
        requested = "cpu"
    elif requested == "mps":
        mps_ok = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        if not mps_ok:
            print("[warn] MPS not available, falling back to CPU")
            requested = "cpu"
    device = torch.device(requested)
    demo_path = Path(args.demo_path)
    output_path = Path(args.output)

    if not demo_path.exists():
        print(f"Error: Demo file not found: {demo_path}")
        return

    head_dirs = resolve_head_dirs(args)

    print("Loading models...")
    models: dict = {}
    configs: dict = {}
    for head in HEAD_ORDER:
        d = head_dirs[head]
        if d is None:
            models[head] = None
            configs[head] = None
            continue
        m, c = load_model(d, device)
        models[head] = m
        configs[head] = c
    print("Models loaded.")

    # Consistency checks across whichever heads were actually loaded.
    loaded_heads = [h for h in HEAD_ORDER if models[h] is not None]
    ref = loaded_heads[0]
    for h in loaded_heads[1:]:
        if configs[h]["data"]["tick_seq_len"] != configs[ref]["data"]["tick_seq_len"]:
            raise RuntimeError(f"tick_seq_len mismatch ({h} vs {ref})")
        if configs[h]["data"]["temporal_seq_len"] != configs[ref]["data"]["temporal_seq_len"]:
            raise RuntimeError(f"temporal_seq_len mismatch ({h} vs {ref})")
        if configs[h]["model"]["pad_token_id"] != configs[ref]["model"]["pad_token_id"]:
            raise RuntimeError(f"pad_token_id mismatch ({h} vs {ref})")

    # Tokenizer lives next to any head's config — use the first loaded one.
    tokenizer_cfg = load_config(find_tokenizer_yaml(head_dirs[ref]))
    tokenizer = TickTokenizer(tokenizer_cfg)
    valid_maps = set(tokenizer_cfg["maps"].keys())

    print(f"Processing demo: {demo_path}")
    print(f"Sampling ticks every 0.25 seconds...")

    demo_parser = DemoParser(str(demo_path))
    ticks_by_round = get_important_ticks_by_round(demo_parser, interval=0.25)

    ticks_group = [ticks_by_round[r] for r in sorted(ticks_by_round.keys())]
    print(f"Total rounds with sampled ticks: {len(ticks_group)}")

    results_group = extract_states_by_group(str(demo_path), ticks_group)
    print(f"Extracted states for {len(results_group)} rounds. Starting inference...")

    duel_fallback = [[0.5] * 10 for _ in range(10)]
    compute_duel = (not args.skip_duel) and models.get("duel") is not None

    results = {}
    for i, round_id in enumerate(
        tqdm(sorted(ticks_by_round.keys()), desc="rounds")
    ):
        round_ticks = ticks_by_round[round_id]
        round_states = results_group[i]

        print(f"Processing round {round_id} with {len(round_ticks)} ticks...")

        if not round_states:
            results[f"error_round_{round_id}"] = "No states extracted"
            print(f"  Error in round {round_id}: No states extracted")
            continue

        round_result = process_round_states(
            round_id,
            round_states,
            models,
            configs,
            tokenizer,
            valid_maps,
            device,
            batch_size=args.batch_size,
            compute_duel=compute_duel,
            duel_fallback=duel_fallback,
        )

        if isinstance(round_result, dict) and "error" in round_result:
            results[f"error_round_{round_id}"] = round_result["error"]
            print(f"  Error in round {round_id}: {round_result['error']}")
        else:
            results[str(round_id)] = round_result
            n_ticks = len(round_result.get("ticks", []))
            print(f"  Round {round_id}: {n_ticks} ticks enriched")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    results = to_jsonable(results)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
