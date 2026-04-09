import yaml
import argparse
import torch
import torch.nn as nn
import os
import json

from models.tfm_model import TickTransformerModel
from models.tfm_model_rope import TickTransformerModelRope
from demoparser_utils.tick_tokenizer import TickTokenizer
from data.create_training_data import process_json_bytes, group_by_round


# =========================================================
# utils
# =========================================================

BLUE_BG = "\033[44m"
YELLOW_BG = "\033[43m"
GRAY_BG = "\033[100m"
RESET = "\033[0m"


def yaw_to_arrow_4(yaw_deg):
    """
    Map yaw to 4-direction arrow.
    Adjust OFFSET if orientation appears rotated.
    """
    OFFSET = 0
    yaw = (yaw_deg + OFFSET) % 360

    if 45 <= yaw < 135:
        return "^"
    elif 135 <= yaw < 225:
        return "<"
    elif 225 <= yaw < 315:
        return "v"
    else:
        return ">"


def render_ascii_radar(players_info, width=50, height=20):
    xs = [p["X"] for p in players_info]
    ys = [p["Y"] for p in players_info]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    if max_x - min_x < 1e-6:
        max_x += 1
    if max_y - min_y < 1e-6:
        max_y += 1

    grid = [["  " for _ in range(width)] for _ in range(height)]

    for p in players_info:
        gx = int((p["X"] - min_x) / (max_x - min_x) * (width - 1))
        gy = int((p["Y"] - min_y) / (max_y - min_y) * (height - 1))
        gy = height - 1 - gy

        if not p["is_alive"]:
            symbol = f"{GRAY_BG}x {RESET}"
        else:
            arrow = yaw_to_arrow_4(p["yaw"])

            if p["team_num"] == "CT":
                symbol = f"{BLUE_BG}{arrow} {RESET}"
            else:
                symbol = f"{YELLOW_BG}{arrow} {RESET}"

        grid[gy][gx] = symbol

    lines = []
    lines.append("+" + "-" * (width * 2) + "+")

    for row in grid:
        lines.append("|" + "".join(row) + "|")

    lines.append("+" + "-" * (width * 2) + "+")

    return "\n".join(lines)


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def find_yaml(folder):
    for f in os.listdir(folder):
        if f.endswith(".yaml") and f != "tokenizer.yaml":
            return os.path.join(folder, f)
    raise RuntimeError(f"No yaml found in {folder}")


def load_checkpoint(folder):
    for f in os.listdir(folder):
        if f.endswith(".pth"):
            return os.path.join(folder, f)
    raise RuntimeError(f"No checkpoint found in {folder}")


# =========================================================
# heads
# =========================================================

class AliveHead(nn.Module):
    def __init__(self, dim, hidden, layers):
        super().__init__()
        net = []
        for _ in range(layers):
            net += [nn.Linear(dim, hidden), nn.GELU()]
            dim = hidden
        net.append(nn.Linear(dim, 10))
        self.head = nn.Sequential(*net)

    def forward(self, x):
        return self.head(x)


class KillHead(nn.Module):
    def __init__(self, dim, hidden, layers):
        super().__init__()
        net = []
        for _ in range(layers):
            net += [nn.Linear(dim, hidden), nn.GELU()]
            dim = hidden
        net.append(nn.Linear(dim, 22))
        self.head = nn.Sequential(*net)

    def forward(self, x):
        return self.head(x)


class WinRateHead(nn.Module):
    def __init__(self, dim, hidden, layers):
        super().__init__()
        net = []
        for _ in range(layers):
            net += [nn.Linear(dim, hidden), nn.GELU()]
            dim = hidden
        net.append(nn.Linear(dim, 1))
        self.head = nn.Sequential(*net)

    def forward(self, x):
        return self.head(x).squeeze(-1)


class DuelPredictionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_hidden_layers, embedding_dim):
        super().__init__()

        layers = []
        prev_dim = input_dim + embedding_dim * 2

        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.GELU())
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))

        self.head = nn.Sequential(*layers)
        self.embedding = nn.Embedding(10, embedding_dim)

    def forward(self, x, i, j):
        B = x.size(0)

        i = torch.full((B,), i, dtype=torch.long, device=x.device)
        j = torch.full((B,), j, dtype=torch.long, device=x.device)

        emb_i = self.embedding(i)
        emb_j = self.embedding(j)

        x = torch.cat([x, emb_i, emb_j], dim=-1)

        return self.head(x).squeeze(-1)


# =========================================================
# wrappers
# =========================================================

class PredictionModel(nn.Module):
    def __init__(self, base_model, head):
        super().__init__()
        self.base_model = base_model
        self.prediction_head = head

    def forward(self, x):
        feat = self.base_model.get_intermediate_data(x)
        last = feat[:, -1, :]
        mean = feat.mean(dim=1)
        tick_feat = torch.cat([last, mean], dim=-1)
        return self.prediction_head(tick_feat)


class DuelPredictionModel(nn.Module):
    def __init__(self, base_model, head):
        super().__init__()
        self.base_model = base_model
        self.prediction_head = head

    def forward(self, x, i, j):
        feat = self.base_model.get_intermediate_data(x)
        last = feat[:, -1, :]
        mean = feat.mean(dim=1)
        tick_feat = torch.cat([last, mean], dim=-1)
        return self.prediction_head(tick_feat, i, j)


# =========================================================
# load model
# =========================================================

def load_model(folder, head_type, device):
    yaml_path = find_yaml(folder)
    cfg = load_config(yaml_path)

    if cfg["model"]["model_name"] == "TickTransformerModel":
        base = TickTransformerModel(cfg["model"]).to(device)
    else:
        base = TickTransformerModelRope(cfg["model"]).to(device)

    embed = cfg["model"]["embed_dim"]

    if head_type == "alive":
        head = AliveHead(embed * 2,
                         cfg["model"]["alive_hidden_dim"],
                         cfg["model"]["alive_hidden_layers"])
        model = PredictionModel(base, head)

    elif head_type == "kill":
        head = KillHead(embed * 2,
                        cfg["model"]["nxt_kill_hidden_dim"],
                        cfg["model"]["nxt_kill_hidden_layers"])
        model = PredictionModel(base, head)

    elif head_type == "win":
        head = WinRateHead(embed * 2,
                           cfg["model"]["win_rate_hidden_dim"],
                           cfg["model"]["win_rate_hidden_layers"])
        model = PredictionModel(base, head)

    else:
        head = DuelPredictionHead(
            embed * 2,
            cfg["model"]["duel_hidden_dim"],
            cfg["model"]["duel_hidden_layers"],
            cfg["model"]["duel_player_embedding_dim"]
        )
        model = DuelPredictionModel(base, head)

    ckpt = load_checkpoint(folder)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    return model.to(device), cfg


# =========================================================
# build padded input exactly like inference
# =========================================================

def build_input_window(round_tensor, tick_idx, ticks_per_sample, seq_len, pad_token):
    pad_front = max(0, ticks_per_sample - 1 - tick_idx)

    if pad_front > 0:
        pad = torch.full(
            (pad_front, round_tensor.shape[1]),
            pad_token,
            dtype=round_tensor.dtype
        )
        inp = torch.cat([pad, round_tensor[:tick_idx + 1]], dim=0)
    else:
        inp = round_tensor[tick_idx + 1 - ticks_per_sample:tick_idx + 1]

    if inp.shape[1] < seq_len:
        pad_len = seq_len - inp.shape[1]
        pad = torch.full(
            (inp.shape[0], pad_len),
            pad_token,
            dtype=inp.dtype
        )
        inp = torch.cat([inp, pad], dim=1)

    return inp[:, :seq_len]


# =========================================================
# main
# =========================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--json_path", required=True)

    parser.add_argument("--alive_ckpt_dir", required=True)
    parser.add_argument("--kill_ckpt_dir", required=True)
    parser.add_argument("--winrate_ckpt_dir", required=True)
    parser.add_argument("--duel_ckpt_dir", required=True)

    parser.add_argument("--device", default="cuda")

    parser.add_argument(
        "--remove_projectiles",
        action="store_true",
        help="Remove projectiles and grenade entities from json"
    )

    args = parser.parse_args()
    device = torch.device(args.device)

    print("Loading models...")

    alive_model, cfg = load_model(args.alive_ckpt_dir, "alive", device)
    kill_model, _ = load_model(args.kill_ckpt_dir, "kill", device)
    win_model, _ = load_model(args.winrate_ckpt_dir, "win", device)
    duel_model, _ = load_model(args.duel_ckpt_dir, "duel", device)

    print("Models loaded successfully.")

    print("Loading tokenizer...")

    with open(os.path.join(args.alive_ckpt_dir, "tokenizer.yaml")) as f:
        tokenizer_cfg = yaml.safe_load(f)

    tokenizer = TickTokenizer(tokenizer_cfg)

    print("Tokenizer loaded successfully.")

    valid_maps = set(tokenizer_cfg["maps"].keys())

    with open(args.json_path) as f:
        json_data = json.load(f)

    if args.remove_projectiles:
        print("Removing projectiles / grenade entities from json...")

        for tick in json_data:
            tick["projectiles"] = []
            tick["entity_grenades"] = []

    round_id = int(input("Input round number: "))
    target_sec = float(input("Input round seconds: "))

    round_tensors, _, _, _, _, _ = process_json_bytes(
        json.dumps(json_data).encode(),
        tokenizer,
        valid_maps
    )

    rounds = group_by_round(json_data)

    if round_id >= len(rounds):
        raise ValueError("Invalid round id")

    round_ticks = rounds[round_id]

    tick_idx = min(
        range(len(round_ticks)),
        key=lambda i: abs(round_ticks[i]["round_seconds"] - target_sec)
    )

    matched_tick = round_ticks[tick_idx]

    print(
        f"Matched tick: round={round_id}, "
        f"round_seconds={matched_tick['round_seconds']}"
    )

    inp = build_input_window(
        round_tensors[round_id],
        tick_idx,
        cfg["data"]["ticks_per_sample"],
        cfg["data"]["seq_len"],
        cfg["data"]["pad_token"]
    ).unsqueeze(0).to(device)

    print("Running predictions...")

    with torch.no_grad():
        alive = torch.sigmoid(alive_model(inp))[0]
        kill_logits = kill_model(inp)[0]
        win_rate = torch.sigmoid(win_model(inp))[0]

        kill_prob = torch.softmax(kill_logits[:11], dim=0)
        death_prob = torch.softmax(kill_logits[11:], dim=0)

        duel = torch.zeros(10, 10)

        for i in range(10):
            for j in range(10):
                if i == j:
                    duel[i, j] = 0.5
                else:
                    duel[i, j] = torch.sigmoid(
                        duel_model(inp, i, j)
                    )[0]
    
    print("Predictions generated successfully.\n")

    players_info = matched_tick["players_info"]

    ct_players = [
        (i, p["name"])
        for i, p in enumerate(players_info)
        if p["team_num"] == "CT"
    ]

    t_players = [
        (i, p["name"])
        for i, p in enumerate(players_info)
        if p["team_num"] == "T"
    ]

    print("\n" + "=" * 70)
    print("CASE STUDY")

    print("\nRadar View:")
    print(render_ascii_radar(players_info))

    print("=" * 70)

    print(
        f"Round: {matched_tick['round']} | "
        f"Time: {matched_tick['round_seconds']:.2f}s"
    )

    print("\nCT Win Rate:")
    print(f"  {win_rate.item():.4f}")

    # --------------------------------------------------
    # Alive Prediction
    # --------------------------------------------------
    print("\nAlive Prediction:")
    alive_probs = alive.cpu().tolist()

    for i, p in enumerate(players_info):
        if p["is_alive"]:
            print(
                f"  {p['name']:<15} "
                f"{alive_probs[i]:.4f}"
            )
        else:
            print(
                f"  {p['name']:<15} "
                f"DEAD"
            )

    # --------------------------------------------------
    # Next Killer
    # --------------------------------------------------
    print("\nNext Killer Distribution:")
    kill_probs = kill_prob.cpu().tolist()

    for i, p in enumerate(players_info):
        if p["is_alive"]:
            print(
                f"  {p['name']:<15} "
                f"{kill_probs[i]:.4f}"
            )
        else:
            print(
                f"  {p['name']:<15} "
                f"DEAD"
            )

    if len(kill_probs) > 10:
        print(f"  {'<NO KILL>':<15} {kill_probs[10]:.4f}")

    # --------------------------------------------------
    # Next Death
    # --------------------------------------------------
    print("\nNext Death Distribution:")
    death_probs = death_prob.cpu().tolist()

    for i, p in enumerate(players_info):
        if p["is_alive"]:
            print(
                f"  {p['name']:<15} "
                f"{death_probs[i]:.4f}"
            )
        else:
            print(
                f"  {p['name']:<15} "
                f"DEAD"
            )

    if len(death_probs) > 10:
        print(f"  {'<NO DEATH>':<15} {death_probs[10]:.4f}")

    # --------------------------------------------------
    # Duel Matrix
    # only alive CT / alive T
    # --------------------------------------------------
    print("\nDuel Matrix (CT vs T)")
    print("P[CT beats T]\n")

    duel_np = duel.cpu().numpy()

    header = " " * 15
    for t_idx, t_name in t_players:
        if players_info[t_idx]["is_alive"]:
            header += f"{t_name[:10]:>10}"
        else:
            header += f"{'[DEAD]':>10}"
    print(header)

    for ct_idx, ct_name in ct_players:

        if not players_info[ct_idx]["is_alive"]:
            row = f"{ct_name:<15}"
            row += f"{'[DEAD]':>10}" * len(t_players)
            print(row)
            continue

        row = f"{ct_name:<15}"

        for t_idx, _ in t_players:
            if not players_info[t_idx]["is_alive"]:
                row += f"{'[DEAD]':>10}"
            else:
                row += f"{duel_np[ct_idx, t_idx]:10.3f}"

        print(row)

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()