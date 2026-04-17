import yaml
import argparse
import torch
import torch.nn as nn
import os
import json
from pathlib import Path
from tqdm import tqdm
import sys
import traceback
import numpy as np
from copy import deepcopy

# from models.tfm_model import TickTransformerModel
# from models.tfm_model_rope import TickTransformerModelRope
from models.model2 import Model2
from demoparser_utils.tick_tokenizer import TickTokenizer
from demoparser_utils.state_extract import extract_states_by_group
from data.process_demo import get_important_ticks_by_round
from data.create_training_data import process_json_bytes
from demoparser2 import DemoParser


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def find_yaml(folder):
    for f in os.listdir(folder):
        if f.endswith(".yaml") and "tokenizer" not in f:
            return os.path.join(folder, f)
    raise RuntimeError(f"No yaml found in {folder}")


def load_checkpoint(folder):
    for f in os.listdir(folder):
        if f.endswith(".pth") or f.endswith(".pt"):
            return os.path.join(folder, f)
    raise RuntimeError(f"No checkpoint found in {folder}")

def to_jsonable(value):
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def load_model(folder, device):
    """Load win rate model from checkpoint folder."""
    try:
        yaml_path = find_yaml(folder)
        config = load_config(yaml_path)
        
        model = Model2(config).to(device)

        ckpt = load_checkpoint(folder)
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state["model_state_dict"])
        model.eval()

        print(f"Loaded win rate model from {ckpt}")
        return model, config
    except Exception as e:
        print(f"Error loading model from {folder}: {type(e).__name__}: {str(e)}")
        print(f"Please try to run the download_model.py script to download the model files and ensure the folder structure is correct.")
        sys.exit(1)

def process_round_json(round_result):
    """Process a single round's result for visualization."""
    processed_json = {}

    win_rate = []
    for state in round_result:
        win_rate.append({
            "round_seconds": state["round_seconds"],
            "ct_win_rate": state["ct_win_rate"]
        })
    winner = round_result[-1]["round_label"]["round_info"]["winner"]
    if winner not in ["CT", "T"]:
        return {"error": f"Invalid winner: {winner}"}
    last_second = round_result[-1]["round_seconds"]
    if winner == "CT":
        win_rate.append({
            "round_seconds": last_second + 0.25,
            "ct_win_rate": 1.0
        })
    else:
        win_rate.append({
            "round_seconds": last_second + 0.25,
            "ct_win_rate": 0.0
        })

    processed_json["win_rate"] = win_rate

    all_kills = round_result[0]["future_kills"]

    team_ct = []
    team_t = []
    for player_info in round_result[0]["players_info"]:
        if player_info["team_num"] == "CT":
            team_ct.append(player_info["name"])
        elif player_info["team_num"] == "T":
            team_t.append(player_info["name"])

    # Keep round-start inventories so downstream LLM analysis can reference economy/loadout context.
    start_inventory = []
    for player_info in round_result[0]["players_info"]:
        start_inventory.append(
            {
                "player": player_info.get("name", "Unknown"),
                "team_num": player_info.get("team_num", "Unknown"),
                "inventory": player_info.get("inventory", []),
            }
        )

    player_data = {}
    for name in team_ct + team_t:
        player_data[name] = {
            "kill_contribution": 0,
            "tactical_contribution": 0
        }

    processed_json["kills"] = []
    processed_json["player_data"] = []
    processed_json["player_data"].append(deepcopy(player_data))

    for i in range(len(win_rate) - 1):
        d_win = win_rate[i + 1]["ct_win_rate"] - win_rate[i]["ct_win_rate"]
        kill_count = 0

        alive_players = []

        for player_info in round_result[i]["players_info"]:
            if player_info["is_alive"]:
                alive_players.append(player_info["name"])

        for kill in all_kills:
            if kill["time"] >= win_rate[i]["round_seconds"] and kill["time"] < win_rate[i + 1]["round_seconds"]:
                killer = kill["attacker_name"]
                victim = kill["victim_name"]

                team_killer = "CT" if killer in team_ct else "T"
                team_victim = "CT" if victim in team_ct else "T"

                if killer in player_data.keys() and victim in player_data.keys():
                    kill_count += 1

        if kill_count > 0:
            for kill in all_kills:
                if kill["time"] >= win_rate[i]["round_seconds"] and kill["time"] < win_rate[i + 1]["round_seconds"]:
                    killer = kill["attacker_name"]
                    victim = kill["victim_name"]

                    team_killer = "CT" if killer in team_ct else "T"
                    team_victim = "CT" if victim in team_ct else "T"


                    if killer in player_data.keys() and victim in player_data.keys():
                        kill_impact = 0
                        if team_killer == "CT":
                            if team_victim == "T":
                                # CT kill T
                                player_data[killer]["kill_contribution"] += d_win / kill_count
                                player_data[victim]["kill_contribution"] -= d_win / kill_count
                                kill_impact = d_win / kill_count
                            else:
                                # CT kill CT (team kill)
                                player_data[killer]["kill_contribution"] += d_win / (kill_count * 2)
                                player_data[victim]["kill_contribution"] += d_win / (kill_count * 2)
                                kill_impact = d_win / kill_count

                                count_t_alive = sum(1 for p in alive_players if p in team_t)
                                if count_t_alive > 0:
                                    for p in alive_players:
                                        if p in team_t:
                                            player_data[p]["tactical_contribution"] -= d_win / (kill_count * count_t_alive)
                        else:
                            if team_victim == "CT":
                                # T kill CT
                                player_data[killer]["kill_contribution"] -= d_win / kill_count
                                player_data[victim]["kill_contribution"] += d_win / kill_count
                                kill_impact = -d_win / kill_count
                            else:
                                # T kill T (team kill)
                                player_data[killer]["kill_contribution"] -= d_win / (kill_count * 2)
                                player_data[victim]["kill_contribution"] -= d_win / (kill_count * 2)
                                kill_impact = -d_win / kill_count

                                count_ct_alive = sum(1 for p in alive_players if p in team_ct)
                                if count_ct_alive > 0:
                                    for p in alive_players:
                                        if p in team_ct:
                                            player_data[p]["tactical_contribution"] += d_win / (kill_count * count_ct_alive)
                        processed_json["kills"].append({
                            "killer": killer,
                            "assister": kill.get("assister_name"),
                            "victim": victim,
                            "round_seconds": kill["time"],
                            "kill_impact": kill_impact,
                            "weapon": kill["weapon"],
                            "headshot": bool(kill.get("headshot", False)),
                        })
        else:
            alive_t_players = [p for p in alive_players if p in team_t]
            alive_ct_players = [p for p in alive_players if p in team_ct]
            if len(alive_ct_players) > 0:
                for p in alive_ct_players:
                    player_data[p]["tactical_contribution"] += d_win / len(alive_ct_players)
            if len(alive_t_players) > 0:
                for p in alive_t_players:
                    player_data[p]["tactical_contribution"] -= d_win / len(alive_t_players)

        processed_json["player_data"].append(deepcopy(player_data))

    processed_json["winner"] = winner
    processed_json["CT_players"] = team_ct
    processed_json["T_players"] = team_t
    processed_json["start_inventory"] = start_inventory

    return processed_json

def process_round_states(
    round_id,
    round_states,
    model,
    config,
    tokenizer,
    valid_maps,
    device,
    batch_size=32,
):
    """
    Process states for a single round and get win rate predictions.
    
    Args:
        round_id: round number
        round_states: list of state dicts for this round
        model: win rate model
        config: model config
        tokenizer: tick tokenizer
        valid_maps: set of valid maps
        device: torch device
        batch_size: batch size for inference
    
    Returns:
        list of [round_seconds, ct_win_rate] or error message
    """
    try:
        # Convert round_states to json bytes format and tokenize
        json_bytes = json.dumps(to_jsonable(round_states)).encode()
        round_tensors, _, _, _, _, _, _ = process_json_bytes(
            json_bytes,
            tokenizer,
            valid_maps
        )
        
        if len(round_tensors) == 0:
            return {"error": "No valid tensors generated"}
        
        tensor = round_tensors[0]  # Should be only one round
        
        pad_token = config["model"]["pad_token_id"]
        seq_len = config["data"]["tick_seq_len"]
        ticks_per_sample = config["data"]["temporal_seq_len"]
        
        results = []
        inputs = []
        round_seconds_list = []
        
        for i in range(tensor.shape[0]):
            round_seconds_list.append(round_states[i]["round_seconds"])

        if tensor.shape[1] < seq_len:
            pad_len = seq_len - tensor.shape[1]
            pad = torch.full((tensor.shape[0], pad_len), pad_token, dtype=tensor.dtype)
            tensor = torch.cat([tensor, pad], 1)

        pad_front = ticks_per_sample - 1
        pad = torch.full((pad_front, tensor.shape[1]), pad_token, dtype=tensor.dtype)
        tensor = torch.cat([pad, tensor], 0)
        
        # print(tensor.shape)

        # use get_tick_embeddings to get embeddings for all the ticks
        # tick_embeddings = model.base_model.get_tick_embeddings(tensor.to(device))

        outputs = []
        masks = []

        for i in range(0, tensor.shape[0], batch_size):
            batch = tensor[i:i + batch_size].to(device)

            with torch.no_grad():
                out, mask = model.get_tick_embeddings(batch)

            outputs.append(out.cpu())
            masks.append(mask.cpu())


        tick_embeddings = torch.cat(outputs, dim=0)
        masks = torch.cat(masks, dim=0)
        # print(tick_embeddings.shape)

        input_mask = []
        for i in range(tick_embeddings.shape[0] - ticks_per_sample + 1):
            inputs.append(tick_embeddings[i:i + ticks_per_sample])
            input_mask.append(masks[i:i + ticks_per_sample])
        
        # Batch inference
        inputs = torch.stack(inputs)
        input_mask = torch.stack(input_mask)
        
        for i in tqdm(range(0, inputs.shape[0], batch_size), desc=f"Round {round_id} Inference"):
            inp = inputs[i:i+batch_size].to(device)
            inp_mask = input_mask[i:i+batch_size].to(device)
            
            with torch.no_grad():
                win_logit = model.get_predictions_from_tick_emb(inp, inp_mask, None).squeeze(-1) # (batch_size,)
                win_rate = torch.sigmoid(win_logit).cpu().numpy().tolist() # (batch_size,)
            
            for j, wr in enumerate(win_rate):
                results.append([round_seconds_list[i + j], wr])

        assert len(results) == len(round_states), f"Number of results {len(results)} does not match number of states {len(round_states)}"

        for idx, win_rate in enumerate(results):
            round_states[idx]["ct_win_rate"] = win_rate[1]  # win_rate is [t, ct_win_rate]
        
        return process_round_json(round_states)
    
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        return {"error": error_msg, "traceback": traceback.format_exc()}


def main():
    parser = argparse.ArgumentParser(description="Get win rate predictions for each round in a demo")
    parser.add_argument("--demo_path", required=True, help="Path to .dem file")
    parser.add_argument("--model_path", required=True, help="Path to win rate model checkpoint directory")
    parser.add_argument("--device", default="cuda", help="Device to run inference on")
    parser.add_argument("--output", type=str, required=True, help="Output JSON file path")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference")
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    demo_path = Path(args.demo_path)
    model_path = Path(args.model_path)
    output_path = Path(args.output)
    
    if not demo_path.exists():
        print(f"Error: Demo file not found: {demo_path}")
        return
    
    if not model_path.exists():
        print(f"Error: Model path not found: {model_path}")
        return
    
    print(f"Loading model from {model_path}...")
    model, config = load_model(str(model_path), device)
    
    # Load tokenizer config
    tokenizer_yaml = model_path / "tokenizer.yaml"
    if not tokenizer_yaml.exists():
        print(f"Error: tokenizer.yaml not found in {model_path}")
        return
    
    with open(tokenizer_yaml, "r", encoding="utf-8") as f:
        tokenizer_cfg = yaml.safe_load(f)
    
    tokenizer = TickTokenizer(tokenizer_cfg)
    valid_maps = set(tokenizer_cfg["maps"].keys())
    
    print(f"Processing demo: {demo_path}")
    
    # Get important ticks
    print(f"Sampling ticks every 0.25 seconds...")
    demo_parser = DemoParser(str(demo_path))
    ticks_by_round = get_important_ticks_by_round(demo_parser, interval=0.25)

    # Process each round independently so one broken round does not fail the whole demo
    print(f"Total rounds with sampled ticks: {len(ticks_by_round)}")
    results = {}

    ticks_group = []
    for round_id in sorted(ticks_by_round.keys()):
        ticks_group.append(ticks_by_round[round_id])

    results_group = extract_states_by_group(str(demo_path), ticks_group)

    print(f"Extracted states for {len(results_group)} rounds. Starting inference...")
        
    for i, round_id in enumerate(sorted(ticks_by_round.keys())):
        round_ticks = ticks_by_round[round_id]
        print(f"Processing round {round_id} with {len(round_ticks)} ticks...")

        round_states = results_group[i]

        if not round_states or len(round_states) == 0:
            results[f"error_round_{round_id}"] = "No states extracted"
            print(f"  Error in round {round_id}: No states extracted")
            continue

        round_result = process_round_states(
            round_id,
            round_states,
            model,
            config,
            tokenizer,
            valid_maps,
            device,
            batch_size=args.batch_size,
        )
        
        if isinstance(round_result, dict) and "error" in round_result:
            results[f"error_round_{round_id}"] = round_result["error"]
            print(f"  Error in round {round_id}: {round_result['error']}")
        else:
            results[str(round_id)] = round_result
            print(f"  Round {round_id}: {len(round_result)} predictions")
    
    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results = to_jsonable(results)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
