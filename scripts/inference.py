import yaml
import argparse
import torch
import torch.nn as nn
import os
import json
from pathlib import Path
from tqdm import tqdm
import sys
import subprocess

from models.tfm_model import TickTransformerModel
from models.tfm_model_rope import TickTransformerModelRope
from demoparser_utils.tick_tokenizer import TickTokenizer
from data.create_training_data import process_json_bytes, group_by_round


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def find_yaml(folder):
    for f in os.listdir(folder):
        if f.endswith(".yaml"):
            return os.path.join(folder, f)
    raise RuntimeError(f"No yaml found in {folder}")


def load_checkpoint(folder):
    best = os.path.join(folder, "best_checkpoint.pth")
    latest = os.path.join(folder, "latest_checkpoint.pth")
    # if os.path.exists(best):
    #     return best
    return latest


# ------------------------
# prediction heads
# ------------------------

class AliveHead(nn.Module):
    def __init__(self, dim, hidden, layers):
        super().__init__()
        net=[]
        for _ in range(layers):
            net += [nn.Linear(dim, hidden), nn.GELU()]
            dim = hidden
        net.append(nn.Linear(dim,10))
        self.head = nn.Sequential(*net)

    def forward(self,x):
        return self.head(x)


class KillHead(nn.Module):
    def __init__(self, dim, hidden, layers):
        super().__init__()
        net=[]
        for _ in range(layers):
            net += [nn.Linear(dim, hidden), nn.GELU()]
            dim = hidden
        net.append(nn.Linear(dim,22))
        self.head = nn.Sequential(*net)

    def forward(self,x):
        return self.head(x)


class WinRateHead(nn.Module):
    def __init__(self, dim, hidden, layers):
        super().__init__()
        net=[]
        for _ in range(layers):
            net += [nn.Linear(dim, hidden), nn.GELU()]
            dim = hidden
        net.append(nn.Linear(dim,1))
        self.head = nn.Sequential(*net)

    def forward(self,x):
        return self.head(x).squeeze(-1)


# ------------------------
# model wrapper
# ------------------------

class PredictionModel(nn.Module):

    def __init__(self, base_model, head):
        super().__init__()
        self.base_model = base_model
        self.prediction_head = head

    def forward(self,x):

        features = self.base_model.get_intermediate_data(x)
        last = features[:,-1,:]
        mean_feat = features.mean(dim=1)
        tick_feat = torch.cat([last, mean_feat], dim=-1)

        return self.prediction_head(tick_feat)


class Duel_Prediction_Head(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_hidden_layers, embedding_dim):
        super().__init__()
        layers = []
        prev_dim = input_dim + embedding_dim * 2  # concatenate player embeddings to the input features
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.GELU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))  # Output layer for binary classification (win/loss)
        self.head = nn.Sequential(*layers)

        # map 0~9 to player embedding
        self.embedding = nn.Embedding(10, embedding_dim)


    def forward(self, x, player_i, player_j):

        # x shape: (batch, feature_dim * 2)
        # player_i, player_j shape: (batch, )

        B = x.size(0)
        device = x.device

        if isinstance(player_i, int):
            player_i = torch.full((B,), player_i, dtype=torch.long, device=device)
        elif player_i.dim() == 0:
            player_i = player_i.expand(B)

        if isinstance(player_j, int):
            player_j = torch.full((B,), player_j, dtype=torch.long, device=device)
        elif player_j.dim() == 0:
            player_j = player_j.expand(B)

        player_i_emb = self.embedding(player_i)  # shape: (batch, embedding_dim)
        player_j_emb = self.embedding(player_j)  # shape: (batch, embedding_dim)
        x = torch.cat([x, player_i_emb, player_j_emb], dim=-1)  # shape: (batch, feature_dim * 2 + embedding_dim * 2)

        return self.head(x).squeeze(-1)  # shape: (batch_size, )

class Duel_Prediction_Model(nn.Module):
    def __init__(self, base_model, prediction_head):
        super().__init__()
        self.base_model = base_model
        self.prediction_head = prediction_head

    def forward(self, x, player_i, player_j):
        # x shape: (batch, ticks, seq_len)
        features = self.base_model.get_intermediate_data(x)  # shape: (batch, ticks, feature_dim)
        # only keep the last ticks' features for prediction
        last_tick_features = features[:, -1, :]  # shape: (batch, feature_dim)
        # cat the mean of all ticks' features to the last tick's features
        mean_features = features.mean(dim=1)  # shape: (batch, feature_dim)
        tick_features = torch.cat([last_tick_features, mean_features], dim=-1)  # shape: (batch, feature_dim * 2)
        logits = self.prediction_head(tick_features, player_i, player_j)  # shape: (batch,)
        return logits

def build_i_j(nxt_kill, nxt_death):
    # random torch 0 or 1, (batch, )
    label = torch.randint(0, 2, (nxt_kill.size(0),), device=nxt_kill.device)
    player_i = torch.where(label == 1, nxt_kill, nxt_death)  # shape: (batch, ) 
    player_j = torch.where(label == 0, nxt_kill, nxt_death)  # shape: (batch, )
    return player_i, player_j, label

# ------------------------
# demo processing
# ------------------------

def process_dem(dem_path, json_path):

    cmd=[
        sys.executable,
        "-m","data.process_demo",
        "-path",str(dem_path),
        "-interval","0.25",
        "-out",str(json_path),
    ]

    subprocess.run(cmd,check=True)


# ------------------------
# load model from checkpoint folder
# ------------------------

def load_model(folder, head_type, device):

    yaml_path = find_yaml(folder)
    config = load_config(yaml_path)
    if config['model']['model_name'] == "TickTransformerModel":
        base_model = TickTransformerModel(config["model"]).to(device)
    elif config['model']['model_name'] == "TickTransformerModelROPE":
        base_model = TickTransformerModelRope(config["model"]).to(device)
    else:
        raise ValueError(f"Unsupported model name: {config['model']['model_name']}")

    embed = config["model"]["embed_dim"]

    if head_type == "duel":
        head = Duel_Prediction_Head(
            embed * 2,
            hidden_dim=config['model']['duel_hidden_dim'],
            num_hidden_layers=config['model']['duel_hidden_layers'],
            embedding_dim=config['model']['duel_player_embedding_dim']
        )
        model = Duel_Prediction_Model(base_model, head).to(device)
    else:
        if head_type=="alive":
            head = AliveHead(
                embed * 2,
                config["model"]["alive_hidden_dim"],
                config["model"]["alive_hidden_layers"]
            )

        elif head_type=="kill":

            head = KillHead(
                embed * 2,
                config["model"]["nxt_kill_hidden_dim"],
                config["model"]["nxt_kill_hidden_layers"]
            )
        else:

            head = WinRateHead(
                embed * 2,
                config["model"]["win_rate_hidden_dim"],
                config["model"]["win_rate_hidden_layers"]
            )

        model = PredictionModel(base_model,head).to(device)

    ckpt = load_checkpoint(folder)

    state = torch.load(ckpt,map_location=device)
    model.load_state_dict(state["model_state_dict"])

    model.eval()

    print("Loaded",head_type,"model from",ckpt)

    return model,config


# ------------------------
# main
# ------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--demo_path",required=True)
    parser.add_argument("--json_path",required=True)

    parser.add_argument("--alive_ckpt_dir",required=True)
    parser.add_argument("--kill_ckpt_dir",required=True)
    parser.add_argument("--winrate_ckpt_dir",required=True)
    parser.add_argument("--duel_ckpt_dir",required=True)

    parser.add_argument("--device",default="cuda")
    parser.add_argument("--batch_size",default=1, type=int)

    args=parser.parse_args()

    device=torch.device(args.device)

    # --------------------
    # load models
    # --------------------

    alive_model,alive_cfg = load_model(args.alive_ckpt_dir,"alive",device)
    kill_model,_ = load_model(args.kill_ckpt_dir,"kill",device)
    win_model,_ = load_model(args.winrate_ckpt_dir,"win",device)
    duel_model,_ = load_model(args.duel_ckpt_dir, "duel", device)

    config = alive_cfg

    # --------------------
    # process demo
    # --------------------

    demo_name = os.path.basename(args.demo_path)
    json_path = Path(args.json_path)

    if not json_path.exists():
        process_dem(Path(args.demo_path),json_path)

    print("JSON ready:",json_path)

    TEAM1 = []
    TEAM2 = []

    # --------------------
    # tokenizer
    # --------------------

    with open(os.path.join(args.alive_ckpt_dir, "tokenizer.yaml")) as f:
        tokenizer_cfg=yaml.safe_load(f)

    tokenizer=TickTokenizer(tokenizer_cfg)

    valid_maps=set(tokenizer_cfg["maps"].keys())

    # --------------------
    # read json
    # --------------------

    with open(json_path) as f:
        json_data=json.load(f)

    for idx, playerinfo in enumerate(json_data[0]["players_info"]):
        if playerinfo["team_num"] == "CT":
            TEAM1.append((idx, playerinfo["name"]))
        else:
            TEAM2.append((idx, playerinfo["name"]))

    assert len(TEAM1) == 5
    assert len(TEAM2) == 5
    print(TEAM1)
    print(TEAM2)

    round_tensors, _, _, _, _, _ = process_json_bytes(
        json.dumps(json_data).encode(),
        tokenizer,
        valid_maps
    )

    rounds = group_by_round(json_data)

    pad_token = config["data"]["pad_token"]
    seq_len = config["data"]["seq_len"]
    ticks_per_sample = config["data"]["ticks_per_sample"]

    # --------------------
    # inference
    # --------------------

    idx_global=0

    for round_id in range(len(round_tensors)):

        tensor=round_tensors[round_id]
        inputs = []

        for i in range(tensor.shape[0]):

            pad_front=max(0,ticks_per_sample-1-i)

            if pad_front>0:
                pad=torch.full(
                    (pad_front,tensor.shape[1]),
                    pad_token,
                    dtype=tensor.dtype
                )
                inp=torch.cat([pad,tensor[:i+1]],0)

            else:
                inp=tensor[i+1-ticks_per_sample:i+1]

            if inp.shape[1]<seq_len:
                pad_len=seq_len-inp.shape[1]
                pad=torch.full((inp.shape[0],pad_len),pad_token,dtype=inp.dtype)
                inp=torch.cat([inp,pad],1)

            inp=inp[:,:seq_len]

            inputs.append(inp)
    
        inputs = torch.stack(inputs)
        for i in tqdm(range(0,inputs.shape[0],args.batch_size), desc=f"Round {round_id+1}/{len(round_tensors)}"):
            inp = inputs[i:i+args.batch_size].to(device)
            duel_results = [[None for _ in range(10)] for __ in range(10)]

            with torch.no_grad():

                alive_logits=alive_model(inp)
                kill_logits=kill_model(inp)
                win_logit=win_model(inp)
                if round_id <= 2:
                    for player1 in TEAM1:
                        for player2 in TEAM2:
                            duel_results[player1[0]][player2[0]] = duel_model(inp, player1[0], player2[0])
                        

            for j in range(alive_logits.shape[0]):

                alive_pred=torch.sigmoid(alive_logits[j]).cpu().numpy().tolist()

                kill=torch.softmax(kill_logits[j][:11],dim=0).cpu().numpy().tolist()
                death=torch.softmax(kill_logits[j][11:],dim=0).cpu().numpy().tolist()

                win_rate=torch.sigmoid(win_logit[j]).item()

                tick=json_data[idx_global]

                tick["alive_pred"]=alive_pred
                tick["next_kill"]=kill
                tick["next_death"]=death
                tick["win_rate"]=win_rate
                tick["duel"]= [[None for _ in range(10)] for __ in range(10)]
                for playeri in range(10):
                    for playerj in range(10):
                        if duel_results[playeri][playerj] is not None:
                            tick["duel"][playeri][playerj] = torch.sigmoid(duel_results[playeri][playerj][j]).item()
                            tick["duel"][playerj][playeri] = 1 - tick["duel"][playeri][playerj]

                idx_global+=1

    out=json_path.parent/(json_path.stem+"_pred.json")

    with open(out,"w") as f:
        json.dump(json_data,f,indent=2)

    print("Saved:",out)


if __name__=="__main__":
    main()