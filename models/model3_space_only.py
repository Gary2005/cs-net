import torch
import torch.nn as nn
import torch.nn.functional as F
class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, num_layers):
        super().__init__()

        assert num_layers >= 1

        layers = []

        if num_layers == 1:
            layers.append(nn.Linear(in_dim, out_dim))

        else:
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.GELU())

            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden, hidden))
                layers.append(nn.GELU())

            layers.append(nn.Linear(hidden, out_dim))

        self.net = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        """
        x: [B, D_in]
        output: [B, D_out]
        """

        x = self.net(x)
        x = self.norm(x)
        return x
    
import torch
import torch.nn as nn


class SpaceTransformer(nn.Module):
    def __init__(
        self,
        d_model,
        n_heads,
        num_layers,
        ffn_dim,
        n_tokens,
        dropout,
    ):
        super().__init__()

        self.pos_emb = nn.Embedding(n_tokens, d_model)

        self.pad_token = nn.Parameter(torch.randn(d_model))
        self.dead_token = nn.Parameter(torch.randn(d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

    def forward(self, x, pad_mask, dead_mask):
        """
        x: [B, N, D]
        pad_mask:  [B, N]  (True = PAD)
        dead_mask: [B, N]  (True = DEAD)

        output: [B, N, D]
        """

        B, N, D = x.shape

        pad_token = self.pad_token.view(1, 1, D)
        dead_token = self.dead_token.view(1, 1, D)

        x = torch.where(
            dead_mask.unsqueeze(-1),
            dead_token,
            x
        )

        x = torch.where(
            pad_mask.unsqueeze(-1),
            pad_token,
            x
        )


        token_ids = torch.arange(N, device=x.device).unsqueeze(0).expand(B, N)
        pos = self.pos_emb(token_ids)

        x = x + pos

        out = self.transformer(
            x,
            src_key_padding_mask=pad_mask
        )

        return out
    
class CSModelV3(nn.Module):
    def __init__(self, cfg):

        super().__init__()
        
        self.cfg = cfg
        d = cfg["d"]
        mlp_cfg = cfg["MLP"]

        def make_mlp(in_dim):
            return MLP(
                in_dim=in_dim,
                hidden=mlp_cfg["hidden_dim"],
                out_dim=d,
                num_layers=mlp_cfg["layers"]
            )
        
        self.map_emb = nn.Embedding(cfg["n_maps"], cfg["d_map_embedding"])
        self.weapon_emb = nn.Embedding(cfg["n_weapons"], cfg["d_inventory_embedding"])
        self.proj_emb = nn.Embedding(cfg["n_projectiles"], cfg["d_projectile_embedding"])
        self.index_emb = nn.Embedding(32, cfg["d_index_embedding"])

        self.weapon_proj = make_mlp(cfg["d_inventory_embedding"])
        self.map_proj = make_mlp(cfg["d_map_embedding"])

        
        self.MLP1 = make_mlp(3 + cfg["d_map_embedding"])
        self.MLP2 = make_mlp(14)
        self.MLP3 = make_mlp(1 + cfg["d_projectile_embedding"])
        self.MLP4 = make_mlp(4)
        self.MLP5 = make_mlp(13 + cfg["d_index_embedding"])

        self.final_norm = nn.LayerNorm(d)

        st = cfg["space_transformer"]
        self.space_tf = SpaceTransformer(
            d_model=st["d_model"],
            n_heads=st["n_heads"],
            num_layers=st["layers"],
            ffn_dim=st["ffn_dim"],
            n_tokens=cfg["Space_Size"],
            dropout=st["dropout"],
        )

        def make_head(in_dim, out_dim):
            layers = []
            layers.append(nn.Linear(in_dim, mlp_cfg["hidden_dim"]))
            layers.append(nn.GELU())
            layers.append(nn.Linear(mlp_cfg["hidden_dim"], out_dim))
            return nn.Sequential(*layers)
        
        if cfg["task"] == "winrate":
            self.head = make_head(d, 1)
        elif cfg["task"] == "duel":
            self.head = make_head(d * 2, 1)
        elif cfg["task"] == "nxt_kill" or cfg["task"] == "nxt_death":
            self.head = make_head(d, 11)
        elif cfg["task"] == "alive_in_the_end":
            self.head = make_head(d, 1)

    def encode_tick(self, batch):
        B, S, _ = batch["mlp1_f"].shape

        result = torch.zeros(B, S, self.cfg["d"], device=next(self.parameters()).device)
        count = torch.zeros(B, S, 1, device=next(self.parameters()).device)

        """
        MLP1
        """
        embedding_map = self.map_emb(batch["mlp1_i"]) # (B, S, d_map_embedding)
        mlp1_input = torch.cat([batch["mlp1_f"], embedding_map], dim=-1)  # (B, S, 3 + d_map_embedding)
        mlp1_output = self.MLP1(mlp1_input)  # (B, S, d)
        kept = batch["mlp1_mask"] # (B, S)
        result = result + mlp1_output * kept.unsqueeze(-1)
        count = count + kept.unsqueeze(-1)

        """
        MLP2
        """
        mlp2_input = batch["mlp2_f"]  # (B, S, 14)
        mlp2_output = self.MLP2(mlp2_input)  # (B, S, d)
        kept = batch["mlp2_mask"] # (B, S)
        result = result + mlp2_output * kept.unsqueeze(-1)
        count = count + kept.unsqueeze(-1)

        """
        MLP3
        """
        embedding_proj = self.proj_emb(batch["mlp3_i"]) # (B, S, d_projectile_embedding)
        mlp3_input = torch.cat([batch["mlp3_f"], embedding_proj], dim=-1)  # (B, S, 1 + d_projectile_embedding)
        mlp3_output = self.MLP3(mlp3_input)  # (B, S, d)
        kept = batch["mlp3_mask"] # (B, S)
        result = result + mlp3_output * kept.unsqueeze(-1)
        count = count + kept.unsqueeze(-1)

        """
        MLP4
        """
        mlp4_input = batch["mlp4_f"]  # (B, S, 4)
        mlp4_output = self.MLP4(mlp4_input)  # (B, S, d)
        kept = batch["mlp4_mask"] # (B, S)
        result = result + mlp4_output * kept.unsqueeze(-1)
        count = count + kept.unsqueeze(-1)

        """
        MLP5
        """
        embedding_index = self.index_emb(batch["mlp5_i"]) # (B, S, 9, d_index_embedding)
        mlp5_input = torch.cat([batch["mlp5_f"], embedding_index], dim=-1)  # (B, S, 9, 13 + d_index_embedding)
        mlp5_output = self.MLP5(mlp5_input)  # (B, S, 9, d)
        kept = batch["mlp5_mask"] # (B, S, 9)
        mlp5_output = mlp5_output * kept.unsqueeze(-1)  # (B, S, 9, d)
        # mlp5_output = mlp5_output.sum(dim=2)  # (B, S, d)
        valid_count = kept.sum(dim=2, keepdim=True).clamp(min=1) # (B, S, 1)
        mlp5_output = mlp5_output.sum(dim=2) / valid_count
        result = result + mlp5_output
        branch_mask = (kept.sum(dim=2) > 0).float() # (B, S)
        count = count + branch_mask.unsqueeze(-1)

        """
        Map_embedding
        """
        embedding_map = self.map_emb(batch["emb2_i"]) # (B, S, d_map_embedding)
        map_emb_output = self.map_proj(embedding_map)  # (B, S, d)
        kept = batch["emb2_mask"] # (B, S)
        map_emb_output = map_emb_output * kept.unsqueeze(-1)
        result = result + map_emb_output
        count = count + kept.unsqueeze(-1)

        """
        Weapon_embedding
        """
        embedding_weapon = self.weapon_emb(batch["emb1_i"]) # (B, S, 9, d_inventory_embedding)
        weapon_emb_output = self.weapon_proj(embedding_weapon)  # (B, S, 9, d)
        kept = batch["emb1_mask"] # (B, S, 9)
        weapon_emb_output = weapon_emb_output * kept.unsqueeze(-1)  # (B, S, 9, d)
        # weapon_emb_output = weapon_emb_output.sum(dim=2)  # (B, S, d)
        valid_count = kept.sum(dim=2, keepdim=True).clamp(min=1) # (B, S, 1)
        weapon_emb_output = weapon_emb_output.sum(dim=2) / valid_count # (B, S, d)
        result = result + weapon_emb_output
        branch_mask = (kept.sum(dim=2) > 0).float() # (B, S)
        count = count + branch_mask.unsqueeze(-1)


        count = count.clamp(min=1)
        result = result / count

        result = self.final_norm(result)

        return result
    
    def forward(self, batch):

        B, S, _ = batch["mlp1_f"].shape

        x = self.encode_tick(batch)  # (B, S, d)
        
        x = x.view(B, S, self.cfg["d"])  # (B, S, d)
        pad_mask = batch["pad_mask"].view(B, S)  # (B, S)
        dead_mask = batch["dead_mask"].view(B, S)  # (B, S)
        x = self.space_tf(x, pad_mask, dead_mask)  # (B, S, d)
        
        x = x.view(B, S, self.cfg["d"])  # (B, S, d)

        if self.cfg["task"] == "winrate":
            global_tokens = x[:, 10, :]  # (B, d)
            out = self.head(global_tokens)  # (B, 1)
            return out.squeeze(-1), batch['label'] # (B,), (B,) binary classification
        elif self.cfg["task"] == "duel":
            duel = batch["duel"] # (B, 3)  [a, b, is_a_win]
            idx_a = duel[:, 0]  # (B,)
            idx_b = duel[:, 1]  # (B,)
            is_a_win = duel[:, 2]  # (B,)
            
            batch_idx = torch.arange(B, device=x.device)
            token_a = x[batch_idx, idx_a, :]  # (B, d)
            token_b = x[batch_idx, idx_b, :]  # (B, d)

            out = torch.cat([token_a, token_b], dim=-1)  # (B, 2d)
            out = self.head(out)  # (B, 1)
            return out.squeeze(-1), is_a_win  # (B,), (B,) binary classification
        elif self.cfg["task"] == "nxt_kill":
            global_tokens = x[:, 10, :]  # (B, d)
            out = self.head(global_tokens)  # (B, 11)
            return out, batch['nxt_kill']  # (B, 11), (B,) multi-class classification
        elif self.cfg["task"] == "nxt_death":
            global_tokens = x[:, 10, :]  # (B, d)
            out = self.head(global_tokens)  # (B, 11)
            return out, batch['nxt_death']  # (B, 11), (B,) multi-class classification
        elif self.cfg["task"] == "alive_in_the_end":
            players_tokens = x[:, :10, :]  # (B, 10, d)
            # 变换维度：(B, 10, d) -> (B, 10, d)
            players_tokens = players_tokens.contiguous().view(B * 10, self.cfg["d"])  # (B*10, d)
            out = self.head(players_tokens)  # (B * 10, 1)
            out = out.view(B, 10)  # (B, 10)
            return out, batch['alive_in_the_end']  # (B, 10), (B,10) binary classification
        else:
            raise ValueError(f"Unsupported task: {self.cfg['task']}")
        
if __name__ == "__main__":
    
    from dataset.model3_wds_space_only import build_dataloader, move_to_device
    import yaml

    device = "cuda"
    cfg_path = "config/model3.yaml"
    test_data_dir = "/data/dataset"
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    tasks = ["winrate", "duel", "nxt_kill", "nxt_death", "alive_in_the_end"]

    for task in tasks:
        cfg["task"] = task
        model = CSModelV3(cfg).to(device)

        print(f"Number of parameters: {sum(p.numel() for p in model.parameters())} for task: {task}")

        dataloader = build_dataloader(test_data_dir, split="test", batch_size=8, num_workers=0, task=task)
        for batch in dataloader:
            batch = move_to_device(batch, device)
            out, label = model(batch)
            print(f"Task: {task}, out shape: {out.shape}, label shape: {label.shape}")
            print(f"Out: {out}")
            print(f"Label: {label}")
            break