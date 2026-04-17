# Shout-out to OTaTatU for helpful suggestions on improving the model!

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.tfm_model_rope import TickTransformerModelRope
import os

# ==========================================
# RoPE (Rotary Position Embedding) 组件
# ==========================================

class RotaryEmbedding(nn.Module):
    """
    预计算 RoPE 的 cos 和 sin 频率矩阵
    """
    def __init__(self, dim, max_seq_len=10000):
        super().__init__()
        # 频率公式: 1 / (10000 ** (2i / d))
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_seq_len = max_seq_len

    def forward(self, seq_len, device):
        """返回形状为 (1, 1, seq_len, dim) 的 cos 和 sin 矩阵"""
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos()[None, None, :, :], emb.sin()[None, None, :, :]

def apply_rotary_pos_emb(q, k, cos, sin):
    """
    将旋转位置编码应用于 Query 和 Key
    """
    def rotate_half(x):
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class RoPEAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1, is_causal=False):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim 必须能被 num_heads 整除"
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.is_causal = is_causal
        self.dropout_p = dropout

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x, rotary_emb_fn):
        B, T, C = x.shape
        # 计算 Q, K, V
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # 形状: (B, num_heads, T, head_dim)

        # 获取并应用 RoPE
        cos, sin = rotary_emb_fn(T, x.device)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # 使用 PyTorch 原生的高效 Attention (支持 FlashAttention)
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=self.is_causal
        )

        y = y.transpose(1, 2).contiguous().reshape(B, T, C)
        return self.resid_drop(self.proj(y))

class RoPETransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1, is_causal=False):
        super().__init__()
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.attn = RoPEAttention(embed_dim, num_heads, dropout, is_causal)
        self.ln_2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, rotary_emb_fn):
        x = x + self.attn(self.ln_1(x), rotary_emb_fn)
        x = x + self.mlp(self.ln_2(x))
        return x
    

class AttentionPooling(nn.Module):
    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or dim

        self.score_net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, mask=None):
        """
        x: (B, L, C)
        mask: (B, L)  1=valid, 0=pad
        """
        # (B, L, 1) → (B, L)
        scores = self.score_net(x).squeeze(-1)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # (B, L)
        weights = F.softmax(scores, dim=1)

        # (B, L, 1)
        weights = weights.unsqueeze(-1)

        pooled = torch.sum(weights * x, dim=1)

        return pooled, weights

class FirstTokenPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, mask=None):
        """
        x: (B, L, C)
        mask: (B, L)  1=valid, 0=pad
        """
        return x[:, 0], None
    
# class TickEncoder(nn.Module):
#     """
#     Non-causal transformer embedder with RoPE.
#     """
#     def __init__(self, vocab_size, embed_dim, num_heads, num_layers, dropout=0.1):
#         super().__init__()
#         self.embed_dim = embed_dim
        
#         self.token_embedding = nn.Embedding(vocab_size, embed_dim)
#         self.dropout = nn.Dropout(dropout)
        
#         # RoPE 实例 (头部维度)
#         self.rotary_emb = RotaryEmbedding(embed_dim // num_heads, max_seq_len=10000)
        
#         # Transformer encoder (non-causal)
#         self.layers = nn.ModuleList([
#             RoPETransformerBlock(embed_dim, num_heads, dropout, is_causal=False)
#             for _ in range(num_layers)
#         ])
#         self.norm = nn.LayerNorm(embed_dim)
#         self.pool = FirstTokenPooling()
        
#     def forward(self, x, mask=None):
#         """
#         x: (B, L)
#         mask: (B, L)  1=valid, 0=pad
#         """
#         x = self.token_embedding(x)  # (B, L, C)
#         x = self.dropout(x)
        
#         for layer in self.layers:
#             x = layer(x, self.rotary_emb)
            
#         x = self.norm(x)

#         pooled, weights = self.pool(x, mask)

#         return pooled   # (B, C)
    
class TemporalEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        
        self.rotary_emb = RotaryEmbedding(embed_dim // num_heads, max_seq_len=10000)
        
        # Transformer encoder (non-causal)
        self.layers = nn.ModuleList([
            RoPETransformerBlock(embed_dim, num_heads, dropout, is_causal=False)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.pool = AttentionPooling(embed_dim)
        
    def forward(self, x, mask=None):
        """
        x: (B, L, C)
        mask: (B, L)  1=valid, 0=pad
        """
        
        for layer in self.layers:
            x = layer(x, self.rotary_emb)
            
        x = self.norm(x)
        pooled, weights = self.pool(x, mask)
        return pooled   # (B, C)
    
class Model2(nn.Module):
    """
    Model2:
    Input: game states and conditions
    Output: logits of shape (B, n_logits)
    """

    def __init__(self, config):
        super().__init__()

        pretrained_model = TickTransformerModelRope(config['pretrain'])

        config = config['model']

        pretrained_path = config['pretrained_path']

        if os.path.exists(pretrained_path):
            state_dict = torch.load(pretrained_path, map_location='cpu')
            pretrained_model.load_state_dict(state_dict['model_state_dict'])

        self.tick_encoder = pretrained_model.embedder

        for param in self.tick_encoder.parameters():
            param.requires_grad = False

        self.temporal_encoder = TemporalEncoder(
            embed_dim=config['embed_dim'],
            num_heads=config['num_heads'],
            num_layers=config['temporal_num_layers'],
            dropout=config['dropout']
        )
        
        if config['num_cond'] > 0:
            self.num_cond = config['num_cond']
            self.cond_vocab_size = config['cond_vocab_size']
            self.cond_embedding = nn.Embedding(
                config['num_cond'] * config['cond_vocab_size'],
                config['embed_dim']
            )

        # config['embed_dim'] -> 1 (binary classification)
        self.head = nn.Sequential(
            nn.Linear(config['embed_dim'] + config['embed_dim'] * config['num_cond'], 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, config['n_logits']),
        )

        self.pad_token_id = config['pad_token_id']

    def get_tick_embeddings(self, x):
        """
        x: (B, L)
        output: (B, C)
        """
        mask = (x != self.pad_token_id).long()  # (B, L)
        mask = mask.any(dim=-1)  # (B,)
        with torch.no_grad():
            tick_emb = self.tick_encoder(x)  # (B, C)
        return tick_emb, mask
    
    def get_predictions_from_tick_emb(self, tick_emb, mask, cond):
        """
        tick_emb: (B, T, C)
        mask: (B, T) 1=valid, 0=pad
        cond: (B, num_cond) or None
        """
        temporal_emb = self.temporal_encoder(tick_emb, mask)  # (B, C)

        if cond is not None:
            offset = torch.arange(self.num_cond, device=cond.device) * self.cond_vocab_size
            cond_offset = cond + offset.unsqueeze(0)   # (B, num_cond)
            cond_emb = self.cond_embedding(cond_offset)  # (B, num_cond, C)
            cond_emb = cond_emb.view(cond_emb.size(0), -1)  # (B, num_cond * C)
        else:
            cond_emb = None

        if cond_emb is not None:
            combined_emb = torch.cat([temporal_emb, cond_emb], dim=1)
        else:
            combined_emb = temporal_emb

        logits = self.head(combined_emb)  # (B, n_logits)

        return logits

    def forward(self, x, cond):
        """
        x: (B, T, L)
        cond: (B, num_cond), integer < cond_vocab_size or None if no conditions

        output: (B, n_logits) logits
        """
        B, T, L = x.shape
        x = x.view(B * T, L)  # (B*T, L)

        # mask (1=valid, 0=pad)
        mask = (x != self.pad_token_id).long()  # (B*T, L)
        with torch.no_grad():
            tick_emb = self.tick_encoder(x)  # (B*T, C)
        tick_emb = tick_emb.view(B, T, -1)     # (B, T, C)
        
        mask = mask.view(B, T, L).any(dim=-1)  # (B, T)

        if cond is not None:
            offset = torch.arange(self.num_cond, device=cond.device) * self.cond_vocab_size
            cond_offset = cond + offset.unsqueeze(0)   # (B, num_cond)
            cond_emb = self.cond_embedding(cond_offset)  # (B, num_cond, C)
            cond_emb = cond_emb.view(B, -1)  # (B, num_cond * C)
        else:
            cond_emb = None

        temporal_emb = self.temporal_encoder(tick_emb, mask)  # (B, C)

        if cond_emb is not None:
            combined_emb = torch.cat([temporal_emb, cond_emb], dim=1)
        else:
            combined_emb = temporal_emb

        logits = self.head(combined_emb)  # (B, n_logits)

        return logits
    
    def train(self, mode=True):
        super().train(mode)
        self.tick_encoder.eval() 
        return self