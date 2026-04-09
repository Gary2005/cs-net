import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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

class Embedder(nn.Module):
    """
    Non-causal transformer embedder with RoPE.
    """
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, seq_len, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # RoPE 实例 (头部维度)
        self.rotary_emb = RotaryEmbedding(embed_dim // num_heads, max_seq_len=seq_len)
        
        # Transformer encoder (non-causal)
        self.layers = nn.ModuleList([
            RoPETransformerBlock(embed_dim, num_heads, dropout, is_causal=False)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        x = self.token_embedding(x)  # (batch, seq_len, embed_dim)
        x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(x, self.rotary_emb)
            
        x = self.norm(x)
        
        # Extract first token as representation
        x = x[:, 0, :]  # (batch, embed_dim)
        return x


class Processor(nn.Module):
    """
    GPT-style causal transformer processor with RoPE.
    """
    def __init__(self, embed_dim, num_heads, num_layers, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        
        # RoPE for time ticks (支持更长的序列长度，例如10000)
        self.rotary_emb = RotaryEmbedding(embed_dim // num_heads, max_seq_len=10000)
        
        # Causal transformer encoder
        self.layers = nn.ModuleList([
            RoPETransformerBlock(embed_dim, num_heads, dropout, is_causal=True)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(x, self.rotary_emb)
            
        x = self.norm(x)
        return x


class Decoder(nn.Module):
    """
    Causal transformer decoder for in-tick token generation with RoPE.
    """
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, seq_len, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.vocab_size = vocab_size

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(dropout)

        # 序列长度为 seq_len + 1 (包含 condition)
        self.rotary_emb = RotaryEmbedding(embed_dim // num_heads, max_seq_len=seq_len + 1)

        self.layers = nn.ModuleList([
            RoPETransformerBlock(embed_dim, num_heads, dropout, is_causal=True)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        
        self.output_proj = nn.Linear(embed_dim, vocab_size)

    def forward(self, condition, tokens):
        tok_emb = self.token_embedding(tokens)
        tok_emb = self.dropout(tok_emb)

        cond = condition.unsqueeze(1)
        x = torch.cat([cond, tok_emb], dim=1) # (batch, seq_len+1, embed_dim)

        for layer in self.layers:
            x = layer(x, self.rotary_emb)
            
        x = self.norm(x)

        # Take only the first seq_len output positions
        x = x[:, :self.seq_len, :]
        logits = self.output_proj(x)

        return logits

    def generate_tokens(self, condition, temperature=1.0):
        cond = condition.unsqueeze(1)  # (batch, 1, embed_dim)
        generated = []

        for i in range(self.seq_len):
            if i == 0:
                x = cond
            else:
                prev = torch.stack(generated, dim=1)            # (batch, i)
                tok_emb = self.token_embedding(prev)            # (batch, i, embed_dim)
                x = torch.cat([cond, tok_emb], dim=1)           # (batch, i+1, embed_dim)

            # Pass through RoPE layers (RoPE 自动处理序列长度变化)
            for layer in self.layers:
                x = layer(x, self.rotary_emb)
            x = self.norm(x)

            logits = self.output_proj(x[:, -1, :])              # 预测最后一个 token (batch, vocab_size)

            if temperature <= 0 or temperature == 1.0:
                next_tok = torch.argmax(logits, dim=-1)
            else:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_tok = torch.multinomial(probs, 1).squeeze(-1)
            generated.append(next_tok)

        return torch.stack(generated, dim=1)


class TickTransformerModelRope(nn.Module):
    """
    Complete model combining Embedder, Processor, and Decoder (RoPE version).
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.embedder = Embedder(
            vocab_size=config['vocab_size'],
            embed_dim=config['embed_dim'],
            num_heads=config['embedder_heads'],
            num_layers=config['embedder_layers'],
            seq_len=config['seq_len'],
            dropout=config['dropout']
        )
        
        self.processor = Processor(
            embed_dim=config['embed_dim'],
            num_heads=config['processor_heads'],
            num_layers=config['processor_layers'],
            dropout=config['dropout']
        )
        
        self.decoder = Decoder(
            vocab_size=config['vocab_size'],
            embed_dim=config['embed_dim'],
            num_heads=config['decoder_heads'],
            num_layers=config['decoder_layers'],
            seq_len=config['seq_len'],
            dropout=config['dropout']
        )
        
    def forward(self, x, teacher_forcing=True):
        batch_size, ticks, seq_len = x.shape

        x_flat = x.view(batch_size * ticks, seq_len)
        embedded_flat = self.embedder(x_flat)
        embedded = embedded_flat.view(batch_size, ticks, self.embedder.embed_dim)

        processed = self.processor(embedded)

        conditions = processed[:, :-1, :]
        target_ticks = x[:, 1:, :]

        cond_flat = conditions.contiguous().view(batch_size * (ticks - 1), self.embedder.embed_dim)
        tgt_flat  = target_ticks.contiguous().view(batch_size * (ticks - 1), seq_len)

        logits_flat = self.decoder(cond_flat, tgt_flat)
        logits = logits_flat.view(batch_size, ticks - 1, seq_len, -1)

        return logits
    
    def get_intermediate_data(self, x):
        batch_size, ticks, seq_len = x.shape
        x_flat = x.view(batch_size * ticks, seq_len)
        embedded_flat = self.embedder(x_flat)
        embedded = embedded_flat.view(batch_size, ticks, self.embedder.embed_dim)
        processed = self.processor(embedded)
        return processed
    
    def generate(self, x, num_ticks_to_generate):
        self.eval()
        with torch.no_grad():
            current_seq = x

            for _ in range(num_ticks_to_generate):
                batch_size, ticks, seq_len = current_seq.shape

                x_flat = current_seq.view(batch_size * ticks, seq_len)
                embedded_flat = self.embedder(x_flat)
                embedded = embedded_flat.view(batch_size, ticks, self.embedder.embed_dim)

                processed = self.processor(embedded)
                condition = processed[:, -1, :]

                next_tokens = self.decoder.generate_tokens(condition)
                next_tokens = next_tokens.unsqueeze(1)

                current_seq = torch.cat([current_seq, next_tokens], dim=1)

            return current_seq