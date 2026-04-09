import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LearnablePositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len):
        super().__init__()
        self.pos_emb = nn.Embedding(max_len, d_model)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        """
        B, L, D = x.shape

        pos = torch.arange(L, device=x.device)

        pos_embedding = self.pos_emb(pos).unsqueeze(0)

        return x + pos_embedding

class PositionalEncoding(nn.Module):
    """Position encoding for non-causal transformer."""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor shape (batch, seq_len, d_model)
        """
        return x + self.pe[:x.size(1), :]


class TimeEncoding(nn.Module):
    """Time encoding for causal transformer processing ticks."""
    def __init__(self, d_model, max_len=10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor shape (batch, ticks, d_model)
        """
        return x + self.pe[:x.size(1), :]


class Embedder(nn.Module):
    """
    Non-causal transformer embedder.
    Input: (batch, seq_len) - integer tokens
    Output: (batch, embed_dim) - embedded representation
    """
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, seq_len, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Position encoding
        self.pos_encoding = PositionalEncoding(embed_dim, max_len=seq_len)
        
        # Transformer encoder (non-causal)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len) - integer tokens
        Returns:
            (batch, embed_dim) - embedded representation
        """
        # Token embedding
        x = self.token_embedding(x)  # (batch, seq_len, embed_dim)
        x = self.dropout(x)
        
        # Add position encoding
        x = self.pos_encoding(x)  # (batch, seq_len, embed_dim)
        
        # Transformer encoding (non-causal, full attention)
        x = self.transformer(x)  # (batch, seq_len, embed_dim)
        
        # Extract first token as representation
        x = x[:, 0, :]  # (batch, embed_dim)
        
        return x


class EmbedderLearnablePositional(nn.Module):
    """
    Non-causal transformer embedder.
    Input: (batch, seq_len) - integer tokens
    Output: (batch, embed_dim) - embedded representation
    """
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, seq_len, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Position encoding
        self.pos_encoding = LearnablePositionalEncoding(embed_dim, max_len=seq_len)
        
        # Transformer encoder (non-causal)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len) - integer tokens
        Returns:
            (batch, embed_dim) - embedded representation
        """
        # Token embedding
        x = self.token_embedding(x)  # (batch, seq_len, embed_dim)
        x = self.dropout(x)
        
        # Add position encoding
        x = self.pos_encoding(x)  # (batch, seq_len, embed_dim)
        
        # Transformer encoding (non-causal, full attention)
        x = self.transformer(x)  # (batch, seq_len, embed_dim)
        
        # Extract first token as representation
        x = x[:, 0, :]  # (batch, embed_dim)
        
        return x



class Processor(nn.Module):
    """
    GPT-style causal transformer processor for next token prediction.
    Input: (batch, ticks, embed_dim)
    Output: (batch, ticks, embed_dim)
    """
    def __init__(self, embed_dim, num_heads, num_layers, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Time encoding
        self.time_encoding = TimeEncoding(embed_dim, max_len=10000)
        
        # GPT-style causal transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: (batch, ticks, embed_dim)
        Returns:
            (batch, ticks, embed_dim)
        """
        # Add time encoding
        x = self.time_encoding(x)
        x = self.dropout(x)
        
        # Create causal mask for GPT-style autoregressive modeling
        ticks = x.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(ticks).to(x.device)
        
        # Process with causal transformer encoder
        x = self.transformer(x, mask=causal_mask)
        
        return x
    


class ProcessorLearnablePositional(nn.Module):
    """
    GPT-style causal transformer processor for next token prediction.
    Input: (batch, ticks, embed_dim)
    Output: (batch, ticks, embed_dim)
    """
    def __init__(self, embed_dim, num_heads, num_layers, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Time encoding
        self.time_encoding = LearnablePositionalEncoding(embed_dim, max_len=65)
        
        # GPT-style causal transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: (batch, ticks, embed_dim)
        Returns:
            (batch, ticks, embed_dim)
        """
        # Add time encoding
        x = self.time_encoding(x)
        x = self.dropout(x)
        
        # Create causal mask for GPT-style autoregressive modeling
        ticks = x.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(ticks).to(x.device)
        
        # Process with causal transformer encoder
        x = self.transformer(x, mask=causal_mask)
        
        return x


class Decoder(nn.Module):
    """
    Causal transformer decoder for in-tick token generation.

    forward():
        condition (N, embed_dim) is prepended to the full embedded input tokens
        (N, seq_len, embed_dim), forming a sequence of length seq_len+1.
        After causal attention the first seq_len output positions are projected
        to vocabulary logits (N, seq_len, vocab_size) and compared against the
        original input tokens directly.

        Causal structure (0-indexed):
            output[0]   sees only condition              -> predicts tokens[0]
            output[i]   sees condition + tokens[0..i-1]  -> predicts tokens[i]

    generate_tokens() runs autoregressive decoding at inference time.
    """
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, seq_len, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.vocab_size = vocab_size

        # Token embedding for input tokens
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)

        # Position encoding – sequence length is seq_len+1 (condition + seq_len tokens)
        self.pos_encoding = PositionalEncoding(embed_dim, max_len=seq_len + 1)

        # Causal transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection
        self.output_proj = nn.Linear(embed_dim, vocab_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, condition, tokens):
        """
        Teacher-forced forward pass.

        Args:
            condition: (batch, embed_dim) – context vector from Processor
            tokens:    (batch, seq_len)   – ground-truth token ids for the tick
        Returns:
            logits: (batch, seq_len, vocab_size)
                    logits[:, i, :] predicts tokens[:, i]
        """
        # Embed all input tokens: (batch, seq_len, embed_dim)
        tok_emb = self.token_embedding(tokens)
        tok_emb = self.dropout(tok_emb)

        # Prepend condition as position 0: (batch, seq_len+1, embed_dim)
        cond = condition.unsqueeze(1)
        x = torch.cat([cond, tok_emb], dim=1)

        # Positional encoding
        x = self.pos_encoding(x)

        # Causal mask of size seq_len+1
        L = x.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(L).to(x.device)

        # Causal transformer: (batch, seq_len+1, embed_dim)
        x = self.transformer(x, mask=causal_mask)

        # Take only the first seq_len output positions (drop the last one)
        x = x[:, :self.seq_len, :]  # (batch, seq_len, embed_dim)

        # Project to vocabulary
        logits = self.output_proj(x)  # (batch, seq_len, vocab_size)

        return logits

    def generate_tokens(self, condition, temperature=1.0):
        """
        Autoregressive token generation (inference).

        Args:
            condition:   (batch, embed_dim)
            temperature: sampling temperature (<=0 or 1.0 => greedy argmax)
        Returns:
            (batch, seq_len) – generated token ids
        """
        cond = condition.unsqueeze(1)  # (batch, 1, embed_dim)
        generated = []  # list of (batch,) tensors

        for i in range(self.seq_len):
            if i == 0:
                # Input: just the condition, length 1
                x = self.pos_encoding(cond)
                mask = nn.Transformer.generate_square_subsequent_mask(1).to(x.device)
                h = self.transformer(x, mask=mask)
                logits = self.output_proj(h[:, 0, :])   # (batch, vocab_size)
            else:
                # Input: [condition, tok_0, ..., tok_{i-1}], length i+1
                prev = torch.stack(generated, dim=1)            # (batch, i)
                tok_emb = self.token_embedding(prev)            # (batch, i, embed_dim)
                x = torch.cat([cond, tok_emb], dim=1)           # (batch, i+1, embed_dim)
                x = self.pos_encoding(x)
                L = x.size(1)
                mask = nn.Transformer.generate_square_subsequent_mask(L).to(x.device)
                h = self.transformer(x, mask=mask)
                logits = self.output_proj(h[:, -1, :])          # (batch, vocab_size)

            if temperature <= 0 or temperature == 1.0:
                next_tok = torch.argmax(logits, dim=-1)         # (batch,)
            else:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_tok = torch.multinomial(probs, 1).squeeze(-1)
            generated.append(next_tok)

        return torch.stack(generated, dim=1)  # (batch, seq_len)
    

class DecoderLearnablePositional(nn.Module):
    """
    Causal transformer decoder for in-tick token generation.

    forward():
        condition (N, embed_dim) is prepended to the full embedded input tokens
        (N, seq_len, embed_dim), forming a sequence of length seq_len+1.
        After causal attention the first seq_len output positions are projected
        to vocabulary logits (N, seq_len, vocab_size) and compared against the
        original input tokens directly.

        Causal structure (0-indexed):
            output[0]   sees only condition              -> predicts tokens[0]
            output[i]   sees condition + tokens[0..i-1]  -> predicts tokens[i]

    generate_tokens() runs autoregressive decoding at inference time.
    """
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, seq_len, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.vocab_size = vocab_size

        # Token embedding for input tokens
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)

        # Position encoding – sequence length is seq_len+1 (condition + seq_len tokens)
        self.pos_encoding = LearnablePositionalEncoding(embed_dim, max_len=seq_len + 1)

        # Causal transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection
        self.output_proj = nn.Linear(embed_dim, vocab_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, condition, tokens):
        """
        Teacher-forced forward pass.

        Args:
            condition: (batch, embed_dim) – context vector from Processor
            tokens:    (batch, seq_len)   – ground-truth token ids for the tick
        Returns:
            logits: (batch, seq_len, vocab_size)
                    logits[:, i, :] predicts tokens[:, i]
        """
        # Embed all input tokens: (batch, seq_len, embed_dim)
        tok_emb = self.token_embedding(tokens)
        tok_emb = self.dropout(tok_emb)

        # Prepend condition as position 0: (batch, seq_len+1, embed_dim)
        cond = condition.unsqueeze(1)
        x = torch.cat([cond, tok_emb], dim=1)

        # Positional encoding
        x = self.pos_encoding(x)

        # Causal mask of size seq_len+1
        L = x.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(L).to(x.device)

        # Causal transformer: (batch, seq_len+1, embed_dim)
        x = self.transformer(x, mask=causal_mask)

        # Take only the first seq_len output positions (drop the last one)
        x = x[:, :self.seq_len, :]  # (batch, seq_len, embed_dim)

        # Project to vocabulary
        logits = self.output_proj(x)  # (batch, seq_len, vocab_size)

        return logits

    def generate_tokens(self, condition, temperature=1.0):
        """
        Autoregressive token generation (inference).

        Args:
            condition:   (batch, embed_dim)
            temperature: sampling temperature (<=0 or 1.0 => greedy argmax)
        Returns:
            (batch, seq_len) – generated token ids
        """
        cond = condition.unsqueeze(1)  # (batch, 1, embed_dim)
        generated = []  # list of (batch,) tensors

        for i in range(self.seq_len):
            if i == 0:
                # Input: just the condition, length 1
                x = self.pos_encoding(cond)
                mask = nn.Transformer.generate_square_subsequent_mask(1).to(x.device)
                h = self.transformer(x, mask=mask)
                logits = self.output_proj(h[:, 0, :])   # (batch, vocab_size)
            else:
                # Input: [condition, tok_0, ..., tok_{i-1}], length i+1
                prev = torch.stack(generated, dim=1)            # (batch, i)
                tok_emb = self.token_embedding(prev)            # (batch, i, embed_dim)
                x = torch.cat([cond, tok_emb], dim=1)           # (batch, i+1, embed_dim)
                x = self.pos_encoding(x)
                L = x.size(1)
                mask = nn.Transformer.generate_square_subsequent_mask(L).to(x.device)
                h = self.transformer(x, mask=mask)
                logits = self.output_proj(h[:, -1, :])          # (batch, vocab_size)

            if temperature <= 0 or temperature == 1.0:
                next_tok = torch.argmax(logits, dim=-1)         # (batch,)
            else:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_tok = torch.multinomial(probs, 1).squeeze(-1)
            generated.append(next_tok)

        return torch.stack(generated, dim=1)  # (batch, seq_len)


class TickTransformerModel(nn.Module):
    """
    Complete model combining Embedder, Processor, and Decoder.
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
        """
        Args:
            x: (batch, ticks, seq_len) - input token sequences
            teacher_forcing: kept for API compatibility; always True in this design
        Returns:
            logits: (batch, ticks-1, seq_len, vocab_size)
                    logits[:, t, i, :] is the prediction for token i of tick t+1
        """
        batch_size, ticks, seq_len = x.shape

        # ── Embedder: encode every tick into a single vector ────────────────
        x_flat = x.view(batch_size * ticks, seq_len)          # (B*T, S)
        embedded_flat = self.embedder(x_flat)                 # (B*T, D)
        embedded = embedded_flat.view(batch_size, ticks, self.embedder.embed_dim)  # (B, T, D)

        # ── Processor: causal transformer over the tick sequence ─────────────
        processed = self.processor(embedded)                  # (B, T, D)

        # ── Decoder: condition = processed[t-1], target tokens = x[t] ───────
        # Conditions for predicting ticks 1..T-1
        conditions = processed[:, :-1, :]                     # (B, T-1, D)
        target_ticks = x[:, 1:, :]                            # (B, T-1, S)

        # Flatten batch and tick dimensions for a single decoder call
        cond_flat = conditions.contiguous().view(batch_size * (ticks - 1), self.embedder.embed_dim)
        tgt_flat  = target_ticks.contiguous().view(batch_size * (ticks - 1), seq_len)

        # Teacher-forced causal decoding: (B*(T-1), S, V)
        logits_flat = self.decoder(cond_flat, tgt_flat)

        # Reshape: (B, T-1, S, V)
        logits = logits_flat.view(batch_size, ticks - 1, seq_len, -1)

        return logits
    
    def get_intermediate_data(self, x):
        """
        Get intermediate representations for analysis.
        x: (batch, ticks, seq_len) - input token sequences
        returns: (batch, ticks, embed_dim) - processed representations from the Processor
        """
        batch_size, ticks, seq_len = x.shape

        # ── Embedder: encode every tick into a single vector ────────────────
        x_flat = x.view(batch_size * ticks, seq_len)          # (B*T, S)
        embedded_flat = self.embedder(x_flat)                 # (B*T, D)
        embedded = embedded_flat.view(batch_size, ticks, self.embedder.embed_dim)  # (B, T, D)
        # ── Processor: causal transformer over the tick sequence ─────────────
        processed = self.processor(embedded)                  # (B, T, D)
        return processed
    
    def generate(self, x, num_ticks_to_generate):
        """
        Autoregressive generation of future ticks.
        Args:
            x: (batch, ticks, seq_len) - initial sequence
            num_ticks_to_generate: number of ticks to generate
        Returns:
            (batch, ticks + num_ticks_to_generate, seq_len)
        """
        self.eval()
        with torch.no_grad():
            current_seq = x

            for _ in range(num_ticks_to_generate):
                batch_size, ticks, seq_len = current_seq.shape

                # Embed all ticks
                x_flat = current_seq.view(batch_size * ticks, seq_len)
                embedded_flat = self.embedder(x_flat)           # (B*T, D)
                embedded = embedded_flat.view(batch_size, ticks, self.embedder.embed_dim)

                # Process through causal transformer
                processed = self.processor(embedded)            # (B, T, D)

                # Condition for the next tick = last processed representation
                condition = processed[:, -1, :]                 # (B, D)

                # Autoregressively generate the token sequence for the next tick
                next_tokens = self.decoder.generate_tokens(condition)  # (B, S)
                next_tokens = next_tokens.unsqueeze(1)                 # (B, 1, S)

                # Append the new tick
                current_seq = torch.cat([current_seq, next_tokens], dim=1)

            return current_seq



class TickTransformerModelLearnablePositional(nn.Module):
    """
    Complete model combining Embedder, Processor, and Decoder.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.embedder = EmbedderLearnablePositional(
            vocab_size=config['vocab_size'],
            embed_dim=config['embed_dim'],
            num_heads=config['embedder_heads'],
            num_layers=config['embedder_layers'],
            seq_len=config['seq_len'],
            dropout=config['dropout']
        )
        
        self.processor = ProcessorLearnablePositional(
            embed_dim=config['embed_dim'],
            num_heads=config['processor_heads'],
            num_layers=config['processor_layers'],
            dropout=config['dropout']
        )
        
        self.decoder = DecoderLearnablePositional(
            vocab_size=config['vocab_size'],
            embed_dim=config['embed_dim'],
            num_heads=config['decoder_heads'],
            num_layers=config['decoder_layers'],
            seq_len=config['seq_len'],
            dropout=config['dropout']
        )
        
    def forward(self, x, teacher_forcing=True):
        """
        Args:
            x: (batch, ticks, seq_len) - input token sequences
            teacher_forcing: kept for API compatibility; always True in this design
        Returns:
            logits: (batch, ticks-1, seq_len, vocab_size)
                    logits[:, t, i, :] is the prediction for token i of tick t+1
        """
        batch_size, ticks, seq_len = x.shape

        # ── Embedder: encode every tick into a single vector ────────────────
        x_flat = x.view(batch_size * ticks, seq_len)          # (B*T, S)
        embedded_flat = self.embedder(x_flat)                 # (B*T, D)
        embedded = embedded_flat.view(batch_size, ticks, self.embedder.embed_dim)  # (B, T, D)

        # ── Processor: causal transformer over the tick sequence ─────────────
        processed = self.processor(embedded)                  # (B, T, D)

        # ── Decoder: condition = processed[t-1], target tokens = x[t] ───────
        # Conditions for predicting ticks 1..T-1
        conditions = processed[:, :-1, :]                     # (B, T-1, D)
        target_ticks = x[:, 1:, :]                            # (B, T-1, S)

        # Flatten batch and tick dimensions for a single decoder call
        cond_flat = conditions.contiguous().view(batch_size * (ticks - 1), self.embedder.embed_dim)
        tgt_flat  = target_ticks.contiguous().view(batch_size * (ticks - 1), seq_len)

        # Teacher-forced causal decoding: (B*(T-1), S, V)
        logits_flat = self.decoder(cond_flat, tgt_flat)

        # Reshape: (B, T-1, S, V)
        logits = logits_flat.view(batch_size, ticks - 1, seq_len, -1)

        return logits
    
    def get_intermediate_data(self, x):
        """
        Get intermediate representations for analysis.
        x: (batch, ticks, seq_len) - input token sequences
        returns: (batch, ticks, embed_dim) - processed representations from the Processor
        """
        batch_size, ticks, seq_len = x.shape

        # ── Embedder: encode every tick into a single vector ────────────────
        x_flat = x.view(batch_size * ticks, seq_len)          # (B*T, S)
        embedded_flat = self.embedder(x_flat)                 # (B*T, D)
        embedded = embedded_flat.view(batch_size, ticks, self.embedder.embed_dim)  # (B, T, D)
        # ── Processor: causal transformer over the tick sequence ─────────────
        processed = self.processor(embedded)                  # (B, T, D)
        return processed
    
    def generate(self, x, num_ticks_to_generate):
        """
        Autoregressive generation of future ticks.
        Args:
            x: (batch, ticks, seq_len) - initial sequence
            num_ticks_to_generate: number of ticks to generate
        Returns:
            (batch, ticks + num_ticks_to_generate, seq_len)
        """
        self.eval()
        with torch.no_grad():
            current_seq = x

            for _ in range(num_ticks_to_generate):
                batch_size, ticks, seq_len = current_seq.shape

                # Embed all ticks
                x_flat = current_seq.view(batch_size * ticks, seq_len)
                embedded_flat = self.embedder(x_flat)           # (B*T, D)
                embedded = embedded_flat.view(batch_size, ticks, self.embedder.embed_dim)

                # Process through causal transformer
                processed = self.processor(embedded)            # (B, T, D)

                # Condition for the next tick = last processed representation
                condition = processed[:, -1, :]                 # (B, D)

                # Autoregressively generate the token sequence for the next tick
                next_tokens = self.decoder.generate_tokens(condition)  # (B, S)
                next_tokens = next_tokens.unsqueeze(1)                 # (B, 1, S)

                # Append the new tick
                current_seq = torch.cat([current_seq, next_tokens], dim=1)

            return current_seq
