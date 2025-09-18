import torch
import torch.nn as nn
import math

class Perceiver(nn.Module):
    """
    Minimal Perceiver-style resampler:
    - Learned query tokens attend over input sequence (K,V) via multi-head attention
    - Returns fixed number of query embeddings (B, Nq, D)

    Args:
      pre_attention_query_token: number of learned queries (Nq)
      pre_attention_query_size: unused (kept for cfg compatibility)
      embedding_dim: model dimension (D)
      num_attn_heads: number of attention heads
      dropout: attention/ffn dropout
    """
    def __init__(
        self,
        pre_attention_query_token: int = 32,
        pre_attention_query_size: int = 512,
        embedding_dim: int = 512,
        num_attn_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_queries = pre_attention_query_token
        self.dim = embedding_dim

        # Learned queries (Nq, D)
        self.query = nn.Parameter(torch.randn(self.n_queries, self.dim) * 0.02)

        # Cross-attention: Q=learned queries, K/V=input sequence
        self.attn = nn.MultiheadAttention(
            embed_dim=self.dim,
            num_heads=num_attn_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.ln_q = nn.LayerNorm(self.dim)
        self.ln_kv = nn.LayerNorm(self.dim)

        # Lightweight FFN with residual
        self.ffn = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, 4 * self.dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * self.dim, self.dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, D)
        returns: (B, Nq, D)
        """
        assert x.dim() == 3 and x.size(-1) == self.dim, f"Expected (B,T,{self.dim}), got {tuple(x.shape)}"
        B, T, D = x.shape

        # Prepare queries per batch: (B, Nq, D)
        q = self.query.unsqueeze(0).expand(B, -1, -1)

        # LayerNorm
        q_ln = self.ln_q(q)
        kv_ln = self.ln_kv(x)

        # Cross-attention
        attn_out, _ = self.attn(q_ln, kv_ln, kv_ln, need_weights=False)
        y = q + attn_out  # residual on queries

        # Feed-forward with residual
        y = y + self.ffn(y)
        return y
