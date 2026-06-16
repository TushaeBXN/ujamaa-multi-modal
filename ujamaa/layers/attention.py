import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class MultiHeadAttention(nn.Module):
    """Multi-head attention with Grouped Query Attention (GQA) support"""

    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, dropout: float = 0.0):
        super().__init__()
        assert dim % n_heads == 0
        assert n_heads % n_kv_heads == 0

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        self.kv_head_dim = dim // n_heads
        self.repeat_factor = n_heads // n_kv_heads

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

        self.scale = math.sqrt(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch, seq, dim = x.shape

        Q = self.q_proj(x).view(batch, seq, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch, seq, self.n_kv_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch, seq, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # GQA: expand KV heads to match Q heads
        if self.repeat_factor > 1:
            K = K.repeat_interleave(self.repeat_factor, dim=1)
            V = V.repeat_interleave(self.repeat_factor, dim=1)

        scores = (Q @ K.transpose(-2, -1)) / self.scale

        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ V).transpose(1, 2).contiguous().view(batch, seq, dim)
        return self.out_proj(out)
