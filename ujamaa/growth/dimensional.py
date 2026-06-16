import torch
import torch.nn as nn
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..model import UjamaaMultiModal


class DimensionGrower:
    """Expands model hidden dimension while preserving learned weights"""

    def grow(self, model: "UjamaaMultiModal", new_dim: int):
        old_dim = model.config.dim

        # Token embedding
        old_w = model.token_emb.weight.data
        new_emb = nn.Embedding(model.config.vocab_size, new_dim)
        new_emb.weight.data[:, :old_dim] = old_w
        new_emb.weight.data[:, old_dim:].normal_(0, 0.01)
        model.token_emb = new_emb

        # Positional embedding
        old_pe = model.pos_emb.data
        new_pe = nn.Parameter(torch.zeros(1, model.config.max_seq_len, new_dim))
        new_pe.data[:, :, :old_dim] = old_pe
        new_pe.data[:, :, old_dim:].normal_(0, 0.02)
        model.pos_emb = new_pe

        # Each layer
        for layer in model.layers:
            _grow_layer(layer, old_dim, new_dim, model.config)

        # LM head
        old_head_w = model.lm_head.weight.data
        new_head = nn.Linear(new_dim, model.config.vocab_size, bias=False)
        new_head.weight.data[:, :old_dim] = old_head_w
        new_head.weight.data[:, old_dim:].normal_(0, 0.01)
        model.lm_head = new_head
        model.token_emb.weight = model.lm_head.weight

        model.ln_final = nn.LayerNorm(new_dim)
        model.config.dim = new_dim


def _grow_layer(layer, old_dim: int, new_dim: int, config):
    """Expand a single UjamaaLayer's projections to new_dim"""
    kv_dim = new_dim // (config.n_heads // config.n_kv_heads)

    old_q = layer.q_proj.weight.data
    old_k = layer.k_proj.weight.data
    old_v = layer.v_proj.weight.data
    old_o = layer.out_proj.weight.data

    layer.q_proj = nn.Linear(new_dim, new_dim, bias=False)
    layer.k_proj = nn.Linear(new_dim, kv_dim, bias=False)
    layer.v_proj = nn.Linear(new_dim, kv_dim, bias=False)
    layer.out_proj = nn.Linear(new_dim, new_dim, bias=False)

    layer.q_proj.weight.data[:old_dim, :old_dim] = old_q
    layer.k_proj.weight.data[:old_k.size(0), :old_dim] = old_k
    layer.v_proj.weight.data[:old_v.size(0), :old_dim] = old_v
    layer.out_proj.weight.data[:old_dim, :old_dim] = old_o

    layer.norm1 = nn.LayerNorm(new_dim)
    layer.norm2 = nn.LayerNorm(new_dim)
