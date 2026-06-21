"""
Ujamaa → Anthos Projection Bridge

Maps Ujamaa-3B's hidden states (dim=2560) into Anthos's embedding space
(dim=512 for current checkpoints, or 2048/3072 for future 1B/3B Anthos).

The bridge outputs a sequence of "perception tokens" formatted with Anthos's
THT (thought-token) special token so Anthos treats Ujamaa's visual/audio
understanding as pre-formed thoughts it can reason over.

Usage:
    bridge = AnthosProjectionBridge(ujamaa_dim=2560, anthos_dim=512)
    # ujamaa_hidden: (batch, seq, 2560) — last hidden state from Ujamaa
    perception_tokens = bridge(ujamaa_hidden)
    # perception_tokens: (batch, n_perception, 512) — prepend to Anthos input
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# Anthos special token IDs (from anthos tokenizer)
THT_TOKEN_ID = 50259   # thought token — Anthos treats content after this as internal reasoning
AST_TOKEN_ID = 50260   # assistant token
END_TOKEN_ID = 50261   # end token


class CommunityPooling(nn.Module):
    """
    Compress Ujamaa's variable-length hidden sequence into a fixed number of
    perception tokens using learned cross-attention — the Ujamaa way, tokens
    cooperating to summarize themselves.
    """

    def __init__(self, ujamaa_dim: int, n_perception_tokens: int = 16):
        super().__init__()
        self.n_perception_tokens = n_perception_tokens
        # Learnable query vectors — one per perception slot
        self.queries = nn.Parameter(torch.randn(1, n_perception_tokens, ujamaa_dim) * 0.02)
        self.attn = nn.MultiheadAttention(ujamaa_dim, num_heads=8, batch_first=True, dropout=0.0)
        self.norm = nn.LayerNorm(ujamaa_dim)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        # hidden: (B, S, ujamaa_dim)
        B = hidden.size(0)
        queries = self.queries.expand(B, -1, -1)
        pooled, _ = self.attn(queries, hidden, hidden)
        return self.norm(pooled)   # (B, n_perception_tokens, ujamaa_dim)


class AnthosProjectionBridge(nn.Module):
    """
    Full Ujamaa → Anthos bridge.

    Pipeline:
        Ujamaa hidden states
            → CommunityPooling (compress sequence)
            → Linear projection (ujamaa_dim → anthos_dim)
            → LayerNorm + GeLU gate
            → perception token embeddings in Anthos space
    """

    def __init__(
        self,
        ujamaa_dim: int = 2560,       # Ujamaa-3B hidden dim
        anthos_dim: int = 512,         # Anthos current dim (512); set 2048/3072 for future Anthos
        n_perception_tokens: int = 16, # how many tokens Anthos sees from Ujamaa
        dropout: float = 0.05,
    ):
        super().__init__()
        self.ujamaa_dim = ujamaa_dim
        self.anthos_dim = anthos_dim
        self.n_perception_tokens = n_perception_tokens

        self.pooling = CommunityPooling(ujamaa_dim, n_perception_tokens)

        # Two-layer projection with gating — gives the bridge capacity to
        # do non-trivial remapping between the two latent spaces
        self.proj1 = nn.Linear(ujamaa_dim, anthos_dim * 2)
        self.proj2 = nn.Linear(anthos_dim * 2, anthos_dim)
        self.norm = nn.LayerNorm(anthos_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, ujamaa_hidden: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ujamaa_hidden: (B, S, ujamaa_dim) — Ujamaa last hidden states
        Returns:
            (B, n_perception_tokens, anthos_dim) — ready to prepend to Anthos input_embeds
        """
        pooled = self.pooling(ujamaa_hidden)          # (B, N, ujamaa_dim)
        x = self.proj1(pooled)                        # (B, N, anthos_dim*2)
        x = F.gelu(x)
        x = self.proj2(x)                             # (B, N, anthos_dim)
        x = self.drop(self.norm(x))
        return x


class UjamaaBridge(nn.Module):
    """
    Top-level module that wraps Ujamaa + bridge for joint inference.

    Ujamaa encodes vision/audio/text → bridge produces perception tokens →
    caller prepends those tokens to Anthos's embedding input.

    This keeps Ujamaa and Anthos as separate models (no weight sharing),
    which means:
      - Either can be updated independently
      - Bridge is the only trained connector
      - Bridge checkpoint is small (~5M params for 3B→512 config)
    """

    def __init__(self, ujamaa_model, bridge: AnthosProjectionBridge):
        super().__init__()
        self.ujamaa = ujamaa_model
        self.bridge = bridge

    def encode(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor | None = None,
        audio_values: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Run Ujamaa and return perception tokens for Anthos.
        Returns: (B, n_perception_tokens, anthos_dim)
        """
        with torch.no_grad():
            outputs = self.ujamaa(
                input_ids=input_ids,
                pixel_values=pixel_values,
                audio_values=audio_values,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
        # Use last hidden state; fall back to logits-derived hidden if needed
        hidden = outputs.hidden_states[-1] if hasattr(outputs, "hidden_states") else outputs[0]
        return self.bridge(hidden)

    def forward(self, *args, **kwargs):
        return self.encode(*args, **kwargs)


def build_bridge(anthos_dim: int = 512, n_perception_tokens: int = 16) -> AnthosProjectionBridge:
    """Convenience constructor for Ujamaa-3B → current Anthos checkpoint config."""
    return AnthosProjectionBridge(
        ujamaa_dim=2560,
        anthos_dim=anthos_dim,
        n_perception_tokens=n_perception_tokens,
    )


def prepend_perception_tokens(
    perception: torch.Tensor,   # (B, N, anthos_dim)
    anthos_embeds: torch.Tensor # (B, S, anthos_dim)
) -> torch.Tensor:
    """
    Concatenate perception tokens before Anthos's normal token embeddings.
    Call this after anthos_model.embed(input_ids) and before the transformer layers.
    """
    return torch.cat([perception, anthos_embeds], dim=1)
