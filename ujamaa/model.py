"""
Ujamaa Multi-Modal Foundation Model
Built by Brian Tushae Thomas for Anthos Intelligence Company
© 2024-2025 Anthos Intelligence. All rights reserved.
"""
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import UjamaaConfig, CONFIG_MAP
from .layers.vision import VisionEncoder, VisionLanguageConnector
from .layers.audio import AudioEncoder
from .layers.community import CommunityGate
from .layers.moe import MixtureOfExperts
from .growth import GrowthManager


class UjamaaLayer(nn.Module):
    """Single transformer layer with community routing and MoE FFN"""

    def __init__(self, config: UjamaaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        kv_dim = config.dim // (config.n_heads // config.n_kv_heads)
        self.q_proj = nn.Linear(config.dim, config.dim, bias=False)
        self.k_proj = nn.Linear(config.dim, kv_dim, bias=False)
        self.v_proj = nn.Linear(config.dim, kv_dim, bias=False)
        self.out_proj = nn.Linear(config.dim, config.dim, bias=False)

        self.thought_tokens = nn.Parameter(
            torch.randn(1, config.n_thought_tokens, config.dim) * 0.02
        )

        self.moe = MixtureOfExperts(
            dim=config.dim,
            n_experts=config.n_experts,
            n_vision_experts=config.n_vision_experts,
            n_audio_experts=config.n_audio_experts,
            n_shared_experts=config.n_shared_experts,
            n_experts_per_tok=config.n_experts_per_tok,
        )

        self.community_gate = CommunityGate(config.dim, config.community_window)

        self.norm1 = nn.LayerNorm(config.dim)
        self.norm2 = nn.LayerNorm(config.dim)
        self.dropout = nn.Dropout(config.dropout)

        self._scale = math.sqrt(config.dim // config.n_heads)
        self._repeat_factor = config.n_heads // config.n_kv_heads

    def _attention(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, seq, _ = x.shape
        n_thought = self.config.n_thought_tokens

        thoughts = self.thought_tokens.expand(batch, -1, -1)
        combined = torch.cat([thoughts, x], dim=1)
        comb_len = combined.shape[1]

        Q = self.q_proj(combined).view(batch, comb_len, self.config.n_heads, -1).transpose(1, 2)
        K = self.k_proj(combined).view(batch, comb_len, self.config.n_kv_heads, -1).transpose(1, 2)
        V = self.v_proj(combined).view(batch, comb_len, self.config.n_kv_heads, -1).transpose(1, 2)

        if self._repeat_factor > 1:
            K = K.repeat_interleave(self._repeat_factor, dim=1)
            V = V.repeat_interleave(self._repeat_factor, dim=1)

        scores = (Q @ K.transpose(-2, -1)) / self._scale

        if mask is not None:
            # Extend mask to include thought token positions
            full_mask = torch.zeros(comb_len, comb_len, dtype=torch.bool, device=x.device)
            full_mask[n_thought:, n_thought:] = mask
            scores = scores.masked_fill(full_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn = F.softmax(scores, dim=-1)
        out = (attn @ V).transpose(1, 2).contiguous().view(batch, comb_len, self.config.dim)

        new_thoughts = out[:, :n_thought, :]
        seq_out = out[:, n_thought:, :]
        return self.out_proj(seq_out), new_thoughts

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        residual = x
        x = self.norm1(x)

        x, community_stats = self.community_gate(x)

        attn_out, new_thoughts = self._attention(x, mask)
        x = residual + self.dropout(attn_out)

        # Update thought tokens (EMA with detached new thoughts)
        with torch.no_grad():
            self.thought_tokens.data = (self.thought_tokens.data + new_thoughts.mean(0, keepdim=True)) / 2

        residual = x
        x = self.norm2(x)
        x = self.moe(x)
        x = residual + self.dropout(x)

        stats = {
            "community_impact": community_stats["community_impact"],
            "moe_loss": self.moe.get_aux_loss().item(),
        }

        return x, stats


class UjamaaMultiModal(nn.Module):
    """
    Ujamaa Multi-Modal Foundation Model.

    A cooperative, community-driven model processing text, images, and audio.
    Tokens cooperate, share resources, and lift each other up.

    Built by Brian Tushae Thomas for Anthos Intelligence Company.
    """

    def __init__(self, config: UjamaaConfig):
        super().__init__()
        self.config = config
        self.growth_manager = GrowthManager(config)

        self.token_emb = nn.Embedding(config.vocab_size, config.dim)
        self.pos_emb = nn.Parameter(torch.randn(1, config.max_seq_len, config.dim) * 0.02)

        self.vision_encoder = VisionEncoder(config)
        self.audio_encoder = AudioEncoder(config)
        self.vision_connector = VisionLanguageConnector(config)

        self.layers = nn.ModuleList([UjamaaLayer(config, i) for i in range(config.n_layers)])

        self.global_community = CommunityGate(config.dim, config.community_window)

        self.ln_final = nn.LayerNorm(config.dim)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)

        # Weight tying
        self.token_emb.weight = self.lm_head.weight

        self._init_weights()

    def _init_weights(self):
        std = 0.02 / math.sqrt(2 * self.config.n_layers)
        for p in self.parameters():
            if p.dim() > 1 and p is not self.token_emb.weight:
                nn.init.normal_(p, mean=0.0, std=std)

    def _embed_text(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.token_emb(input_ids)
        return x + self.pos_emb[:, : input_ids.size(1), :]

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        audio_features: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_stats: bool = False,
        return_embeddings: bool = False,
    ):
        """
        Args:
            input_ids:       [batch, seq_len]
            pixel_values:    [batch, 3, H, W]
            audio_features:  [batch, frames, audio_dim]
            attention_mask:  [batch, seq_len]
            return_stats:    also return per-layer stats and community stats
            return_embeddings: return final hidden states instead of logits
        """
        parts = []

        text_features = None
        if input_ids is not None:
            text_features = self._embed_text(input_ids)
            parts.append(text_features)

        if pixel_values is not None:
            vis = self.vision_encoder(pixel_values)
            vis = self.vision_connector(vis, text_features)
            parts.append(vis)

        if audio_features is not None:
            aud = self.audio_encoder(audio_features)
            parts.append(aud)

        combined = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]

        combined, community_stats = self.global_community(combined)

        seq_len = combined.size(1)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=combined.device),
            diagonal=1,
        )

        layer_stats: List[Dict] = []
        for layer in self.layers:
            combined, stats = layer(combined, causal_mask)
            layer_stats.append(stats)

        combined = self.ln_final(combined)

        if return_embeddings:
            return combined

        logits = self.lm_head(combined)

        if return_stats:
            return logits, layer_stats, community_stats

        return logits

    @torch.no_grad()
    def generate_text(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        pixel_values: Optional[torch.Tensor] = None,
        audio_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self.eval()
        generated = input_ids

        for _ in range(max_new_tokens):
            context = generated[:, -self.config.max_seq_len :]

            logits = self.forward(
                input_ids=context,
                pixel_values=pixel_values,
                audio_features=audio_features,
            )

            next_logits = logits[:, -1, :] / temperature

            if top_k > 0:
                threshold = torch.topk(next_logits, top_k)[0][..., -1, None]
                next_logits = next_logits.masked_fill(next_logits < threshold, float("-inf"))

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

            if next_token.item() == 0:  # EOS
                break

        return generated

    def get_growth_report(self) -> Dict:
        return self.growth_manager.get_report()

    def get_model_size(self) -> str:
        params = sum(p.numel() for p in self.parameters())
        return f"{params / 1e9:.2f}B" if params >= 1e9 else f"{params / 1e6:.1f}M"


def ujamaa_mm(config_size: str = "1.5b") -> UjamaaMultiModal:
    if config_size not in CONFIG_MAP:
        raise ValueError(f"config_size must be one of {list(CONFIG_MAP.keys())}")
    return UjamaaMultiModal(CONFIG_MAP[config_size]())


def ujamaa_mm_1_5b() -> UjamaaMultiModal:
    return ujamaa_mm("1.5b")


def ujamaa_mm_3b() -> UjamaaMultiModal:
    return ujamaa_mm("3b")


def ujamaa_mm_7b() -> UjamaaMultiModal:
    return ujamaa_mm("7b")


def ujamaa_mm_13b() -> UjamaaMultiModal:
    return ujamaa_mm("13b")


def ujamaa_mm_34b() -> UjamaaMultiModal:
    return ujamaa_mm("34b")


def ujamaa_mm_70b() -> UjamaaMultiModal:
    return ujamaa_mm("70b")


def ujamaa_mm_100b() -> UjamaaMultiModal:
    return ujamaa_mm("100b")
