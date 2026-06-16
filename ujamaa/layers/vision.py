import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class VisionEncoder(nn.Module):
    """
    Vision encoder for Ujamaa multi-modal model.
    Wraps CLIP ViT and projects to model dimension.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        from transformers import CLIPVisionModel
        self.encoder = CLIPVisionModel.from_pretrained(config.vision_encoder)
        self.vision_dim = self.encoder.config.hidden_size

        self.projector = nn.Sequential(
            nn.Linear(self.vision_dim, config.dim),
            nn.LayerNorm(config.dim),
            nn.GELU(),
            nn.Linear(config.dim, config.dim),
            nn.LayerNorm(config.dim),
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: [batch, 3, H, W]
        Returns:
            visual_features: [batch, num_patches, dim]
        """
        outputs = self.encoder(pixel_values)
        features = outputs.last_hidden_state  # [batch, num_patches, vision_dim]
        return self.projector(features)


class VisionLanguageConnector(nn.Module):
    """
    Aligns visual features with text feature space.
    Supports cross_attention, gated, and concat fusion.
    """

    def __init__(self, config):
        super().__init__()
        self.fusion_type = config.fusion_type

        if self.fusion_type == "cross_attention":
            self.cross_attn = nn.MultiheadAttention(
                config.dim,
                num_heads=config.n_heads,
                batch_first=True,
            )
        elif self.fusion_type == "gated":
            self.gate = nn.Sequential(
                nn.Linear(config.dim, config.dim // 4),
                nn.SiLU(),
                nn.Linear(config.dim // 4, 1),
                nn.Sigmoid(),
            )

    def forward(
        self,
        visual_features: torch.Tensor,
        text_features: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if text_features is None or self.fusion_type == "concat":
            return visual_features

        if self.fusion_type == "cross_attention":
            fused, _ = self.cross_attn(
                text_features,
                visual_features,
                visual_features,
                attn_mask=attention_mask,
            )
            return torch.cat([fused, visual_features], dim=1)

        # gated
        gate = self.gate(visual_features.mean(dim=1, keepdim=True))
        visual_boost = gate * visual_features.mean(dim=1, keepdim=True)
        text_boosted = text_features + visual_boost.expand(-1, text_features.size(1), -1)
        return torch.cat([text_boosted, visual_features], dim=1)
