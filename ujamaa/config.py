from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class UjamaaConfig:
    """Configuration for Ujamaa multi-modal model with growth support"""

    # Text backbone
    vocab_size: int = 50277
    dim: int = 2048
    n_layers: int = 24
    n_heads: int = 16
    n_kv_heads: int = 8
    max_seq_len: int = 8192
    dropout: float = 0.1

    # MoE
    n_experts: int = 8
    n_shared_experts: int = 2
    n_experts_per_tok: int = 2
    expert_dim_scale: int = 4

    # Ujamaa-specific
    n_thought_tokens: int = 8
    community_window: int = 512

    # Multi-modal
    vision_encoder: str = "openai/clip-vit-large-patch14"
    vision_dim: int = 1024
    audio_encoder: str = "openai/whisper-large-v3"
    audio_dim: int = 1280
    fusion_type: str = "cross_attention"  # "cross_attention" | "gated" | "concat"

    # Multi-modal experts
    n_vision_experts: int = 4
    n_audio_experts: int = 4
    n_text_experts: int = 8

    # Growth
    growth_stage: int = 0
    base_config: Optional["UjamaaConfig"] = field(default=None, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vocab_size": self.vocab_size,
            "dim": self.dim,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "n_kv_heads": self.n_kv_heads,
            "max_seq_len": self.max_seq_len,
            "dropout": self.dropout,
            "n_experts": self.n_experts,
            "n_shared_experts": self.n_shared_experts,
            "n_experts_per_tok": self.n_experts_per_tok,
            "n_vision_experts": self.n_vision_experts,
            "n_audio_experts": self.n_audio_experts,
            "n_text_experts": self.n_text_experts,
            "n_thought_tokens": self.n_thought_tokens,
            "community_window": self.community_window,
            "vision_encoder": self.vision_encoder,
            "vision_dim": self.vision_dim,
            "audio_encoder": self.audio_encoder,
            "audio_dim": self.audio_dim,
            "fusion_type": self.fusion_type,
            "growth_stage": self.growth_stage,
        }


def config_1_5b() -> UjamaaConfig:
    return UjamaaConfig(
        dim=2048,
        n_layers=24,
        n_heads=16,
        n_kv_heads=8,
        n_experts=8,
        n_shared_experts=2,
        n_experts_per_tok=2,
        n_thought_tokens=8,
        vision_dim=768,
        audio_dim=512,
    )


def config_3b() -> UjamaaConfig:
    return UjamaaConfig(
        dim=2560,
        n_layers=28,
        n_heads=20,
        n_kv_heads=10,
        n_experts=16,
        n_shared_experts=2,
        n_experts_per_tok=2,
        n_thought_tokens=12,
        vision_dim=1024,
        audio_dim=768,
    )


def config_7b() -> UjamaaConfig:
    return UjamaaConfig(
        dim=4096,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        n_experts=32,
        n_shared_experts=4,
        n_experts_per_tok=3,
        n_thought_tokens=16,
        vision_dim=1024,
        audio_dim=768,
        max_seq_len=8192,
    )


def config_13b() -> UjamaaConfig:
    return UjamaaConfig(
        dim=5120,
        n_layers=40,
        n_heads=40,
        n_kv_heads=10,
        n_experts=48,
        n_shared_experts=6,
        n_experts_per_tok=4,
        n_thought_tokens=24,
        vision_dim=1408,
        audio_dim=1024,
        max_seq_len=16384,
    )


def config_34b() -> UjamaaConfig:
    return UjamaaConfig(
        dim=8192,
        n_layers=48,
        n_heads=64,
        n_kv_heads=8,
        n_experts=64,
        n_shared_experts=8,
        n_experts_per_tok=4,
        n_thought_tokens=32,
        vision_dim=1664,
        audio_dim=1152,
        max_seq_len=32768,
    )


def config_70b() -> UjamaaConfig:
    return UjamaaConfig(
        dim=10240,
        n_layers=80,
        n_heads=80,
        n_kv_heads=8,
        n_experts=128,
        n_shared_experts=12,
        n_experts_per_tok=6,
        n_thought_tokens=48,
        vision_dim=2048,
        audio_dim=1408,
        max_seq_len=131072,
    )


def config_100b() -> UjamaaConfig:
    return UjamaaConfig(
        dim=12288,
        n_layers=96,
        n_heads=96,
        n_kv_heads=8,
        n_experts=256,
        n_shared_experts=16,
        n_experts_per_tok=8,
        n_thought_tokens=64,
        vision_dim=2560,
        audio_dim=1664,
        max_seq_len=131072,
    )


CONFIG_MAP = {
    "1.5b": config_1_5b,
    "3b": config_3b,
    "7b": config_7b,
    "13b": config_13b,
    "34b": config_34b,
    "70b": config_70b,
    "100b": config_100b,
}
