import pytest
import torch
from ujamaa.layers.community import CommunityGate, TokenCollective
from ujamaa.layers.moe import MixtureOfExperts, Expert
from ujamaa.layers.fusion import MultiModalFusion
from ujamaa.layers.attention import MultiHeadAttention


def test_expert_forward():
    expert = Expert(64)
    x = torch.randn(2, 8, 64)
    out = expert(x)
    assert out.shape == x.shape


def test_moe_forward():
    moe = MixtureOfExperts(dim=64, n_experts=4, n_vision_experts=1, n_audio_experts=1,
                           n_shared_experts=1, n_experts_per_tok=2)
    x = torch.randn(2, 8, 64)
    out = moe(x)
    assert out.shape == x.shape


def test_community_gate():
    gate = CommunityGate(dim=64, window=16)
    x = torch.randn(2, 8, 64)
    out, stats = gate(x)
    assert out.shape == x.shape
    assert "community_impact" in stats


def test_token_collective_variable_seq():
    tc = TokenCollective(dim=64, window=16)
    x = torch.randn(1, 32, 64)  # seq > window
    out = tc(x)
    assert out.shape == x.shape


def test_fusion():
    fusion = MultiModalFusion(dim=64, n_modalities=3)
    features = [torch.randn(2, 8, 64) for _ in range(3)]
    out = fusion(features)
    assert out.shape == (2, 8, 64)


def test_attention_gqa():
    attn = MultiHeadAttention(dim=64, n_heads=8, n_kv_heads=2)
    x = torch.randn(2, 10, 64)
    out = attn(x)
    assert out.shape == x.shape
