import pytest
import torch
from ujamaa import UjamaaMultiModal
from ujamaa.config import config_1_5b


@pytest.fixture
def model():
    cfg = config_1_5b()
    cfg.n_layers = 2
    cfg.n_experts = 4
    cfg.n_vision_experts = 1
    cfg.n_audio_experts = 1
    cfg.max_seq_len = 64
    return UjamaaMultiModal(cfg)


def test_model_size(model):
    size = model.get_model_size()
    assert "M" in size or "B" in size


def test_text_forward(model):
    input_ids = torch.randint(0, 100, (2, 16))
    logits = model(input_ids=input_ids)
    assert logits.shape == (2, 16, model.config.vocab_size)


def test_forward_with_stats(model):
    input_ids = torch.randint(0, 100, (1, 8))
    logits, layer_stats, community_stats = model(input_ids=input_ids, return_stats=True)
    assert len(layer_stats) == model.config.n_layers
    assert "community_impact" in community_stats


def test_generate(model):
    input_ids = torch.randint(0, 100, (1, 4))
    output = model.generate_text(input_ids, max_new_tokens=5, temperature=1.0, top_k=10)
    assert output.shape[1] >= 4


def test_growth_report(model):
    report = model.get_growth_report()
    assert "current_dim" in report
    assert "growth_stage" in report
