import pytest
import torch
from ujamaa import UjamaaMultiModal
from ujamaa.config import config_1_5b


@pytest.fixture
def small_model():
    cfg = config_1_5b()
    cfg.n_layers = 2
    cfg.n_experts = 4
    cfg.n_vision_experts = 1
    cfg.n_audio_experts = 1
    cfg.max_seq_len = 32
    return UjamaaMultiModal(cfg)


def test_grow_layers(small_model):
    initial_layers = len(small_model.layers)
    small_model.growth_manager.grow_layers(small_model, initial_layers + 2)
    assert len(small_model.layers) == initial_layers + 2
    assert small_model.config.growth_stage == 1


def test_grow_experts(small_model):
    initial_experts = len(small_model.layers[0].moe.experts)
    small_model.growth_manager.grow_experts(small_model, initial_experts + 2)
    for layer in small_model.layers:
        assert len(layer.moe.experts) == initial_experts + 2


def test_growth_report(small_model):
    small_model.growth_manager.grow_layers(small_model, len(small_model.layers) + 1)
    report = small_model.get_growth_report()
    assert report["growth_stage"] == 1
    assert len(report["growth_history"]) == 2
