import torch
import torch.nn as nn
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..model import UjamaaMultiModal


class ExpertGrower:
    """Adds MoE experts to each layer, expanding the router accordingly"""

    def grow(self, model: "UjamaaMultiModal", new_experts: int):
        from ..layers.moe import ModalityExpert

        for layer in model.layers:
            moe = layer.moe
            current = len(moe.experts)
            if new_experts <= current:
                continue

            new_list = list(moe.experts)
            for i in range(current, new_experts):
                expert = ModalityExpert(layer.config.dim, "text")
                expert.load_state_dict(moe.experts[0].state_dict())
                for p in expert.parameters():
                    p.data += torch.randn_like(p) * 0.01
                new_list.append(expert)
            moe.experts = nn.ModuleList(new_list)

            # Expand router
            old_w = moe.router.weight.data
            new_router = nn.Linear(layer.config.dim, new_experts, bias=False)
            new_router.weight.data[:current] = old_w
            new_router.weight.data[current:].normal_(0, 0.01)
            moe.router = new_router
            moe.n_experts = new_experts
