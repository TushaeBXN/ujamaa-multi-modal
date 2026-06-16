import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..model import UjamaaMultiModal


class LayerGrower:
    """Adds layers to the model, initializing from the last existing layer"""

    def grow(self, model: "UjamaaMultiModal", new_layers: int):
        from ..model import UjamaaLayer

        current = len(model.layers)
        if new_layers <= current:
            return

        reference = model.layers[-1]
        for i in range(current, new_layers):
            new_layer = UjamaaLayer(model.config, i)
            new_layer.load_state_dict(reference.state_dict())
            for p in new_layer.parameters():
                p.data += torch.randn_like(p) * 0.01
            model.layers.append(new_layer)

        model.config.n_layers = new_layers
