from typing import Dict, List, TYPE_CHECKING
from ..config import UjamaaConfig

if TYPE_CHECKING:
    from ..model import UjamaaMultiModal


class GrowthManager:
    """
    Manages progressive growth of Ujamaa from 1.5B to 100B+.
    Delegates to specialized growers for dimension, layer, and expert expansion.
    """

    def __init__(self, config: UjamaaConfig):
        self.config = config
        self.growth_history: List[int] = [config.growth_stage]

    def grow_dimension(self, model: "UjamaaMultiModal", new_dim: int) -> "UjamaaMultiModal":
        from .dimensional import DimensionGrower
        DimensionGrower().grow(model, new_dim)
        self.config.dim = new_dim
        self._record_stage()
        return model

    def grow_layers(self, model: "UjamaaMultiModal", new_layers: int) -> "UjamaaMultiModal":
        from .layer_growth import LayerGrower
        LayerGrower().grow(model, new_layers)
        self.config.n_layers = new_layers
        self._record_stage()
        return model

    def grow_experts(self, model: "UjamaaMultiModal", new_experts: int) -> "UjamaaMultiModal":
        from .expert_growth import ExpertGrower
        ExpertGrower().grow(model, new_experts)
        self.config.n_experts = new_experts
        self._record_stage()
        return model

    def _record_stage(self):
        self.config.growth_stage += 1
        self.growth_history.append(self.config.growth_stage)

    def get_report(self) -> Dict:
        return {
            "current_dim": self.config.dim,
            "current_layers": self.config.n_layers,
            "current_experts": self.config.n_experts,
            "growth_stage": self.config.growth_stage,
            "growth_history": self.growth_history,
        }
