import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class Expert(nn.Module):
    """Feed-forward expert with SwiGLU activation"""

    def __init__(self, dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class ModalityExpert(Expert):
    """Expert with a learnable modality bias"""

    def __init__(self, dim: int, modality: str):
        super().__init__(dim)
        self.modality = modality
        self.modality_bias = nn.Parameter(torch.randn(1, 1, dim) * 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x) + self.modality_bias


class MixtureOfExperts(nn.Module):
    """
    Mixture of Experts with modality-specific routing.
    Includes shared experts that always activate.
    """

    def __init__(
        self,
        dim: int,
        n_experts: int = 8,
        n_vision_experts: int = 4,
        n_audio_experts: int = 4,
        n_shared_experts: int = 2,
        n_experts_per_tok: int = 2,
    ):
        super().__init__()
        self.dim = dim
        self.n_experts = n_experts
        self.n_vision_experts = n_vision_experts
        self.n_audio_experts = n_audio_experts
        self.n_shared_experts = n_shared_experts
        self.n_experts_per_tok = n_experts_per_tok

        experts = []
        for i in range(n_experts):
            if i < n_vision_experts:
                modality = "vision"
            elif i < n_vision_experts + n_audio_experts:
                modality = "audio"
            else:
                modality = "text"
            experts.append(ModalityExpert(dim, modality))
        self.experts = nn.ModuleList(experts)

        self.shared_experts = nn.ModuleList([Expert(dim) for _ in range(n_shared_experts)])

        self.router = nn.Linear(dim, n_experts, bias=False)

        self._load_balance_loss = torch.tensor(0.0)

    def forward(self, x: torch.Tensor, layer_idx: int = 0, expert_loader: Optional[callable] = None) -> torch.Tensor:
        batch, seq, dim = x.shape

        router_logits = self.router(x)
        router_probs = F.softmax(router_logits, dim=-1)

        top_k_probs, top_k_indices = torch.topk(router_probs, self.n_experts_per_tok, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        output = torch.zeros_like(x)
        expert_usage = torch.zeros(self.n_experts, device=x.device)

        for expert_idx, expert in enumerate(self.experts):
            mask = (top_k_indices == expert_idx).any(dim=-1)
            if mask.any():
                expert_input = x[mask]

                # Stream expert weights from disk if loader provides them
                streamed = None
                if expert_loader is not None:
                    streamed = expert_loader(layer_idx, expert_idx)

                if streamed is not None:
                    expert.load_state_dict(streamed)

                expert_output = expert(expert_input)

                slot = (top_k_indices[mask] == expert_idx).nonzero(as_tuple=True)[1]
                expert_weight = top_k_probs[mask].gather(1, slot.unsqueeze(1)).squeeze(1)

                output[mask] += expert_output * expert_weight.unsqueeze(-1)
                expert_usage[expert_idx] = mask.float().mean()

        for shared_expert in self.shared_experts:
            output = output + shared_expert(x)

        self._load_balance_loss = expert_usage.var()

        return output

    def get_aux_loss(self) -> torch.Tensor:
        return self._load_balance_loss
