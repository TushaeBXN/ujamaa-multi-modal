import torch
import torch.nn as nn


class MultiModalFusion(nn.Module):
    """Fuses multiple modality feature tensors with learned gating"""

    def __init__(self, dim: int, n_modalities: int = 3):
        super().__init__()
        self.n_modalities = n_modalities

        self.proj = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(n_modalities)])

        self.fusion_gate = nn.Sequential(
            nn.Linear(dim * n_modalities, dim),
            nn.SiLU(),
            nn.Linear(dim, n_modalities),
            nn.Softmax(dim=-1),
        )

    def forward(self, features: list) -> torch.Tensor:
        """
        Args:
            features: list of [batch, seq, dim] tensors, one per modality
        Returns:
            fused: [batch, seq, dim]
        """
        projected = [proj(f) for proj, f in zip(self.proj, features)]

        # [batch, seq, dim, n_modalities]
        stacked = torch.stack(projected, dim=-1)

        flat = stacked.view(stacked.size(0), stacked.size(1), -1)
        weights = self.fusion_gate(flat)  # [batch, seq, n_modalities]

        return (stacked * weights.unsqueeze(2)).sum(dim=-1)
