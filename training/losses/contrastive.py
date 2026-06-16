import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """CLIP-style symmetric contrastive loss for modality alignment"""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings_a: torch.Tensor, embeddings_b: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings_a: [batch, dim] — e.g. text embeddings
            embeddings_b: [batch, dim] — e.g. vision embeddings
        """
        a = F.normalize(embeddings_a, dim=-1)
        b = F.normalize(embeddings_b, dim=-1)

        sim = torch.matmul(a, b.T) / self.temperature
        labels = torch.arange(sim.size(0), device=sim.device)

        loss_ab = F.cross_entropy(sim, labels)
        loss_ba = F.cross_entropy(sim.T, labels)
        return (loss_ab + loss_ba) / 2
