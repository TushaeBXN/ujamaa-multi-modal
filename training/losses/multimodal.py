import torch
import torch.nn as nn
from .contrastive import ContrastiveLoss
from .captioning import CaptioningLoss


class MultiModalLoss(nn.Module):
    """
    Combined loss: captioning + contrastive + MoE load-balancing auxiliary loss
    """

    def __init__(self, config: dict):
        super().__init__()
        self.captioning = CaptioningLoss(vocab_size=config.get("vocab_size", 50277))
        self.contrastive = ContrastiveLoss(temperature=config.get("temperature", 0.07))
        self.lm_weight = config.get("lm_weight", 1.0)
        self.contrastive_weight = config.get("contrastive_weight", 0.1)
        self.aux_weight = config.get("aux_weight", 0.01)

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        text_emb: torch.Tensor = None,
        vision_emb: torch.Tensor = None,
        aux_loss: torch.Tensor = None,
    ) -> torch.Tensor:
        loss = self.lm_weight * self.captioning(logits, labels)

        if text_emb is not None and vision_emb is not None:
            loss = loss + self.contrastive_weight * self.contrastive(
                text_emb.mean(1), vision_emb.mean(1)
            )

        if aux_loss is not None:
            loss = loss + self.aux_weight * aux_loss

        return loss
