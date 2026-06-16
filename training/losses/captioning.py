import torch
import torch.nn as nn
import torch.nn.functional as F


class CaptioningLoss(nn.Module):
    """Next-token prediction loss for captioning / pretraining"""

    def __init__(self, vocab_size: int, ignore_index: int = -100):
        super().__init__()
        self.vocab_size = vocab_size
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [batch, seq, vocab_size]
            labels: [batch, seq]
        """
        shift_logits = logits[:, :-1, :].contiguous().view(-1, self.vocab_size)
        shift_labels = labels[:, 1:].contiguous().view(-1)
        return F.cross_entropy(shift_logits, shift_labels, ignore_index=self.ignore_index)
