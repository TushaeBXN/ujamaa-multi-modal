import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from training.data import MultiModalDataset, multi_modal_collate


class Phase2Pretraining:
    """Multi-modal pretraining — next-token prediction across all modalities"""

    def __init__(self, config: dict):
        self.config = config
        self.batch_size = config.get("pretraining_batch_size", 16)

    def create_dataloader(self) -> DataLoader:
        dataset = MultiModalDataset(self.config.get("pretraining_data_path"))
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=multi_modal_collate,
            num_workers=self.config.get("num_workers", 4),
        )

    def compute_loss(self, logits, batch, model) -> torch.Tensor:
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = batch["input_ids"][:, 1:].contiguous()
        return F.cross_entropy(
            shift_logits.view(-1, model.config.vocab_size),
            shift_labels.view(-1),
        )
