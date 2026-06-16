import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from training.data import InstructionDataset, instruction_collate


class Phase3Instruction:
    """Multi-modal instruction tuning — loss only on response tokens"""

    def __init__(self, config: dict):
        self.config = config
        self.batch_size = config.get("instruction_batch_size", 8)

    def create_dataloader(self) -> DataLoader:
        dataset = InstructionDataset(self.config.get("instruction_data_path"))
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=instruction_collate,
            num_workers=self.config.get("num_workers", 4),
        )

    def compute_loss(self, logits, batch, model) -> torch.Tensor:
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = batch["input_ids"][:, 1:].contiguous()
        response_mask = batch.get("response_mask")

        if response_mask is not None:
            shift_mask = response_mask[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, model.config.vocab_size),
                shift_labels.view(-1),
                reduction="none",
            )
            return (loss * shift_mask.view(-1)).sum() / shift_mask.sum().clamp(min=1)

        return F.cross_entropy(
            shift_logits.view(-1, model.config.vocab_size),
            shift_labels.view(-1),
        )
