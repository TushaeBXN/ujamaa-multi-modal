import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from training.data import AlignmentDataset, alignment_collate


class Phase1Alignment:
    """Feature alignment phase — align vision/audio embeddings with text space"""

    def __init__(self, config: dict):
        self.config = config
        self.batch_size = config.get("alignment_batch_size", 32)

    def create_dataloader(self) -> DataLoader:
        dataset = AlignmentDataset(self.config.get("alignment_data_path"))
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=alignment_collate,
            num_workers=self.config.get("num_workers", 4),
        )

    def compute_loss(self, logits, batch, model) -> torch.Tensor:
        embeddings = model(
            input_ids=batch.get("input_ids"),
            pixel_values=batch.get("pixel_values"),
            audio_features=batch.get("audio_features"),
            return_embeddings=True,
        )

        text_len = batch.get("text_len", embeddings.size(1) // 2)
        vision_len = batch.get("vision_len", embeddings.size(1) - text_len)

        text_emb = embeddings[:, :text_len, :].mean(dim=1)
        vision_emb = embeddings[:, text_len: text_len + vision_len, :].mean(dim=1)

        text_emb = F.normalize(text_emb, dim=-1)
        vision_emb = F.normalize(vision_emb, dim=-1)

        sim = torch.matmul(text_emb, vision_emb.T) / 0.07
        labels = torch.arange(sim.size(0), device=sim.device)
        return (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2
