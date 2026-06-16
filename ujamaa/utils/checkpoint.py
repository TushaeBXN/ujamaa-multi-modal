import os
import torch
from typing import Dict, Optional


class CheckpointManager:
    """Handles saving and loading Ujamaa checkpoints"""

    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save(
        self,
        model,
        optimizer,
        scheduler,
        step: int,
        phase: str,
        extra: Optional[Dict] = None,
    ) -> str:
        checkpoint = {
            "step": step,
            "phase": phase,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": model.config.to_dict(),
            "model_size": model.get_model_size(),
        }
        if extra:
            checkpoint.update(extra)

        path = os.path.join(self.checkpoint_dir, f"ujamaa_{phase}_step_{step}.pt")
        torch.save(checkpoint, path)
        return path

    def load(self, path: str, model, optimizer=None, scheduler=None, device="cpu") -> Dict:
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        return checkpoint

    def latest(self) -> Optional[str]:
        """Return path to the most recently modified checkpoint file"""
        files = [
            os.path.join(self.checkpoint_dir, f)
            for f in os.listdir(self.checkpoint_dir)
            if f.endswith(".pt")
        ]
        if not files:
            return None
        return max(files, key=os.path.getmtime)
