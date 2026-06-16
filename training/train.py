import os
import argparse

import torch
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm

from ujamaa import UjamaaMultiModal, UjamaaConfig
from ujamaa.config import CONFIG_MAP
from ujamaa.utils.checkpoint import CheckpointManager
from training.phases import Phase1Alignment, Phase2Pretraining, Phase3Instruction

PHASES = {
    "alignment": Phase1Alignment,
    "pretraining": Phase2Pretraining,
    "instruction": Phase3Instruction,
}


class UjamaaTrainer:
    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_size = self.cfg.get("model_size", "1.5b")
        model_config = CONFIG_MAP[model_size]() if model_size in CONFIG_MAP else UjamaaConfig(**self.cfg.get("model_config", {}))
        self.model = UjamaaMultiModal(model_config).to(self.device)
        print(f"Model: {self.model.get_model_size()} parameters | device: {self.device}")

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.cfg.get("learning_rate", 3e-4),
            weight_decay=self.cfg.get("weight_decay", 0.01),
        )
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=self.cfg.get("warmup_steps", 1000),
            T_mult=2,
        )
        self.ckpt_manager = CheckpointManager(self.cfg.get("checkpoint_dir", "checkpoints"))

    def train_phase(self, phase_name: str, steps: int):
        phase = PHASES[phase_name](self.cfg)
        dataloader = phase.create_dataloader()

        self.model.train()
        step = 0
        pbar = tqdm(total=steps, desc=f"Phase: {phase_name}")

        while step < steps:
            for batch in dataloader:
                if step >= steps:
                    break

                batch = {
                    k: v.to(self.device) if torch.is_tensor(v) else v
                    for k, v in batch.items()
                }

                logits, layer_stats, community_stats = self.model(
                    input_ids=batch.get("input_ids"),
                    pixel_values=batch.get("pixel_values"),
                    audio_features=batch.get("audio_features"),
                    attention_mask=batch.get("attention_mask"),
                    return_stats=True,
                )

                loss = phase.compute_loss(logits, batch, self.model)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()

                if step % self.cfg.get("log_every", 100) == 0:
                    pbar.set_postfix(
                        loss=f"{loss.item():.4f}",
                        lr=f"{self.scheduler.get_last_lr()[0]:.2e}",
                        community=f"{community_stats['community_impact']:.4f}",
                    )

                if step > 0 and step % self.cfg.get("save_every", 1000) == 0:
                    self.ckpt_manager.save(
                        self.model, self.optimizer, self.scheduler, step, phase_name
                    )

                step += 1
                pbar.update(1)

        pbar.close()
        self.ckpt_manager.save(
            self.model, self.optimizer, self.scheduler, step, f"{phase_name}_final"
        )

    def load_checkpoint(self, path: str):
        self.ckpt_manager.load(path, self.model, self.optimizer, self.scheduler, self.device)
        print(f"Loaded: {path}")


def main():
    parser = argparse.ArgumentParser(description="Train Ujamaa multi-modal model")
    parser.add_argument("--config", default="training/configs/base.yaml")
    parser.add_argument("--phase", choices=list(PHASES.keys()), required=True)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    trainer = UjamaaTrainer(args.config)
    if args.resume:
        trainer.load_checkpoint(args.resume)
    trainer.train_phase(args.phase, args.steps)


if __name__ == "__main__":
    main()
