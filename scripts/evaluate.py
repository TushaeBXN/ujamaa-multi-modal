"""
Evaluate a trained Ujamaa model on a held-out text dataset.
Usage: python scripts/evaluate.py --checkpoint PATH --data PATH
"""
import argparse
import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from ujamaa import UjamaaMultiModal, UjamaaConfig
from training.data import MultiModalDataset, multi_modal_collate


def compute_perplexity(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            logits = model(input_ids=batch["input_ids"])
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = batch["input_ids"][:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, model.config.vocab_size),
                shift_labels.view(-1),
                reduction="sum",
            )
            total_loss += loss.item()
            total_tokens += shift_labels.numel()

    return math.exp(total_loss / total_tokens)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = UjamaaConfig(**checkpoint["config"])
    model = UjamaaMultiModal(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    dataset = MultiModalDataset(args.data)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=multi_modal_collate)

    ppl = compute_perplexity(model, dataloader, device)
    print(f"Perplexity: {ppl:.2f}")


if __name__ == "__main__":
    main()
