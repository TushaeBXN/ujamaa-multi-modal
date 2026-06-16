"""
Export Ujamaa checkpoint for deployment (weights only, no optimizer state).
Usage: python scripts/export_model.py --checkpoint PATH --output PATH
"""
import argparse
import torch
from ujamaa import UjamaaMultiModal, UjamaaConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = UjamaaConfig(**checkpoint["config"])
    model = UjamaaMultiModal(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    export = {
        "model_state_dict": model.state_dict(),
        "config": checkpoint["config"],
        "model_size": model.get_model_size(),
    }
    torch.save(export, args.output)
    print(f"Exported {model.get_model_size()} model → {args.output}")


if __name__ == "__main__":
    main()
