"""
Visualize Ujamaa community routing for a given prompt.
Usage: python examples/visualize_routing.py [--size 1.5b]
"""
import argparse
import torch
import matplotlib.pyplot as plt
from ujamaa import ujamaa_mm
from ujamaa.utils.tokenizers import MultiModalTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", default="1.5b")
    parser.add_argument("--prompt", default="Ujamaa means cooperative economics.")
    args = parser.parse_args()

    model = ujamaa_mm(args.size)
    model.eval()
    tokenizer = MultiModalTokenizer()

    enc = tokenizer.encode(args.prompt, return_tensors="pt")
    input_ids = enc["input_ids"]

    with torch.no_grad():
        _, layer_stats, community_stats = model(input_ids=input_ids, return_stats=True)

    impacts = [s["community_impact"] for s in layer_stats]

    plt.figure(figsize=(10, 4))
    plt.bar(range(len(impacts)), impacts)
    plt.xlabel("Layer")
    plt.ylabel("Community Impact")
    plt.title(f"Ujamaa Community Routing — '{args.prompt[:40]}...'")
    plt.tight_layout()
    plt.savefig("outputs/community_routing.png", dpi=150)
    print("Saved: outputs/community_routing.png")


if __name__ == "__main__":
    main()
