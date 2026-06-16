"""
Multi-modal chat example for Ujamaa.
Usage: python examples/chat.py [--checkpoint PATH] [--image PATH] [--size 1.5b]
"""
import argparse
import torch
from ujamaa import ujamaa_mm
from ujamaa.utils.tokenizers import MultiModalTokenizer
from inference.multimodal import MultiModalGenerator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--size", type=str, default="1.5b")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ujamaa_mm(args.size)

    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location=device)
        key = "model_state_dict" if "model_state_dict" in state else None
        model.load_state_dict(state[key] if key else state)

    model = model.to(device)
    tokenizer = MultiModalTokenizer()
    gen = MultiModalGenerator(model, tokenizer, device)

    image = None
    if args.image:
        from PIL import Image
        image = Image.open(args.image).convert("RGB")
        print(f"Loaded image: {args.image}")

    print(f"\nUjamaa {args.size} ({model.get_model_size()}) — Multi-Modal Chat")
    print("Built by Brian Tushae Thomas for Anthos Intelligence Company")
    print("Type 'quit' to exit.\n")

    while True:
        prompt = input("You: ").strip()
        if prompt.lower() in ("quit", "exit", "q"):
            break
        if not prompt:
            continue

        response = gen.generate(prompt, image=image, max_new_tokens=200, temperature=0.8)
        print(f"Ujamaa: {response}\n")


if __name__ == "__main__":
    main()
