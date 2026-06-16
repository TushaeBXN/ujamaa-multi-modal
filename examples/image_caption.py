"""
Image captioning example.
Usage: python examples/image_caption.py --image PATH [--size 1.5b]
"""
import argparse
import torch
from PIL import Image
from ujamaa import ujamaa_mm
from ujamaa.utils.tokenizers import MultiModalTokenizer
from inference.multimodal import MultiModalGenerator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--size", default="1.5b")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ujamaa_mm(args.size)

    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location=device)
        key = "model_state_dict" if "model_state_dict" in state else None
        model.load_state_dict(state[key] if key else state)

    image = Image.open(args.image).convert("RGB")
    tokenizer = MultiModalTokenizer()
    gen = MultiModalGenerator(model.to(device), tokenizer, device)

    caption = gen.generate("Describe this image in detail.", image=image, max_new_tokens=150)
    print(f"Caption: {caption}")


if __name__ == "__main__":
    main()
