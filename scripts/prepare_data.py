"""
Prepare and tokenize training data.
Usage: python scripts/prepare_data.py --input data/raw --output data/processed
"""
import os
import argparse
import json
import torch
from ujamaa.utils.tokenizers import MultiModalTokenizer


def tokenize_text_samples(input_path: str, output_path: str, max_length: int = 2048):
    tokenizer = MultiModalTokenizer()
    os.makedirs(output_path, exist_ok=True)

    with open(input_path) as f:
        samples = json.load(f)

    processed = []
    for sample in samples:
        text = sample.get("text", sample.get("instruction", ""))
        enc = tokenizer.encode(text, max_length=max_length, padding=False, return_tensors="pt")
        processed.append({"input_ids": enc["input_ids"][0].tolist()})

    out_file = os.path.join(output_path, "tokenized.json")
    with open(out_file, "w") as f:
        json.dump(processed, f)
    print(f"Processed {len(processed)} samples → {out_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-length", type=int, default=2048)
    args = parser.parse_args()

    if os.path.isfile(args.input):
        tokenize_text_samples(args.input, args.output, args.max_length)
    else:
        for fname in os.listdir(args.input):
            if fname.endswith(".json"):
                tokenize_text_samples(
                    os.path.join(args.input, fname),
                    args.output,
                    args.max_length,
                )


if __name__ == "__main__":
    main()
