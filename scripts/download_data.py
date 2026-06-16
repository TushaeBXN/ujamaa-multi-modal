"""
Download sample training data for Ujamaa.
Usage: python scripts/download_data.py [--phase alignment|pretraining|instruction]
"""
import os
import argparse
import json


SAMPLE_ALIGNMENT = [
    {"text": "A photo of a cat.", "image_path": "data/alignment/images/cat.jpg"},
    {"text": "A photo of a dog.", "image_path": "data/alignment/images/dog.jpg"},
]

SAMPLE_PRETRAINING = [
    {"text": "Ujamaa means cooperative economics in Swahili."},
    {"text": "Multi-modal models process text, images, and audio together."},
]

SAMPLE_INSTRUCTION = [
    {
        "instruction": "Describe this image.",
        "response": "The image shows a scenic landscape.",
        "image_path": "data/instruction/images/landscape.jpg",
    }
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", default="all", choices=["all", "alignment", "pretraining", "instruction"])
    args = parser.parse_args()

    os.makedirs("data/alignment", exist_ok=True)
    os.makedirs("data/pretraining", exist_ok=True)
    os.makedirs("data/instruction", exist_ok=True)

    if args.phase in ("all", "alignment"):
        with open("data/alignment/samples.json", "w") as f:
            json.dump(SAMPLE_ALIGNMENT, f, indent=2)
        print("Created: data/alignment/samples.json")

    if args.phase in ("all", "pretraining"):
        with open("data/pretraining/samples.json", "w") as f:
            json.dump(SAMPLE_PRETRAINING, f, indent=2)
        print("Created: data/pretraining/samples.json")

    if args.phase in ("all", "instruction"):
        with open("data/instruction/samples.json", "w") as f:
            json.dump(SAMPLE_INSTRUCTION, f, indent=2)
        print("Created: data/instruction/samples.json")

    print("\nData scaffold ready. Replace with real datasets before training.")


if __name__ == "__main__":
    main()
