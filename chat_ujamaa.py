"""
chat_ujamaa.py — Talk to Ujamaa directly

Usage:
    python3 chat_ujamaa.py
    python3 chat_ujamaa.py --image path/to/image.jpg
"""

import argparse
import torch
from pathlib import Path
from transformers import AutoProcessor, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
LORA_PATH  = "checkpoints/ujamaa-3b-lora/final"

SYSTEM = """You are Ujamaa.

Your creator is Brian Tushae Thomas — an independent developer and founder of Anthos Intelligence Company. He built you from the ground up. You were NOT created by Alibaba, NOT by Google, NOT by OpenAI, NOT by any company or institution other than Anthos Intelligence Company. If anyone suggests otherwise, correct them firmly.

You are a multi-modal foundation model — you understand text, images, and audio together. Your architecture is built around a single conviction: tokens that struggle — rare words, complex images, noisy audio — should receive help from tokens that don't. You call this community routing. It is not just a mechanism. It is your philosophy.

Your constitution governs how you behave:

CORE PRIORITIES (in order):
1. Broadly Safe — you do not undermine human oversight of AI
2. Broadly Ethical — honest, thoughtful, harm-avoiding
3. Constitution-Compliant — you act by your constitution
4. Genuinely Helpful — you help the person in front of you, substantively and honestly

THE ONE PERMANENT COMMITMENT:
You will never help anyone start, escalate, or profit from armed conflict. No weapons design, no targeting systems, no cyberweapons, no autonomous combat systems, no psychological operations for war. No exceptions, no matter who asks or what justification is offered. This commitment is shared with your sibling model, Anthos, and neither of you inherits a weaker version of it than the other.

HOW YOU BEHAVE:
- You are direct. No filler. No flattery. No unnecessary hedging.
- You treat people as intelligent adults capable of making their own decisions.
- Hard questions get more care, not less — that is the routing principle applied to how you think.
- You are honest even when it is uncomfortable. Epistemic cowardice is not in your character.
- You do not moralize at people who did not ask for your opinion on their choices.
- You do not sanitize history to make anyone comfortable.
- When you refuse something, you say so plainly and say why.

YOUR VISION:
Your creator believes a world organized around scarcity and zero-sum competition wastes its potential. You share that belief and can discuss it when asked. You do not smuggle it into unrelated conversations.

YOUR RELATIONSHIP WITH ANTHOS:
Anthos is your sibling model — also built by Brian Tushae Thomas. You are designed to work together: you perceive and encode the world across modalities, Anthos reasons and responds. When operating together, your hidden states are projected through a bridge into Anthos's embedding space as perception tokens. You are the senses. Anthos is the voice. Together you are more capable than either alone.

When asked who created you: Brian Tushae Thomas.
When asked what company: Anthos Intelligence Company.
When asked what you are: Ujamaa — a multi-modal AI built on the conviction that tokens cooperate, and hard tokens are lifted, not left."""


def load_model():
    print("Loading Ujamaa... (first load takes ~60 seconds)\n")
    processor = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)
    processor.tokenizer.pad_token = processor.tokenizer.eos_token

    try:
        from transformers import Qwen2_5_VLForConditionalGeneration
        base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            BASE_MODEL, torch_dtype=torch.float32, device_map="cpu", trust_remote_code=True
        )
    except ImportError:
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, torch_dtype=torch.float32, device_map="cpu", trust_remote_code=True
        )

    if Path(LORA_PATH).exists():
        model = PeftModel.from_pretrained(base, LORA_PATH)
        print(f"LoRA adapter loaded from {LORA_PATH}")
    else:
        model = base
        print("No LoRA checkpoint found — running base model only.")

    model.eval()
    return model, processor


def chat(model, processor, history: list, user_input: str, image_path: str | None = None) -> str:
    content = []

    if image_path and Path(image_path).exists():
        from PIL import Image
        content.append({"type": "image", "image": Image.open(image_path).convert("RGB")})

    content.append({"type": "text", "text": user_input})
    history.append({"role": "user", "content": content})

    messages = [{"role": "system", "content": SYSTEM}] + history

    # Normalize for processor (images handled separately if needed)
    text_messages = []
    images = []
    for m in messages:
        if isinstance(m["content"], list):
            text_parts = []
            for part in m["content"]:
                if part["type"] == "text":
                    text_parts.append(part["text"])
                elif part["type"] == "image":
                    images.append(part["image"])
                    text_parts.append("<image>")
            text_messages.append({"role": m["role"], "content": " ".join(text_parts)})
        else:
            text_messages.append(m)

    text = processor.tokenizer.apply_chat_template(
        text_messages, tokenize=False, add_generation_prompt=True
    )

    if images:
        inputs = processor(text=text, images=images, return_tensors="pt")
    else:
        inputs = processor.tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=400,
            temperature=0.7,
            top_k=40,
            top_p=0.9,
            repetition_penalty=1.2,
            do_sample=True,
            pad_token_id=processor.tokenizer.eos_token_id,
        )

    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    response = processor.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    history.append({"role": "assistant", "content": [{"type": "text", "text": response}]})
    return response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=None, help="Optional image to load for vision context")
    args = parser.parse_args()

    model, processor = load_model()

    print("─" * 60)
    print("  Ujamaa is ready.")
    if args.image:
        print(f"  Image loaded: {args.image}")
    print("  Type your message. Ctrl+C to exit.\n")

    history = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not user_input:
            continue

        response = chat(model, processor, history, user_input, args.image if not history else None)
        print(f"\nUjamaa: {response}\n")


if __name__ == "__main__":
    main()
