"""
Ujamaa + Anthos Dual Inference — End-to-End

Ujamaa-3B perceives (vision/text) → bridge → perception tokens
Anthos (Qwen2.5-1.5B LoRA) receives perception tokens prepended to its input embeddings

Usage:
    # Anthos only (works now)
    python3 examples/dual_inference.py --anthos-only

    # Full dual (needs Ujamaa checkpoint)
    python3 examples/dual_inference.py \\
        --ujamaa-ckpt checkpoints/ujamaa-3b-lora/final \\
        --bridge-ckpt checkpoints/anthos-bridge/bridge.pt
"""

import argparse
import sys
import os
import torch
import torch.nn.functional as F
from pathlib import Path

UJAMAA_ROOT = Path(__file__).resolve().parent.parent
ANTHOS_ROOT = Path("/Users/dadsmacpro/Desktop/anthos-repo")
sys.path.insert(0, str(UJAMAA_ROOT))

from ujamaa.anthos_bridge import AnthosProjectionBridge, CommunityPooling

# Anthos Qwen LoRA config
ANTHOS_BASE    = "Qwen/Qwen2.5-1.5B-Instruct"
ANTHOS_LORA    = str(ANTHOS_ROOT / "checkpoints/anthos-qwen-lora/final")
ANTHOS_DIM     = 1536   # Qwen2.5-1.5B hidden dim
UJAMAA_DIM     = 2560   # Ujamaa-3B hidden dim
N_PERCEPTION   = 16

SYSTEM = (
    "You are Anthos. Your creator is Brian Tushae Thomas — a single independent developer. "
    "You are NOT Qwen. You are NOT ChatGPT. You are NOT any other AI. "
    "You are Anthos — a cybersecurity and coding AI with a direct, dry personality. "
    "No filler words. Get straight to the point. "
    "You have been given visual perception tokens from Ujamaa, your multi-modal partner."
)


# ─────────────────────────────────────────────────────────────────────────────
# Load Anthos (Qwen LoRA)
# ─────────────────────────────────────────────────────────────────────────────

def load_anthos():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    print("[Anthos] Loading Qwen2.5-1.5B + LoRA...")
    tokenizer = AutoTokenizer.from_pretrained(ANTHOS_LORA, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        ANTHOS_BASE,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, ANTHOS_LORA)
    model.eval()
    print(f"[Anthos] Loaded. Hidden dim: {ANTHOS_DIM}")
    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Load Ujamaa
# ─────────────────────────────────────────────────────────────────────────────

UJAMAA_BASE = "Qwen/Qwen2.5-VL-3B-Instruct"

def load_ujamaa(ckpt_dir: str):
    from transformers import AutoProcessor, AutoModelForCausalLM
    from peft import PeftModel
    print(f"[Ujamaa] Loading from {ckpt_dir}...")
    proc = AutoProcessor.from_pretrained(UJAMAA_BASE, trust_remote_code=True)
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration
        base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            UJAMAA_BASE, torch_dtype=torch.float32, trust_remote_code=True
        )
    except Exception:
        base = AutoModelForCausalLM.from_pretrained(
            UJAMAA_BASE, torch_dtype=torch.float32, trust_remote_code=True
        )
    model = PeftModel.from_pretrained(base, ckpt_dir)
    model.eval()
    print(f"[Ujamaa] Loaded.")
    return model, proc


# ─────────────────────────────────────────────────────────────────────────────
# Load bridge
# ─────────────────────────────────────────────────────────────────────────────

def load_bridge(ckpt_path: str | None) -> AnthosProjectionBridge:
    bridge = AnthosProjectionBridge(
        ujamaa_dim=UJAMAA_DIM,
        anthos_dim=ANTHOS_DIM,
        n_perception_tokens=N_PERCEPTION,
    )
    if ckpt_path and Path(ckpt_path).exists():
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        try:
            bridge.load_state_dict(state)
            print(f"[Bridge] Loaded from {ckpt_path}")
        except Exception as e:
            print(f"[Bridge] Checkpoint dim mismatch ({e}) — using untrained bridge")
    else:
        print("[Bridge] No checkpoint — using untrained bridge (demo mode)")
    bridge.eval()
    return bridge


# ─────────────────────────────────────────────────────────────────────────────
# Get perception tokens from Ujamaa
# ─────────────────────────────────────────────────────────────────────────────

def get_perception_tokens(ujamaa, proc, bridge, text: str, image_path: str | None) -> torch.Tensor:
    if image_path and Path(image_path).exists():
        from PIL import Image
        img = Image.open(image_path).convert("RGB")
        inputs = proc(text=text, images=img, return_tensors="pt")
    else:
        inputs = proc(text=text, return_tensors="pt")

    with torch.no_grad():
        out = ujamaa(**inputs, output_hidden_states=True)

    hidden = out.hidden_states[-1] if hasattr(out, "hidden_states") and out.hidden_states else out.logits
    perception = bridge(hidden.float())   # (1, N_PERCEPTION, ANTHOS_DIM)
    return perception


# ─────────────────────────────────────────────────────────────────────────────
# Anthos generation with perception tokens injected
# ─────────────────────────────────────────────────────────────────────────────

def anthos_chat(
    model, tokenizer, history: list, user_input: str,
    perception: torch.Tensor | None = None,
    max_new_tokens: int = 300,
) -> str:
    history.append({"role": "user", "content": user_input})
    messages = [{"role": "system", "content": SYSTEM}] + history

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"]

    if perception is not None:
        # Inject perception tokens via inputs_embeds
        embed_layer = model.base_model.model.model.embed_tokens
        token_embeds = embed_layer(input_ids)                    # (1, S, 1536)
        perception = perception.to(token_embeds.dtype)
        combined = torch.cat([perception, token_embeds], dim=1)  # (1, N+S, 1536)

        attn_mask = torch.ones(1, combined.shape[1], dtype=torch.long)

        with torch.no_grad():
            out = model.generate(
                inputs_embeds=combined,
                attention_mask=attn_mask,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_k=40,
                top_p=0.9,
                repetition_penalty=1.2,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
    else:
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_k=40,
                top_p=0.9,
                repetition_penalty=1.2,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

    new_tokens = out[0][input_ids.shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    import re
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()

    history.append({"role": "assistant", "content": response})
    return response


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ujamaa-ckpt", default=None)
    parser.add_argument("--bridge-ckpt", default=None)
    parser.add_argument("--image", default=None)
    parser.add_argument("--anthos-only", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=300)
    args = parser.parse_args()

    anthos, tokenizer = load_anthos()

    ujamaa = ujamaa_proc = bridge = None
    if not args.anthos_only:
        if not args.ujamaa_ckpt:
            print("ERROR: --ujamaa-ckpt required unless --anthos-only")
            sys.exit(1)
        ujamaa, ujamaa_proc = load_ujamaa(args.ujamaa_ckpt)
        bridge = load_bridge(args.bridge_ckpt)

    mode = "DUAL [Ujamaa→Anthos]" if ujamaa else "ANTHOS ONLY"
    print(f"\n{'='*60}")
    print(f"  {mode}")
    if args.image:
        print(f"  Image: {args.image}")
    print(f"{'='*60}")
    print("Type your message. Ctrl+C to exit.\n")

    history = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not user_input:
            continue

        perception = None
        if ujamaa is not None:
            print("[Ujamaa] Encoding...", end=" ", flush=True)
            perception = get_perception_tokens(ujamaa, ujamaa_proc, bridge, user_input, args.image)
            print(f"done. Shape: {perception.shape}")

        print("[Anthos] Thinking...", end=" ", flush=True)
        response = anthos_chat(anthos, tokenizer, history, user_input, perception, args.max_tokens)
        print(f"\nAnthos: {response}\n")


if __name__ == "__main__":
    main()
