"""
Ujamaa-3B HuggingFace Fine-Tune Setup
Run this first to verify your environment and download datasets.

Local (Mac):   python scripts/setup_3b_training.py --mode local
RunPod:        python scripts/setup_3b_training.py --mode runpod
"""

import argparse
import sys

def check_env(mode: str):
    print("=== Checking environment ===")
    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if mode == "runpod" and not torch.cuda.is_available():
        print("WARNING: No CUDA detected. Are you sure you're on RunPod?")
        sys.exit(1)
    if mode == "local":
        print("Local mode — using CPU/MPS, float32, SmolVLM-500M")

def install_deps(mode: str):
    import subprocess
    base = [
        "pip install -q",
        "transformers==4.46.0",   # supports Qwen2.5-VL and SmolVLM
        "peft>=0.12.0",
        "datasets>=2.20.0",
        "trl>=0.9.0",
        '"accelerate>=0.30.0"',
        "pillow",
        "soundfile",
    ]
    if mode == "runpod":
        base += ['"torch>=2.4.0"', "bitsandbytes"]
    cmd = " ".join(base)
    print(f"\n=== Installing deps ===\n{cmd}\n")
    subprocess.run(cmd, shell=True, check=True)

def download_datasets(mode: str):
    from datasets import load_dataset
    print("\n=== Downloading datasets ===")

    configs = [
        ("HuggingFaceH4/ultrachat_200k", "train_sft", 5000 if mode == "local" else 50000),
        ("HuggingFaceH4/no_robots", "train", 500 if mode == "local" else 10000),
    ]
    if mode != "local":
        configs += [
            ("liuhaotian/LLaVA-Instruct-150K", "train", 80000),
            ("lmms-lab/LLaVA-OneVision-Data", "train", 50000),
        ]

    for name, split, n in configs:
        print(f"  {name} ({split}, {n} samples)...", end=" ", flush=True)
        try:
            ds = load_dataset(name, split=split, streaming=True)
            # Just verify it loads; actual caching happens during training
            _ = next(iter(ds))
            print("OK")
        except Exception as e:
            print(f"FAILED: {e}")

def verify_bridge():
    print("\n=== Verifying Anthos bridge ===")
    import sys
    sys.path.insert(0, ".")
    import torch
    from ujamaa.anthos_bridge import build_bridge, prepend_perception_tokens

    bridge = build_bridge(anthos_dim=512, n_perception_tokens=16)
    n_params = sum(p.numel() for p in bridge.parameters())
    print(f"Bridge params: {n_params:,} ({n_params/1e6:.1f}M)")

    # Smoke test
    fake_ujamaa_hidden = torch.randn(2, 64, 2560)  # batch=2, seq=64, dim=2560
    perception = bridge(fake_ujamaa_hidden)
    print(f"Input shape:       {fake_ujamaa_hidden.shape}")
    print(f"Perception shape:  {perception.shape}")  # expect (2, 16, 512)

    fake_anthos_embeds = torch.randn(2, 128, 512)
    combined = prepend_perception_tokens(perception, fake_anthos_embeds)
    print(f"Combined shape:    {combined.shape}")  # expect (2, 144, 512)
    print("Bridge OK")

def load_base_model(mode: str):
    print("\n=== Loading base model ===")
    import torch
    from transformers import AutoProcessor, AutoModelForCausalLM

    model_id = (
        "HuggingFaceTB/SmolVLM-500M-Instruct" if mode == "local"
        else "Qwen/Qwen2.5-VL-3B-Instruct"
    )
    dtype = torch.float32 if mode == "local" else torch.bfloat16

    print(f"  Loading {model_id} in {dtype}...")
    try:
        from transformers import AutoProcessor, AutoModelForCausalLM
        proc = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        try:
            from transformers import AutoModelForVision2Seq
            model = AutoModelForVision2Seq.from_pretrained(model_id, torch_dtype=dtype, trust_remote_code=True)
        except (ImportError, Exception):
            model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, trust_remote_code=True)
        n = sum(p.numel() for p in model.parameters())
        print(f"  Loaded. Params: {n/1e9:.2f}B")
        print("  Base model OK")
    except Exception as e:
        print(f"  FAILED: {e}")
        print("  → Try: pip install transformers --upgrade")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["local", "runpod"], default="local")
    parser.add_argument("--skip-deps", action="store_true")
    parser.add_argument("--skip-datasets", action="store_true")
    args = parser.parse_args()

    check_env(args.mode)
    if not args.skip_deps:
        install_deps(args.mode)
    if not args.skip_datasets:
        download_datasets(args.mode)
    verify_bridge()
    load_base_model(args.mode)

    print("\n=== Setup complete ===")
    if args.mode == "local":
        print("Next: python training/train_hf.py --config training/configs/3b_hf.yaml --local")
    else:
        print("Next: python training/train_hf.py --config training/configs/3b_hf.yaml")

if __name__ == "__main__":
    main()
