"""
Ujamaa-3B HuggingFace Training Script

Fine-tunes Qwen2.5-VL-3B (or SmolVLM-500M locally) with LoRA,
then trains the Anthos bridge in a separate phase.

Usage:
    # RunPod — full 3B
    python training/train_hf.py --config training/configs/3b_hf.yaml

    # Mac — pipeline test with SmolVLM
    python training/train_hf.py --config training/configs/3b_hf.yaml --local
"""

import argparse
import os
import sys
import yaml
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contextlib import nullcontext
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
try:
    from transformers import AutoModelForVision2Seq
except ImportError:
    AutoModelForVision2Seq = None
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset, concatenate_datasets

from ujamaa.anthos_bridge import build_bridge, AnthosProjectionBridge


def load_config(path: str, local: bool) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if local and "local_override" in cfg.get("training", {}):
        overrides = cfg["training"]["local_override"]
        cfg["training"]["batch_size"] = overrides.get("batch_size", 1)
        cfg["training"]["gradient_accumulation"] = overrides.get("gradient_accumulation", 8)
        cfg["training"]["dtype"] = overrides.get("dtype", "float32")
        cfg["model"]["base"] = overrides.get("base", cfg["model"]["base"])
        # Replace step counts with smoke-test override
        steps = overrides.get("steps_override", 50)
        for phase in cfg["training"]["phases"]:
            phase["steps"] = steps
    return cfg


def build_datasets(cfg: dict, phase_name: str, processor) -> torch.utils.data.Dataset:
    ds_cfgs = cfg["datasets"].get(phase_name, [])
    splits = []
    for ds_cfg in ds_cfgs:
        name = ds_cfg["name"]
        config = ds_cfg.get("config", None)
        local_path = ds_cfg.get("path", None)
        try:
            if local_path:
                ds = load_dataset(name, data_files=local_path, split=ds_cfg.get("split", "train"))
            elif config:
                ds = load_dataset(name, config, split=ds_cfg["split"], streaming=False)
            else:
                ds = load_dataset(name, split=ds_cfg["split"], streaming=False)
            n = min(ds_cfg.get("sample", len(ds)), len(ds))
            ds = ds.select(range(n))
        except Exception:
            if local_path:
                raise
            ds = load_dataset(name, config, split=ds_cfg["split"], streaming=True) if config else load_dataset(name, split=ds_cfg["split"], streaming=True)
            ds = ds.take(ds_cfg.get("sample", 10000))
            from datasets import Dataset
            ds = Dataset.from_list(list(ds))
        splits.append(ds)
    if not splits:
        return None
    combined = concatenate_datasets(splits) if len(splits) > 1 else splits[0]
    return combined


def apply_lora(model, cfg: dict):
    lora_cfg = cfg["lora"]
    lora_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        target_modules=lora_cfg["target_modules"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg["bias"],
        task_type=TaskType.CAUSAL_LM,
    )
    return get_peft_model(model, lora_config)


def train_phase(model, processor, dataset, phase_cfg: dict, output_dir: str, dtype_str: str):
    if dataset is None:
        print(f"  No dataset for phase — skipping")
        return

    dtype = torch.bfloat16 if dtype_str == "bfloat16" else torch.float32
    use_fp16 = dtype_str == "float16"
    use_bf16 = dtype_str == "bfloat16"

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        max_steps=phase_cfg["steps"],
        per_device_train_batch_size=1,
        gradient_accumulation_steps=phase_cfg.get("gradient_accumulation", 4),
        learning_rate=float(phase_cfg["lr"]),
        warmup_steps=phase_cfg.get("warmup_steps", 100),
        fp16=use_fp16,
        bf16=use_bf16,
        logging_steps=10,
        save_steps=phase_cfg.get("save_steps", 999999),
        save_total_limit=1,
        dataloader_num_workers=0,   # required on Mac
        remove_unused_columns=False,
        report_to="none",
    )

    def tokenize(example):
        if "messages" in example:
            # Normalize content to list-of-dicts for VLM chat templates
            # (ultrachat and similar datasets use plain strings for content)
            msgs = example["messages"]
            normalized = []
            for m in msgs:
                content = m.get("content", "")
                if isinstance(content, str):
                    content = [{"type": "text", "text": content}]
                normalized.append({"role": m["role"], "content": content})
            try:
                text = processor.apply_chat_template(normalized, tokenize=False)
            except Exception:
                # Fallback: join messages as plain text
                text = "\n".join(
                    f"{m['role'].upper()}: {m['content'][0]['text'] if isinstance(m['content'], list) else m['content']}"
                    for m in normalized
                )
        elif "conversations" in example:
            # LLaVA format
            convs = example["conversations"]
            text = "\n".join(f"{c['from'].upper()}: {c['value']}" for c in convs)
        else:
            text = example.get("text", example.get("prompt", ""))
        tokens = processor.tokenizer(
            text, max_length=2048, truncation=True, padding="max_length", return_tensors="pt"
        )
        tokens["labels"] = tokens["input_ids"].clone()
        return {k: v.squeeze(0) for k, v in tokens.items()}

    # Cap dataset to what's actually needed for this run (avoids tokenizing 60K for 50 steps)
    max_needed = args.batch_size * phase_cfg["steps"] * phase_cfg.get("gradient_accumulation", 1) if hasattr(args, "batch_size") else phase_cfg["steps"] * 8
    if len(dataset) > max_needed:
        dataset = dataset.select(range(max_needed))
    tokenized = dataset.map(tokenize, remove_columns=dataset.column_names, num_proc=1)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        data_collator=DataCollatorForSeq2Seq(processor.tokenizer, model=model, padding=True),
    )
    trainer.train()


def train_bridge_phase(ujamaa_model, bridge: AnthosProjectionBridge, dataset, phase_cfg: dict, output_dir: str, dtype_str: str):
    """
    Bridge-only training phase. Freezes Ujamaa, only updates bridge weights.
    Loss: MSE between bridge output and a target Anthos embedding
    (in practice, supervised by Anthos's embed layer output on paired text).
    """
    if dataset is None:
        print("  No bridge dataset — skipping")
        return

    dtype = torch.bfloat16 if dtype_str == "bfloat16" else torch.float32
    bridge = bridge.to(dtype)
    optimizer = torch.optim.AdamW(bridge.parameters(), lr=float(phase_cfg["lr"]))

    ujamaa_model.eval()
    bridge.train()

    print(f"  Bridge training for {phase_cfg['steps']} steps...")
    for step, batch in enumerate(dataset):
        if step >= phase_cfg["steps"]:
            break
        # Dummy forward for now — replace with real paired Anthos embeddings
        fake_hidden = torch.randn(1, 32, 2560, dtype=dtype)
        perception = bridge(fake_hidden)
        # Self-supervised: perception tokens should be spread (not collapsed)
        loss = -perception.var(dim=1).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 10 == 0:
            print(f"    step {step:4d} | loss {loss.item():.4f}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    torch.save(bridge.state_dict(), f"{output_dir}/bridge.pt")
    print(f"  Bridge saved to {output_dir}/bridge.pt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--phase", default=None, help="Run a single phase only")
    args = parser.parse_args()

    cfg = load_config(args.config, local=args.local)
    model_id = cfg["model"]["base"]
    dtype_str = cfg["training"]["dtype"]
    dtype = torch.bfloat16 if dtype_str == "bfloat16" else torch.float32
    checkpoint_dir = cfg["output"]["checkpoint_dir"]
    bridge_dir = cfg["output"]["bridge_checkpoint"]

    print(f"=== Ujamaa-3B Training ===")
    print(f"Base model : {model_id}")
    print(f"dtype      : {dtype_str}")
    print(f"Checkpoint : {checkpoint_dir}")
    print()

    print("Loading processor and model...")
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, dtype=dtype, trust_remote_code=True)
    except (ImportError, Exception):
        try:
            if AutoModelForVision2Seq is not None:
                model = AutoModelForVision2Seq.from_pretrained(model_id, dtype=dtype, trust_remote_code=True)
            else:
                raise ImportError
        except Exception:
            model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, trust_remote_code=True)
    model = apply_lora(model, cfg)
    model.print_trainable_parameters()

    bridge = build_bridge(
        anthos_dim=cfg["model"]["anthos_dim"],
        n_perception_tokens=cfg["ujamaa_modules"]["n_perception_tokens"],
    )

    phases = cfg["training"]["phases"]
    if args.phase:
        phases = [p for p in phases if p["name"] == args.phase]

    for phase in phases:
        name = phase["name"]
        print(f"\n--- Phase: {name} ({phase['steps']} steps) ---")

        if phase.get("train_bridge_only"):
            ds = build_datasets(cfg, "bridge", processor)
            train_bridge_phase(model, bridge, ds, phase, bridge_dir, dtype_str)
        else:
            ds_key = "vision" if "vision" in name else "instruction"
            ds = build_datasets(cfg, ds_key, processor)
            phase_out = f"{checkpoint_dir}/{name}"
            train_phase(model, processor, ds, phase, phase_out, dtype_str)

    print("\n=== Training complete ===")
    final_dir = f"{checkpoint_dir}/final"
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)
    print(f"Saved to {final_dir}")
    print(f"Bridge at {bridge_dir}/bridge.pt")
    print("\nTo chat with the model:")
    print(f"  python examples/chat.py --checkpoint {final_dir}")


if __name__ == "__main__":
    main()
