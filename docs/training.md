# Training Guide

## Prerequisites
```bash
pip install -e .
python scripts/download_data.py  # scaffold data directories
```

## Phase 1: Alignment (10K steps)
Aligns vision and audio encoders with the text space using contrastive loss.
```bash
python training/train.py --config training/configs/1.5b.yaml --phase alignment --steps 10000
```

## Phase 2: Pretraining (50K steps)
Next-token prediction across all modalities on a large multi-modal corpus.
```bash
python training/train.py --config training/configs/1.5b.yaml --phase pretraining --steps 50000
```

## Phase 3: Instruction Tuning (10K steps)
Multi-modal QA fine-tuning with loss masked to response tokens only.
```bash
python training/train.py --config training/configs/1.5b.yaml --phase instruction --steps 10000
```

## Resume from checkpoint
```bash
python training/train.py --config training/configs/1.5b.yaml --phase pretraining --resume checkpoints/1.5b/ujamaa_alignment_final_step_10000.pt
```

## Scale to larger models
Simply swap the config:
```bash
python training/train.py --config training/configs/7b.yaml --phase pretraining --steps 100000
```
