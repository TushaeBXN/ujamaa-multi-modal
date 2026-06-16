<div align="center">

# 𓋴 UJAMAA MULTI-MODAL 𓋴
## *Cooperative Gated Recurrent Attention Foundation Model*

**Built by Brian Tushae Thomas for Anthos Intelligence Company**

[![Anthos Intelligence](https://img.shields.io/badge/Anthos-Intelligence-FF6B35?style=for-the-badge)](https://anthosintelligence.com)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?style=for-the-badge)](https://pytorch.org)
[![Multi-Modal](https://img.shields.io/badge/Multi--Modal-8A2BE2?style=for-the-badge)](#)

*"Tokens cooperating, sharing resources, lifting each other up across vision, audio, and text."*

</div>

---

## The Ujamaa Philosophy

> *"Ujamaa is the foundation of community-based cooperation. Tokens of all modalities — vision, audio, text — cooperate, share resources, and lift each other up."* — Brian Tushae Thomas

Ujamaa is a **multi-modal foundation model** where tokens cooperate through a novel **community routing** mechanism. Hard tokens (rare words, complex images, noisy audio) receive compute from easy tokens through learned resource sharing.

| Principle | Technical Implementation |
|---|---|
| **Community over individual** | Tokens vote on routing collectively |
| **Shared resources** | Compute redistributed from easy to hard tokens |
| **Cross-modal cooperation** | Vision and audio tokens inform text tokens |
| **Collective efficiency** | MoE with modality-specific experts |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  UJAMAA MULTI-MODAL                     │
├─────────────────────────────────────────────────────────┤
│   [Vision]        [Audio]           [Text]              │
│      ↓               ↓                ↓                 │
│  VisionEncoder  AudioEncoder    TokenEmbedding          │
│      ↓               ↓                ↓                 │
│  ┌───────────────────────────────────────────────────┐  │
│  │           UJAMAA COMMUNITY GATE                   │  │
│  │  • Tokens vote on resource allocation             │  │
│  │  • Hard tokens get help from easy tokens          │  │
│  └───────────────────────────────────────────────────┘  │
│                         ↓                               │
│  ┌───────────────────────────────────────────────────┐  │
│  │           UJAMAA LAYERS (×N)                      │  │
│  │   GQA Attention + Community Gate + MoE FFN        │  │
│  └───────────────────────────────────────────────────┘  │
│                         ↓                               │
│              Multi-Modal Output Logits                  │
└─────────────────────────────────────────────────────────┘
```

---

## Model Variants

| Variant | Params | Dim | Layers | Experts | Context |
|---|---|---|---|---|---|
| **Ujamaa-MM-1.5B** | 1.5B | 2048 | 24 | 8 | 8K |
| **Ujamaa-MM-3B** | 3B | 2560 | 28 | 16 | 8K |
| **Ujamaa-MM-7B** | 7B | 4096 | 32 | 32 | 8K |
| **Ujamaa-MM-13B** | 13B | 5120 | 40 | 48 | 16K |
| **Ujamaa-MM-34B** | 34B | 8192 | 48 | 64 | 32K |
| **Ujamaa-MM-70B** | 70B | 10240 | 80 | 128 | 128K |
| **Ujamaa-MM-100B+** | 100B+ | 12288 | 96 | 256 | 128K |

---

## Growth Pathway

```
1.5B ──grow_dim──▶ 3B ──grow_dim──▶ 7B ──grow_dim──▶ 13B
       grow_layers        grow_layers        grow_layers
       grow_experts       grow_experts       grow_experts
                                                  ↓
100B ◀──────────── 70B ◀──────────── 34B ◀────────┘
```

---

## Quick Start

```bash
git clone https://github.com/AnthosIntelligence/ujamaa-multi-modal
cd ujamaa-multi-modal
pip install -e .
python scripts/download_data.py

# Train 1.5B model
python training/train.py --config training/configs/1.5b.yaml --phase alignment --steps 10000
python training/train.py --config training/configs/1.5b.yaml --phase pretraining --steps 50000
python training/train.py --config training/configs/1.5b.yaml --phase instruction --steps 10000

# Chat
python examples/chat.py --checkpoint checkpoints/ujamaa_instruction_final_step_10000.pt

# Multi-modal chat
python examples/chat.py --checkpoint checkpoints/... --image photo.jpg
```

## Python API

```python
from ujamaa import ujamaa_mm

model = ujamaa_mm("7b")
output = model.generate_text(input_ids, pixel_values=image, max_new_tokens=100)
```

---

## Training Phases

| Phase | Objective | Loss | Duration |
|---|---|---|---|
| 1. Alignment | Align vision/audio with text | Contrastive | 10K steps |
| 2. Pretraining | Next-token prediction | Cross-entropy | 50K steps |
| 3. Instruction | Multi-modal QA | Masked cross-entropy | 10K steps |

---

## Citation

```bibtex
@software{thomas2024ujamaa,
  author    = {Brian Tushae Thomas},
  title     = {Ujamaa Multi-Modal: Cooperative Gated Recurrent Attention Foundation Model},
  year      = {2024},
  publisher = {Anthos Intelligence Company},
  url       = {https://github.com/AnthosIntelligence/ujamaa-multi-modal},
}
```

---

## License

Copyright © 2024-2025 Anthos Intelligence Company. All rights reserved.

<div align="center">
<sub>Built with ❤️ by Brian Tushae Thomas for Anthos Intelligence Company</sub><br>
<sub>𓋴 Tokens of all modalities cooperating, lifting each other up 𓋴</sub>
</div>
