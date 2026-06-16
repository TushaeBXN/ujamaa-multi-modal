# Ujamaa Multi-Modal Architecture

Built by Brian Tushae Thomas for Anthos Intelligence Company.

## Overview

Ujamaa is a cooperative multi-modal foundation model where tokens from different modalities
(text, vision, audio) share computation through a novel **Community Routing** mechanism.

## Core Components

### 1. Multi-Modal Encoders
- **VisionEncoder**: CLIP ViT-L/14 → projected to model dimension
- **AudioEncoder**: Whisper large-v3 encoder → projected to model dimension
- **VisionLanguageConnector**: aligns visual features with text space (cross-attention, gated, or concat)

### 2. Ujamaa Community Gate
Each token computes a "need score" — how uncertain or complex it is. The collective then
redistributes compute from low-need tokens to high-need tokens. This is the core Ujamaa
innovation: tokens cooperate rather than compete.

### 3. Mixture of Experts (MoE)
Modality-specific experts handle vision, audio, and text tokens differently:
- Vision experts: biased for spatial/pixel features
- Audio experts: biased for temporal/spectral features
- Text experts: general language modeling
- Shared experts: always active for all tokens

### 4. Grouped Query Attention (GQA)
Reduces KV cache memory by sharing key/value heads across query groups.
Thought tokens are prepended to each sequence and attend to everything.

### 5. Growth System
Models grow progressively from 1.5B to 100B+ by:
- `grow_dimension`: expand hidden dim, copy existing weights
- `grow_layers`: add new layers initialized from the last layer + noise
- `grow_experts`: add MoE experts with expanded router

## Training Phases
1. **Alignment**: contrastive loss to align vision/audio with text space
2. **Pretraining**: next-token prediction on multi-modal corpus
3. **Instruction**: masked loss on response tokens only
