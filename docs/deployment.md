# Deployment Guide

## Export a trained checkpoint
```bash
python scripts/export_model.py \
  --checkpoint checkpoints/ujamaa_instruction_final_step_10000.pt \
  --output checkpoints/ujamaa_7b_deploy.pt
```

## Python API
```python
import torch
from ujamaa import ujamaa_mm
from ujamaa.utils.tokenizers import MultiModalTokenizer
from inference.generator import TextGenerator

model = ujamaa_mm("7b")
model.load_state_dict(torch.load("checkpoints/ujamaa_7b_deploy.pt")["model_state_dict"])

tokenizer = MultiModalTokenizer()
gen = TextGenerator(model, tokenizer, device="cuda")
print(gen.generate("What is cooperative economics?"))
```

## Multi-modal API
```python
from PIL import Image
from inference.multimodal import MultiModalGenerator

gen = MultiModalGenerator(model, tokenizer, device="cuda")
image = Image.open("photo.jpg")
print(gen.generate("Describe this image.", image=image))
```

## REST API server
```bash
pip install fastapi uvicorn
uvicorn inference.server:app --host 0.0.0.0 --port 8080
```

Then POST to `http://localhost:8080/generate`:
```json
{"prompt": "Explain Ujamaa.", "max_new_tokens": 200}
```

## Evaluate perplexity
```bash
python scripts/evaluate.py \
  --checkpoint checkpoints/ujamaa_7b_deploy.pt \
  --data data/eval/samples.json
```
