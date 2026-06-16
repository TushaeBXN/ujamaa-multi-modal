"""
Simple FastAPI inference server for Ujamaa.
Run: uvicorn inference.server:app --host 0.0.0.0 --port 8080
"""
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

from ujamaa import ujamaa_mm
from ujamaa.utils.tokenizers import MultiModalTokenizer
from inference.generator import TextGenerator

app = FastAPI(title="Ujamaa Multi-Modal API", version="0.1.0")

_generator: Optional[TextGenerator] = None


def get_generator() -> TextGenerator:
    global _generator
    if _generator is None:
        model = ujamaa_mm("1.5b")
        tokenizer = MultiModalTokenizer()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _generator = TextGenerator(model, tokenizer, device)
    return _generator


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 200
    temperature: float = 0.8
    top_k: int = 50


class GenerateResponse(BaseModel):
    text: str
    model_size: str


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    gen = get_generator()
    text = gen.generate(req.prompt, req.max_new_tokens, req.temperature, req.top_k)
    return GenerateResponse(text=text, model_size=gen.model.get_model_size())


@app.get("/health")
def health():
    return {"status": "ok"}
