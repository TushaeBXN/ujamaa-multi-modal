"""
MultiModalGenerator — vision + audio + text with expert streaming and Engram.

Always uses StreamingUjamaa for efficient inference on consumer hardware.
Tracks active modalities and persists sessions to Engram memory.

Built by Brian Tushae Thomas for Anthos Intelligence Company.
"""
from pathlib import Path
from typing import Optional

import torch
from PIL import Image

from ujamaa import UjamaaMultiModal
from ujamaa.utils.tokenizers import MultiModalTokenizer
from inference.streaming import StreamingUjamaa, EngramBridge


class MultiModalGenerator:
    """Multi-modal generation with text, image, and audio inputs."""

    def __init__(
        self,
        model: UjamaaMultiModal,
        tokenizer: MultiModalTokenizer,
        device: str = "cpu",
        expert_dir: Optional[str] = None,
        max_cache_gb: Optional[float] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        engram = EngramBridge()
        self._engine = StreamingUjamaa(
            model=model,
            tokenizer=tokenizer,
            expert_dir=Path(expert_dir) if expert_dir else None,
            device=torch.device(device) if isinstance(device, str) else device,
            max_cache_gb=max_cache_gb,
            engram=engram,
        )

    def generate(
        self,
        prompt: str,
        image: Optional[Image.Image] = None,
        audio_features: Optional[torch.Tensor] = None,
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_k: int = 50,
        save_to_memory: bool = True,
    ) -> str:
        pixel_values = None
        if image is not None:
            from transformers import CLIPProcessor
            processor = CLIPProcessor.from_pretrained(self.model.config.vision_encoder)
            pixel_values = processor(images=image, return_tensors="pt")["pixel_values"]

        return self._engine.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            pixel_values=pixel_values,
            audio_features=audio_features,
            save_to_memory=save_to_memory,
        )

    def cache_stats(self) -> dict:
        """Return ExpertCache statistics."""
        return self._engine.cache_stats()
