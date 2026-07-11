"""
TextGenerator — drop-in replacement with optional Colibri streaming + Engram.

streaming=True (default): experts stream from disk, Engram memory active.
streaming=False: all weights in RAM, no Engram — original behavior.

Built by Brian Tushae Thomas for Anthos Intelligence Company.
"""
from pathlib import Path
from typing import Optional

import torch

from ujamaa import UjamaaMultiModal
from ujamaa.utils.tokenizers import MultiModalTokenizer


class TextGenerator:
    """Simple text generation wrapper with optional expert streaming."""

    def __init__(
        self,
        model: UjamaaMultiModal,
        tokenizer: MultiModalTokenizer,
        device: str = "cpu",
        streaming: bool = True,
        expert_dir: Optional[str] = None,
        max_cache_gb: Optional[float] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.streaming = streaming
        self._engine = None

        if streaming:
            from inference.streaming import StreamingUjamaa, EngramBridge

            engram = EngramBridge()
            self._engine = StreamingUjamaa(
                model=model,
                tokenizer=tokenizer,
                expert_dir=Path(expert_dir) if expert_dir else None,
                device=torch.device(device) if isinstance(device, str) else device,
                max_cache_gb=max_cache_gb,
                engram=engram,
            )
        else:
            self.model = model.to(device)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_k: int = 50,
        save_to_memory: bool = True,
    ) -> str:
        if self._engine is not None:
            return self._engine.generate(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                save_to_memory=save_to_memory,
            )

        enc = self.tokenizer.encode(prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to(self.device)

        output_ids = self.model.generate_text(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )

        return self.tokenizer.decode(output_ids[0])

    def cache_stats(self) -> dict:
        """Return ExpertCache statistics. Only available in streaming mode."""
        if self._engine is not None:
            return self._engine.cache_stats()
        return {"streaming": False}
