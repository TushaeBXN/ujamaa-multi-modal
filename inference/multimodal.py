from typing import Optional
import torch
from PIL import Image
from ujamaa import UjamaaMultiModal
from ujamaa.utils.tokenizers import MultiModalTokenizer


class MultiModalGenerator:
    """Multi-modal generation with text, image, and audio inputs"""

    def __init__(self, model: UjamaaMultiModal, tokenizer: MultiModalTokenizer, device: str = "cpu"):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device

    def generate(
        self,
        prompt: str,
        image: Optional[Image.Image] = None,
        audio_features: Optional[torch.Tensor] = None,
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_k: int = 50,
    ) -> str:
        enc = self.tokenizer.encode(prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to(self.device)

        pixel_values = None
        if image is not None:
            from transformers import CLIPProcessor
            processor = CLIPProcessor.from_pretrained(self.model.config.vision_encoder)
            pixel_values = processor(images=image, return_tensors="pt")["pixel_values"].to(self.device)

        if audio_features is not None:
            audio_features = audio_features.to(self.device)

        output_ids = self.model.generate_text(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            pixel_values=pixel_values,
            audio_features=audio_features,
        )

        return self.tokenizer.decode(output_ids[0])
