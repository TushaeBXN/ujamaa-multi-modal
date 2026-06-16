import torch
from ujamaa import UjamaaMultiModal
from ujamaa.utils.tokenizers import MultiModalTokenizer


class TextGenerator:
    """Simple text generation wrapper"""

    def __init__(self, model: UjamaaMultiModal, tokenizer: MultiModalTokenizer, device: str = "cpu"):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_k: int = 50,
    ) -> str:
        enc = self.tokenizer.encode(prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to(self.device)

        output_ids = self.model.generate_text(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )

        return self.tokenizer.decode(output_ids[0])
