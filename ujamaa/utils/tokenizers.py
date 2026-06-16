from typing import List, Optional, Union
import torch


class MultiModalTokenizer:
    """
    Wraps a text tokenizer and adds special tokens for vision/audio modalities.
    """

    VISION_START = "<vision>"
    VISION_END = "</vision>"
    AUDIO_START = "<audio>"
    AUDIO_END = "</audio>"

    def __init__(self, base_tokenizer_name: str = "gpt2"):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_tokenizer_name)
        special_tokens = [
            self.VISION_START,
            self.VISION_END,
            self.AUDIO_START,
            self.AUDIO_END,
        ]
        self.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def encode(
        self,
        text: Union[str, List[str]],
        max_length: Optional[int] = None,
        padding: bool = False,
        return_tensors: Optional[str] = None,
    ):
        return self.tokenizer(
            text,
            max_length=max_length,
            padding=padding,
            truncation=max_length is not None,
            return_tensors=return_tensors,
        )

    def decode(self, token_ids: Union[List[int], torch.Tensor], skip_special_tokens: bool = True) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    @property
    def vocab_size(self) -> int:
        return len(self.tokenizer)
