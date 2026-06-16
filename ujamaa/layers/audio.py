import torch
import torch.nn as nn


class AudioEncoder(nn.Module):
    """
    Audio encoder for Ujamaa multi-modal model.
    Wraps Whisper encoder and projects to model dimension.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        from transformers import WhisperModel
        self.encoder = WhisperModel.from_pretrained(config.audio_encoder).encoder
        self.audio_dim = 1280  # Whisper large hidden size

        self.projector = nn.Sequential(
            nn.Linear(self.audio_dim, config.dim),
            nn.LayerNorm(config.dim),
            nn.GELU(),
            nn.Linear(config.dim, config.dim),
            nn.LayerNorm(config.dim),
        )

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_features: [batch, num_frames, audio_dim] mel spectrogram
        Returns:
            audio_features: [batch, num_frames, dim]
        """
        outputs = self.encoder(input_features)
        features = outputs.last_hidden_state
        return self.projector(features)
