"""
Audio transcription example.
Usage: python examples/audio_transcribe.py --audio PATH [--size 1.5b]
"""
import argparse
import torch
from ujamaa import ujamaa_mm
from ujamaa.utils.tokenizers import MultiModalTokenizer
from inference.multimodal import MultiModalGenerator


def load_audio_features(path: str) -> torch.Tensor:
    import librosa
    import numpy as np
    audio, sr = librosa.load(path, sr=16000)
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=80)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return torch.tensor(mel_db.T, dtype=torch.float32).unsqueeze(0)  # [1, frames, 80]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--size", default="1.5b")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ujamaa_mm(args.size)

    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location=device)
        key = "model_state_dict" if "model_state_dict" in state else None
        model.load_state_dict(state[key] if key else state)

    audio_features = load_audio_features(args.audio).to(device)
    tokenizer = MultiModalTokenizer()
    gen = MultiModalGenerator(model.to(device), tokenizer, device)

    result = gen.generate("Transcribe the audio:", audio_features=audio_features, max_new_tokens=200)
    print(f"Transcription: {result}")


if __name__ == "__main__":
    main()
