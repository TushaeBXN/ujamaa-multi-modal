"""
Colibri-Style Expert Streaming + Engram Memory for Ujamaa

Dense layers stay in RAM. MoE experts stream from disk via LRU cache.
Engram provides persistent memory context across sessions.

Built by Brian Tushae Thomas for Anthos Intelligence Company.
"""
import os
import sys
import time
import subprocess
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Dict, Tuple, List

import torch
import torch.nn as nn

DEFAULT_EXPERT_DIR = Path("/Volumes/1TB Drive/anthos-data/models/ujamaa-experts")
MIN_RESERVED_RAM_GB = 4.0


def _get_available_ram_gb() -> float:
    """Get available RAM in GB. macOS uses vm_stat, Linux uses /proc/meminfo."""
    if sys.platform == "darwin":
        try:
            output = subprocess.check_output(["vm_stat"], text=True)
            page_size = 4096
            free = 0
            for line in output.splitlines():
                if "Pages free" in line:
                    free += int(line.split(":")[1].strip().rstrip("."))
                elif "Pages inactive" in line:
                    free += int(line.split(":")[1].strip().rstrip("."))
                elif "Pages speculative" in line:
                    free += int(line.split(":")[1].strip().rstrip("."))
            return (free * page_size) / (1024 ** 3)
        except Exception:
            return 8.0
    else:
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if "MemAvailable" in line:
                        kb = int(line.split()[1])
                        return kb / (1024 ** 2)
        except Exception:
            return 8.0
    return 8.0


def _detect_device() -> torch.device:
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class ExpertCache:
    """LRU cache for expert tensors streamed from disk.

    Auto-sizes from available RAM minus a safety reserve.
    Experts are stored as individual .pt shards on disk.
    """

    def __init__(
        self,
        expert_dir: Path,
        device: torch.device,
        max_cache_gb: Optional[float] = None,
    ):
        self.expert_dir = Path(expert_dir)
        self.device = device
        self._cache: OrderedDict[Tuple[int, int], nn.Module] = OrderedDict()
        self._hits = 0
        self._misses = 0

        if max_cache_gb is not None:
            self._max_bytes = int(max_cache_gb * 1024 ** 3)
        else:
            available = _get_available_ram_gb()
            usable = max(available - MIN_RESERVED_RAM_GB, 1.0)
            self._max_bytes = int(usable * 1024 ** 3)

        self._current_bytes = 0

    def _shard_path(self, layer_idx: int, expert_id: int) -> Path:
        return self.expert_dir / f"layer_{layer_idx:03d}_expert_{expert_id:04d}.pt"

    def _estimate_size(self, state_dict: dict) -> int:
        return sum(t.nelement() * t.element_size() for t in state_dict.values())

    def get(self, layer_idx: int, expert_id: int) -> Optional[torch.Tensor]:
        """Load expert weights. Returns state_dict or None if shard doesn't exist."""
        key = (layer_idx, expert_id)

        if key in self._cache:
            self._hits += 1
            self._cache.move_to_end(key)
            return self._cache[key]

        path = self._shard_path(layer_idx, expert_id)
        if not path.exists():
            return None

        self._misses += 1
        state_dict = torch.load(path, map_location=self.device, weights_only=True)
        size = self._estimate_size(state_dict)

        while self._current_bytes + size > self._max_bytes and self._cache:
            _, evicted = self._cache.popitem(last=False)
            evicted_size = self._estimate_size(
                evicted if isinstance(evicted, dict) else evicted.state_dict()
            )
            self._current_bytes -= evicted_size

        self._cache[key] = state_dict
        self._current_bytes += size
        return state_dict

    def cache_stats(self) -> Dict[str, float]:
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "cache_size_gb": self._current_bytes / (1024 ** 3),
            "max_cache_gb": self._max_bytes / (1024 ** 3),
            "entries": len(self._cache),
        }

    def clear(self):
        self._cache.clear()
        self._current_bytes = 0


class EngramBridge:
    """Thin wrapper around Engram memory system. Falls back silently if unavailable."""

    def __init__(self, wing: str = "ujamaa", room: str = "inference", hall: str = "discoveries"):
        self.wing = wing
        self.room = room
        self.hall = hall
        self._available = False
        self._layer_stack = None
        self._miner = None

        try:
            from engram.chateau import Chateau
            from engram.searcher import Searcher
            from engram.layers import LayerStack
            from engram.miner import Miner
            from engram.backends import get_backend

            backend = get_backend("chromadb")
            palace = Chateau(wing=self.wing)
            searcher = Searcher(backend=backend)
            self._layer_stack = LayerStack(palace, searcher)
            self._miner = Miner(palace=palace)
            self._available = True
        except Exception as e:
            print(f"[EngramBridge] Engram not available ({e}) — memory disabled, continuing without it.")

    @property
    def available(self) -> bool:
        return self._available

    def wake_up(self) -> str:
        """Load L0+L1 context (~170 tokens) for cold start."""
        if not self._available:
            return ""
        try:
            return self._layer_stack.wake_up(wing=self.wing)
        except Exception:
            return ""

    def search(self, query: str, top_k: int = 3) -> str:
        """Semantic search for relevant past context."""
        if not self._available:
            return ""
        try:
            results = self._layer_stack.deep_search(query, wing=self.wing)
            return results if isinstance(results, str) else str(results)
        except Exception:
            return ""

    def save(self, prompt: str, response: str, modalities: Optional[List[str]] = None):
        """Save a session to Engram memory."""
        if not self._available or self._miner is None:
            return
        try:
            metadata = {"modalities": ",".join(modalities)} if modalities else {}
            self._miner.add_memory(
                content=f"User: {prompt}\nAssistant: {response}",
                wing=self.wing,
                room=self.room,
                hall=self.hall,
                metadata=metadata,
            )
        except Exception:
            pass


class StreamingUjamaa:
    """Unified inference engine: ExpertCache + EngramBridge + model generation.

    Dense layers stay in RAM. Experts stream from disk on demand.
    Engram provides persistent memory context.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        expert_dir: Optional[Path] = None,
        device: Optional[torch.device] = None,
        max_cache_gb: Optional[float] = None,
        engram: Optional[EngramBridge] = None,
    ):
        self.device = device or _detect_device()
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.engram = engram or EngramBridge()

        expert_path = Path(expert_dir) if expert_dir else DEFAULT_EXPERT_DIR
        self.cache = ExpertCache(expert_path, self.device, max_cache_gb)

        self._cold_start_context = ""
        if self.engram.available:
            self._cold_start_context = self.engram.wake_up()
            if self._cold_start_context:
                print(f"[StreamingUjamaa] Engram L0+L1 loaded ({len(self._cold_start_context)} chars)")

    def _expert_loader(self, layer_idx: int, expert_id: int) -> Optional[dict]:
        """Callback passed to model.generate_text() for on-demand expert loading."""
        return self.cache.get(layer_idx, expert_id)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_k: int = 50,
        pixel_values: Optional[torch.Tensor] = None,
        audio_features: Optional[torch.Tensor] = None,
        save_to_memory: bool = True,
    ) -> str:
        """Generate with expert streaming and memory context."""
        full_prompt = prompt

        if self._cold_start_context:
            full_prompt = f"{self._cold_start_context}\n\n{prompt}"

        engram_context = self.engram.search(prompt)
        if engram_context:
            full_prompt = f"{engram_context}\n\n{full_prompt}"

        enc = self.tokenizer.encode(full_prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to(self.device)

        if pixel_values is not None:
            pixel_values = pixel_values.to(self.device)
        if audio_features is not None:
            audio_features = audio_features.to(self.device)

        start = time.perf_counter()

        with torch.inference_mode():
            output_ids = self.model.generate_text(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                pixel_values=pixel_values,
                audio_features=audio_features,
                expert_loader=self._expert_loader,
            )

        elapsed = time.perf_counter() - start
        new_tokens = output_ids.shape[1] - input_ids.shape[1]
        tok_per_sec = new_tokens / elapsed if elapsed > 0 else 0

        stats = self.cache.cache_stats()
        print(f"[StreamingUjamaa] {new_tokens} tokens in {elapsed:.2f}s "
              f"({tok_per_sec:.1f} tok/s) | cache hit rate: {stats['hit_rate']:.1%}")

        response = self.tokenizer.decode(output_ids[0])

        if save_to_memory:
            modalities = ["text"]
            if pixel_values is not None:
                modalities.append("vision")
            if audio_features is not None:
                modalities.append("audio")
            self.save_session(prompt, response, modalities)

        return response

    def save_session(self, prompt: str, response: str, modalities: Optional[List[str]] = None):
        """Persist conversation to Engram memory."""
        self.engram.save(prompt, response, modalities)

    def cache_stats(self) -> Dict[str, float]:
        """Return ExpertCache statistics."""
        return self.cache.cache_stats()

    def shard_experts(self):
        """Write all expert weights to disk as individual shards. Idempotent."""
        expert_dir = self.cache.expert_dir
        expert_dir.mkdir(parents=True, exist_ok=True)

        for layer_idx, layer in enumerate(self.model.layers):
            if not hasattr(layer, "moe"):
                continue
            moe = layer.moe
            for expert_id, expert in enumerate(moe.experts):
                path = expert_dir / f"layer_{layer_idx:03d}_expert_{expert_id:04d}.pt"
                if path.exists():
                    continue
                state = expert.state_dict()
                torch.save(state, path)
                print(f"[shard_experts] layer {layer_idx} expert {expert_id} → {path}")

        print(f"[shard_experts] Done. Shards at {expert_dir}")
