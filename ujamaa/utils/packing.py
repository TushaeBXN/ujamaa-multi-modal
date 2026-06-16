from typing import List, Dict
import torch


class DynamicPacker:
    """
    Packs multiple short sequences into a single context window
    to maximize GPU utilization during training.
    """

    def __init__(self, max_seq_len: int):
        self.max_seq_len = max_seq_len

    def pack(self, sequences: List[Dict]) -> List[Dict]:
        """
        Greedily pack sequences into bins of max_seq_len.

        Args:
            sequences: list of dicts with 'input_ids' (1D tensor) and optional other keys
        Returns:
            packed: list of packed dicts with concatenated input_ids and position_ids
        """
        bins: List[Dict] = []
        current_bin: List[torch.Tensor] = []
        current_len = 0

        for seq in sequences:
            ids = seq["input_ids"]
            seq_len = ids.size(0)

            if seq_len > self.max_seq_len:
                ids = ids[: self.max_seq_len]
                seq_len = self.max_seq_len

            if current_len + seq_len > self.max_seq_len:
                if current_bin:
                    bins.append(self._finalize(current_bin))
                current_bin = [ids]
                current_len = seq_len
            else:
                current_bin.append(ids)
                current_len += seq_len

        if current_bin:
            bins.append(self._finalize(current_bin))

        return bins

    def _finalize(self, bin_seqs: List[torch.Tensor]) -> Dict:
        packed_ids = torch.cat(bin_seqs)
        position_ids = torch.arange(packed_ids.size(0))
        return {"input_ids": packed_ids, "position_ids": position_ids}
