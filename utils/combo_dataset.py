"""
ComboDataset: lazy-loading dataset backed by pre-computed combo shard files.

Each shard stores 32 combos (query + candidate raw features) as stacked tensors.
The dataset loads shards on-demand with an LRU cache, pools text features in
__getitem__, and returns dicts matching the model's expected format.
"""

import bisect
import gc
from collections import OrderedDict
from pathlib import Path

import torch
from torch.utils.data import Dataset


# ---- Mean pooling: done offline during data preparation ----
# TEXT_DIM = 768
# VISUAL_DIM = 4096
# MAX_IMAGE_DESCS = 15
# MAX_VISUAL = 5
# TEXT_SEQ_LEN = 512
#
# def masked_mean_pool(embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
#     """Pool [T, D] + mask [T] -> [D], ignoring padding."""
#     valid = mask.sum()
#     if valid == 0:
#         return torch.zeros(embeddings.size(1), dtype=embeddings.dtype)
#     return (embeddings * mask.unsqueeze(-1).float()).sum(dim=0) / valid


class PooledComboDataset(Dataset):
    """
    Dataset backed by pre-pooled combo shard files (shard_*_pooled.pt).

    Pooled shards already have mean-pooled text features, so no runtime pooling.
    Keys in each shard:
      query_tokens, candidate_title, candidate_features, candidate_description,
      candidate_image_desc, candidate_image_desc_mask,
      candidate_visual, candidate_visual_mask, labels
    """

    def __init__(self, shard_dir: str, cache_size: int = 2):
        self.shard_dir = Path(shard_dir)
        self.cache_size = cache_size
        self._cache: OrderedDict[str, dict] = OrderedDict()

        self._shard_paths = sorted(
            str(p) for p in self.shard_dir.glob("shard_*_pooled.pt")
        )
        if not self._shard_paths:
            raise FileNotFoundError(
                f"No shard_*_pooled.pt files found in {shard_dir}."
            )

        self._num_shards = len(self._shard_paths)

        first = torch.load(self._shard_paths[0], map_location="cpu", weights_only=True)
        self._shard_size = first["labels"].size(0)
        if self._num_shards > 1:
            last = torch.load(self._shard_paths[-1], map_location="cpu", weights_only=True)
            last_size = last["labels"].size(0)
        else:
            last_size = self._shard_size

        self._total_samples = self._shard_size * (self._num_shards - 1) + last_size

        self._cum = [0]
        for i in range(self._num_shards):
            size = self._shard_size if i < self._num_shards - 1 else last_size
            self._cum.append(self._cum[-1] + size)

    def __len__(self) -> int:
        return self._total_samples

    def __getitem__(self, idx: int) -> dict:
        if idx < 0 or idx >= self._total_samples:
            raise IndexError(f"Index {idx} out of range [0, {self._total_samples})")

        shard_idx = bisect.bisect_right(self._cum, idx) - 1
        local_idx = idx - self._cum[shard_idx]
        shard = self._load_shard(shard_idx)

        return {
            "query_tokens": shard["query_tokens"][local_idx],
            "candidate_title": shard["candidate_title"][local_idx],
            "candidate_features": shard["candidate_features"][local_idx],
            "candidate_description": shard["candidate_description"][local_idx],
            "candidate_image_desc": shard["candidate_image_desc"][local_idx],
            "candidate_image_desc_mask": shard["candidate_image_desc_mask"][local_idx],
            "candidate_visual": shard["candidate_visual"][local_idx],
            "candidate_visual_mask": shard["candidate_visual_mask"][local_idx],
            "label": shard["labels"][local_idx],
        }

    def _load_shard(self, shard_idx: int) -> dict:
        shard_path = self._shard_paths[shard_idx]

        if shard_path in self._cache:
            self._cache.move_to_end(shard_path)
            return self._cache[shard_path]

        shard = torch.load(shard_path, map_location="cpu", weights_only=True)
        self._cache[shard_path] = shard
        self._cache.move_to_end(shard_path)

        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)

        return shard

    def clear_cache(self):
        self._cache.clear()
        gc.collect()
