"""
Dataset for precomputed multi-view candidate reranking.

Expected data (dict of tensors or .pt file):
    query_tokens:              [N, Tq, Dq]
    candidate_title:           [N, K, title_dim]       pooled title embedding
    candidate_features:        [N, K, feat_dim]        pooled features embedding
    candidate_description:     [N, K, desc_dim]        pooled description embedding
    candidate_image_desc:      [N, K, max_img, idim]   padded image desc embeddings
    candidate_image_desc_mask: [N, K, max_img]         1=valid, 0=padding
    candidate_visual:          [N, K, 5, vdim]         visual CLS tokens (padded to 5)
    candidate_visual_mask:     [N, K, 5]               1=valid, 0=padding
    labels:                    [N]
"""

import torch
from torch.utils.data import Dataset


class RerankingDataset(Dataset):
    """
    Dataset backed by precomputed feature tensors.

    Args:
        data_path: path to a .pt file containing a dict of tensors
        data_dict: dict of tensors directly (takes priority over data_path)
    """

    def __init__(self, data_path: str = None, data_dict: dict = None):
        if data_dict is not None:
            self.data = data_dict
        elif data_path is not None:
            self.data = torch.load(data_path, map_location="cpu", weights_only=False)
        else:
            raise ValueError("Provide either data_path or data_dict")

        # Validate keys and sizes
        N = self.data["query_tokens"].size(0)
        for key in ("labels", "candidate_title", "candidate_features",
                     "candidate_description", "candidate_image_desc",
                     "candidate_image_desc_mask", "candidate_visual",
                     "candidate_visual_mask"):
            assert key in self.data, f"Missing key: {key}"
            assert self.data[key].size(0) == N, f"Size mismatch for {key}"

        # Infer K and shapes
        self.num_samples = N
        self.num_candidates = self.data["candidate_visual"].size(1)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        return {
            "query_tokens": self.data["query_tokens"][idx],
            "candidate_title": self.data["candidate_title"][idx],
            "candidate_features": self.data["candidate_features"][idx],
            "candidate_description": self.data["candidate_description"][idx],
            "candidate_image_desc": self.data["candidate_image_desc"][idx],
            "candidate_image_desc_mask": self.data["candidate_image_desc_mask"][idx],
            "candidate_visual": self.data["candidate_visual"][idx],
            "candidate_visual_mask": self.data["candidate_visual_mask"][idx],
            "label": self.data["labels"][idx],
        }


# ---------------------------------------------------------------------------
# Dummy data factory for testing and dry-run
# ---------------------------------------------------------------------------


def create_dummy_data(
    num_samples: int = 100,
    num_candidates: int = 10,
    query_seq_len: int = 32,
    query_dim: int = 768,
    text_dim: int = 768,
    visual_dim: int = 4096,
    max_image_descs: int = 15,
) -> dict:
    """
    Create dummy data for testing. All tensors on CPU.

    Args:
        num_samples:     number of instances (N)
        num_candidates:  K
        query_seq_len:   Tq
        query_dim:       Dq
        text_dim:        dim for title/features/description/image_desc (default 768)
        visual_dim:      dim for visual CLS tokens (default 4096)
        max_image_descs: max number of image description tokens

    Returns:
        dict of tensors with shapes:
            query_tokens:              [N, Tq, Dq]
            candidate_title:           [N, K, text_dim]
            candidate_features:        [N, K, text_dim]
            candidate_description:     [N, K, text_dim]
            candidate_image_desc:      [N, K, max_image_descs, text_dim]
            candidate_image_desc_mask: [N, K, max_image_descs]
            candidate_visual:          [N, K, 5, visual_dim]
            candidate_visual_mask:     [N, K, 5]
            labels:                    [N]
    """
    data = {
        "query_tokens": torch.randn(num_samples, query_seq_len, query_dim),
        "candidate_title": torch.randn(num_samples, num_candidates, text_dim),
        "candidate_features": torch.randn(num_samples, num_candidates, text_dim),
        "candidate_description": torch.randn(num_samples, num_candidates, text_dim),
        "candidate_image_desc": torch.randn(
            num_samples, num_candidates, max_image_descs, text_dim
        ),
        "candidate_image_desc_mask": torch.randint(
            0, 2, (num_samples, num_candidates, max_image_descs)
        ).float(),
        "candidate_visual": torch.randn(
            num_samples, num_candidates, 5, visual_dim
        ),
        "candidate_visual_mask": torch.ones(
            num_samples, num_candidates, 5
        ),
        "labels": torch.randint(0, num_candidates, (num_samples,)),
    }

    return data
