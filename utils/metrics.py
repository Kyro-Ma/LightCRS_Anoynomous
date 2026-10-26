"""
Evaluation metrics for reranking: HR@K, NDCG@K, MRR.
"""

import torch


def hr_at_k(logits: torch.Tensor, labels: torch.Tensor, k: int = 1) -> float:
    """
    Hit Rate at K.

    Args:
        logits: [B, K] predicted scores
        labels: [B]    ground-truth candidate index
        k:      cutoff

    Returns:
        float: HR@K
    """
    _, topk = logits.topk(k, dim=1)             # [B, k]
    labels_exp = labels.unsqueeze(1).expand_as(topk)  # [B, k]
    hits = (topk == labels_exp).any(dim=1).float()
    return hits.mean().item()


def ndcg_at_k(logits: torch.Tensor, labels: torch.Tensor, k: int = 1) -> float:
    """
    Normalized DCG at K (single relevant item per query).

    Args:
        logits: [B, K]
        labels: [B]
        k:      cutoff

    Returns:
        float: NDCG@K
    """
    _, topk = logits.topk(k, dim=1)                  # [B, k]
    labels_exp = labels.unsqueeze(1).expand_as(topk)
    hits = (topk == labels_exp).float()              # [B, k]

    ranks = torch.arange(1, k + 1, device=logits.device, dtype=torch.float)
    discounts = 1.0 / torch.log2(ranks + 1)          # [k]

    dcg = (hits * discounts.unsqueeze(0)).sum(dim=1)  # [B]
    # IDCG for single relevant item at rank 1 = 1 / log2(2) = 1.0
    return dcg.mean().item()


def mrr(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Mean Reciprocal Rank.

    Args:
        logits: [B, K]
        labels: [B]

    Returns:
        float: MRR
    """
    _, sorted_idx = logits.sort(dim=1, descending=True)     # [B, K]
    labels_exp = labels.unsqueeze(1).expand_as(sorted_idx)
    matches = sorted_idx == labels_exp                       # [B, K]

    # Find rank of first match per row
    # argmax on boolean finds first True
    ranks = matches.float().argmax(dim=1) + 1  # 1-indexed [B]
    reciprocal = 1.0 / ranks.float()

    # Handle cases where no match (shouldn't happen with valid data)
    any_match = matches.any(dim=1)
    reciprocal = reciprocal * any_match.float()

    return reciprocal.mean().item()


def valid_ratio(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Valid Ratio (VR): proportion of top-1 predictions that are valid candidates.

    For discriminative reranking models, all predictions are valid by
    construction since the model selects from a known candidate set (VR = 1.0).
    This metric is included for consistency with generative baselines where
    the model may produce out-of-distribution IDs.

    Args:
        logits: [B, K] predicted scores
        labels: [B]    ground-truth candidate index

    Returns:
        float: VR (always 1.0 for reranking)
    """
    return 1.0


def compute_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ks: tuple = (1, 5, 10),
) -> dict:
    """
    Compute HR@K, NDCG@K for multiple cutoffs, plus MRR and VR.

    Args:
        logits: [B, K]
        labels: [B]
        ks:     tuple of cutoff values

    Returns:
        dict of metric name → float value
    """
    metrics = {}
    for k in ks:
        metrics[f"hr@{k}"] = hr_at_k(logits, labels, k)
        metrics[f"ndcg@{k}"] = ndcg_at_k(logits, labels, k)
    metrics["mrr"] = mrr(logits, labels)
    metrics["vr"] = valid_ratio(logits, labels)
    return metrics
