"""
Query-conditioned multi-view candidate reranking model.

Architecture (pooled mode):
    Input → QueryEncoder → CandidateFusion (cross-attn) → ComparisonTransformer → ScoringHead → logits

Architecture (token-level mode):
    Input → TokenLevelCandidateFusion (cross-attn with compressed query tokens) → ComparisonTransformer → ScoringHead → logits

All encoders are frozen; only projection / fusion / comparison / scoring layers are trained.
Supports fp16 / bf16 mixed precision.

Ablation flags:
    ablation.no_visual   — ignore visual tokens (description-only)
    ablation.no_desc     — ignore description embedding (visual-only)
    ablation.no_cross_attn — skip CandidateFusion, use simple concat+pool
    ablation.no_comparison — skip CandidateComparisonTransformer
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. QueryEncoder
# ---------------------------------------------------------------------------


class QueryEncoder(nn.Module):
    """
    Encode a precomputed query token sequence into a fixed-size vector.

    Pipeline:
        [B, Tq, Dq] --linear--> [B, Tq, D]
                         --attn_pool--> [B, D]   (or mean pool)

    Args:
        query_dim:    input dimension of query tokens (Dq)
        hidden_dim:   projection dimension (D)
        use_attn_pool: if True, learnable attention pooling; else mean pool
        num_heads:    heads for attention pooling (only when use_attn_pool=True)
    """

    def __init__(
        self,
        query_dim: int,
        hidden_dim: int,
        use_attn_pool: bool = True,
        num_heads: int = 4,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_attn_pool = use_attn_pool

        # Project input query tokens to common hidden dim
        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.query_norm = nn.LayerNorm(hidden_dim)

        if use_attn_pool:
            # Learnable attention pooling: scalar attention over Tq positions
            # Inner dim must be divisible by num_heads — but we use a simple
            # single-vector query so we only need 1 effective head.
            self.attn_pool_q = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
            self.attn_scale = 1.0 / math.sqrt(hidden_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.query_proj.weight)
        nn.init.zeros_(self.query_proj.bias)

    def forward(self, query_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query_tokens: [B, Tq, Dq]  precomputed query token embeddings

        Returns:
            query_vec: [B, D]  compressed query representation
        """
        B, Tq, Dq = query_tokens.shape
        # --- project & normalise ---
        h = self.query_proj(query_tokens)       # [B, Tq, D]
        h = self.query_norm(h)                  # [B, Tq, D]

        if self.use_attn_pool:
            # Expand learnable query to batch
            q = self.attn_pool_q.expand(B, -1, -1)  # [B, 1, D]
            # Scaled dot-product attention: q attends over Tq positions
            scores = torch.bmm(q, h.transpose(1, 2)) * self.attn_scale  # [B, 1, Tq]
            weights = F.softmax(scores, dim=-1)                         # [B, 1, Tq]
            query_vec = torch.bmm(weights, h).squeeze(1)                # [B, D]
        else:
            # Simple mean pooling
            query_vec = h.mean(dim=1)  # [B, D]

        assert query_vec.shape == (B, self.hidden_dim), (
            f"QueryEncoder output shape mismatch: {query_vec.shape} vs ({B}, {self.hidden_dim})"
        )
        return query_vec


# ---------------------------------------------------------------------------
# 2. CandidateFusion  (cross-attention: query → all candidate tokens)
# ---------------------------------------------------------------------------


class CandidateFusion(nn.Module):
    """
    Fuse each candidate's multi-modal tokens using query-conditioned cross-attention.

    Each candidate has:
        - 1 pooled title token         [B, K, title_dim]    → project to D
        - 1 pooled features token      [B, K, feat_dim]     → project to D
        - 1 pooled description token   [B, K, desc_dim]     → project to D
        - N image desc tokens          [B, K, max_img, idim] → project to D (with mask)
        - 5 visual CLS tokens          [B, K, 5, vdim]      → project to D (with mask)

    Total tokens per candidate: 3 + max_image_descs + 5

    The query vector attends to these tokens (with key_padding_mask) to produce
    a fused representation.  Attention masks ensure padding positions are ignored.

    Args:
        title_dim:      input dim of pooled title embedding (default 768)
        feat_dim:       input dim of pooled features embedding (default 768)
        desc_dim:       input dim of pooled description embedding (default 768)
        img_desc_dim:   input dim of image description embeddings (default 768)
        visual_dim:     input dim of visual CLS tokens (default 4096)
        hidden_dim:     common projection dim (D)
        num_heads:      cross-attention heads
        num_layers:     number of cross-attention layers
        max_image_descs: max number of image description tokens (padded)
        dropout:        attention dropout
    """

    def __init__(
        self,
        title_dim: int = 768,
        feat_dim: int = 768,
        desc_dim: int = 768,
        img_desc_dim: int = 768,
        visual_dim: int = 4096,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        max_image_descs: int = 15,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_image_descs = max_image_descs

        # --- input projections for each modality ---
        self.title_proj = nn.Linear(title_dim, hidden_dim)
        self.feat_proj = nn.Linear(feat_dim, hidden_dim)
        self.desc_proj = nn.Linear(desc_dim, hidden_dim)
        self.img_desc_proj = nn.Linear(img_desc_dim, hidden_dim)
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)

        self.title_norm = nn.LayerNorm(hidden_dim)
        self.feat_norm = nn.LayerNorm(hidden_dim)
        self.desc_norm = nn.LayerNorm(hidden_dim)
        self.img_desc_norm = nn.LayerNorm(hidden_dim)
        self.visual_norm = nn.LayerNorm(hidden_dim)

        # --- cross-attention layers ---
        # Each layer: query (Q) attends to candidate tokens (K, V)
        self.cross_attn_layers = nn.ModuleList()
        self.cross_attn_norms = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.ffn_norms = nn.ModuleList()

        for _ in range(num_layers):
            self.cross_attn_layers.append(
                nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
            )
            self.cross_attn_norms.append(nn.LayerNorm(hidden_dim))
            self.ffn_layers.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.Dropout(dropout),
            ))
            self.ffn_norms.append(nn.LayerNorm(hidden_dim))

        self._reset_parameters()

    def _reset_parameters(self):
        for proj in (self.title_proj, self.feat_proj, self.desc_proj,
                     self.img_desc_proj, self.visual_proj):
            nn.init.xavier_uniform_(proj.weight)
            nn.init.zeros_(proj.bias)

    def forward(
        self,
        query_vec: torch.Tensor,
        candidate_title: torch.Tensor,
        candidate_features: torch.Tensor,
        candidate_description: torch.Tensor,
        candidate_image_desc: torch.Tensor,
        candidate_image_desc_mask: torch.Tensor,
        candidate_visual: torch.Tensor,
        candidate_visual_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            query_vec:                 [B, D]
            candidate_title:           [B, K, title_dim]    pooled title embedding
            candidate_features:        [B, K, feat_dim]     pooled features embedding
            candidate_description:     [B, K, desc_dim]     pooled description embedding
            candidate_image_desc:      [B, K, max_img, idim] padded image desc embeddings
            candidate_image_desc_mask: [B, K, max_img]      1=valid, 0=padding
            candidate_visual:          [B, K, 5, vdim]      visual CLS tokens (padded to 5)
            candidate_visual_mask:     [B, K, 5]            1=valid, 0=padding

        Returns:
            fused: [B, K, D]  query-aware fused candidate representation
        """
        B, K, _ = candidate_title.shape
        D = self.hidden_dim
        max_img = self.max_image_descs

        # --- project all modalities to hidden_dim ---
        # Text modalities: [B, K, dim] → [B, K, 1, D]
        title_h = self.title_norm(self.title_proj(candidate_title)).unsqueeze(2)
        feat_h = self.feat_norm(self.feat_proj(candidate_features)).unsqueeze(2)
        desc_h = self.desc_norm(self.desc_proj(candidate_description)).unsqueeze(2)

        # Image descriptions: [B*K, max_img, img_desc_dim] → [B, K, max_img, D]
        img_flat = candidate_image_desc.reshape(B * K, max_img, -1)
        img_h = self.img_desc_norm(self.img_desc_proj(img_flat))
        img_h = img_h.reshape(B, K, max_img, D)

        # Visual: [B*K, 5, visual_dim] → [B, K, 5, D]
        vis_flat = candidate_visual.reshape(B * K, 5, -1)
        vis_h = self.visual_norm(self.visual_proj(vis_flat))
        vis_h = vis_h.reshape(B, K, 5, D)

        # --- concatenate all tokens: [B, K, Tc, D] ---
        cand_tokens = torch.cat([title_h, feat_h, desc_h, img_h, vis_h], dim=2)
        Tc = cand_tokens.size(2)  # 3 + max_img + 5

        # --- build combined attention mask: [B, K, Tc] ---
        # Pooled text tokens are always valid (may be zero vectors but still present)
        ones = torch.ones(B, K, 1, device=query_vec.device, dtype=torch.bool)
        combined_mask = torch.cat([
            ones, ones, ones,                       # title, feat, desc
            candidate_image_desc_mask.bool(),       # [B, K, max_img]
            candidate_visual_mask.bool(),            # [B, K, 5]
        ], dim=2)  # [B, K, Tc]

        # Reshape for parallel processing: [B*K, Tc, D]
        cand_tokens_flat = cand_tokens.reshape(B * K, Tc, D)

        # key_padding_mask: True = ignore (inverse of validity mask)
        key_padding_mask = ~combined_mask.reshape(B * K, Tc)  # [B*K, Tc]

        # Query: [B, D] → [B*K, 1, D]
        q = query_vec.unsqueeze(1).expand(B, K, D).reshape(B * K, 1, D)

        # --- cross-attention layers (post-norm style) ---
        for i in range(self.num_layers):
            attn_out, _ = self.cross_attn_layers[i](
                q, cand_tokens_flat, cand_tokens_flat,
                key_padding_mask=key_padding_mask,
            )  # attn_out: [B*K, 1, D]
            q = self.cross_attn_norms[i](q + attn_out)
            q = self.ffn_norms[i](q + self.ffn_layers[i](q))

        # q is now the fused representation: [B*K, 1, D] → [B, K, D]
        fused = q.squeeze(1).reshape(B, K, D)

        assert fused.shape == (B, K, D), (
            f"CandidateFusion output shape mismatch: {fused.shape} vs ({B}, {K}, {D})"
        )
        return fused


# ---------------------------------------------------------------------------
# 2b. TokenLevelCandidateFusion  (token-level cross-attention)
# ---------------------------------------------------------------------------


class QueryCompressor(nn.Module):
    """
    Compress a variable-length query token sequence into a fixed number of
    learned summary positions via cross-attention.

    [B, Tq, D] → [B, Tq', D]  where Tq' = num_queries (learned).

    Args:
        hidden_dim:    model dimension (D)
        num_queries:   number of output query positions (Tq')
        num_heads:     attention heads
    """

    def __init__(self, hidden_dim: int, num_queries: int = 32, num_heads: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.learned_queries = nn.Parameter(torch.randn(1, num_queries, hidden_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm_q = nn.LayerNorm(hidden_dim)
        self.norm_kv = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.norm_ffn = nn.LayerNorm(hidden_dim)

    def forward(self, query_tokens: torch.Tensor, query_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            query_tokens: [B, Tq, D]  projected query token embeddings
            query_mask:   [B, Tq]     1=valid, 0=padding (optional)

        Returns:
            compressed: [B, Tq', D]
        """
        B = query_tokens.size(0)
        q = self.learned_queries.expand(B, -1, -1)  # [B, Tq', D]
        kv = query_tokens  # [B, Tq, D]

        key_padding_mask = None
        if query_mask is not None:
            key_padding_mask = ~query_mask.bool()  # True = ignore

        # Cross-attention: learned queries attend to query tokens
        q_normed = self.norm_q(q)
        kv_normed = self.norm_kv(kv)
        attn_out, _ = self.cross_attn(q_normed, kv_normed, kv_normed,
                                       key_padding_mask=key_padding_mask)
        q = q + attn_out
        q = q + self.ffn(self.norm_ffn(q))

        return q


class TokenLevelCandidateFusion(nn.Module):
    """
    Token-level cross-attention fusion between compressed query tokens and
    per-field candidate token sequences.

    Unlike CandidateFusion (which uses a single pooled query vector), this
    preserves multi-token representations on both sides, allowing each query
    token to individually attend to specific candidate tokens.

    Pipeline:
        Query: [B, Tq, Dq] → project → [B, Tq, D] → compress → [B, Tq'=32, D]
             → expand to [B*K, Tq', D]

        Candidate tokens per field: [B, K, MT, 768] → project → [B*K, MT, D]
        Image desc:  [B, K, 15, 768]  → project → [B*K, 15, D]
        Visual:      [B, K, 5, 4096]  → project → [B*K, 5, D]
        → concatenate → [B*K, Tc=MT*3+15+5, D]

        Cross-attention: Q=[B*K, 32, D] x K/V=[B*K, Tc, D], 2 layers
        Aggregate: [B*K, 32, D] → [B, K, D] via learned query attention

    Args:
        query_dim:       input dim of query tokens
        title_dim:       input dim of title token embeddings
        feat_dim:        input dim of feature token embeddings
        desc_dim:        input dim of description token embeddings
        img_desc_dim:    input dim of image description embeddings
        visual_dim:      input dim of visual CLS tokens
        hidden_dim:      common projection dim (D)
        num_heads:       cross-attention heads
        num_layers:      number of cross-attention layers
        max_image_descs: max number of image description tokens
        max_text_tokens: max tokens per text field (title/feat/desc)
        num_query_tokens: number of compressed query positions (Tq')
        dropout:         attention dropout
        aggregation:     how to reduce 32 query outputs → single vec per candidate
                         "learned_query" | "mean_pool" | "attention_pool"
    """

    def __init__(
        self,
        query_dim: int = 768,
        title_dim: int = 768,
        feat_dim: int = 768,
        desc_dim: int = 768,
        img_desc_dim: int = 768,
        visual_dim: int = 4096,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        max_image_descs: int = 15,
        max_text_tokens: int = 64,
        num_query_tokens: int = 32,
        dropout: float = 0.1,
        aggregation: str = "learned_query",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_image_descs = max_image_descs
        self.max_text_tokens = max_text_tokens
        self.num_query_tokens = num_query_tokens
        self.aggregation = aggregation

        # --- query projection and compression ---
        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.query_compressor = QueryCompressor(
            hidden_dim, num_queries=num_query_tokens, num_heads=num_heads,
        )

        # --- separate projections for each text field ---
        self.title_proj = nn.Linear(title_dim, hidden_dim)
        self.feat_proj = nn.Linear(feat_dim, hidden_dim)
        self.desc_proj = nn.Linear(desc_dim, hidden_dim)
        self.img_desc_proj = nn.Linear(img_desc_dim, hidden_dim)
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)

        self.title_norm = nn.LayerNorm(hidden_dim)
        self.feat_norm = nn.LayerNorm(hidden_dim)
        self.desc_norm = nn.LayerNorm(hidden_dim)
        self.img_desc_norm = nn.LayerNorm(hidden_dim)
        self.visual_norm = nn.LayerNorm(hidden_dim)

        # --- cross-attention layers ---
        self.cross_attn_layers = nn.ModuleList()
        self.cross_attn_norms = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.ffn_norms = nn.ModuleList()

        for _ in range(num_layers):
            self.cross_attn_layers.append(
                nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
            )
            self.cross_attn_norms.append(nn.LayerNorm(hidden_dim))
            self.ffn_layers.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.Dropout(dropout),
            ))
            self.ffn_norms.append(nn.LayerNorm(hidden_dim))

        # --- aggregation ---
        if aggregation == "learned_query":
            self.agg_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
            self.agg_attn = nn.MultiheadAttention(hidden_dim, num_heads=1, batch_first=True)
            self.agg_norm = nn.LayerNorm(hidden_dim)
        elif aggregation == "attention_pool":
            self.attn_pool_q = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
            self.attn_scale = 1.0 / math.sqrt(hidden_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        for proj in (self.query_proj, self.title_proj, self.feat_proj, self.desc_proj,
                     self.img_desc_proj, self.visual_proj):
            nn.init.xavier_uniform_(proj.weight)
            nn.init.zeros_(proj.bias)

    def forward(
        self,
        query_tokens: torch.Tensor,
        query_attention_mask: torch.Tensor,
        candidate_title_tokens: torch.Tensor,
        candidate_title_mask: torch.Tensor,
        candidate_feature_tokens: torch.Tensor,
        candidate_feature_mask: torch.Tensor,
        candidate_description_tokens: torch.Tensor,
        candidate_description_mask: torch.Tensor,
        candidate_image_desc: torch.Tensor,
        candidate_image_desc_mask: torch.Tensor,
        candidate_visual: torch.Tensor,
        candidate_visual_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            query_tokens:                    [B, Tq, Dq]
            query_attention_mask:            [B, Tq]  1=valid, 0=padding
            candidate_title_tokens:          [B, K, MT, title_dim]
            candidate_title_mask:            [B, K, MT]  1=valid, 0=padding
            candidate_feature_tokens:        [B, K, MT, feat_dim]
            candidate_feature_mask:          [B, K, MT]
            candidate_description_tokens:    [B, K, MT, desc_dim]
            candidate_description_mask:      [B, K, MT]
            candidate_image_desc:            [B, K, max_img, img_desc_dim]
            candidate_image_desc_mask:       [B, K, max_img]
            candidate_visual:                [B, K, 5, visual_dim]
            candidate_visual_mask:           [B, K, 5]

        Returns:
            fused: [B, K, D]
        """
        B, K = candidate_title_tokens.shape[:2]
        D = self.hidden_dim
        MT = self.max_text_tokens
        max_img = self.max_image_descs

        # --- project and compress query ---
        q_proj = self.query_norm(self.query_proj(query_tokens))  # [B, Tq, D]
        compressed_q = self.query_compressor(q_proj, query_attention_mask)  # [B, Tq', D]

        # Expand to [B*K, Tq', D]
        q_expanded = compressed_q.unsqueeze(1).expand(B, K, -1, -1).reshape(B * K, self.num_query_tokens, D)

        # --- project candidate text tokens per field ---
        # title: [B, K, MT, title_dim] → [B*K, MT, D]
        title_flat = candidate_title_tokens.reshape(B * K, MT, -1)
        title_h = self.title_norm(self.title_proj(title_flat))

        # features: [B, K, MT, feat_dim] → [B*K, MT, D]
        feat_flat = candidate_feature_tokens.reshape(B * K, MT, -1)
        feat_h = self.feat_norm(self.feat_proj(feat_flat))

        # description: [B, K, MT, desc_dim] → [B*K, MT, D]
        desc_flat = candidate_description_tokens.reshape(B * K, MT, -1)
        desc_h = self.desc_norm(self.desc_proj(desc_flat))

        # image descriptions: [B*K, max_img, img_desc_dim] → [B*K, max_img, D]
        img_flat = candidate_image_desc.reshape(B * K, max_img, -1)
        img_h = self.img_desc_norm(self.img_desc_proj(img_flat))

        # visual: [B*K, 5, visual_dim] → [B*K, 5, D]
        vis_flat = candidate_visual.reshape(B * K, 5, -1)
        vis_h = self.visual_norm(self.visual_proj(vis_flat))

        # --- concatenate all candidate tokens: [B*K, Tc, D] ---
        cand_tokens = torch.cat([title_h, feat_h, desc_h, img_h, vis_h], dim=1)
        Tc = cand_tokens.size(1)  # MT*3 + max_img + 5

        # --- build key_padding_mask: [B, K, Tc] → [B*K, Tc] ---
        text_title_mask = candidate_title_mask.bool()    # [B, K, MT]
        text_feat_mask = candidate_feature_mask.bool()    # [B, K, MT]
        text_desc_mask = candidate_description_mask.bool()  # [B, K, MT]

        combined_mask = torch.cat([
            text_title_mask,
            text_feat_mask,
            text_desc_mask,
            candidate_image_desc_mask.bool(),   # [B, K, max_img]
            candidate_visual_mask.bool(),       # [B, K, 5]
        ], dim=2)  # [B, K, Tc]

        key_padding_mask = ~combined_mask.reshape(B * K, Tc)  # True = ignore

        # --- cross-attention layers (post-norm) ---
        q = q_expanded  # [B*K, Tq', D]
        for i in range(self.num_layers):
            attn_out, _ = self.cross_attn_layers[i](
                q, cand_tokens, cand_tokens,
                key_padding_mask=key_padding_mask,
            )
            q = self.cross_attn_norms[i](q + attn_out)
            q = self.ffn_norms[i](q + self.ffn_layers[i](q))

        # q: [B*K, Tq', D] → aggregate to [B*K, D] → reshape to [B, K, D]

        if self.aggregation == "learned_query":
            agg_q = self.agg_token.expand(B * K, -1, -1)  # [B*K, 1, D]
            agg_out, _ = self.agg_attn(
                self.agg_norm(agg_q), q, q,
            )  # [B*K, 1, D]
            fused = agg_out.squeeze(1).reshape(B, K, D)
        elif self.aggregation == "attention_pool":
            pool_q = self.attn_pool_q.expand(B * K, -1, -1)  # [B*K, 1, D]
            scores = torch.bmm(pool_q, q.transpose(1, 2)) * self.attn_scale  # [B*K, 1, Tq']
            weights = F.softmax(scores, dim=-1)
            fused = torch.bmm(weights, q).squeeze(1).reshape(B, K, D)
        else:  # mean_pool
            fused = q.mean(dim=1).reshape(B, K, D)

        assert fused.shape == (B, K, D), (
            f"TokenLevelCandidateFusion output shape mismatch: {fused.shape} vs ({B}, {K}, {D})"
        )
        return fused


# ---------------------------------------------------------------------------
# 3. CandidateComparisonTransformer
# ---------------------------------------------------------------------------


class CandidateComparisonTransformer(nn.Module):
    """
    Transformer that operates across K candidates so the model can compare them.

    Input:  [B, K, D]  (one fused vector per candidate)
    Output: [B, K, D]  (comparison-aware representation)

    Uses a standard TransformerEncoder with positional encoding over K positions.

    Args:
        hidden_dim:    model dimension (D)
        num_heads:     attention heads
        num_layers:    transformer layers
        dropout:       dropout rate
        max_candidates: maximum K (for positional encoding)
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_candidates: int = 50,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Learnable positional encoding for up to max_candidates positions
        self.pos_embed = nn.Parameter(torch.randn(1, max_candidates, hidden_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # pre-norm for stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_dim),
        )

        self._reset_parameters()

    def _reset_parameters(self):
        # Xavier init is handled by TransformerEncoderLayer defaults
        pass

    def forward(self, candidate_repr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            candidate_repr: [B, K, D]

        Returns:
            comparison_repr: [B, K, D]
        """
        B, K, D = candidate_repr.shape

        # Add positional encoding
        x = candidate_repr + self.pos_embed[:, :K, :]  # [B, K, D]

        # Transformer across K candidates
        x = self.transformer(x)  # [B, K, D]

        assert x.shape == (B, K, D), (
            f"ComparisonTransformer output shape mismatch: {x.shape} vs ({B}, {K}, {D})"
        )
        return x


# ---------------------------------------------------------------------------
# 4. ScoringHead
# ---------------------------------------------------------------------------


class ScoringHead(nn.Module):
    """
    Map each candidate's representation to a scalar score.

    MLP: [B, K, D] → [B, K, D//2] → [B, K]

    Args:
        hidden_dim: input dimension (D)
        dropout:    dropout before final linear
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, candidate_repr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            candidate_repr: [B, K, D]

        Returns:
            logits: [B, K]
        """
        logits = self.mlp(candidate_repr).squeeze(-1)  # [B, K]
        return logits


# ---------------------------------------------------------------------------
# 5. AttributeHead
# ---------------------------------------------------------------------------


class AttributeHead(nn.Module):
    """
    Auxiliary attribute prediction head.

    Predicts binary presence of each modality token (image_desc + visual masks)
    from the fused candidate representation — a self-supervised auxiliary task
    that encourages the model to preserve modality information.

    MLP: [B, K, D] → [B, K, D//2] → [B, K, num_attrs]
    """

    def __init__(self, hidden_dim: int, num_attrs: int = 20, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_attrs),
        )

    def forward(self, candidate_repr: torch.Tensor) -> torch.Tensor:
        return self.mlp(candidate_repr)  # [B, K, num_attrs]


# ---------------------------------------------------------------------------
# 6. RerankingModel  (assembled)
# ---------------------------------------------------------------------------


class RerankingModel(nn.Module):
    """
    Full query-conditioned multi-view candidate reranking model.

    Supports two modes:
        - Pooled (default): query is pooled to a single vector, text features
          are pre-pooled. Uses QueryEncoder + CandidateFusion.
        - Token-level (--token_level_cross_attn): query tokens are compressed
          via learned queries, text features are kept at token level. Uses
          TokenLevelCandidateFusion.

    Data flow (pooled mode):
        query_tokens [B, Tq, Dq]
            → QueryEncoder → query_vec [B, D]
        All candidate features (title, features, description, image_desc, visual)
            → CandidateFusion(query_vec) → fused [B, K, D]

    Data flow (token-level mode):
        query_tokens [B, Tq, Dq] + query_attention_mask [B, Tq]
            → TokenLevelCandidateFusion → fused [B, K, D]

    Common:
        fused → CandidateComparisonTransformer → compared [B, K, D]
             → ScoringHead → logits [B, K]

    Args:
        query_dim:    dimension of query tokens (Dq)
        title_dim:    dimension of pooled title embeddings (default 768)
        feat_dim:     dimension of pooled features embeddings (default 768)
        desc_dim:     dimension of pooled description embeddings (default 768)
        img_desc_dim: dimension of image description embeddings (default 768)
        visual_dim:   dimension of visual CLS tokens (default 4096)
        hidden_dim:   internal projection dim (D)
        num_candidates: default K for shape assertions
        num_cross_attn_heads: heads in CandidateFusion
        num_cross_attn_layers: layers in CandidateFusion
        num_comparison_heads: heads in ComparisonTransformer
        num_comparison_layers: layers in ComparisonTransformer
        max_candidates: max K for positional encoding
        max_image_descs: max number of image descriptions per candidate
        dropout:       dropout rate
        use_attn_pool: attention pooling in QueryEncoder
        token_level_cross_attn: enable token-level cross-attention mode
        max_text_tokens: max tokens per text field (token-level mode only)
        num_query_tokens: compressed query positions (token-level mode only)
        token_aggregation: aggregation method for token-level mode
    """

    def __init__(
        self,
        query_dim: int = 768,
        title_dim: int = 768,
        feat_dim: int = 768,
        desc_dim: int = 768,
        img_desc_dim: int = 768,
        visual_dim: int = 4096,
        hidden_dim: int = 256,
        num_candidates: int = 10,
        num_cross_attn_heads: int = 4,
        num_cross_attn_layers: int = 2,
        num_comparison_heads: int = 4,
        num_comparison_layers: int = 2,
        max_candidates: int = 50,
        max_image_descs: int = 15,
        dropout: float = 0.1,
        use_attn_pool: bool = True,
        token_level_cross_attn: bool = False,
        max_text_tokens: int = 64,
        num_query_tokens: int = 32,
        token_aggregation: str = "learned_query",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_candidates = num_candidates
        self.token_level_cross_attn = token_level_cross_attn

        if token_level_cross_attn:
            self.candidate_fusion = TokenLevelCandidateFusion(
                query_dim=query_dim,
                title_dim=title_dim,
                feat_dim=feat_dim,
                desc_dim=desc_dim,
                img_desc_dim=img_desc_dim,
                visual_dim=visual_dim,
                hidden_dim=hidden_dim,
                num_heads=num_cross_attn_heads,
                num_layers=num_cross_attn_layers,
                max_image_descs=max_image_descs,
                max_text_tokens=max_text_tokens,
                num_query_tokens=num_query_tokens,
                dropout=dropout,
                aggregation=token_aggregation,
            )
        else:
            self.query_encoder = QueryEncoder(
                query_dim=query_dim,
                hidden_dim=hidden_dim,
                use_attn_pool=use_attn_pool,
            )
            self.candidate_fusion = CandidateFusion(
                title_dim=title_dim,
                feat_dim=feat_dim,
                desc_dim=desc_dim,
                img_desc_dim=img_desc_dim,
                visual_dim=visual_dim,
                hidden_dim=hidden_dim,
                num_heads=num_cross_attn_heads,
                num_layers=num_cross_attn_layers,
                max_image_descs=max_image_descs,
                dropout=dropout,
            )

        self.comparison_transformer = CandidateComparisonTransformer(
            hidden_dim=hidden_dim,
            num_heads=num_comparison_heads,
            num_layers=num_comparison_layers,
            dropout=dropout,
            max_candidates=max_candidates,
        )

        self.scoring_head = ScoringHead(
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        self.attr_head = AttributeHead(
            hidden_dim=hidden_dim,
            num_attrs=max_image_descs + 5,
            dropout=dropout,
        )

    def forward(
        self,
        query_tokens: torch.Tensor,
        candidate_title: torch.Tensor,
        candidate_features: torch.Tensor,
        candidate_description: torch.Tensor,
        candidate_image_desc: torch.Tensor,
        candidate_image_desc_mask: torch.Tensor,
        candidate_visual: torch.Tensor,
        candidate_visual_mask: torch.Tensor,
        query_attention_mask: torch.Tensor = None,
        candidate_title_tokens: torch.Tensor = None,
        candidate_title_mask: torch.Tensor = None,
        candidate_feature_tokens: torch.Tensor = None,
        candidate_feature_mask: torch.Tensor = None,
        candidate_description_tokens: torch.Tensor = None,
        candidate_description_mask: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Dispatches to pooled or token-level mode based on self.token_level_cross_attn.

        Pooled mode args:
            query_tokens:               [B, Tq, Dq]
            candidate_title:            [B, K, title_dim]
            candidate_features:         [B, K, feat_dim]
            candidate_description:      [B, K, desc_dim]
            candidate_image_desc:       [B, K, max_img, img_desc_dim]
            candidate_image_desc_mask:  [B, K, max_img]
            candidate_visual:           [B, K, 5, visual_dim]
            candidate_visual_mask:      [B, K, 5]

        Token-level mode args (in addition to shared ones):
            query_attention_mask:            [B, Tq]
            candidate_title_tokens:          [B, K, MT, title_dim]
            candidate_title_mask:            [B, K, MT]
            candidate_feature_tokens:        [B, K, MT, feat_dim]
            candidate_feature_mask:          [B, K, MT]
            candidate_description_tokens:    [B, K, MT, desc_dim]
            candidate_description_mask:      [B, K, MT]

        Returns:
            logits: [B, K]
            attr_predictions: [B, K, num_attrs]
        """
        B = query_tokens.size(0)
        K = candidate_visual.size(1)

        if self.token_level_cross_attn:
            fused = self.candidate_fusion(
                query_tokens=query_tokens,
                query_attention_mask=query_attention_mask,
                candidate_title_tokens=candidate_title_tokens,
                candidate_title_mask=candidate_title_mask,
                candidate_feature_tokens=candidate_feature_tokens,
                candidate_feature_mask=candidate_feature_mask,
                candidate_description_tokens=candidate_description_tokens,
                candidate_description_mask=candidate_description_mask,
                candidate_image_desc=candidate_image_desc,
                candidate_image_desc_mask=candidate_image_desc_mask,
                candidate_visual=candidate_visual,
                candidate_visual_mask=candidate_visual_mask,
            )
        else:
            query_vec = self.query_encoder(query_tokens)
            fused = self.candidate_fusion(
                query_vec,
                candidate_title,
                candidate_features,
                candidate_description,
                candidate_image_desc,
                candidate_image_desc_mask,
                candidate_visual,
                candidate_visual_mask,
            )

        compared = self.comparison_transformer(fused)
        logits = self.scoring_head(compared)
        attr_preds = self.attr_head(compared)

        assert logits.shape == (B, K), (
            f"RerankingModel output shape mismatch: {logits.shape} vs ({B}, {K})"
        )
        return logits, attr_preds
