"""
Offline combo generation: encode queries and candidate features together in batches.

Workflow per shard_size (32) queries:
  1. Encode shard_size queries using GTE-base (batch)
  2. For each query's candidates, generate text features from item metadata
     using the same GTE-base encoder (one by one, cached)
  3. Bundle query + candidates into combo dicts
  4. Stack and save as one shard file
  5. Move to next shard_size queries

Candidate features are generated fresh from item2meta metadata files,
NOT fetched from existing ./features/. Generated features are cached
so the same item is not re-encoded across batches.

Usage:
    python generate_combos.py --candidate_source st
    python generate_combos.py --candidate_source gpt_large
    python generate_combos.py --candidate_source st --dry_run
"""

import argparse
import gc
import hashlib
import io
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
import torch
import torch.nn.functional as F
from PIL import Image

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

os.environ["HF_HOME"] = os.path.join(_PROJECT_ROOT, ".cache", "huggingface")
os.environ["HF_HUB_OFFLINE"] = "1"

from transformers import AutoTokenizer, AutoModel, AutoProcessor


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TEXT_DIM = 768
VISUAL_DIM = 4096
MAX_IMAGE_DESCS = 15
MAX_VISUAL = 5
TEXT_SEQ_LEN = 512

IMAGE_TOKENS = [
    "<ItemImageEmb1>", "<ItemImageEmb2>", "<ItemImageEmb3>",
    "<ItemImageEmb4>", "<ItemImageEmb5>",
]


# ---------------------------------------------------------------------------
# Reused image / text helpers (copied from generate_item_multiview_features.py)
# ---------------------------------------------------------------------------

def to_image(path_or_url: str) -> Image.Image:
    """Load RGB image from local path or URL."""
    if isinstance(path_or_url, str) and path_or_url.startswith("http"):
        r = requests.get(path_or_url, timeout=30)
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert("RGB")
    return Image.open(path_or_url).convert("RGB")


def prepare_five_views(image: Image.Image, size: int = 336) -> List[Image.Image]:
    """Generate 5 views: resize to 336x336, replicate 5 times."""
    img = image.resize((size, size))
    return [img, img, img, img, img]


def normalize_pixel_values(
    pixel_values: torch.Tensor, target_batch: int = 1,
) -> Tuple[torch.Tensor, Optional[int]]:
    """Normalize pixel_values to (B, V, C, H, W)."""
    num_crops = None
    if pixel_values.ndim == 4:
        pixel_values = pixel_values.unsqueeze(0)
    elif pixel_values.ndim == 5:
        if pixel_values.shape[0] != target_batch and target_batch == 1:
            pixel_values = pixel_values[:1]
    elif pixel_values.ndim == 6:
        num_crops = pixel_values.shape[2]
    else:
        raise ValueError(f"Unexpected pixel_values shape: {tuple(pixel_values.shape)}")
    return pixel_values, num_crops


def extract_projected_features(
    model, pixel_values: torch.Tensor, num_crops: Optional[int],
) -> torch.Tensor:
    """Run vision tower + projector -> (B, V, hidden)."""
    device = next(model.parameters()).device

    if pixel_values.ndim == 6:
        B, V, Nc, C, H, W = pixel_values.shape
        pv = pixel_values.view(B * V * Nc, C, H, W).to(device, dtype=torch.float16)
        with torch.no_grad():
            vision_outputs = model.vision_tower(pv, return_dict=False)
        last_hidden = vision_outputs[0]
        cls = last_hidden[:, 0, :].view(B, V, Nc, -1)
        cls = cls.mean(dim=2)
        cls_proj = model.multi_modal_projector(cls)
        return cls_proj

    elif pixel_values.ndim == 5:
        B, V, C, H, W = pixel_values.shape
        pv = pixel_values.view(B * V, C, H, W).to(device, dtype=torch.float16)
        with torch.no_grad():
            vision_outputs = model.vision_tower(pv, return_dict=False)
        last_hidden = vision_outputs[0]
        cls = last_hidden[:, 0, :].view(B, V, -1)
        cls_proj = model.multi_modal_projector(cls)
        return cls_proj

    else:
        raise ValueError(f"Unexpected pixel_values ndim: {pixel_values.ndim}")


def pick_image_url(images_list: List[dict]) -> Tuple[Optional[str], int]:
    """Select the best image URL following resolution priority."""
    if not images_list:
        return None, -1
    for idx, img in enumerate(images_list):
        url = img.get("hi_res") or img.get("large") or img.get("thumb")
        if url:
            return url, idx
    return None, -1


def clean_features(features: Optional[list]) -> str:
    """Join feature bullet strings. Handle missing/empty fields."""
    if not features:
        return ""
    if isinstance(features, str):
        return features.strip()
    parts = [f.strip() for f in features if f and f.strip()]
    return "\n".join(parts)


def clean_description(description: Optional[list]) -> str:
    """Join description paragraphs. Handle missing/empty fields."""
    if not description:
        return ""
    if isinstance(description, str):
        return description.strip()
    parts = [d.strip() for d in description if d and d.strip()]
    return "\n".join(parts)


def get_image_descriptions(item: dict) -> List[str]:
    """Extract ordered image descriptions from image_descriptions_llava_cleaned."""
    desc_map = item.get("image_descriptions_llava_cleaned") or {}
    if not desc_map:
        return []

    def sort_key(kv):
        k = kv[0]
        try:
            num = int(k.rsplit("_", 1)[-1].replace(".jpg", ""))
            return num
        except (ValueError, IndexError):
            return 0

    sorted_descs = sorted(desc_map.items(), key=sort_key)
    return [v for k, v in sorted_descs if v and v.strip()]


# ---------------------------------------------------------------------------
# TextEncoder: Alibaba-NLP/gte-base-en-v1.5
# ---------------------------------------------------------------------------

class TextEncoder:
    """
    Wrapper for Alibaba-NLP/gte-base-en-v1.5 token-level encoding.

    Encodes text fields to:
      - Token-level embeddings: [T, D]  (preserves per-token information)
      - Attention mask: [T]
      - Pooled embeddings: [D]  (mean pool over valid tokens)
    """

    def __init__(self, device: torch.device, max_length: int = 512):
        self.device = device
        self.max_length = max_length

        print(f"[TextEncoder] Loading Alibaba-NLP/gte-base-en-v1.5 ...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            "Alibaba-NLP/gte-base-en-v1.5", trust_remote_code=True,
        )
        self.model = AutoModel.from_pretrained(
            "Alibaba-NLP/gte-base-en-v1.5",
            trust_remote_code=True,
            dtype=torch.float16,
        ).to(device)
        self.model.eval()

        self.hidden_dim = self.model.config.hidden_size
        print(f"[TextEncoder] Ready. hidden_dim={self.hidden_dim}, device={device}")

    @torch.no_grad()
    def encode_token_level(self, text: str) -> Dict[str, torch.Tensor]:
        """
        Encode text -> token-level embeddings + attention_mask.

        Returns:
            {
                "embeddings":      [T, D]  token-level hidden states (fp32)
                "attention_mask":  [T]     binary mask (1 = valid token)
            }
        """
        if not text or not text.strip():
            return {
                "embeddings": torch.zeros(1, self.hidden_dim, dtype=torch.float32),
                "attention_mask": torch.zeros(1, dtype=torch.long),
            }

        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            token_embs = outputs.last_hidden_state.squeeze(0).float()

        return {
            "embeddings": token_embs.cpu(),
            "attention_mask": attention_mask.squeeze(0).cpu(),
        }

    @torch.no_grad()
    def encode_pooled(self, text: str) -> torch.Tensor:
        """Encode text -> single pooled embedding (mean pool over valid tokens). [D]"""
        if not text or not text.strip():
            return torch.zeros(self.hidden_dim, dtype=torch.float32)

        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="longest",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            token_embs = outputs.last_hidden_state

        mask_exp = attention_mask.unsqueeze(-1).float()
        pooled = (token_embs.float() * mask_exp).sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1)
        return pooled.squeeze(0).cpu()

    @torch.no_grad()
    def encode_pooled_batch(self, texts: List[str]) -> torch.Tensor:
        """Encode a batch of texts -> pooled embeddings. [B, D]"""
        valid = [t if t and t.strip() else " " for t in texts]

        encoded = self.tokenizer(
            valid,
            max_length=self.max_length,
            padding="longest",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            token_embs = outputs.last_hidden_state

        mask_exp = attention_mask.unsqueeze(-1).float()
        pooled = (token_embs.float() * mask_exp).sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1)
        return pooled.cpu()


# ---------------------------------------------------------------------------
# VisualEncoder: LLaVA v1.6
# ---------------------------------------------------------------------------

class VisualEncoder:
    """
    Wraps LLaVA v1.6 for 5 CLS token extraction.

    Pipeline:
      1. prepare_five_views(image, size=336)
      2. processor.image_processor(views) -> pixel_values
      3. normalize_pixel_values -> (1, 5, C, H, W)
      4. vision_tower -> CLS tokens
      5. multi_modal_projector -> projected features
    """

    def __init__(self, device: torch.device, model_name: str = "llava-hf/llava-v1.6-mistral-7b-hf"):
        self.device = device
        print(f"[VisualEncoder] Loading {model_name} on {device} ...")

        from transformers import LlavaNextForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            model_name,
            dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).to(device)

        self._ensure_special_tokens()
        self.model.eval()

        proj = self.model.multi_modal_projector
        if hasattr(proj, 'proj') and hasattr(proj.proj, 'out_features'):
            self.hidden_dim = proj.proj.out_features
        elif hasattr(proj, 'linear_2') and hasattr(proj.linear_2, 'out_features'):
            self.hidden_dim = proj.linear_2.out_features
        else:
            self.hidden_dim = self.model.config.text_config.hidden_size
        print(f"[VisualEncoder] Ready. hidden_dim={self.hidden_dim}, device={device}")

    def _ensure_special_tokens(self):
        """Ensure image tokens are in tokenizer, resize embeddings if needed."""
        tokenizer = self.processor.tokenizer
        added = []
        for tk in IMAGE_TOKENS:
            tid = tokenizer.convert_tokens_to_ids(tk)
            if tid is None or tid == tokenizer.unk_token_id:
                added.append(tk)
        if added:
            tokenizer.add_special_tokens({"additional_special_tokens": added})
            self.model.resize_token_embeddings(len(tokenizer))
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token

    @torch.no_grad()
    def extract_cls_tokens(self, image: Image.Image) -> torch.Tensor:
        """
        Extract exactly 5 CLS token embeddings from one image.

        Processes each of 5 views individually to guarantee exactly 5 CLS tokens.

        Returns:
            cls_tokens: [5, D]  (fp32 for offline storage)
        """
        imgs5 = prepare_five_views(image, size=336)
        device = next(self.model.parameters()).device

        cls_tokens = []
        for single_img in imgs5:
            pv = self.processor.image_processor([single_img], return_tensors="pt")["pixel_values"]
            pv, num_crops = normalize_pixel_values(pv, target_batch=1)

            if pv.ndim == 6:
                B, V, Nc, C, H, W = pv.shape
                pv_flat = pv.view(B * V * Nc, C, H, W).to(device, dtype=torch.float16)
                out = self.model.vision_tower(pv_flat, return_dict=False)
                cls = out[0][:, 0, :].view(B, V, Nc, -1).mean(dim=2)
                proj = self.model.multi_modal_projector(cls)
            elif pv.ndim == 5:
                B, V, C, H, W = pv.shape
                pv_flat = pv.view(B * V, C, H, W).to(device, dtype=torch.float16)
                out = self.model.vision_tower(pv_flat, return_dict=False)
                cls = out[0][:, 0, :].view(B, V, -1)
                proj = self.model.multi_modal_projector(cls)
            else:
                raise ValueError(f"Unexpected pv ndim: {pv.ndim}")

            cls_tokens.append(proj.mean(dim=1).squeeze(0).float().cpu())

        result = torch.stack(cls_tokens, dim=0)
        assert result.shape[0] == 5, f"Expected 5 CLS tokens, got {result.shape[0]}"
        return result


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def discover_categories(data_dir: str) -> list[str]:
    categories = []
    for p in sorted(Path(data_dir).iterdir()):
        if p.is_dir() and (p / "train.jsonl").exists():
            categories.append(p.name)
    return categories


def read_jsonl(jsonl_path: str, candidate_source: str = "st"):
    cand_key = f"candidates_{candidate_source}"
    query_texts = []
    candidate_asins_list = []
    labels = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            candidates = sample.get(cand_key, [])
            if not candidates:
                continue
            gt_items = set(sample.get("gt_items", []))
            if not gt_items:
                continue

            label = -1
            for i, asin in enumerate(candidates):
                if asin in gt_items:
                    label = i
                    break
            if label < 0:
                continue

            query_texts.append(sample["context"])
            candidate_asins_list.append(candidates)
            labels.append(label)

    return query_texts, candidate_asins_list, labels


# ---------------------------------------------------------------------------
# Metadata loading
# ---------------------------------------------------------------------------

def load_all_metadata(metadata_dir: str, categories: list[str]) -> dict:
    """Load item metadata from item2meta_train_*.with_desc.json files."""
    all_meta = {}
    meta_dir = Path(metadata_dir)

    for cat in categories:
        path = meta_dir / f"item2meta_train_{cat}.with_desc.json"
        if not path.exists():
            print(f"  [WARN] metadata not found: {path}")
            continue
        print(f"  Loading {path.name} ...", end=" ")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for asin, item in data.items():
            all_meta[asin] = item
        print(f"{len(data)} items")

    print(f"  Total: {len(all_meta)} items in metadata")
    return all_meta


# ---------------------------------------------------------------------------
# Feature generation for one item (same methods as generate_item_multiview_features.py)
# ---------------------------------------------------------------------------

def generate_item_features(
    asin: str,
    item_meta: dict,
    text_enc: TextEncoder,
    vis_enc: VisualEncoder = None,
    pool: bool = False,
) -> dict:
    """Generate features for one item from its metadata."""
    features = {}

    if pool:
        title = (item_meta.get("title") or "").strip()
        features["title"] = text_enc.encode_pooled(title)

        feat_text = clean_features(item_meta.get("features"))
        features["features"] = text_enc.encode_pooled(feat_text)

        desc_text = clean_description(item_meta.get("description"))
        features["description"] = text_enc.encode_pooled(desc_text)
    else:
        # Title (token-level)
        title = (item_meta.get("title") or "").strip()
        title_result = text_enc.encode_token_level(title)
        features["title_token_embeddings"] = title_result["embeddings"]
        features["title_attention_mask"] = title_result["attention_mask"]

        # Features bullet points (token-level)
        feat_text = clean_features(item_meta.get("features"))
        feat_result = text_enc.encode_token_level(feat_text)
        features["feature_token_embeddings"] = feat_result["embeddings"]
        features["feature_attention_mask"] = feat_result["attention_mask"]

        # Description (token-level)
        desc_text = clean_description(item_meta.get("description"))
        desc_result = text_enc.encode_token_level(desc_text)
        features["description_token_embeddings"] = desc_result["embeddings"]
        features["description_attention_mask"] = desc_result["attention_mask"]

    # Image descriptions (pooled per image)
    img_descs = get_image_descriptions(item_meta)
    if img_descs:
        pooled = text_enc.encode_pooled_batch(img_descs)
        features["image_description_embeddings"] = pooled
    else:
        features["image_description_embeddings"] = torch.zeros(
            0, text_enc.hidden_dim, dtype=torch.float32,
        )

    # Visual CLS tokens via LLaVA
    if vis_enc is not None:
        images_list = item_meta.get("images") or []
        url, _ = pick_image_url(images_list)
        if url:
            try:
                img = to_image(url)
                features["visual_cls_tokens"] = vis_enc.extract_cls_tokens(img)
            except Exception as e:
                print(f"  [WARN] {asin}: visual failed: {e}")
                features["visual_cls_tokens"] = torch.zeros(0, VISUAL_DIM, dtype=torch.float32)
        else:
            features["visual_cls_tokens"] = torch.zeros(0, VISUAL_DIM, dtype=torch.float32)
    else:
        features["visual_cls_tokens"] = torch.zeros(0, VISUAL_DIM, dtype=torch.float32)

    return features


# ---------------------------------------------------------------------------
# Combo assembly
# ---------------------------------------------------------------------------

def assemble_combo(
    query_tensor: torch.Tensor,
    candidate_asins: list[str],
    label: int,
    all_meta: dict,
    text_enc: TextEncoder,
    vis_enc: VisualEncoder = None,
    pool: bool = False,
) -> dict:
    """
    Assemble one combo: query tokens + candidate features.
    If pool=True, text fields are mean-pooled to single vectors (no masks).
    """
    K = len(candidate_asins)

    if pool:
        title_emb = torch.zeros(K, TEXT_DIM)
        feat_emb = torch.zeros(K, TEXT_DIM)
        desc_emb = torch.zeros(K, TEXT_DIM)
    else:
        title_tok = torch.zeros(K, TEXT_SEQ_LEN, TEXT_DIM)
        title_mask = torch.zeros(K, TEXT_SEQ_LEN, dtype=torch.long)
        feat_tok = torch.zeros(K, TEXT_SEQ_LEN, TEXT_DIM)
        feat_mask = torch.zeros(K, TEXT_SEQ_LEN, dtype=torch.long)
        desc_tok = torch.zeros(K, TEXT_SEQ_LEN, TEXT_DIM)
        desc_mask = torch.zeros(K, TEXT_SEQ_LEN, dtype=torch.long)

    img_desc = torch.zeros(K, MAX_IMAGE_DESCS, TEXT_DIM)
    img_desc_mask = torch.zeros(K, MAX_IMAGE_DESCS)

    vis = torch.zeros(K, MAX_VISUAL, VISUAL_DIM)
    vis_mask = torch.zeros(K, MAX_VISUAL)

    for j, asin in enumerate(candidate_asins):
        f = None
        if asin in all_meta:
            f = generate_item_features(asin, all_meta[asin], text_enc, vis_enc, pool=pool)
        if f is None:
            continue

        if pool:
            if "title" in f:
                title_emb[j] = f["title"]
            if "features" in f:
                feat_emb[j] = f["features"]
            if "description" in f:
                desc_emb[j] = f["description"]
        else:
            for field, tok_buf, msk_buf in [
                ("title", title_tok, title_mask),
                ("feature", feat_tok, feat_mask),
                ("description", desc_tok, desc_mask),
            ]:
                ek = f"{field}_token_embeddings"
                mk = f"{field}_attention_mask"
                if ek in f:
                    emb = f[ek]
                    n = min(emb.size(0), TEXT_SEQ_LEN)
                    tok_buf[j, :n] = emb[:n]
                    m = f.get(mk)
                    if m is not None:
                        msk_buf[j, :n] = m[:n]
                    else:
                        msk_buf[j, :n] = 1

        # Image descriptions
        ide = f.get("image_description_embeddings")
        if ide is not None and ide.size(0) > 0:
            n = min(ide.size(0), MAX_IMAGE_DESCS)
            img_desc[j, :n] = ide[:n]
            img_desc_mask[j, :n] = 1.0

        # Visual CLS tokens
        vc = f.get("visual_cls_tokens")
        if vc is not None and vc.size(0) > 0:
            n = min(vc.size(0), MAX_VISUAL)
            vis[j, :n] = vc[:n]
            vis_mask[j, :n] = 1.0

    if pool:
        return {
            "query_tokens": query_tensor,
            "candidate_title": title_emb,
            "candidate_features": feat_emb,
            "candidate_description": desc_emb,
            "candidate_image_desc": img_desc,
            "candidate_image_desc_mask": img_desc_mask,
            "candidate_visual": vis,
            "candidate_visual_mask": vis_mask,
            "label": label,
        }
    else:
        return {
            "query_tokens": query_tensor,
            "candidate_title_token_embeddings": title_tok,
            "candidate_title_attention_mask": title_mask,
            "candidate_feature_token_embeddings": feat_tok,
            "candidate_feature_attention_mask": feat_mask,
            "candidate_description_token_embeddings": desc_tok,
            "candidate_description_attention_mask": desc_mask,
            "candidate_image_description_embeddings": img_desc,
            "candidate_image_desc_mask": img_desc_mask,
            "candidate_visual_cls_tokens": vis,
            "candidate_visual_mask": vis_mask,
            "label": label,
        }


# ---------------------------------------------------------------------------
# Shard helpers
# ---------------------------------------------------------------------------

def stack_combos(combos: list[dict], pool: bool = False) -> dict:
    """Stack N combo dicts into one shard dict of stacked tensors."""
    if pool:
        return {
            "query_tokens": torch.stack([c["query_tokens"] for c in combos]),
            "candidate_title": torch.stack([c["candidate_title"] for c in combos]),
            "candidate_features": torch.stack([c["candidate_features"] for c in combos]),
            "candidate_description": torch.stack([c["candidate_description"] for c in combos]),
            "candidate_image_desc": torch.stack([c["candidate_image_desc"] for c in combos]),
            "candidate_image_desc_mask": torch.stack([c["candidate_image_desc_mask"] for c in combos]),
            "candidate_visual": torch.stack([c["candidate_visual"] for c in combos]),
            "candidate_visual_mask": torch.stack([c["candidate_visual_mask"] for c in combos]),
            "labels": torch.tensor([c["label"] for c in combos], dtype=torch.long),
        }
    return {
        "query_tokens": torch.stack([c["query_tokens"] for c in combos]),
        "candidate_title_token_embeddings": torch.stack([c["candidate_title_token_embeddings"] for c in combos]),
        "candidate_title_attention_mask": torch.stack([c["candidate_title_attention_mask"] for c in combos]),
        "candidate_feature_token_embeddings": torch.stack([c["candidate_feature_token_embeddings"] for c in combos]),
        "candidate_feature_attention_mask": torch.stack([c["candidate_feature_attention_mask"] for c in combos]),
        "candidate_description_token_embeddings": torch.stack([c["candidate_description_token_embeddings"] for c in combos]),
        "candidate_description_attention_mask": torch.stack([c["candidate_description_attention_mask"] for c in combos]),
        "candidate_image_description_embeddings": torch.stack([c["candidate_image_description_embeddings"] for c in combos]),
        "candidate_image_desc_mask": torch.stack([c["candidate_image_desc_mask"] for c in combos]),
        "candidate_visual_cls_tokens": torch.stack([c["candidate_visual_cls_tokens"] for c in combos]),
        "candidate_visual_mask": torch.stack([c["candidate_visual_mask"] for c in combos]),
        "labels": torch.tensor([c["label"] for c in combos], dtype=torch.long),
    }


def flush_shard(buffer: list[dict], output_dir: Path, shard_idx: int, pool: bool = False):
    """Stack and save buffered combos as a shard file."""
    stacked = stack_combos(buffer, pool=pool)
    suffix = "_pooled.pt" if pool else ".pt"
    path = output_dir / f"shard_{shard_idx:04d}{suffix}"
    torch.save(stacked, path)
    print(f"  [SAVE] {path} ({len(buffer)} combos)")


def compute_data_fingerprint(
    query_texts: list[str],
    candidate_asins_list: list[list[str]],
    labels: list[int],
) -> str:
    """Deterministic SHA-256 over all input samples."""
    h = hashlib.sha256()
    for q, asins, lab in zip(query_texts, candidate_asins_list, labels):
        h.update(q.encode("utf-8"))
        h.update("|".join(asins).encode("utf-8"))
        h.update(str(lab).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def write_metadata(output_dir: Path, num_samples: int, shard_size: int,
                   num_shards: int, fingerprint: str):
    meta = {
        "num_samples": num_samples,
        "shard_size": shard_size,
        "num_shards": num_shards,
        "data_fingerprint": fingerprint,
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)


def find_start_shard(output_dir: Path, fingerprint: str, pool: bool = False) -> int:
    """Return the next shard index to write (skips existing shards for resume).

    If the stored fingerprint doesn't match the current input data, returns 0
    to start over.
    """
    if not output_dir.exists():
        return 0

    meta_path = output_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path, "r") as f:
            meta = json.load(f)
        if meta.get("data_fingerprint") != fingerprint:
            print(f"  [FINGERPRINT MISMATCH] input data changed, regenerating from scratch")
            return 0

    suffix = "_pooled.pt" if pool else ".pt"
    idx = 0
    while (output_dir / f"shard_{idx:04d}{suffix}").exists():
        idx += 1
    return idx


# ---------------------------------------------------------------------------
# Batch processing: encode queries + generate candidate features together
# ---------------------------------------------------------------------------

def process_in_batches(
    query_texts: list[str],
    candidate_asins_list: list[list[str]],
    labels: list[int],
    all_meta: dict,
    text_enc: TextEncoder,
    output_dir: Path,
    encode_device: torch.device,
    shard_size: int = 32,
    encode_batch_size: int = 32,
    label: str = "",
    vis_enc: VisualEncoder = None,
    pool: bool = False,
    start_shard_override: int = None,
    end_shard_override: int = None,
    write_meta: bool = True,
):
    """
    Process queries in batches of shard_size:
      1. Encode shard_size queries using GTE-base
      2. For each query's candidates, generate features from metadata
      3. Bundle into combos, stack, save as one shard
      4. Move to next batch

    Skips already-written shards on resume.
    """
    n = len(query_texts)
    if n == 0:
        return

    print(f"  Processing {label}: {n} samples in batches of {shard_size}...")
    output_dir.mkdir(parents=True, exist_ok=True)

    fingerprint = compute_data_fingerprint(query_texts, candidate_asins_list, labels)
    if start_shard_override is not None:
        start_shard = start_shard_override
    else:
        start_shard = find_start_shard(output_dir, fingerprint, pool=pool)
    skip_samples = start_shard * shard_size

    end_sample = n
    if end_shard_override is not None:
        end_sample = min(end_shard_override * shard_size, n)

    if skip_samples >= end_sample:
        print(f"  [SKIP] {label}: shard range [{start_shard}, {end_shard_override}) empty")
        return
    if start_shard > 0:
        print(f"  [RESUME] {label}: skipping {start_shard} shards ({skip_samples}/{n} samples)")

    use_cuda = encode_device.type == "cuda"
    tokenizer = AutoTokenizer.from_pretrained(
        "Alibaba-NLP/gte-base-en-v1.5", trust_remote_code=True,
    )
    query_model = AutoModel.from_pretrained(
        "Alibaba-NLP/gte-base-en-v1.5",
        trust_remote_code=True,
        dtype=torch.float16,
    ).to(encode_device)
    query_model.eval()

    shard_idx = start_shard
    total_saved = skip_samples
    t0 = time.time()

    for batch_start in range(skip_samples, end_sample, shard_size):
        batch_end = min(batch_start + shard_size, n)
        batch_qtexts = query_texts[batch_start:batch_end]
        batch_casins = candidate_asins_list[batch_start:batch_end]
        batch_labels = labels[batch_start:batch_end]
        batch_n = len(batch_qtexts)

        # Step 1: Encode this batch of queries
        encoded = tokenizer(
            batch_qtexts, max_length=TEXT_SEQ_LEN,
            padding="max_length", truncation=True, return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(encode_device)
        attention_mask = encoded["attention_mask"].to(encode_device)

        ctx = (torch.amp.autocast(device_type="cuda", dtype=torch.float16)
               if use_cuda else torch.amp.autocast(device_type="cpu", enabled=False))
        with torch.no_grad(), ctx:
            outputs = query_model(input_ids=input_ids, attention_mask=attention_mask)
        query_tokens = outputs.last_hidden_state.float().cpu()  # [batch_n, 512, 768]

        # Step 2: For each query, assemble combo (generates candidate features)
        combos = []
        for i in range(batch_n):
            combo = assemble_combo(
                query_tokens[i],
                batch_casins[i],
                batch_labels[i],
                all_meta,
                text_enc,
                vis_enc,
                pool=pool,
            )
            combos.append(combo)

        # Step 3: Stack and save
        flush_shard(combos, output_dir, shard_idx, pool=pool)
        total_saved += batch_n
        shard_idx += 1

        del query_tokens, combos

        if batch_end % 320 == 0 or batch_end >= n:
            elapsed = time.time() - t0
            rate = (total_saved - skip_samples) / max(elapsed, 1)
            print(f"    {label}: {batch_end}/{n} combos | "
                  f"{rate:.1f} combos/s")

    total_samples = len(query_texts)
    total_shards = (total_samples + shard_size - 1) // shard_size
    if write_meta:
        write_metadata(output_dir, total_samples, shard_size, total_shards, fingerprint)
    print(f"  {label}: {total_saved} combos -> {shard_idx} shards (total dataset: {total_samples} samples / {total_shards} shards)")

    del query_model, tokenizer
    if use_cuda:
        torch.cuda.empty_cache()
    gc.collect()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate pre-computed combo shards")

    parser.add_argument("--data_dir", type=str, default="../data")
    parser.add_argument("--metadata_dir", type=str, default="../data",
                        help="Dir with item2meta_train_{category}.with_desc.json files")
    parser.add_argument("--output_dir", type=str, default="combos")
    parser.add_argument("--candidate_source", type=str, default="st",
                        choices=["st", "gpt_large"])
    parser.add_argument("--shard_size", type=int, default=32)
    parser.add_argument("--encode_batch_size", type=int, default=32)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--categories", type=str, nargs="+", default=None)
    parser.add_argument("--pool", action="store_true",
                        help="Mean-pool text token embeddings into single vectors (produces *_pooled.pt)")
    parser.add_argument("--splits", type=str, nargs="+", default=["train", "valid", "test"],
                        help="Which splits to generate (default: train valid test)")
    parser.add_argument("--skip_visual", action="store_true",
                        help="Skip LLaVA visual feature generation (text-only, much faster)")
    parser.add_argument("--visual_device", type=str, default=None,
                        help="Device for VisualEncoder (default: same as text device)")
    parser.add_argument("--start_shard", type=int, default=None,
                        help="Explicit start shard index (bypasses resume logic; for parallel runs).")
    parser.add_argument("--end_shard", type=int, default=None,
                        help="Exclusive end shard index. Processes shards [start_shard, end_shard).")
    parser.add_argument("--no_metadata", action="store_true",
                        help="Skip writing metadata.json (set on all but one parallel worker).")
    args = parser.parse_args()

    n_gpus = torch.cuda.device_count()
    device = torch.device("cuda:0" if n_gpus > 0 else "cpu")
    print(f"Device: {device} ({n_gpus} GPUs visible)")

    # Discover categories
    all_categories = discover_categories(args.data_dir)
    if args.categories:
        categories = [c for c in all_categories if c in args.categories]
        missing = set(args.categories) - set(categories)
        if missing:
            print(f"Warning: categories not found: {missing}")
    else:
        categories = all_categories
    print(f"Categories ({len(categories)}): {', '.join(categories)}")

    source_tag = f"candidates_{args.candidate_source}"
    print(f"Candidate source: {source_tag}, shard size: {args.shard_size}")

    # Load item metadata
    print(f"\nLoading item metadata from {args.metadata_dir}/...")
    all_meta = load_all_metadata(args.metadata_dir, categories)

    # Initialize TextEncoder (for candidate text feature generation)
    print(f"\nInitializing TextEncoder on {device}...")
    text_enc = TextEncoder(device=device, max_length=TEXT_SEQ_LEN)

    # Initialize VisualEncoder (for candidate visual feature generation)
    vis_enc = None
    if not args.skip_visual:
        vis_device = torch.device(args.visual_device) if args.visual_device else device
        print(f"\nInitializing VisualEncoder on {vis_device}...")
        vis_enc = VisualEncoder(device=vis_device)

    t_start = time.time()

    # ---- Train split: merge all categories ----
    if "train" in args.splits:
        print("\n=== Train split ===")
        train_qtexts, train_casins, train_labels, train_cats = [], [], [], []
        for cat in categories:
            jsonl_path = Path(args.data_dir) / cat / "train.jsonl"
            if not jsonl_path.exists():
                continue
            qtexts, casins, labs = read_jsonl(str(jsonl_path), args.candidate_source)
            n_cat = len(qtexts)
            if n_cat == 0:
                continue
            train_qtexts.extend(qtexts)
            train_casins.extend(casins)
            train_labels.extend(labs)
            train_cats.extend([cat] * n_cat)
            print(f"  train/{cat}: {n_cat} samples")

        if args.dry_run:
            train_qtexts = train_qtexts[:5]
            train_casins = train_casins[:5]
            train_labels = train_labels[:5]
            train_cats = train_cats[:5]
            print(f"  [DRY RUN] truncated to {len(train_qtexts)} samples")

        train_dir = Path(args.output_dir) / source_tag / "train"
        process_in_batches(
            train_qtexts, train_casins, train_labels,
            all_meta, text_enc,
            train_dir, device, args.shard_size, args.encode_batch_size,
            label="train", vis_enc=vis_enc, pool=args.pool,
            start_shard_override=args.start_shard,
            end_shard_override=args.end_shard,
            write_meta=not args.no_metadata,
        )
    else:
        print("\n=== Train split: SKIPPED (--splits) ===")

    # ---- Valid and Test splits: per-category ----
    for split_name in ("valid", "test"):
        if split_name not in args.splits:
            print(f"\n=== {split_name} split: SKIPPED (--splits) ===")
            continue
        print(f"\n=== {split_name} split ===")
        for cat in categories:
            jsonl_path = Path(args.data_dir) / cat / f"{split_name}.jsonl"
            if not jsonl_path.exists():
                continue
            qtexts, casins, labs = read_jsonl(str(jsonl_path), args.candidate_source)
            if not qtexts:
                continue
            print(f"  {split_name}/{cat}: {len(qtexts)} samples")

            if args.dry_run:
                qtexts = qtexts[:3]
                casins = casins[:3]
                labs = labs[:3]

            split_dir = Path(args.output_dir) / source_tag / split_name / cat
            process_in_batches(
                qtexts, casins, labs,
                all_meta, text_enc,
                split_dir, device, args.shard_size, args.encode_batch_size,
                label=f"{split_name}/{cat}", vis_enc=vis_enc, pool=args.pool,
                start_shard_override=args.start_shard,
                end_shard_override=args.end_shard,
                write_meta=not args.no_metadata,
            )

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()
