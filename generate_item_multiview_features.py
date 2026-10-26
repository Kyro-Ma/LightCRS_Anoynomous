"""
Offline item multi-view feature generation.

Generates per-item features from item2meta_train_*.with_desc.json:
  A. Text-side (Alibaba-NLP/gte-base-en-v1.5):
     - title:              token-level embeddings + attention_mask
     - features:           token-level embeddings + attention_mask (joined)
     - description:        token-level embeddings + attention_mask (joined)
     - image_descriptions: pooled embedding per image description
  B. Visual-side (LLaVA v1.6 vision tower + projector, reused from step2):
     - 5 CLS token embeddings from primary image

GPU constraint: uses ONLY GPUs 1, 2, 3.
  - Text encoder on cuda:0 (mapped to physical GPU 1)
  - Visual pipeline on cuda:1 (mapped to physical GPU 2)

Usage:
    # Dry-run: process 5 items, print shapes
    python generate_item_multiview_features.py --dry_run

    # Full run
    python generate_item_multiview_features.py \
        --input_data_path data/item2meta_train_all_beauty.with_desc.json \
        --output_dir features/all_beauty \
        --shard_size 500

    # Text-only (skip heavy LLaVA model)
    python generate_item_multiview_features.py --dry_run --skip_visual

Output: sharded .pt files in output_dir/
    shard_0000.pt  ->  {ASIN: {field: tensor, ...}, ...}
"""

import argparse
import io
import json
import math
import os
import sys
import time
import concurrent.futures
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
import torch
import torch.nn.functional as F
from PIL import Image

# ---- GPU constraint: default to all GPUs; override via CUDA_VISIBLE_DEVICES env ----
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ---- Model cache: load from ./.cache/huggingface/hub/ ----
os.environ["HF_HOME"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache", "huggingface")
os.environ["HF_HUB_OFFLINE"] = "1"

from transformers import AutoTokenizer, AutoModel, AutoProcessor

# ---------------------------------------------------------------------------
# Reused from src/scripts/step2_add_llava_descriptions.py
# (copied inline to avoid path dependency issues)
# ---------------------------------------------------------------------------

IMAGE_TOKENS = [
    "<ItemImageEmb1>", "<ItemImageEmb2>", "<ItemImageEmb3>",
    "<ItemImageEmb4>", "<ItemImageEmb5>",
]


def to_image(path_or_url: str) -> Image.Image:
    """Load RGB image from local path or URL.  [REUSED from step2]"""
    if isinstance(path_or_url, str) and path_or_url.startswith("http"):
        r = requests.get(path_or_url, timeout=30)
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert("RGB")
    return Image.open(path_or_url).convert("RGB")


def prepare_five_views(image: Image.Image, size: int = 336) -> List[Image.Image]:
    """
    Generate 5 views: resize to 336x336, replicate 5 times.
    [REUSED from step2 — exact same function]
    """
    img = image.resize((size, size))
    return [img, img, img, img, img]


def normalize_pixel_values(
    pixel_values: torch.Tensor, target_batch: int = 1,
) -> Tuple[torch.Tensor, Optional[int]]:
    """
    Normalize pixel_values to (B, V, C, H, W).
    [REUSED from step2 — exact same function]
    
    符号	含义	一句话理解
    B	batch size	一次处理多少个样本（商品数）
    V	views	每个样本有多少张图（或复制的视角）
    Nc	crops	每张图被切成多少个小块（可选）
    C	channels	图像颜色通道（RGB=3）
    H, W	height, width	图像尺寸（高×宽）
    """
    num_crops = None
    if pixel_values.ndim == 4:
        # (V, C, H, W) -> (1, V, C, H, W)
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
    """
    Run vision tower + projector -> (B, V, hidden).
    [REUSED from step2 — exact same logic, line-by-line]

    Pipeline:
        vision_tower(pixel_values) -> last_hidden_state
        CLS = last_hidden[:, 0, :]   (first token)
        multi_modal_projector(CLS) -> projected features
    """
    device = next(model.parameters()).device

    if pixel_values.ndim == 6:
        # (B, V, Nc, C, H, W) — multi-crop case
        B, V, Nc, C, H, W = pixel_values.shape
        pv = pixel_values.view(B * V * Nc, C, H, W).to(device, dtype=torch.float16)
        with torch.no_grad():
            vision_outputs = model.vision_tower(pv, return_dict=False)
        last_hidden = vision_outputs[0]                          # (B*V*Nc, seq_len, hidden)
        cls = last_hidden[:, 0, :].view(B, V, Nc, -1)           # (B, V, Nc, hidden)
        cls = cls.mean(dim=2)                                    # (B, V, hidden)
        cls_proj = model.multi_modal_projector(cls)              # (B, V, hidden)
        return cls_proj

    elif pixel_values.ndim == 5:
        # (B, V, C, H, W) — standard case
        B, V, C, H, W = pixel_values.shape
        pv = pixel_values.view(B * V, C, H, W).to(device, dtype=torch.float16)
        with torch.no_grad():
            vision_outputs = model.vision_tower(pv, return_dict=False)
        last_hidden = vision_outputs[0]                          # (B*V, seq_len, hidden)
        cls = last_hidden[:, 0, :].view(B, V, -1)               # (B, V, hidden)
        cls_proj = model.multi_modal_projector(cls)              # (B, V, hidden)
        return cls_proj

    else:
        raise ValueError(f"Unexpected pixel_values ndim: {pixel_values.ndim}")


# ---------------------------------------------------------------------------
# Image selection: local first, URL fallback
# Same convention as step2_add_llava_descriptions.py lines 284-306
# ---------------------------------------------------------------------------

def find_local_image(asin: str, category: str, images_dir: Path) -> Optional[str]:
    """
    Try to find a local image file for this item.
    Convention from step2 lines 287-289:
        fp = base / cate_name / f"{asin}_{k}.jpg"
    We use k=0 (first/primary image).
    """
    fp = images_dir / category / f"{asin}_0.jpg"
    if fp.exists():
        return str(fp)
    return None


def pick_image_url(images_list: List[dict]) -> Tuple[Optional[str], int]:
    """
    Select the best image URL following resolution priority.

    Iterates images in order (MAIN first). For each image, tries:
        hi_res -> large -> thumb
    Returns the first image that has any valid URL.

    Convention from step2_add_llava_descriptions.py line 300:
        url = im.get("hi_res") or im.get("large") or im.get("thumb")

    Returns:
        (url, index)  where index is the position in images_list
        (None, -1)    if no image has a valid URL
    """
    if not images_list:
        return None, -1
    for idx, img in enumerate(images_list):
        url = img.get("hi_res") or img.get("large") or img.get("thumb")
        if url:
            return url, idx
    return None, -1


# ---------------------------------------------------------------------------
# Text encoding: Alibaba-NLP/gte-base-en-v1.5
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

        # hidden_size from config: 768 for gte-base
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
            # Missing text -> single zero embedding with mask=0
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
        input_ids = encoded["input_ids"].to(self.device)           # [1, T]
        attention_mask = encoded["attention_mask"].to(self.device)  # [1, T]

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            token_embs = outputs.last_hidden_state.squeeze(0).float()  # [T, D]

        return {
            "embeddings": token_embs.cpu(),                          # [T, D]
            "attention_mask": attention_mask.squeeze(0).cpu(),       # [T]
        }

    @torch.no_grad()
    def encode_pooled(self, text: str) -> torch.Tensor:
        """
        Encode text -> single pooled embedding (mean pool over valid tokens).

        Returns:
            [D] pooled embedding (fp32)
        """
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
            token_embs = outputs.last_hidden_state  # [1, T, D]

        # Mean pool over valid tokens
        mask_exp = attention_mask.unsqueeze(-1).float()       # [1, T, 1]
        pooled = (token_embs.float() * mask_exp).sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1)
        return pooled.squeeze(0).cpu()  # [D]

    @torch.no_grad()
    def encode_pooled_batch(self, texts: List[str]) -> torch.Tensor:
        """
        Encode a batch of texts -> pooled embeddings.

        Returns:
            [B, D] pooled embeddings (fp32)
        """
        # Replace empty texts with a space to avoid tokenizer errors
        valid = [t if t and t.strip() else " " for t in texts]

        encoded = self.tokenizer(
            valid,
            max_length=self.max_length,
            padding="longest",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(self.device)           # [B, T]
        attention_mask = encoded["attention_mask"].to(self.device)  # [B, T]

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            token_embs = outputs.last_hidden_state  # [B, T, D]

        # Mean pool over valid tokens
        mask_exp = attention_mask.unsqueeze(-1).float()  # [B, T, 1]
        pooled = (token_embs.float() * mask_exp).sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1)
        return pooled.cpu()  # [B, D]


# ---------------------------------------------------------------------------
# Visual feature extraction: LLaVA v1.6 (reused pattern from step2)
# ---------------------------------------------------------------------------

class VisualEncoder:
    """
    Wraps LLaVA v1.6 for 5 CLS token extraction.

    Reuses the EXACT same pipeline from step2_add_llava_descriptions.py:
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

        # Same special token handling as step2
        self._ensure_special_tokens()
        self.model.eval()

        # Determine hidden dim from projector
        # For LLaVA v1.6 Mistral 7B: projector output = 4096
        proj = self.model.multi_modal_projector
        if hasattr(proj, 'proj') and hasattr(proj.proj, 'out_features'):
            self.hidden_dim = proj.proj.out_features
        elif hasattr(proj, 'linear_2') and hasattr(proj.linear_2, 'out_features'):
            self.hidden_dim = proj.linear_2.out_features
        else:
            self.hidden_dim = self.model.config.text_config.hidden_size
        print(f"[VisualEncoder] Ready. hidden_dim={self.hidden_dim}, device={device}")

    def _ensure_special_tokens(self):
        """Same as ensure_tokens_and_resize in step2."""
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

        Pipeline (same components as step2, processed per-view to guarantee 5):
            For each of 5 views:
              1. processor.image_processor([view]) -> pixel_values
              2. normalize_pixel_values -> (1, 1, C, H, W)
              3. vision_tower(pixel_values) -> CLS token
              4. multi_modal_projector(CLS) -> projected feature
            Stack 5 results -> [5, D]

        Processing individually guarantees exactly 5 CLS tokens regardless of
        how LLaVA-Next's any-res processor batches multiple images.

        Args:
            image: PIL Image (RGB)

        Returns:
            cls_tokens: [5, D]  (fp32 for offline storage)
        """
        imgs5 = prepare_five_views(image, size=336)
        device = next(self.model.parameters()).device

        cls_tokens = []
        for single_img in imgs5:
            # Process one view at a time
            pv = self.processor.image_processor([single_img], return_tensors="pt")["pixel_values"]
            pv, num_crops = normalize_pixel_values(pv, target_batch=1)

            # vision tower + projector (same as step2)
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

            # proj shape: (1, V, D) — average across V (anyres crops) then squeeze
            cls_tokens.append(proj.mean(dim=1).squeeze(0).float().cpu())  # [D]

        result = torch.stack(cls_tokens, dim=0)  # [5, D]
        assert result.shape[0] == 5, f"Expected 5 CLS tokens, got {result.shape[0]}"
        return result


# ---------------------------------------------------------------------------
# Data cleaning helpers
# ---------------------------------------------------------------------------

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
    """
    Extract ordered image descriptions from image_descriptions_llava_cleaned.
    Returns list of description strings, ordered by image index.
    """
    desc_map = item.get("image_descriptions_llava_cleaned") or {}
    if not desc_map:
        return []

    # Sort by numeric index in key (e.g., "B0058L24O6_3.jpg" -> 3)
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
# Feature generation for a single item
# ---------------------------------------------------------------------------

def process_text_features(
    asin: str,
    item: dict,
    text_enc: TextEncoder,
) -> dict:
    """
    Text-only feature extraction (runs on cuda:0, fast).
    Returns partial feature dict (no visual).
    """
    features = {}

    # --- A1: title (token-level) ---
    title = (item.get("title") or "").strip()
    title_result = text_enc.encode_token_level(title)
    features["title_token_embeddings"] = title_result["embeddings"]     # [T, D]
    features["title_attention_mask"] = title_result["attention_mask"]   # [T]

    # --- A2: features bullet points (token-level, joined with newlines) ---
    feat_text = clean_features(item.get("features"))
    feat_result = text_enc.encode_token_level(feat_text)
    features["feature_token_embeddings"] = feat_result["embeddings"]    # [T, D]
    features["feature_attention_mask"] = feat_result["attention_mask"]  # [T]

    # --- A3: description (token-level, joined paragraphs) ---
    desc_text = clean_description(item.get("description"))
    desc_result = text_enc.encode_token_level(desc_text)
    features["description_token_embeddings"] = desc_result["embeddings"]    # [T, D]
    features["description_attention_mask"] = desc_result["attention_mask"]  # [T]

    # --- A4: image descriptions (pooled embedding per image description) ---
    img_descs = get_image_descriptions(item)
    if img_descs:
        pooled = text_enc.encode_pooled_batch(img_descs)  # [num_imgs, D]
        features["image_description_embeddings"] = pooled
    else:
        features["image_description_embeddings"] = torch.zeros(
            0, text_enc.hidden_dim, dtype=torch.float32,
        )

    features["item_id"] = asin
    return features


def process_visual_features(
    asin: str,
    item: dict,
    vis_enc: VisualEncoder,
    category: str = "",
    images_dir: Path = None,
) -> torch.Tensor:
    """
    Visual CLS extraction for one item (runs on one visual GPU).
    Uses URL with hi_res → large → thumb priority to guarantee best resolution.
    Returns [5, D] tensor, or [0, D] on failure.
    """
    images_list = item.get("images") or []
    url, _ = pick_image_url(images_list)
    if not url:
        return torch.zeros(0, vis_enc.hidden_dim, dtype=torch.float32)
    try:
        img = to_image(url)
        return vis_enc.extract_cls_tokens(img)  # [5, D]
    except Exception as e:
        print(f"  [WARN] {asin}: visual extraction failed: {e}")
        return torch.zeros(0, vis_enc.hidden_dim, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Sharding & saving
# ---------------------------------------------------------------------------

def save_shard(shard_data: dict, output_dir: Path, shard_idx: int):
    """Save a shard dict to a .pt file."""
    path = output_dir / f"shard_{shard_idx:04d}.pt"
    torch.save(shard_data, path)
    print(f"  [SAVE] {path} ({len(shard_data)} items)")


def load_existing_shards(output_dir: Path) -> set:
    """Return set of ASINs already processed in existing shards."""
    existing = set()
    if not output_dir.exists():
        return existing
    for pt_file in sorted(output_dir.glob("shard_*.pt")):
        try:
            shard = torch.load(pt_file, map_location="cpu", weights_only=False)
            existing.update(shard.keys())
        except Exception:
            print(f"  [WARN] Could not load {pt_file}, skipping")
    return existing


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate offline item multi-view features")

    # I/O
    parser.add_argument("--input_data_path", type=str,
                        default="data/item2meta_train_all_beauty.with_desc.json",
                        help="Path to item2meta JSON file")
    parser.add_argument("--output_dir", type=str,
                        default="features/all_beauty",
                        help="Output directory for sharded .pt files")
    parser.add_argument("--images_dir", type=str, default="data/train_images",
                        help="Root directory for local images (subdirs per category)")
    parser.add_argument("--category", type=str, default=None,
                        help="Category name (auto-derived from input filename if not set)")
    parser.add_argument("--shard_size", type=int, default=500,
                        help="Number of items per shard file")

    # Mode
    parser.add_argument("--dry_run", action="store_true",
                        help="Process only 5 items, print details, then exit")
    parser.add_argument("--skip_visual", action="store_true",
                        help="Skip visual CLS extraction (text-only, much faster)")

    # Models
    parser.add_argument("--text_model", type=str,
                        default="Alibaba-NLP/gte-base-en-v1.5")
    parser.add_argument("--visual_model", type=str,
                        default="llava-hf/llava-v1.6-mistral-7b-hf")
    parser.add_argument("--text_max_length", type=int, default=512)
    parser.add_argument("--text_device", type=str, default="cuda:0",
                        help="Device for text encoder (GPU 1)")
    parser.add_argument("--visual_devices", type=str, default="cuda:0",
                        help="Comma-separated devices for visual encoders")

    args = parser.parse_args()

    # ---- Derive category from input filename (same as step2 line 287) ----
    # e.g. "item2meta_train_all_beauty.with_desc.json" -> "all_beauty"
    if args.category:
        category = args.category
    else:
        stem = Path(args.input_data_path).stem  # "item2meta_train_all_beauty.with_desc"
        category = stem.replace("item2meta_train_", "").replace(".with_desc", "")
    images_dir = Path(args.images_dir)
    print(f"Category: {category}")
    print(f"Images dir: {images_dir}")
    print(f"Local image path pattern: {images_dir}/{category}/{{ASIN}}_0.jpg")

    # ---- Device setup ----
    n_gpus = torch.cuda.device_count()
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"HF_HOME={os.environ.get('HF_HOME')}")
    print(f"Visible GPUs: {n_gpus}")
    for i in range(n_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    text_device = torch.device(args.text_device if n_gpus > 0 else "cpu")
    visual_device_names = args.visual_devices.split(",")
    visual_devices = [torch.device(d.strip()) for d in visual_device_names]
    print(f"Text encoder:  {text_device}")
    print(f"Visual encoders: {visual_devices}")

    # ---- Load models ----
    text_enc = TextEncoder(device=text_device, max_length=args.text_max_length)

    # Determine if text and visual share the same device
    shared_device = (text_device in visual_devices) if not args.skip_visual else False

    # ---- Load input data ----
    input_path = Path(args.input_data_path)
    if not input_path.exists():
        print(f"ERROR: input file not found: {input_path}")
        sys.exit(1)

    print(f"\nLoading {input_path} ...")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_asins = list(data.keys())
    print(f"Total items: {len(all_asins)}")

    # ---- Resume support ----
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    existing_asins = load_existing_shards(output_dir)
    if existing_asins:
        print(f"Found {len(existing_asins)} already-processed items. Resuming...")

    # Filter out already-processed items
    to_process = [(asin, data[asin]) for asin in all_asins if asin not in existing_asins]
    print(f"Items to process: {len(to_process)}")

    if args.dry_run:
        to_process = to_process[:5]
        print(f"[DRY RUN] Processing only {len(to_process)} items")

    # ---- Process items ----
    # Strategy depends on whether text and visual share a single GPU:
    #   shared_device: Phase 1 = text only, free text enc, Phase 2 = visual only
    #   separate GPUs: text + visual in parallel (original behavior)
    shard_data = {}
    shard_idx = len(list(output_dir.glob("shard_*.pt")))
    total_processed = 0
    t_start = time.time()

    if args.skip_visual:
        # ---- Text-only mode (no threading needed) ----
        for i, (asin, item) in enumerate(to_process):
            t0 = time.time()
            try:
                feat = process_text_features(asin, item, text_enc)
                feat["visual_cls_tokens"] = torch.zeros(0, 4096, dtype=torch.float32)
            except Exception as e:
                print(f"  [ERROR] {asin}: {e}")
                continue
            shard_data[asin] = feat
            total_processed += 1
            elapsed = time.time() - t0
            if total_processed <= 3 or args.dry_run:
                print(f"\n--- [{i+1}/{len(to_process)}] {asin} ({elapsed:.1f}s) ---")
                for k, v in feat.items():
                    if isinstance(v, torch.Tensor):
                        print(f"  {k}: shape={list(v.shape)}, dtype={v.dtype}")
                    else:
                        print(f"  {k}: {v}")
            if len(shard_data) >= args.shard_size:
                save_shard(shard_data, output_dir, shard_idx)
                shard_idx += 1
                shard_data = {}
            if (i + 1) % 50 == 0:
                avg = (time.time() - t_start) / max(total_processed, 1)
                eta = avg * (len(to_process) - i - 1) / 60
                print(f"  Progress: {total_processed}/{len(to_process)} | {avg:.1f}s/item | ETA: {eta:.0f}min")

    elif shared_device:
        # ---- Single-GPU mode: Phase 1 (text), free, Phase 2 (visual) ----
        import gc

        # Phase 1: text features
        print(f"\n=== Phase 1: Text features on {text_device} ===")
        for i, (asin, item) in enumerate(to_process):
            try:
                feat = process_text_features(asin, item, text_enc)
                shard_data[asin] = feat
                total_processed += 1
            except Exception as e:
                print(f"  [ERROR] {asin}: text failed: {e}")
                continue
            if len(shard_data) >= args.shard_size:
                save_shard(shard_data, output_dir, shard_idx)
                shard_idx += 1
                shard_data = {}
            if (i + 1) % 50 == 0:
                avg = (time.time() - t_start) / max(i + 1, 1)
                eta = avg * (len(to_process) - i - 1) / 60
                print(f"  Text: {i+1}/{len(to_process)} | {avg:.1f}s/item | ETA: {eta:.0f}min")
        if shard_data:
            save_shard(shard_data, output_dir, shard_idx)
            shard_idx += 1
            shard_data = {}

        # Free text encoder, load visual encoder
        print(f"\nFreeing text encoder, loading visual encoder...")
        del text_enc
        gc.collect()
        torch.cuda.empty_cache()

        vis_enc = VisualEncoder(device=text_device, model_name=args.visual_model)

        # Phase 2: visual features (update existing shards)
        print(f"\n=== Phase 2: Visual features on {text_device} ===")
        shard_files = sorted(output_dir.glob("shard_*.pt"))
        vis_done = 0
        vis_start = time.time()
        for sf in shard_files:
            shard = torch.load(sf, map_location="cpu", weights_only=False)
            updated = False
            for asin, feat in shard.items():
                # Skip if already has visual features
                if feat.get("visual_cls_tokens") is not None and feat["visual_cls_tokens"].numel() > 0:
                    continue
                # Find the item in the original data
                item = data.get(asin)
                if item is None:
                    continue
                try:
                    feat["visual_cls_tokens"] = process_visual_features(
                        asin, item, vis_enc, category, images_dir,
                    )
                    updated = True
                except Exception as e:
                    print(f"  [WARN] {asin}: visual failed: {e}")
                    feat["visual_cls_tokens"] = torch.zeros(0, vis_enc.hidden_dim, dtype=torch.float32)
                    updated = True
                vis_done += 1
            if updated:
                torch.save(shard, sf)
            if (vis_done) % 50 == 0:
                avg = (time.time() - vis_start) / max(vis_done, 1)
                print(f"  Visual: {vis_done} items | {avg:.1f}s/item")

        del vis_enc
        gc.collect()
        torch.cuda.empty_cache()

    else:
        # ---- Multi-GPU mode: text (main thread) + visual (parallel on separate GPUs) ----
        vis_encs: List[VisualEncoder] = []
        for dev in visual_devices:
            enc = VisualEncoder(device=dev, model_name=args.visual_model)
            vis_encs.append(enc)

        n_vis = len(vis_encs)
        print(f"\nUsing {n_vis} visual encoder(s) in parallel with ThreadPoolExecutor")

        with concurrent.futures.ThreadPoolExecutor(max_workers=n_vis) as executor:
            # Submit visual work in advance, collect as we go
            pending_visual = {}  # future -> asin

            # Helper: submit visual work for one item
            def submit_visual(asin, item):
                vis_idx = len(pending_visual) % n_vis  # round-robin
                future = executor.submit(
                    process_visual_features, asin, item, vis_encs[vis_idx],
                    category, images_dir,
                )
                pending_visual[future] = asin

            for i, (asin, item) in enumerate(to_process):
                t0 = time.time()

                # 1) Submit visual work to thread pool (runs on cuda:1 or cuda:2)
                submit_visual(asin, item)

                # 2) Process text on main thread (cuda:0) while visual runs
                try:
                    feat = process_text_features(asin, item, text_enc)
                except Exception as e:
                    print(f"  [ERROR] {asin}: text failed: {e}")
                    # Still need to collect the visual future to avoid leaking
                    continue

                # 3) Collect visual result (blocks until this item's visual is done)
                #    Find the future for this asin
                this_future = None
                for f, a in list(pending_visual.items()):
                    if a == asin:
                        this_future = f
                        break

                if this_future is not None:
                    try:
                        visual_cls = this_future.result()
                    except Exception as e:
                        print(f"  [WARN] {asin}: visual future failed: {e}")
                        visual_cls = torch.zeros(0, vis_encs[0].hidden_dim, dtype=torch.float32)
                    del pending_visual[this_future]

                feat["visual_cls_tokens"] = visual_cls
                shard_data[asin] = feat
                total_processed += 1

                elapsed = time.time() - t0
                if total_processed <= 3 or args.dry_run:
                    print(f"\n--- [{i+1}/{len(to_process)}] {asin} ({elapsed:.1f}s) ---")
                    for k, v in feat.items():
                        if isinstance(v, torch.Tensor):
                            print(f"  {k}: shape={list(v.shape)}, dtype={v.dtype}")
                        else:
                            print(f"  {k}: {v}")

                # Save shard when full
                if len(shard_data) >= args.shard_size:
                    save_shard(shard_data, output_dir, shard_idx)
                    shard_idx += 1
                    shard_data = {}

                # Periodic progress
                if (i + 1) % 50 == 0:
                    avg = (time.time() - t_start) / max(total_processed, 1)
                    eta = avg * (len(to_process) - i - 1) / 60
                    print(f"  Progress: {total_processed}/{len(to_process)} | "
                          f"{avg:.1f}s/item | ETA: {eta:.0f}min")

    # Save remaining items
    if shard_data:
        save_shard(shard_data, output_dir, shard_idx)

    # ---- Summary ----
    total_time = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Done. Processed {total_processed} items in {total_time/60:.1f} min")
    print(f"Output: {output_dir}")
    print(f"Shard files: {len(list(output_dir.glob('shard_*.pt')))}")

    # Print sample output structure for dry-run
    if args.dry_run and total_processed > 0:
        shard_files = sorted(output_dir.glob("shard_*.pt"))
        if shard_files:
            sample_shard = torch.load(
                shard_files[-1], map_location="cpu", weights_only=False,
            )
            sample_asin = list(sample_shard.keys())[0]
            sample = sample_shard[sample_asin]
            print(f"\n--- Sample output structure for {sample_asin} ---")
            for k, v in sample.items():
                if isinstance(v, torch.Tensor):
                    print(f"  {k}: shape={list(v.shape)}, dtype={v.dtype}")
                else:
                    print(f"  {k}: {v}")
            print(f"--- End sample ---")

    print(f"{'='*60}")


if __name__ == "__main__":
    main()
