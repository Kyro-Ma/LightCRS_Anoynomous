"""
Fix candidate feature inconsistency across pooled .pt shards.

Same ASIN can appear as a candidate in multiple queries/shards but get different
feature tensors because generate_combos.py encodes features independently per query.
This script reads source JSONL files to reconstruct the ASIN→tensor mapping, then
ensures every ASIN has identical feature tensors everywhere it appears.

Processes pooled shards in order: train → valid → test.
First occurrence of each ASIN is canonical; all subsequent occurrences are replaced.

Usage:
    python fix_candidate_consistency.py \
        --data_dir ./data \
        --combos_dir ./combos/candidates_st
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SHARD_SIZE = 32
CANDIDATE_KEYS_POOLED = [
    "candidate_title",
    "candidate_features",
    "candidate_description",
    "candidate_image_desc",
    "candidate_image_desc_mask",
    "candidate_visual",
    "candidate_visual_mask",
]
DATA_DIR = Path("./data")
COMBOS_DIR = Path("./combos/candidates_st")


# ---------------------------------------------------------------------------
# JSONL reading — mirrors generate_combos.py:read_jsonl() exactly
# ---------------------------------------------------------------------------
def read_jsonl(jsonl_path: str, candidate_source: str = "st"):
    """Read JSONL, apply same filtering as generate_combos.py lines 408-439."""
    cand_key = f"candidates_{candidate_source}"
    candidate_asins_list = []

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

            candidate_asins_list.append(candidates)

    return candidate_asins_list


def discover_categories(data_dir: Path) -> list[str]:
    categories = []
    for p in sorted(data_dir.iterdir()):
        if p.is_dir() and (p / "train.jsonl").exists():
            categories.append(p.name)
    return categories


# ---------------------------------------------------------------------------
# Build ASIN mapping: for each shard, for each sample, list the 10 ASINs
# ---------------------------------------------------------------------------
def build_train_asin_mapping(categories: list[str]) -> list[list[list[str]]]:
    """
    Returns: list of shards, each shard is a list of samples,
    each sample is a list of 10 ASIN strings.
    Ordered exactly as generate_combos.py processes train (all cats concatenated).
    """
    all_asins = []
    for cat in categories:
        jsonl_path = DATA_DIR / cat / "train.jsonl"
        if not jsonl_path.exists():
            continue
        cat_asins = read_jsonl(str(jsonl_path))
        all_asins.extend(cat_asins)
        print(f"  train/{cat}: {len(cat_asins)} samples")

    # Split into shards
    shards = []
    for i in range(0, len(all_asins), SHARD_SIZE):
        shards.append(all_asins[i : i + SHARD_SIZE])
    return shards


def build_split_asin_mapping(split: str, categories: list[str]) -> dict[str, list[list[list[str]]]]:
    """
    Returns: dict[category] -> list of shards -> list of samples -> list of 10 ASINs.
    """
    result = {}
    for cat in categories:
        jsonl_path = DATA_DIR / cat / f"{split}.jsonl"
        if not jsonl_path.exists():
            continue
        cat_asins = read_jsonl(str(jsonl_path))
        if not cat_asins:
            continue
        shards = []
        for i in range(0, len(cat_asins), SHARD_SIZE):
            shards.append(cat_asins[i : i + SHARD_SIZE])
        result[cat] = shards
        print(f"  {split}/{cat}: {len(cat_asins)} samples -> {len(shards)} shards")
    return result


# ---------------------------------------------------------------------------
# Fix logic
# ---------------------------------------------------------------------------
def get_shard_path_train(shard_idx: int) -> Path:
    return COMBOS_DIR / "train" / f"shard_{shard_idx:04d}_pooled.pt"


def get_shard_path_split(split: str, category: str, shard_idx: int) -> Path:
    return COMBOS_DIR / split / category / f"shard_{shard_idx:04d}_pooled.pt"


def extract_candidate(data: dict, sample_idx: int, cand_idx: int) -> dict:
    """Extract all candidate feature tensors for one candidate slot."""
    features = {}
    for key in CANDIDATE_KEYS_POOLED:
        if key in data:
            features[key] = data[key][sample_idx, cand_idx].clone()
    return features


def inject_candidate(data: dict, sample_idx: int, cand_idx: int, features: dict):
    """Replace candidate feature tensors for one candidate slot."""
    for key, tensor in features.items():
        data[key][sample_idx, cand_idx] = tensor


def process_shard(shard_path: Path, asin_map: list[list[str]], asin_store: dict) -> tuple[int, int]:
    """
    Process one shard file.

    Returns (num_replacements, num_new_asins).
    """
    data = torch.load(shard_path, map_location="cpu", weights_only=True)
    n_samples = data["labels"].shape[0]

    replacements = 0
    new_asins = 0

    for i in range(n_samples):
        if i >= len(asin_map):
            break
        candidates = asin_map[i]
        for j, asin in enumerate(candidates):
            if asin in asin_store:
                inject_candidate(data, i, j, asin_store[asin])
                replacements += 1
            else:
                asin_store[asin] = extract_candidate(data, i, j)
                new_asins += 1

    torch.save(data, shard_path)
    return replacements, new_asins


def main():
    parser = argparse.ArgumentParser(
        description="Fix candidate feature inconsistency across pooled combo shards"
    )
    parser.add_argument(
        "--data_dir", type=str, default=str(DATA_DIR),
        help="Root dir with per-category train/valid/test.jsonl files",
    )
    parser.add_argument(
        "--combos_dir", type=str, default=str(COMBOS_DIR),
        help="Root dir of pooled combo shards (e.g. ./combos/candidates_st)",
    )
    args = parser.parse_args()

    global DATA_DIR, COMBOS_DIR
    DATA_DIR = Path(args.data_dir)
    COMBOS_DIR = Path(args.combos_dir)

    t0 = time.time()

    # No full backup — 1.9 TB data, not enough disk space.
    # The script only overwrites candidate feature tensors in-place.
    # Re-run generate_combos.py if you need to restore originals.

    categories = discover_categories(DATA_DIR)
    print(f"Categories ({len(categories)}): {', '.join(categories)}")

    asin_store: dict[str, dict] = {}
    total_replacements = 0
    total_new = 0

    # --- Train (flat shards) ---
    print("\n=== Train ===")
    train_shards = build_train_asin_mapping(categories)
    print(f"Train: {sum(len(s) for s in train_shards)} samples in {len(train_shards)} shards")

    for shard_idx, asin_map in enumerate(train_shards):
        shard_path = get_shard_path_train(shard_idx)
        if not shard_path.exists():
            print(f"  WARNING: {shard_path} not found, skipping")
            continue
        reps, news = process_shard(shard_path, asin_map, asin_store)
        total_replacements += reps
        total_new += news
        if (shard_idx + 1) % 200 == 0:
            elapsed = time.time() - t0
            print(f"  Train: {shard_idx + 1}/{len(train_shards)} shards | "
                  f"{len(asin_store)} unique ASINs | "
                  f"{elapsed:.0f}s elapsed")

    # --- Valid (per-category) ---
    print("\n=== Valid ===")
    valid_mapping = build_split_asin_mapping("valid", categories)
    for cat in sorted(valid_mapping.keys()):
        for shard_idx, asin_map in enumerate(valid_mapping[cat]):
            shard_path = get_shard_path_split("valid", cat, shard_idx)
            if not shard_path.exists():
                print(f"  WARNING: {shard_path} not found, skipping")
                continue
            reps, news = process_shard(shard_path, asin_map, asin_store)
            total_replacements += reps
            total_new += news
        print(f"  valid/{cat}: done")

    # --- Test (per-category) ---
    print("\n=== Test ===")
    test_mapping = build_split_asin_mapping("test", categories)
    for cat in sorted(test_mapping.keys()):
        for shard_idx, asin_map in enumerate(test_mapping[cat]):
            shard_path = get_shard_path_split("test", cat, shard_idx)
            if not shard_path.exists():
                print(f"  WARNING: {shard_path} not found, skipping")
                continue
            reps, news = process_shard(shard_path, asin_map, asin_store)
            total_replacements += reps
            total_new += news
        print(f"  test/{cat}: done")

    elapsed = time.time() - t0
    print(f"\n{'='*50}")
    print(f"Done in {elapsed:.1f}s")
    print(f"Unique ASINs: {len(asin_store)}")
    print(f"Replacements: {total_replacements}")
    print(f"New (canonical): {total_new}")


if __name__ == "__main__":
    main()
