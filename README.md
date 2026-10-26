# LightCRS: Lightweight Multi-Domain Conversational Recommendation System

This repository contains the core code for **LightCRS: Lightweight Multi-Domain Conversational Recommendation System**.

LightCRS is a query-conditioned multi-view candidate reranking model for multi-domain conversational recommendation. Given a conversation context (query) and a shortlist of candidate items, the model fuses multi-modal candidate features (title, bullet features, description, image descriptions, visual CLS tokens) via query-conditioned cross-attention, compares candidates with a transformer, and scores them with a lightweight MLP head.

The full pipeline has three stages:

1. **Feature generation** — `generate_item_multiview_features.py` produces per-item multi-view features (text token embeddings + LLaVA visual CLS tokens).
2. **Combo generation** — `generate_combos.py` encodes each query and bundles it with its candidate features into stacked, sharded `*_pooled.pt` shards ready for training.
3. **Consistency fix** — `fix_candidate_consistency.py` ensures the same ASIN has identical feature tensors across every shard and split where it appears.
4. **Training / evaluation** — `train.py` trains the reranker on the pre-computed shards using **cross-entropy loss only** and reports HR@K, NDCG@K, MRR, plus efficiency metrics (params, FLOPs, latency, throughput).


## Dataset

The MM-R2A-25 dataset is not publicly available at this stage. We plan to release the dataset and accompanying documentation following publication to support reproducibility and future research.

Extract it under `./data/` at the repository root.

## End-to-End Usage

### Step 1 — Generate per-item multi-view features (optional standalone use)

Only needed if you want offline per-item features separate from the combo pipeline. `generate_combos.py` already inlines all the same helpers.

```bash
python generate_item_multiview_features.py \
    --input_data_path data/item2meta_train_all_beauty.with_desc.json \
    --output_dir features/all_beauty \
    --shard_size 500
```

Add `--skip_visual` to skip the LLaVA vision tower (text-only, much faster) or `--dry_run` to process 5 items and print shapes.

### Step 2 — Generate combo shards (queries + candidate features bundled)

```bash
# Pooled mode (recommended): text fields are mean-pooled, smallest shards
python generate_combos.py \
    --data_dir ./data \
    --metadata_dir ./data \
    --output_dir ./combos \
    --candidate_source st \
    --pool

# Parallel sharding across N workers (each worker handles a disjoint shard range)
for i in 0 1 2 3; do
    start=$((i * 250))
    end=$(((i + 1) * 250))
    python generate_combos.py --pool \
        --start_shard $start --end_shard $end --no_metadata &
done
wait
```

Output layout:

```
combos/candidates_st/
├── train/ shard_0000_pooled.pt, shard_0001_pooled.pt, ...
├── valid/<category>/ shard_0000_pooled.pt, ...
└── test/<category>/  shard_0000_pooled.pt, ...
```

### Step 3 — Fix candidate feature consistency

The same ASIN appears under many queries and may receive slightly different feature tensors across shards. This step dedupes them so every occurrence shares the first-seen tensor.

```bash
python fix_candidate_consistency.py \
    --data_dir ./data \
    --combos_dir ./combos/candidates_st
```

### Step 4 — Train

```bash
python train.py \
    --combos_dir ./combos \
    --candidate_source st \
    --epochs 20 \
    --batch_size 16 \
    --lr 1e-4 \
    --output_dir ./checkpoints
```
