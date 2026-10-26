"""
Training script for the query-conditioned multi-view candidate reranking model.

Requires pre-computed combo shards from generate_combos.py.

Usage:
    # Quick dry-run with dummy data
    python train.py --dry_run

    # Full training (requires combo shards)
    python train.py --epochs 20 --batch_size 32 --lr 1e-4

    # Eval-only on a saved checkpoint
    python train.py --eval_only --checkpoint ./checkpoints/best_model.pt
"""

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

# ---- Ensure script's directory is on sys.path so sibling packages resolve ----
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ---- GPU constraint: default to GPU 0; override via CUDA_VISIBLE_DEVICES env ----
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ---- HuggingFace cache: use local models, skip slow mirror ----
os.environ["HF_HOME"] = os.path.join(_PROJECT_ROOT, ".cache", "huggingface")
os.environ["HF_HUB_OFFLINE"] = "1"

# ---- Project imports ----
from utils.dataset import RerankingDataset, create_dummy_data
from utils.module import RerankingModel
from utils.metrics import compute_metrics
from utils.combo_dataset import PooledComboDataset


def collate_fn(batch: list[dict]) -> dict:
    """Stack list-of-dicts into dict-of-tensors."""
    keys = batch[0].keys()
    return {k: torch.stack([s[k] for s in batch], dim=0) for k in keys}


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    grad_accum_steps: int = 1,
    max_steps: int = 0,
) -> dict:
    model.train()
    total_loss = 0.0
    total_samples = 0

    optimizer.zero_grad()
    for step, batch in enumerate(dataloader):
        query = batch["query_tokens"].to(device, non_blocking=True)
        title = batch["candidate_title"].to(device, non_blocking=True)
        features = batch["candidate_features"].to(device, non_blocking=True)
        desc = batch["candidate_description"].to(device, non_blocking=True)
        img_desc = batch["candidate_image_desc"].to(device, non_blocking=True)
        img_desc_mask = batch["candidate_image_desc_mask"].to(device, non_blocking=True)
        vis = batch["candidate_visual"].to(device, non_blocking=True)
        vis_mask = batch["candidate_visual_mask"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, _ = model(query, title, features, desc, img_desc, img_desc_mask, vis, vis_mask)
            loss = loss_fn(logits, labels)

        scaled_loss = loss / grad_accum_steps
        scaler.scale(scaled_loss).backward()

        if (step + 1) % grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * query.size(0)
        total_samples += query.size(0)

        if step % 10 == 0:
            print(f"  step {step}/{len(dataloader)}, loss={loss.item():.4f}")

        if max_steps > 0 and (step + 1) >= max_steps:
            print(f"  [max_steps={max_steps}] stopping epoch early at step {step + 1}")
            break
    n = max(total_samples, 1)
    return {"loss": total_loss / n, "ce": total_loss / n}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    dump_file=None,
    category: str = "",
) -> dict:
    model.eval()
    all_logits, all_labels = [], []
    total_loss = 0.0
    total_samples = 0
    t_eval_start = time.time()

    for batch in dataloader:
        query = batch["query_tokens"].to(device, non_blocking=True)
        title = batch["candidate_title"].to(device, non_blocking=True)
        features = batch["candidate_features"].to(device, non_blocking=True)
        desc = batch["candidate_description"].to(device, non_blocking=True)
        img_desc = batch["candidate_image_desc"].to(device, non_blocking=True)
        img_desc_mask = batch["candidate_image_desc_mask"].to(device, non_blocking=True)
        vis = batch["candidate_visual"].to(device, non_blocking=True)
        vis_mask = batch["candidate_visual_mask"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, _ = model(query, title, features, desc, img_desc, img_desc_mask, vis, vis_mask)
            loss = loss_fn(logits, labels)

        total_loss += loss.item() * query.size(0)
        total_samples += query.size(0)
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    t_eval_end = time.time()
    eval_time = t_eval_end - t_eval_start
    metrics = compute_metrics(all_logits, all_labels, ks=(1, 5, 10))

    # Optional per-sample dump (only when dump_file is provided)
    if dump_file is not None:
        n = all_logits.size(0)
        for i in range(n):
            logit_row = all_logits[i].tolist()
            gold_idx = int(all_labels[i].item())
            pred_idx = max(range(len(logit_row)), key=lambda j: logit_row[j])
            sorted_idx = sorted(range(len(logit_row)), key=lambda j: -logit_row[j])
            rank_of_gold = sorted_idx.index(gold_idx) + 1
            dump_file.write(json.dumps({
                "category": category,
                "sample_idx": i,
                "gold_idx": gold_idx,
                "pred_idx": pred_idx,
                "correct": pred_idx == gold_idx,
                "rank_of_gold": rank_of_gold,
                "logits": [round(x, 4) for x in logit_row],
            }, ensure_ascii=False) + "\n")
        dump_file.flush()

    metrics["loss"] = total_loss / max(total_samples, 1)
    metrics["eval_time_s"] = eval_time
    metrics["latency_ms_per_query"] = eval_time / max(total_samples, 1) * 1000
    metrics["throughput_qps"] = total_samples / max(eval_time, 1e-9)
    return metrics


@torch.no_grad()
def estimate_flops_per_query(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> float | None:
    """Estimate MFLOPs per query analytically from model architecture.

    For pooled mode (default): QueryEncoder + CandidateFusion + ComparisonTransformer + ScoringHead.
    Counts multiply-add operations for linear layers, attention, layer norms, and FFNs.
    """
    raw = model.module if isinstance(model, nn.DataParallel) else model

    # Default architecture params
    D = raw.hidden_dim          # 256
    K = raw.num_candidates      # 10
    Tq = 512                    # query seq len (pooled mode: reduced to 1 by QueryEncoder)
    Tc = 3 + 15 + 5             # candidate tokens: title + feat + desc + 15 img_desc + 5 visual
    n_heads_ca = 4              # cross-attn heads
    n_layers_ca = 2             # cross-attn layers
    n_layers_comp = 2           # comparison transformer layers
    Dq = 768                    # query input dim
    D_visual = 4096             # visual input dim
    D_text = 768                # text input dim

    def lin_flops(in_d, out_d, n=1):
        """2 * in * out * n for matmul + add bias."""
        return 2 * in_d * out_d * n

    def attn_flops(D, seq_q, seq_k, n=1):
        """Q*K^T + attn*V per head, summed over heads."""
        return 2 * D * seq_q * seq_k * n

    def ffn_flops(D, seq, n=1):
        """Linear(D, 4D) + Linear(4D, D)."""
        return (2 * D * 4 * D + 2 * 4 * D * D) * seq * n

    def layernorm_flops(D, seq, n=1):
        return 2 * D * seq * n  # mean + var

    flops = 0.0

    # 1. QueryEncoder: project [Tq, Dq] → [Tq, D] + LN + attn_pool → [D]
    flops += lin_flops(Dq, D, Tq)           # query_proj
    flops += layernorm_flops(D, Tq)          # query_norm
    flops += attn_flops(D, 1, Tq)            # attn_pool: [1,D] × [Tq,D]
    flops += 2 * D * 1 * Tq                  # attn_pool value aggregation

    # 2. CandidateFusion projections (per candidate × K)
    for in_d in [D_text, D_text, D_text]:    # title, feat, desc
        flops += lin_flops(in_d, D, K)
        flops += layernorm_flops(D, K)
    flops += lin_flops(D_text, D, K * 15)    # img_desc
    flops += layernorm_flops(D, K * 15)
    flops += lin_flops(D_visual, D, K * 5)   # visual
    flops += layernorm_flops(D, K * 5)

    # 3. Cross-attention layers (query [1,D] attends to candidate tokens [Tc,D])
    # Processed as B*K in parallel, so seq_q=1, seq_k=Tc, repeated K times
    for _ in range(n_layers_ca):
        # MHF: Q*K^T + attn*V, per candidate
        flops += attn_flops(D, 1, Tc, K)         # Q*K^T
        flops += 2 * D * 1 * Tc * K               # attn*V
        flops += layernorm_flops(D, 1, K)          # cross_attn_norm
        flops += ffn_flops(D, 1, K)                # FFN
        flops += layernorm_flops(D, 1, K)          # ffn_norm

    # 4. ComparisonTransformer: [K, D] → [K, D]
    for _ in range(n_layers_comp):
        flops += attn_flops(D, K, K)               # self-attn Q*K^T
        flops += 2 * D * K * K                     # self-attn attn*V
        flops += ffn_flops(D, K)                    # FFN
        flops += layernorm_flops(D, K) * 3          # 3 LN per pre-norm layer

    # 5. ScoringHead: Linear(D, D//2) + Linear(D//2, 1) per candidate
    flops += lin_flops(D, D // 2, K)
    flops += lin_flops(D // 2, 1, K)

    # 6. AttributeHead: Linear(D, D//2) + Linear(D//2, 20) per candidate
    flops += lin_flops(D, D // 2, K)
    flops += lin_flops(D // 2, 20, K)

    return flops / 1e6  # MFLOPs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LightCRS reranking model training")

    # Data
    parser.add_argument("--combos_dir", type=str, default="./combos",
                        help="Root dir with pre-computed combo shards")
    parser.add_argument("--candidate_source", type=str, default="st",
                        choices=["st", "gpt_large"],
                        help="Which candidate list to use")
    parser.add_argument("--combo_cache_size", type=int, default=2,
                        help="Number of combo shards to keep in memory")
    parser.add_argument("--dry_run", action="store_true",
                        help="Run with dummy data for testing")

    # Model
    parser.add_argument("--query_dim", type=int, default=768)
    parser.add_argument("--title_dim", type=int, default=768)
    parser.add_argument("--feat_dim", type=int, default=768)
    parser.add_argument("--desc_dim", type=int, default=768)
    parser.add_argument("--img_desc_dim", type=int, default=768)
    parser.add_argument("--visual_dim", type=int, default=4096)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--max_image_descs", type=int, default=15)
    parser.add_argument("--num_cross_attn_heads", type=int, default=4)
    parser.add_argument("--num_cross_attn_layers", type=int, default=2)
    parser.add_argument("--num_comparison_heads", type=int, default=4)
    parser.add_argument("--num_comparison_layers", type=int, default=2)

    # Training
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=0,
                        help="Stop training after N batches per epoch (0 = no limit)")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=2,
                        help="Stop training after N epochs without improvement (0 = disabled)")
    parser.add_argument("--min_delta", type=float, default=0.0,
                        help="Minimum HR@1 improvement to reset patience counter")

    # Eval-only mode
    parser.add_argument("--eval_only", action="store_true",
                        help="Skip training, load checkpoint and evaluate on test set")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint to load (default: output_dir/best_model.pt)")

    # Per-sample prediction dump (optional, only meaningful with --eval_only)
    parser.add_argument("--dump_predictions", type=str, default=None,
                        help="With --eval_only, dump per-sample predictions to this JSONL. "
                             "Default: off. When set, each test sample writes one line with "
                             "{category, sample_idx, gold_idx, pred_idx, correct, rank_of_gold, logits}.")

    # Category filter (optional, for quick single-category tests)
    parser.add_argument("--categories", type=str, default=None,
                        help="Comma-separated list of test categories to evaluate. "
                             "Default: all categories. "
                             "Example: --categories all_beauty,amazon_fashion")

    # Output / dummy data
    parser.add_argument("--output_dir", type=str, default="./checkpoints",
                        help="Directory to save model checkpoints")
    parser.add_argument("--save_last", action="store_true",
                        help="Also save a 'last' checkpoint at the end of training")
    parser.add_argument("--dummy_samples", type=int, default=500)
    parser.add_argument("--num_candidates", type=int, default=10)
    parser.add_argument("--query_seq_len", type=int, default=32)

    args = parser.parse_args()

    # ---- Device ----
    n_gpus = torch.cuda.device_count()
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"Visible GPUs: {n_gpus}")
    if n_gpus > 0:
        for i in range(n_gpus):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    device = torch.device("cuda" if n_gpus > 0 else "cpu")
    print(f"Using device: {device}")

    # ---- Data ----
    if args.dry_run:
        # ---- Dry-run: dummy data with 80/10/10 split ----
        print("\n=== DRY RUN with dummy data ===")
        data_dict = create_dummy_data(
            num_samples=args.dummy_samples,
            num_candidates=args.num_candidates,
            query_seq_len=args.query_seq_len,
            query_dim=args.query_dim,
            text_dim=args.title_dim,
            visual_dim=args.visual_dim,
            max_image_descs=args.max_image_descs,
        )
        full_ds = RerankingDataset(data_dict=data_dict)
        n = len(full_ds)
        n_test = max(1, n // 10)
        n_val = max(1, n // 10)
        n_train = n - n_val - n_test
        train_ds, val_ds, test_ds = random_split(full_ds, [n_train, n_val, n_test])

        val_loaders = {"dummy": DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                           num_workers=0, collate_fn=collate_fn, pin_memory=True)}
        test_loaders = {"dummy": DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                                            num_workers=0, collate_fn=collate_fn, pin_memory=True)}

        sample = full_ds[0]
        print(f"\n--- Sample tensor shapes ---")
        for k, v in sample.items():
            print(f"  {k}: {v.shape}")
        print(f"--- End sample shapes ---\n")

    else:
        # ---- Full mode: load pre-pooled combo shards ----
        combo_dir = os.path.join(
            args.combos_dir, f"candidates_{args.candidate_source}"
        )

        # 1) Train dataset (flat directory with pooled shards)
        train_combo_dir = os.path.join(combo_dir, "train")
        if not os.path.isdir(train_combo_dir):
            print(f"\nERROR: Pooled shards not found at {train_combo_dir}")
            print("Run generate_combos.py with --pool first.")
            sys.exit(1)

        train_ds = PooledComboDataset(train_combo_dir, cache_size=args.combo_cache_size)
        print(f"\nTrain pool: {len(train_ds)} samples from {train_combo_dir}")

        if len(train_ds) == 0:
            print(f"\nERROR: No training samples found.")
            sys.exit(1)

        # 2) Valid/test datasets (per-category subdirectories with pooled shards)
        val_datasets = {}
        val_root = Path(combo_dir) / "valid"
        if val_root.is_dir():
            for cat_dir in sorted(val_root.iterdir()):
                if cat_dir.is_dir() and list(cat_dir.glob("shard_*_pooled.pt")):
                    val_datasets[cat_dir.name] = PooledComboDataset(str(cat_dir), cache_size=1)

        test_datasets = {}
        test_root = Path(combo_dir) / "test"
        # Optional category filter (comma-separated).
        category_filter = None
        if args.categories:
            category_filter = set(c.strip() for c in args.categories.split(",") if c.strip())
        if test_root.is_dir():
            for cat_dir in sorted(test_root.iterdir()):
                if cat_dir.is_dir() and list(cat_dir.glob("shard_*_pooled.pt")):
                    cat_name = cat_dir.name
                    if category_filter is not None and cat_name not in category_filter:
                        continue
                    test_datasets[cat_name] = PooledComboDataset(str(cat_dir), cache_size=1)

        n_val_total = sum(len(ds) for ds in val_datasets.values())
        n_test_total = sum(len(ds) for ds in test_datasets.values())
        print(f"Valid: {n_val_total} samples across {len(val_datasets)} categories")
        print(f"Test:  {n_test_total} samples across {len(test_datasets)} categories")

        # Print sample shapes
        sample = train_ds[0]
        print(f"\n--- Sample tensor shapes ---")
        for k, v in sample.items():
            print(f"  {k}: {v.shape}")
        print(f"--- End sample shapes ---\n")

    # ---- DataLoaders ----
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn,
        pin_memory=True, drop_last=True,
    )

    if not args.dry_run:
        val_loaders = {
            cat: DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate_fn,
                            pin_memory=True)
            for cat, ds in val_datasets.items()
        }
        test_loaders = {
            cat: DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate_fn,
                            pin_memory=True)
            for cat, ds in test_datasets.items()
        }

    print(f"Batch size: {args.batch_size}")
    print(f"Train steps/epoch: {len(train_loader)}")

    # ---- Model ----
    model = RerankingModel(
        query_dim=args.query_dim,
        title_dim=args.title_dim,
        feat_dim=args.feat_dim,
        desc_dim=args.desc_dim,
        img_desc_dim=args.img_desc_dim,
        visual_dim=args.visual_dim,
        hidden_dim=args.hidden_dim,
        num_cross_attn_heads=args.num_cross_attn_heads,
        num_cross_attn_layers=args.num_cross_attn_layers,
        num_comparison_heads=args.num_comparison_heads,
        num_comparison_layers=args.num_comparison_layers,
        max_image_descs=args.max_image_descs,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    if n_gpus > 1:
        print(f"Using DataParallel over {n_gpus} GPUs")
        model = nn.DataParallel(model)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model = model.to(device)

    # Reset peak memory tracking
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)
    print(f"Model size: {model_size_mb:.2f} MB")

    # ---- Optimiser ----
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )

    # ---- Loss: cross-entropy only ----
    loss_fn = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(enabled=False)

    # ---- Checkpoint dir ----
    ckpt_dir = Path(args.output_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    raw_model = model.module if isinstance(model, nn.DataParallel) else model

    # ---- Eval-only mode: load checkpoint and skip training ----
    if args.eval_only:
        ckpt_path = args.checkpoint or str(ckpt_dir / "best_model.pt")
        print(f"\n=== EVAL ONLY: loading {ckpt_path} ===")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        raw_model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}, "
              f"best avg HR@1={ckpt.get('best_avg_hr1', '?')}")
        best_avg_hr1 = ckpt.get("best_avg_hr1", 0.0)
        total_train_time = 0.0

    # Optional: open per-sample dump file (only meaningful with --eval_only)
    dump_file = None
    if args.eval_only and args.dump_predictions:
        dump_file = open(args.dump_predictions, "w")
        print(f"Dumping per-sample predictions to {args.dump_predictions}")

    if not args.eval_only:
        # ---- Training loop ----
        print(f"\n{'Epoch':>5} | {'Train Loss':>10} | {'Avg Val HR@1':>13} | {'Time':>6}")
        print("-" * 50)

        best_avg_hr1 = 0.0
        epochs_no_improve = 0
        total_train_start = time.time()
        for epoch in range(1, args.epochs + 1):
            t0 = time.time()

            train_metrics = train_one_epoch(
                model, train_loader, optimizer, loss_fn, device, scaler,
                grad_accum_steps=args.grad_accum_steps,
                max_steps=args.max_steps,
            )

            # Per-category validation
            cat_val_metrics = {}
            for cat, loader in sorted(val_loaders.items()):
                cat_val_metrics[cat] = evaluate(model, loader, loss_fn, device)

            avg_hr1 = sum(m["hr@1"] for m in cat_val_metrics.values()) / max(len(cat_val_metrics), 1)
            elapsed = time.time() - t0

            print(
                f"{epoch:5d} | ce={train_metrics['ce']:8.4f} | "
                f"{avg_hr1:13.4f} | {elapsed:5.1f}s"
            )
            # Per-category HR@1
            parts = [f"{cat}={m['hr@1']:.4f}" for cat, m in sorted(cat_val_metrics.items())]
            print(f"      val HR@1: {', '.join(parts)}")

            if avg_hr1 > best_avg_hr1 + args.min_delta:
                best_avg_hr1 = avg_hr1
                epochs_no_improve = 0
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": raw_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_avg_hr1": best_avg_hr1,
                    "args": vars(args),
                }, ckpt_dir / "best_model.pt")
                print(f"      -> saved best checkpoint (avg HR@1={best_avg_hr1:.4f})")
            else:
                epochs_no_improve += 1
                if args.patience > 0 and epochs_no_improve >= args.patience:
                    print(f"  Early stopping: no improvement for {args.patience} epochs")
                    break

        total_train_time = time.time() - total_train_start
        print(f"\nBest avg val HR@1: {best_avg_hr1:.4f}")

        # ---- Save last checkpoint ----
        if args.save_last:
            torch.save({
                "epoch": args.epochs,
                "model_state_dict": raw_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_avg_hr1": best_avg_hr1,
                "args": vars(args),
            }, ckpt_dir / "last_model.pt")
            print(f"Saved last checkpoint to {ckpt_dir / 'last_model.pt'}")

    # ---- Per-category test evaluation ----
    print(f"\n{'='*70}")
    print("Per-category test results:")
    print(f"{'='*70}")
    print(f"  {'Category':<45s} {'HR@1':>6s} {'HR@5':>6s} {'HR@10':>6s} {'NDCG@5':>7s} {'MRR':>6s} {'VR':>6s}")
    print(f"  {'-'*82}")

    test_hr1_values = []
    test_metrics_all = {}
    for cat, loader in sorted(test_loaders.items()):
        m = evaluate(model, loader, loss_fn, device,
                     dump_file=dump_file, category=cat)
        test_hr1_values.append(m["hr@1"])
        test_metrics_all[cat] = m
        print(f"  {cat:<45s} {m['hr@1']:6.4f} {m['hr@5']:6.4f} {m['hr@10']:6.4f} "
              f"{m['ndcg@5']:7.4f} {m['mrr']:6.4f} {m['vr']:6.4f}")

    avg_test_hr1 = sum(test_hr1_values) / max(len(test_hr1_values), 1)
    print(f"  {'-'*82}")
    print(f"  {'Average':<45s} {avg_test_hr1:6.4f}")
    print(f"{'='*70}")

    # ---- Efficiency Metrics ----
    print(f"\n{'='*70}")
    print("Efficiency Metrics")
    print(f"{'='*70}")
    print(f"  Model parameters : {total_params:,} total, {trainable_params:,} trainable")
    print(f"  Model size       : {model_size_mb:.2f} MB")

    peak_mem_gb = 0.0
    if torch.cuda.is_available():
        peak_mem_bytes = torch.cuda.max_memory_allocated(device)
        peak_mem_mb = peak_mem_bytes / (1024 ** 2)
        peak_mem_gb = peak_mem_mb / 1024
        print(f"  Peak GPU memory  : {peak_mem_mb:.1f} MB ({peak_mem_gb:.2f} GB)")

    print(f"  Total train time : {total_train_time:.1f} s")

    gflops = None
    avg_latency = 0.0
    avg_throughput = 0.0

    if test_loaders:
        # Per-category latency (from the evaluation we already ran)
        print(f"\n  Per-category inference latency:")
        print(f"    {'Category':<45s} {'ms/query':>10s} {'q/s':>10s}")
        print(f"    {'-'*67}")
        for cat, m in sorted(test_metrics_all.items()):
            print(f"    {cat:<45s} {m.get('latency_ms_per_query', 0):10.2f} "
                  f"{m.get('throughput_qps', 0):10.1f}")
            avg_latency += m.get('latency_ms_per_query', 0)
            avg_throughput += m.get('throughput_qps', 0)
        n_cats = max(len(test_metrics_all), 1)
        avg_latency /= n_cats
        avg_throughput /= n_cats

        # FLOPs estimation
        bench_cat = list(test_loaders.keys())[0]
        bench_loader = list(test_loaders.values())[0]
        print(f"\n  FLOPs estimation ({bench_cat}):")
        gflops = estimate_flops_per_query(model, bench_loader, device)
        if gflops is not None:
            print(f"    MFLOPs/query      : {gflops:.2f}")
        else:
            print(f"    MFLOPs/query      : N/A")

    # Quality-cost trade-off summary
    print(f"\n  Quality-Cost Trade-off Summary:")
    print(f"    {'Metric':<25s} {'Value':>15s}")
    print(f"    {'-'*42}")
    print(f"    {'HR@1':<25s} {avg_test_hr1:15.4f}")
    print(f"    {'Parameters':<25s} {total_params:>15,}")
    print(f"    {'Model Size (MB)':<25s} {model_size_mb:15.2f}")
    if torch.cuda.is_available():
        print(f"    {'Peak GPU Memory (GB)':<25s} {peak_mem_gb:15.2f}")
    if gflops is not None:
        print(f"    {'MFLOPs/query':<25s} {gflops:15.2f}")
    if test_loaders:
        print(f"    {'Latency (ms/query)':<25s} {avg_latency:15.2f}")
        print(f"    {'Throughput (queries/s)':<25s} {avg_throughput:15.1f}")
    print(f"{'='*70}")

    if dump_file is not None:
        dump_file.close()
        print(f"Closed {args.dump_predictions}")

    print("Training complete.")


if __name__ == "__main__":
    main()
