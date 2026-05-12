"""
Post-training pruning evaluation for NetGPT.

Usage (paper dataset):
    python pruning/run_pruning.py \
        --pretrained_model_path models/finetuned_model.bin \
        --config_path models/gpt2/config.json \
        --vocab_path models/encryptd_vocab.txt \
        --train_path finetune_dataset/train_dataset.tsv \
        --dev_path finetune_dataset/valid_dataset.tsv \
        --test_path finetune_dataset/test_dataset.tsv \
        --seq_length 64 --labels_num 2 --batch_size 32 \
        --metric magnitude --sparsity 0.5 --seed 42 \
        --output_csv results/pruning_results.csv

Metrics: magnitude, wanda, pruner_zero
"""

import sys
import os
import random
import argparse
import time
import csv
import json
import torch
import torch.nn as nn

# Add project root to path for UER imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from uer.utils.constants import *
from uer.utils import *
from uer.utils.config import load_hyperparam
from uer.utils.seed import set_seed
from uer.utils.logging import init_logger
from uer.opts import finetune_opts, tokenizer_opts, adv_opts

# Import from existing pipeline
from finetune.run_understanding import (
    Classifier, read_dataset, batch_loader, evaluate,
    load_or_initialize_parameters
)

# Import pruning module
from pruning.pruner import NetGPTPruner


def make_calibration_loader(dataset, n_calib, batch_size, seed=42):
    """Sample n_calib examples from dataset and return batches."""
    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    calib_data = [dataset[i] for i in indices[:n_calib]]

    src = torch.LongTensor([s[0] for s in calib_data])
    tgt = torch.LongTensor([s[1] for s in calib_data])
    seg = torch.LongTensor([s[2] for s in calib_data])

    batches = list(batch_loader(batch_size, src, tgt, seg))
    print(f"[Calib] Sampled {len(calib_data)} flows, "
          f"{len(batches)} batches (batch_size={batch_size})")
    return batches


def compute_model_size(model):
    """Compute model size stats."""
    total_params = 0
    nonzero_params = 0
    total_bytes = 0
    for p in model.parameters():
        total_params += p.numel()
        nonzero_params += (p != 0).sum().item()
        total_bytes += p.nelement() * p.element_size()
    return {
        "total_params": total_params,
        "nonzero_params": nonzero_params,
        "size_mb": total_bytes / (1024 * 1024),
        "density": nonzero_params / total_params if total_params > 0 else 1.0
    }


def measure_inference_time(model, dataset, device, batch_size=32, n_runs=3):
    """Measure average inference time per sample (ms)."""
    subset = dataset[:min(256, len(dataset))]
    src = torch.LongTensor([s[0] for s in subset])
    tgt = torch.LongTensor([s[1] for s in subset])
    seg = torch.LongTensor([s[2] for s in subset])

    model.eval()
    # Warmup
    with torch.no_grad():
        for src_b, tgt_b, seg_b, _ in batch_loader(batch_size, src, tgt, seg):
            model(src_b.to(device), tgt_b.to(device), seg_b.to(device))
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        with torch.no_grad():
            for src_b, tgt_b, seg_b, _ in batch_loader(batch_size, src, tgt, seg):
                model(src_b.to(device), tgt_b.to(device), seg_b.to(device))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    avg_time = sum(times) / len(times)
    return (avg_time / len(subset)) * 1000


def evaluate_with_metrics(args, dataset):
    """Extended evaluation returning accuracy + per-class P/R/F1."""
    acc, confusion = evaluate(args, dataset, istest=False)
    eps = 1e-9
    results = {"accuracy": acc}
    f1_list = []
    for i in range(confusion.size(0)):
        p = confusion[i, i].item() / (confusion[i, :].sum().item() + eps)
        r = confusion[i, i].item() / (confusion[:, i].sum().item() + eps)
        f1 = 2 * p * r / (p + r + eps)
        results[f"precision_class{i}"] = p
        results[f"recall_class{i}"] = r
        results[f"f1_class{i}"] = f1
        f1_list.append(f1)
    results["f1_macro"] = sum(f1_list) / len(f1_list)
    results["confusion_matrix"] = confusion.tolist()
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Post-training pruning for NetGPT",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # ── Reuse UER's standard opts (adds all required args with defaults) ──
    finetune_opts(parser)
    tokenizer_opts(parser)
    adv_opts(parser)

    # ── Args required by Classifier (from run_understanding.py) ──
    parser.add_argument("--soft_targets", action="store_true", default=False)
    parser.add_argument("--soft_alpha", type=float, default=0.5)

    # ── Add pruning-specific args ──
    parser.add_argument("--labels_num", type=int, default=2)
    parser.add_argument("--metric", type=str, default="magnitude",
                        choices=["magnitude", "wanda", "pruner_zero"])
    parser.add_argument("--sparsity", type=float, default=0.5)
    parser.add_argument("--n_calib", type=int, default=128)
    parser.add_argument("--prune_output", action="store_true",
                        help="Also prune output classification layers")
    parser.add_argument("--output_csv", type=str, default=None)
    parser.add_argument("--output_pruned_model", type=str, default=None)
    parser.add_argument("--exp_id", type=str, default="")

    args = parser.parse_args()

    # ── Load hyperparameters from config ──
    args = load_hyperparam(args)

    # ── Build tokenizer ──
    args.tokenizer = str2tokenizer[args.tokenizer](args)
    set_seed(args.seed)

    # ── Print experiment info ──
    print(f"\n{'='*60}")
    print(f" NetGPT Post-Training Pruning")
    print(f"  Metric:     {args.metric}")
    print(f"  Sparsity:   {args.sparsity*100:.0f}%")
    print(f"  N_calib:    {args.n_calib}")
    print(f"  Seed:       {args.seed}")
    print(f"  Seq_length: {args.seq_length}")
    print(f"  Labels:     {args.labels_num}")
    print(f"{'='*60}\n")

    # ── Build and load model ──
    model = Classifier(args)
    load_or_initialize_parameters(args, model)

    args.logger = init_logger(args)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(args.device)
    args.model = model

    # ── Load datasets ──
    trainset = read_dataset(args, args.train_path)
    devset = read_dataset(args, args.dev_path)
    testset = read_dataset(args, args.test_path) if args.test_path else None

    print(f"Train: {len(trainset)}, Dev: {len(devset)}, "
          f"Test: {len(testset) if testset else 'N/A'}")

    # ═══════════════════════════════════════════════════════
    # [1/4] Evaluate DENSE model
    # ═══════════════════════════════════════════════════════
    print("\n[1/4] Evaluating DENSE model...")
    dense_results = evaluate_with_metrics(args, devset)
    dense_size = compute_model_size(model)
    dense_ms = measure_inference_time(model, devset, args.device, args.batch_size)

    print(f"  Accuracy:  {dense_results['accuracy']:.4f}")
    print(f"  F1-macro:  {dense_results['f1_macro']:.4f}")
    print(f"  Inference: {dense_ms:.2f} ms/sample")
    print(f"  Params:    {dense_size['total_params']:,}")

    # ═══════════════════════════════════════════════════════
    # [2/4] Calibration
    # ═══════════════════════════════════════════════════════
    print(f"\n[2/4] Calibrating ({args.n_calib} samples)...")
    calib_loader = make_calibration_loader(
        trainset, args.n_calib, args.batch_size, seed=args.seed
    )

    # ═══════════════════════════════════════════════════════
    # [3/4] Pruning
    # ═══════════════════════════════════════════════════════
    print(f"\n[3/4] Pruning: metric={args.metric}, sparsity={args.sparsity*100:.0f}%")

    pruner = NetGPTPruner(
        model, metric=args.metric, sparsity=args.sparsity,
        prune_output_layers=args.prune_output
    )
    pruner.calibrate(calib_loader, args.device, args)
    pruner.prune()
    pruner.print_stats()

    # ═══════════════════════════════════════════════════════
    # [4/4] Evaluate PRUNED model
    # ═══════════════════════════════════════════════════════
    print("[4/4] Evaluating PRUNED model...")
    pruned_results = evaluate_with_metrics(args, devset)
    pruned_size = compute_model_size(model)
    pruned_ms = measure_inference_time(model, devset, args.device, args.batch_size)

    print(f"  Accuracy:  {pruned_results['accuracy']:.4f}")
    print(f"  F1-macro:  {pruned_results['f1_macro']:.4f}")
    print(f"  Inference: {pruned_ms:.2f} ms/sample")
    print(f"  Density:   {pruned_size['density']*100:.2f}%")

    # ── Retention ──
    retention = (pruned_results['accuracy'] / dense_results['accuracy'] * 100
                 if dense_results['accuracy'] > 0 else 0)
    delta_f1 = pruned_results['f1_macro'] - dense_results['f1_macro']

    print(f"\n{'='*60}")
    print(f"  RETENTION: {retention:.1f}%")
    print(f"  F1 DELTA:  {delta_f1:+.4f}")
    print(f"{'='*60}")

    # ── Test set ──
    test_results = None
    if testset:
        print("\nEvaluating on TEST set...")
        test_results = evaluate_with_metrics(args, testset)
        print(f"  Test Accuracy: {test_results['accuracy']:.4f}")
        print(f"  Test F1-macro: {test_results['f1_macro']:.4f}")

    # ── Save pruned model ──
    if args.output_pruned_model:
        os.makedirs(os.path.dirname(args.output_pruned_model) or ".", exist_ok=True)
        torch.save(model.state_dict(), args.output_pruned_model)
        print(f"\nPruned model saved: {args.output_pruned_model}")

    # ── Append to CSV ──
    if args.output_csv:
        os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
        file_exists = os.path.exists(args.output_csv)

        row = {
            "exp_id": args.exp_id,
            "metric": args.metric,
            "sparsity": args.sparsity,
            "seed": args.seed,
            "n_calib": args.n_calib,
            "seq_length": args.seq_length,
            "dense_accuracy": f"{dense_results['accuracy']:.6f}",
            "dense_f1_macro": f"{dense_results['f1_macro']:.6f}",
            "dense_f1_class0": f"{dense_results.get('f1_class0', 0):.6f}",
            "dense_f1_class1": f"{dense_results.get('f1_class1', 0):.6f}",
            "dense_inference_ms": f"{dense_ms:.4f}",
            "dense_params": dense_size["total_params"],
            "pruned_accuracy": f"{pruned_results['accuracy']:.6f}",
            "pruned_f1_macro": f"{pruned_results['f1_macro']:.6f}",
            "pruned_f1_class0": f"{pruned_results.get('f1_class0', 0):.6f}",
            "pruned_f1_class1": f"{pruned_results.get('f1_class1', 0):.6f}",
            "pruned_inference_ms": f"{pruned_ms:.4f}",
            "pruned_nonzero_params": pruned_size["nonzero_params"],
            "pruned_density": f"{pruned_size['density']:.6f}",
            "retention_pct": f"{retention:.2f}",
            "f1_delta": f"{delta_f1:.6f}",
        }
        if test_results:
            row["test_accuracy"] = f"{test_results['accuracy']:.6f}"
            row["test_f1_macro"] = f"{test_results['f1_macro']:.6f}"

        with open(args.output_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
        print(f"Results appended to {args.output_csv}")


if __name__ == "__main__":
    main()
