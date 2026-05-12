"""
PyTorch Profiler + rigorous latency benchmark for NetGPT.

Usage:
    cd ~/NetGPT_work/
    python pruning/benchmark_profiler.py \
        --pretrained_model_path models/finetuned_model.bin \
        --config_path models/gpt2/config.json \
        --vocab_path models/encryptd_vocab.txt \
        --dev_path finetune_dataset/valid_dataset.tsv \
        --train_path finetune_dataset/train_dataset.tsv \
        --seq_length 64 --labels_num 2 --batch_size 32 \
        --pooling mean --seed 42 \
        --metric wanda --sparsity 0.5 \
        --output_dir results/profiler
"""

import sys, os, argparse, time, copy, json
import numpy as np
import torch
import torch.nn as nn

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from uer.utils.constants import *
from uer.utils import *
from uer.utils.config import load_hyperparam
from uer.utils.seed import set_seed
from uer.utils.logging import init_logger
from uer.opts import finetune_opts, tokenizer_opts, adv_opts
from finetune.run_understanding import (
    Classifier, read_dataset, batch_loader, load_or_initialize_parameters
)
from pruning.pruner import NetGPTPruner
from pruning.run_pruning import make_calibration_loader


@torch.no_grad()
def benchmark_latency(model, src, seg, device, n_warmup=50, n_measure=500):
    model.eval()
    src, seg = src.to(device), seg.to(device)
    for _ in range(n_warmup):
        model(src, None, seg)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    times_ms = []
    for _ in range(n_measure):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(src, None, seg)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)
    t = np.array(times_ms)
    return {
        "median_ms": float(np.median(t)),
        "mean_ms": float(np.mean(t)),
        "std_ms": float(np.std(t)),
        "p05_ms": float(np.percentile(t, 5)),
        "p95_ms": float(np.percentile(t, 95)),
        "min_ms": float(np.min(t)),
        "n_measure": n_measure,
        "raw_times": times_ms,
    }


def get_event_time(evt):
    """Get time from profiler event — try CUDA first, fallback to CPU."""
    for attr in ["cuda_time_total", "self_cuda_time_total",
                 "device_time_total", "self_device_time_total"]:
        try:
            val = getattr(evt, attr, None)
            if val is not None and val > 0:
                return val
        except Exception:
            continue
    for attr in ["cpu_time_total", "self_cpu_time_total"]:
        try:
            val = getattr(evt, attr, None)
            if val is not None and val > 0:
                return val
        except Exception:
            continue
    return 0


def find_sort_key(prof):
    """Find the best available sort key for this PyTorch version."""
    if not prof.key_averages():
        return "self_cpu_time_total"
    evt = prof.key_averages()[0]
    for candidate in ["cuda_time_total", "self_cuda_time_total",
                      "device_time_total", "self_device_time_total"]:
        try:
            if getattr(evt, candidate, 0) > 0:
                return candidate
        except Exception:
            continue
    return "self_cpu_time_total"


def profile_model(model, src, seg, device, output_dir, label="dense"):
    from torch.profiler import profile, ProfilerActivity, schedule

    model.eval()
    src, seg = src.to(device), seg.to(device)
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    with profile(
        activities=activities,
        schedule=schedule(wait=5, warmup=10, active=20, repeat=1),
        record_shapes=True, profile_memory=True,
    ) as prof:
        for step in range(35):
            with torch.no_grad():
                model(src, None, seg)
            prof.step()

    sort_key = find_sort_key(prof)
    table = prof.key_averages().table(sort_by=sort_key, row_limit=25)
    print(f"\n{'='*80}")
    print(f"  PROFILER RESULTS — {label} (sorted by {sort_key})")
    print(f"{'='*80}")
    print(table)

    trace_path = os.path.join(output_dir, f"trace_{label}.json")
    prof.export_chrome_trace(trace_path)
    print(f"Chrome trace: {trace_path}")

    # Breakdown by type
    print(f"\n{'='*80}")
    print(f"  BREAKDOWN BY LAYER TYPE — {label}")
    print(f"{'='*80}")
    categories = {
        "MatMul (attn+FFN)": 0, "Softmax": 0, "LayerNorm": 0,
        "Embedding": 0, "Dropout": 0, "Other": 0,
    }
    for evt in prof.key_averages():
        t = get_event_time(evt)
        key = evt.key.lower()
        if "softmax" in key:
            categories["Softmax"] += t
        elif "layer_norm" in key or "native_layer_norm" in key:
            categories["LayerNorm"] += t
        elif "embedding" in key or "index_select" in key:
            categories["Embedding"] += t
        elif "dropout" in key:
            categories["Dropout"] += t
        elif any(k in key for k in ["mm", "linear", "addmm", "matmul", "bmm"]):
            categories["MatMul (attn+FFN)"] += t
        else:
            categories["Other"] += t
    total = sum(categories.values()) or 1
    for cat, us in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"  {cat:<25} {us/1000:>8.1f} ms  ({us/total*100:>5.1f}%)")

    return prof


def stat_compare(results_dict):
    try:
        from scipy.stats import mannwhitneyu
        has_scipy = True
    except ImportError:
        has_scipy = False
        print("  (scipy not installed — pip install scipy)")

    dense_times = results_dict.get("dense", {}).get("raw_times")
    if dense_times is None:
        return

    print(f"\n{'='*80}")
    print(f"  STATISTICAL COMPARISON (Mann-Whitney U)")
    print(f"{'='*80}")
    print(f"  {'Config':<20} {'Median':>10} {'p5-p95':>18} {'p-value':>10} {'Sig?':>6}")
    print(f"  {'-'*64}")
    for name, res in results_dict.items():
        med = f"{res['median_ms']:.3f}"
        ci = f"[{res['p05_ms']:.3f}, {res['p95_ms']:.3f}]"
        if name == "dense":
            print(f"  {name:<20} {med:>10} {ci:>18} {'—':>10} {'ref':>6}")
        elif has_scipy:
            _, pval = mannwhitneyu(dense_times, res["raw_times"], alternative="two-sided")
            sig = "YES" if pval < 0.05 else "no"
            print(f"  {name:<20} {med:>10} {ci:>18} {pval:>10.4f} {sig:>6}")
        else:
            print(f"  {name:<20} {med:>10} {ci:>18} {'N/A':>10} {'N/A':>6}")


def main():
    parser = argparse.ArgumentParser(description="Profiler + latency benchmark")
    finetune_opts(parser); tokenizer_opts(parser); adv_opts(parser)
    parser.add_argument("--soft_targets", action="store_true", default=False)
    parser.add_argument("--soft_alpha", type=float, default=0.5)
    parser.add_argument("--labels_num", type=int, default=2)
    parser.add_argument("--metric", type=str, default="magnitude",
                        choices=["magnitude", "wanda", "pruner_zero"])
    parser.add_argument("--sparsity", type=float, default=0.5)
    parser.add_argument("--n_calib", type=int, default=128)
    parser.add_argument("--output_dir", type=str, default="results/profiler")
    parser.add_argument("--n_warmup", type=int, default=50)
    parser.add_argument("--n_measure", type=int, default=500)
    parser.add_argument("--skip_profiler", action="store_true")
    args = parser.parse_args()
    args = load_hyperparam(args)
    args.tokenizer = str2tokenizer[args.tokenizer](args)
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    model = Classifier(args)
    load_or_initialize_parameters(args, model)
    args.logger = init_logger(args)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(args.device)
    args.model = model

    devset = read_dataset(args, args.dev_path)
    trainset = read_dataset(args, args.train_path)
    print(f"Dev: {len(devset)}, Train: {len(trainset)}")

    subset = devset[:min(args.batch_size, len(devset))]
    src = torch.LongTensor([s[0] for s in subset])
    seg = torch.LongTensor([s[2] for s in subset])

    if not args.skip_profiler:
        print("\n[1] Profiling DENSE model...")
        profile_model(model, src, seg, args.device, args.output_dir, "dense")

    print(f"\n[2] Benchmarking latency (warmup={args.n_warmup}, measure={args.n_measure})...")
    results = {}
    print("  -> Dense...")
    results["dense"] = benchmark_latency(model, src, seg, args.device, args.n_warmup, args.n_measure)
    print(f"    Median: {results['dense']['median_ms']:.3f} ms")

    calib_loader = make_calibration_loader(trainset, args.n_calib, args.batch_size, seed=args.seed)

    for metric in ["magnitude", "wanda", "pruner_zero"]:
        label = f"{metric}_{int(args.sparsity*100)}pct"
        print(f"\n  -> {label}...")
        model_pruned = copy.deepcopy(model)
        pruner = NetGPTPruner(model_pruned, metric=metric, sparsity=args.sparsity)
        pruner.calibrate(calib_loader, args.device, args)
        pruner.prune()
        if not args.skip_profiler:
            profile_model(model_pruned, src, seg, args.device, args.output_dir, label)
        results[label] = benchmark_latency(model_pruned, src, seg, args.device, args.n_warmup, args.n_measure)
        print(f"    Median: {results[label]['median_ms']:.3f} ms")
        del model_pruned; torch.cuda.empty_cache()

    stat_compare(results)

    save_results = {k: {kk: vv for kk, vv in v.items() if kk != "raw_times"} for k, v in results.items()}
    out_path = os.path.join(args.output_dir, "benchmark_results.json")
    with open(out_path, "w") as f:
        json.dump(save_results, f, indent=2)
    print(f"\nResults saved: {out_path}")
    print("Done.")


if __name__ == "__main__":
    main()
