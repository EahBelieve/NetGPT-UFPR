"""
Sparse inference with torch.sparse.semi_structured (2:4 sparsity).

FIX v2: Uses torch.cuda.amp.autocast(dtype=torch.float16) for the
semi-structured benchmark, because to_sparse_semi_structured converts
weights to FP16 but the input embeddings remain FP32. Without autocast,
F.linear fails with "Expected float, got Half".

Usage:
    cd ~/NetGPT_work/
    python pruning/sparse_inference.py \
        --pretrained_model_path models/finetuned_model.bin \
        --config_path models/gpt2/config.json \
        --vocab_path models/encryptd_vocab.txt \
        --train_path finetune_dataset/train_dataset.tsv \
        --dev_path finetune_dataset/valid_dataset.tsv \
        --seq_length 64 --labels_num 2 --batch_size 32 \
        --pooling mean --seed 42 \
        --metric wanda \
        --output_dir results/sparse_inference
"""

import sys, os, copy, time, json, argparse
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
from pruning.metrics import METRICS
from pruning.run_pruning import make_calibration_loader, evaluate_with_metrics


# ═════════════════════════════════════════════════════════════
# 2:4 Structured Pruning
# ═════════════════════════════════════════════════════════════

def prune_2_4(weight, scores):
    """Apply 2:4 structured pruning: keep top-2 per group of 4."""
    out_f, in_f = weight.shape
    mask = torch.ones_like(weight)
    for row_idx in range(out_f):
        for col in range(0, in_f - 3, 4):
            group = scores[row_idx, col:col + 4]
            _, indices = torch.sort(group)
            mask[row_idx, col + indices[0]] = 0.0
            mask[row_idx, col + indices[1]] = 0.0
    return mask


def apply_2_4_pruning(model, pruner):
    """Apply 2:4 structured pruning to all linear layers."""
    print("[2:4] Applying 2:4 structured pruning...")
    total_params = 0
    total_pruned = 0
    for name, module in pruner.linear_layers.items():
        weight = module.weight.data
        kwargs = {}
        if name in pruner.activation_norms:
            kwargs["activation_norms"] = pruner.activation_norms[name]
        if name in pruner.gradients:
            kwargs["gradients"] = pruner.gradients[name]
        scores = pruner.metric_fn(weight, **kwargs)
        mask = prune_2_4(weight, scores)
        module.weight.data *= mask
        pruner.masks[name] = mask
        zeros = (mask == 0).sum().item()
        total_params += weight.numel()
        total_pruned += zeros
    print(f"[2:4] Effective sparsity: {total_pruned/total_params*100:.2f}%")


def verify_2_4_pattern(weight):
    """Check if weight follows 2:4 pattern."""
    out_f, in_f = weight.shape
    for row in range(min(out_f, 10)):  # spot-check first 10 rows
        for col in range(0, in_f - 3, 4):
            if (weight[row, col:col + 4] == 0).sum().item() != 2:
                return False
    return True


def convert_to_semi_structured(model, verbose=True):
    """Convert 2:4-pruned model to semi-structured sparse format (FP16)."""
    try:
        from torch.sparse import to_sparse_semi_structured
    except ImportError:
        print("ERROR: torch.sparse.to_sparse_semi_structured unavailable (need PyTorch >= 2.1)")
        return []

    converted = []
    skipped = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if "output_layer" in name:
            skipped.append(name)
            continue
        w = module.weight.data
        if not verify_2_4_pattern(w):
            skipped.append(name)
            if verbose: print(f"  SKIP {name}: not 2:4")
            continue
        try:
            w_fp16 = w.to(torch.float16)
            w_sparse = to_sparse_semi_structured(w_fp16)
            module.weight = nn.Parameter(w_sparse)
            converted.append(name)
            if verbose: print(f"  OK   {name}")
        except Exception as e:
            skipped.append(name)
            if verbose: print(f"  FAIL {name}: {e}")

    print(f"[Sparse] Converted {len(converted)}/{len(converted)+len(skipped)} layers")
    return converted


# ═════════════════════════════════════════════════════════════
# Benchmark (with optional FP16 autocast)
# ═════════════════════════════════════════════════════════════

@torch.no_grad()
def benchmark(model, src, seg, device, n_warmup=50, n_measure=500,
              label="", use_fp16=False):
    """
    Rigorous latency benchmark.
    
    use_fp16: wrap forward in autocast(float16) — required when
              weights are semi-structured FP16 but inputs are FP32.
    """
    model.eval()
    src, seg = src.to(device), seg.to(device)

    for _ in range(n_warmup):
        if use_fp16:
            with torch.cuda.amp.autocast(dtype=torch.float16):
                model(src, None, seg)
        else:
            model(src, None, seg)
    torch.cuda.synchronize()

    times_ms = []
    for _ in range(n_measure):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        if use_fp16:
            with torch.cuda.amp.autocast(dtype=torch.float16):
                model(src, None, seg)
        else:
            model(src, None, seg)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    t = np.array(times_ms)
    result = {
        "median_ms": float(np.median(t)),
        "mean_ms": float(np.mean(t)),
        "std_ms": float(np.std(t)),
        "p05_ms": float(np.percentile(t, 5)),
        "p95_ms": float(np.percentile(t, 95)),
    }
    print(f"  {label:<35} median={result['median_ms']:.3f} ms  "
          f"[{result['p05_ms']:.3f}, {result['p95_ms']:.3f}]")
    return result


# ═════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Sparse 2:4 inference benchmark")
    finetune_opts(parser); tokenizer_opts(parser); adv_opts(parser)
    parser.add_argument("--soft_targets", action="store_true", default=False)
    parser.add_argument("--soft_alpha", type=float, default=0.5)
    parser.add_argument("--labels_num", type=int, default=2)
    parser.add_argument("--metric", type=str, default="wanda",
                        choices=["magnitude", "wanda", "pruner_zero"])
    parser.add_argument("--n_calib", type=int, default=128)
    parser.add_argument("--output_dir", type=str, default="results/sparse_inference")
    parser.add_argument("--n_warmup", type=int, default=50)
    parser.add_argument("--n_measure", type=int, default=500)
    args = parser.parse_args()
    args = load_hyperparam(args)
    args.tokenizer = str2tokenizer[args.tokenizer](args)
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    if not torch.cuda.is_available():
        print("ERROR: CUDA required"); sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    compute = torch.cuda.get_device_capability(0)
    print(f"GPU: {gpu_name} (compute {compute[0]}.{compute[1]})")
    if compute[0] < 8:
        print("WARNING: Sparse TC requires Ampere+ (compute >= 8.0)")

    has_semi = False
    try:
        from torch.sparse import to_sparse_semi_structured
        has_semi = True
        print("torch.sparse.semi_structured: AVAILABLE")
    except ImportError:
        print(f"torch.sparse.semi_structured: NOT AVAILABLE (torch {torch.__version__})")

    model = Classifier(args)
    load_or_initialize_parameters(args, model)
    args.logger = init_logger(args)
    args.device = torch.device("cuda")
    model = model.to(args.device)
    args.model = model

    trainset = read_dataset(args, args.train_path)
    devset = read_dataset(args, args.dev_path)
    print(f"Train: {len(trainset)}, Dev: {len(devset)}")

    subset = devset[:min(args.batch_size, len(devset))]
    src = torch.LongTensor([s[0] for s in subset])
    seg = torch.LongTensor([s[2] for s in subset])

    calib_loader = make_calibration_loader(trainset, args.n_calib, args.batch_size, seed=args.seed)
    results = {}

    # ═══════════════════════════════════════════════════
    # 1. Dense FP32
    # ═══════════════════════════════════════════════════
    print(f"\n{'='*60}\n  1. DENSE FP32\n{'='*60}")
    results["dense_fp32"] = benchmark(
        model, src, seg, args.device, args.n_warmup, args.n_measure, "Dense FP32")

    # ═══════════════════════════════════════════════════
    # 2. Dense FP16 (autocast) — fair comparison for sparse FP16
    # ═══════════════════════════════════════════════════
    print(f"\n{'='*60}\n  2. DENSE FP16 (autocast)\n{'='*60}")
    results["dense_fp16"] = benchmark(
        model, src, seg, args.device, args.n_warmup, args.n_measure,
        "Dense FP16 (autocast)", use_fp16=True)

    # ═══════════════════════════════════════════════════
    # 3. Unstructured 50% FP32 (no speedup expected)
    # ═══════════════════════════════════════════════════
    print(f"\n{'='*60}\n  3. UNSTRUCTURED 50%\n{'='*60}")
    model_u = copy.deepcopy(model)
    p_u = NetGPTPruner(model_u, metric=args.metric, sparsity=0.5)
    p_u.calibrate(calib_loader, args.device, args)
    p_u.prune()
    results["unstructured_50_fp32"] = benchmark(
        model_u, src, seg, args.device, args.n_warmup, args.n_measure,
        "Unstructured 50% FP32")
    args.model = model_u
    u_res = evaluate_with_metrics(args, devset)
    print(f"  Accuracy: {u_res['accuracy']:.4f}")
    del model_u; torch.cuda.empty_cache()

    # ═══════════════════════════════════════════════════
    # 4. 2:4 Structured — dense format (still FP32)
    # ═══════════════════════════════════════════════════
    print(f"\n{'='*60}\n  4. 2:4 STRUCTURED (dense format, FP32)\n{'='*60}")
    model_24 = copy.deepcopy(model)
    p_24 = NetGPTPruner(model_24, metric=args.metric, sparsity=0.5)
    p_24.calibrate(calib_loader, args.device, args)
    apply_2_4_pruning(model_24, p_24)
    args.model = model_24
    res_24 = evaluate_with_metrics(args, devset)
    print(f"  2:4 Accuracy: {res_24['accuracy']:.4f}")
    results["2_4_dense_fp32"] = benchmark(
        model_24, src, seg, args.device, args.n_warmup, args.n_measure,
        "2:4 pruned (dense FP32)")
    del model_24; torch.cuda.empty_cache()

    # ═══════════════════════════════════════════════════
    # 5. 2:4 Semi-structured sparse (FP16 Tensor Core)
    # ═══════════════════════════════════════════════════
    if has_semi:
        print(f"\n{'='*60}\n  5. 2:4 SEMI-STRUCTURED SPARSE (FP16 TC)\n{'='*60}")
        model_sp = copy.deepcopy(model)
        p_sp = NetGPTPruner(model_sp, metric=args.metric, sparsity=0.5)
        p_sp.calibrate(calib_loader, args.device, args)
        apply_2_4_pruning(model_sp, p_sp)
        converted = convert_to_semi_structured(model_sp, verbose=False)
        print(f"  Converted {len(converted)} layers to semi-structured FP16")

        if converted:
            # Use autocast so embeddings (FP32) are cast to FP16 automatically
            results["2_4_sparse_fp16_tc"] = benchmark(
                model_sp, src, seg, args.device, args.n_warmup, args.n_measure,
                "2:4 sparse FP16 (Tensor Core)", use_fp16=True)

            # Evaluate with autocast
            model_sp.eval()
            args.model = model_sp
            try:
                # Manual eval with autocast
                subset_eval = devset[:min(256, len(devset))]
                src_e = torch.LongTensor([s[0] for s in subset_eval]).to(args.device)
                tgt_e = torch.LongTensor([s[1] for s in subset_eval]).to(args.device)
                seg_e = torch.LongTensor([s[2] for s in subset_eval]).to(args.device)
                correct = 0
                with torch.no_grad():
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        for src_b, tgt_b, seg_b, _ in batch_loader(32, src_e, tgt_e, seg_e):
                            _, logits = model_sp(src_b, None, seg_b)
                            pred = torch.argmax(logits, dim=1)
                            correct += (pred == tgt_b).sum().item()
                acc = correct / len(subset_eval)
                print(f"  Sparse TC Accuracy (256 samples): {acc:.4f}")
            except Exception as e:
                print(f"  Eval failed: {e}")
        else:
            print("  No layers converted — skipping")
        del model_sp; torch.cuda.empty_cache()

    # ═══════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════
    print(f"\n{'='*60}\n  SUMMARY\n{'='*60}")
    print(f"  {'Config':<35} {'Median':>10} {'vs Dense FP32':>14} {'vs Dense FP16':>14}")
    print(f"  {'-'*73}")

    d32 = results["dense_fp32"]["median_ms"]
    d16 = results.get("dense_fp16", {}).get("median_ms", d32)

    for name, res in results.items():
        med = res["median_ms"]
        sp32 = d32 / med if med > 0 else 0
        sp16 = d16 / med if med > 0 else 0
        marker = ""
        if "fp16" in name and sp16 > 1.1:
            marker = " <-- SPEEDUP"
        elif "fp16" not in name and sp32 > 1.1:
            marker = " <-- SPEEDUP"
        print(f"  {name:<35} {med:>8.1f} ms {sp32:>12.2f}x {sp16:>12.2f}x{marker}")

    out_path = os.path.join(args.output_dir, "sparse_benchmark.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults: {out_path}")
    print("Done.")


if __name__ == "__main__":
    main()
