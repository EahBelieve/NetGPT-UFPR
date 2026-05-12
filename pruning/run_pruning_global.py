"""
Global pruning for NetGPT — per-layer sparsity analysis.

Instead of enforcing the same sparsity on every row/layer,
this script uses a SINGLE global threshold across the entire model.
This reveals which layers are most/least redundant.

Usage:
    cd ~/NetGPT_work/
    python pruning/run_pruning_global.py \
        --pretrained_model_path models/finetuned_model.bin \
        --config_path models/gpt2/config.json \
        --vocab_path models/encryptd_vocab.txt \
        --train_path finetune_dataset/train_dataset.tsv \
        --dev_path finetune_dataset/valid_dataset.tsv \
        --test_path finetune_dataset/test_dataset.tsv \
        --seq_length 64 --labels_num 2 --batch_size 32 \
        --pooling mean --seed 42 \
        --metric wanda --sparsity 0.5 \
        --output_dir results/global_pruning
"""

import sys, os, copy, csv, argparse
import torch
import torch.nn as nn
import numpy as np

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
from pruning.run_pruning import (
    make_calibration_loader, evaluate_with_metrics, compute_model_size
)


def prune_global(pruner):
    """GLOBAL pruning: single threshold across entire model."""
    print(f"[Global] Computing scores for {len(pruner.linear_layers)} layers...")
    all_scores = []
    layer_meta = []
    for name, module in pruner.linear_layers.items():
        weight = module.weight.data
        kwargs = {}
        if name in pruner.activation_norms:
            kwargs["activation_norms"] = pruner.activation_norms[name]
        if name in pruner.gradients:
            kwargs["gradients"] = pruner.gradients[name]
        scores = pruner.metric_fn(weight, **kwargs)
        all_scores.append(scores.flatten())
        layer_meta.append((name, module, scores.shape))

    all_cat = torch.cat(all_scores)
    total = len(all_cat)
    k = int(total * pruner.sparsity)
    if k == 0:
        return {}
    threshold = torch.kthvalue(all_cat, k).values.item()
    print(f"[Global] Total: {total:,}, pruning {k:,} ({pruner.sparsity*100:.0f}%)")
    print(f"[Global] Threshold: {threshold:.6f}")

    idx = 0
    layer_sparsities = {}
    total_pruned = 0
    for name, module, shape in layer_meta:
        n = shape[0] * shape[1]
        layer_scores = all_cat[idx:idx + n].view(shape)
        mask = (layer_scores > threshold).float()
        module.weight.data *= mask
        pruner.masks[name] = mask
        zeros = (mask == 0).sum().item()
        layer_sparsities[name] = {
            "shape": tuple(shape), "total": n,
            "pruned": zeros, "sparsity": zeros / n,
        }
        total_pruned += zeros
        idx += n
    print(f"[Global] Effective sparsity: {total_pruned/total*100:.2f}%")
    return layer_sparsities


def classify_layer(name):
    if "self_attn.linear_layers.0" in name: return "Attention Q"
    elif "self_attn.linear_layers.1" in name: return "Attention K"
    elif "self_attn.linear_layers.2" in name: return "Attention V"
    elif "self_attn.final_linear" in name: return "Attention O"
    elif "feed_forward" in name:
        return "FFN up" if "linear_1" in name or "w_1" in name else "FFN down"
    elif "output_layer" in name: return "Output head"
    return "Other"


def get_transformer_layer_idx(name):
    parts = name.split(".")
    for i, part in enumerate(parts):
        if part == "transformer" and i + 1 < len(parts):
            try: return int(parts[i + 1])
            except ValueError: pass
    return -1


def print_analysis(layer_sparsities, label=""):
    if not layer_sparsities: return

    print(f"\n{'='*80}")
    print(f"  PER-LAYER SPARSITY — {label}")
    print(f"{'='*80}")
    print(f"  {'Layer':<50} {'Shape':>12} {'Sparsity':>10}")
    print(f"  {'-'*72}")
    for name, info in layer_sparsities.items():
        sp = info['sparsity'] * 100
        bar = "\u2588" * int(sp / 5) + "\u2591" * (20 - int(sp / 5))
        print(f"  {name:<50} {info['shape'][0]}x{info['shape'][1]:>4} {sp:>6.1f}% {bar}")

    print(f"\n{'='*80}")
    print(f"  GROUPED BY TYPE — {label}")
    print(f"{'='*80}")
    groups = {}
    for name, info in layer_sparsities.items():
        cat = classify_layer(name)
        if cat not in groups: groups[cat] = {"total": 0, "pruned": 0, "layers": 0}
        groups[cat]["total"] += info["total"]
        groups[cat]["pruned"] += info["pruned"]
        groups[cat]["layers"] += 1
    print(f"  {'Category':<25} {'Layers':>7} {'Params':>12} {'Sparsity':>10}")
    print(f"  {'-'*55}")
    for cat in ["Attention Q","Attention K","Attention V","Attention O","FFN up","FFN down","Output head","Other"]:
        if cat in groups:
            g = groups[cat]
            sp = g["pruned"] / g["total"] * 100 if g["total"] > 0 else 0
            print(f"  {cat:<25} {g['layers']:>7} {g['total']:>12,} {sp:>8.1f}%")

    print(f"\n{'='*80}")
    print(f"  GROUPED BY TRANSFORMER LAYER — {label}")
    print(f"{'='*80}")
    lg = {}
    for name, info in layer_sparsities.items():
        idx = get_transformer_layer_idx(name)
        if idx < 0: continue
        if idx not in lg: lg[idx] = {"total": 0, "pruned": 0}
        lg[idx]["total"] += info["total"]; lg[idx]["pruned"] += info["pruned"]
    print(f"  {'Layer':>7} {'Params':>12} {'Sparsity':>10}")
    print(f"  {'-'*30}")
    for idx in sorted(lg.keys()):
        g = lg[idx]
        sp = g["pruned"] / g["total"] * 100 if g["total"] > 0 else 0
        bar = "\u2588" * int(sp / 5) + "\u2591" * (20 - int(sp / 5))
        print(f"  L{idx:>5} {g['total']:>12,} {sp:>8.1f}% {bar}")


def main():
    parser = argparse.ArgumentParser(description="Global pruning + per-layer analysis")
    finetune_opts(parser); tokenizer_opts(parser); adv_opts(parser)
    parser.add_argument("--soft_targets", action="store_true", default=False)
    parser.add_argument("--soft_alpha", type=float, default=0.5)
    parser.add_argument("--labels_num", type=int, default=2)
    parser.add_argument("--metric", type=str, default="wanda",
                        choices=["magnitude", "wanda", "pruner_zero"])
    parser.add_argument("--sparsity", type=float, default=0.5)
    parser.add_argument("--n_calib", type=int, default=128)
    parser.add_argument("--prune_output", action="store_true")
    parser.add_argument("--output_dir", type=str, default="results/global_pruning")
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

    trainset = read_dataset(args, args.train_path)
    devset = read_dataset(args, args.dev_path)
    testset = read_dataset(args, args.test_path) if args.test_path else None
    print(f"Train: {len(trainset)}, Dev: {len(devset)}")

    calib_loader = make_calibration_loader(trainset, args.n_calib, args.batch_size, seed=args.seed)

    # === PER-ROW ===
    print(f"\n{'#'*80}\n  MODE 1: PER-ROW PRUNING\n{'#'*80}")
    model_pr = copy.deepcopy(model)
    pruner_pr = NetGPTPruner(model_pr, metric=args.metric, sparsity=args.sparsity,
                             prune_output_layers=args.prune_output)
    pruner_pr.calibrate(calib_loader, args.device, args)
    pruner_pr.prune()
    pr_stats = pruner_pr.get_stats()
    pr_sp = {n: {"shape": i["shape"], "total": i["total_params"],
                 "pruned": i["zero_params"], "sparsity": i["sparsity"]}
             for n, i in pr_stats.items() if n != "__global__"}
    print_analysis(pr_sp, f"per-row {args.metric} {args.sparsity*100:.0f}%")
    pr_res = evaluate_with_metrics(args, devset)
    print(f"\n  Per-row Accuracy: {pr_res['accuracy']:.4f}, F1: {pr_res['f1_macro']:.4f}")
    del model_pr; torch.cuda.empty_cache()

    # === GLOBAL ===
    print(f"\n{'#'*80}\n  MODE 2: GLOBAL PRUNING\n{'#'*80}")
    model_gl = copy.deepcopy(model)
    pruner_gl = NetGPTPruner(model_gl, metric=args.metric, sparsity=args.sparsity,
                             prune_output_layers=args.prune_output)
    pruner_gl.calibrate(calib_loader, args.device, args)
    gl_sp = prune_global(pruner_gl)
    print_analysis(gl_sp, f"global {args.metric} {args.sparsity*100:.0f}%")
    args.model = model_gl
    gl_res = evaluate_with_metrics(args, devset)
    print(f"\n  Global Accuracy: {gl_res['accuracy']:.4f}, F1: {gl_res['f1_macro']:.4f}")

    print(f"\n{'='*80}\n  COMPARISON\n{'='*80}")
    print(f"  {'Mode':<15} {'Accuracy':>10} {'F1-macro':>10}")
    print(f"  {'-'*35}")
    print(f"  {'Dense':<15} {'1.0000':>10} {'1.0000':>10}")
    print(f"  {'Per-row':<15} {pr_res['accuracy']:>10.4f} {pr_res['f1_macro']:>10.4f}")
    print(f"  {'Global':<15} {gl_res['accuracy']:>10.4f} {gl_res['f1_macro']:>10.4f}")

    csv_path = os.path.join(args.output_dir, "layer_sparsity.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer_name","transformer_idx","category","shape","total_params","perrow_sparsity","global_sparsity"])
        all_names = sorted(set(list(pr_sp.keys()) + list(gl_sp.keys())))
        for name in all_names:
            pr = pr_sp.get(name, {}); gl = gl_sp.get(name, {})
            shape = pr.get("shape", gl.get("shape", (0,0)))
            w.writerow([name, get_transformer_layer_idx(name), classify_layer(name),
                        f"{shape[0]}x{shape[1]}", pr.get("total", gl.get("total",0)),
                        f"{pr.get('sparsity',0):.4f}", f"{gl.get('sparsity',0):.4f}"])
    print(f"\nCSV: {csv_path}")

    if testset:
        args.model = model_gl
        t_res = evaluate_with_metrics(args, testset)
        print(f"  Global Test Acc: {t_res['accuracy']:.4f}, F1: {t_res['f1_macro']:.4f}")
    print("Done.")


if __name__ == "__main__":
    main()
