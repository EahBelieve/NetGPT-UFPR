"""
Global pruning for NetGPT — per-layer contribution to total pruning.

Each layer's pruned weights shown as:
  - Local%   : % of that layer's own weights pruned
  - % modele : % of the TOTAL model weights pruned by this layer
  - % prune  : % of ALL pruned weights that came from this layer

Usage:
    python pruning/run_pruning_global.py \
        --pretrained_model_path teacher_newds.bin \
        --config_path models/gpt2/config.json \
        --vocab_path models/encryptd_vocab.txt \
        --train_path finetune_dataset_newds/train_dataset.tsv \
        --dev_path finetune_dataset_newds/valid_dataset.tsv \
        --test_path finetune_dataset_newds/test_dataset.tsv \
        --seq_length 64 --labels_num 2 --pooling mean \
        --metric pruner_zero --sparsity 0.5
"""

import sys, os, copy, csv, argparse
import torch
import torch.nn as nn
from collections import OrderedDict

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from uer.utils.constants import *
from uer.utils import *
from uer.utils.config import load_hyperparam
from uer.utils.seed import set_seed
from uer.utils.logging import init_logger
from uer.opts import finetune_opts, tokenizer_opts, adv_opts
from finetune.run_understanding import (
    Classifier, read_dataset, batch_loader, evaluate,
    load_or_initialize_parameters
)
from pruning.pruner import NetGPTPruner
from pruning.metrics import METRICS


def make_calibration_loader(dataset, n_calib, batch_size, seed=42):
    import random
    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    calib_data = [dataset[i] for i in indices[:n_calib]]
    src = torch.LongTensor([s[0] for s in calib_data])
    tgt = torch.LongTensor([s[1] for s in calib_data])
    seg = torch.LongTensor([s[2] for s in calib_data])
    batches = []
    for i in range(0, len(calib_data), batch_size):
        batches.append((src[i:i+batch_size], tgt[i:i+batch_size],
                        seg[i:i+batch_size], None))
    return batches


def prune_global(pruner):
    """GLOBAL pruning: single threshold across entire model."""
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

    idx = 0
    layer_info = OrderedDict()
    total_pruned = 0
    for name, module, shape in layer_meta:
        n = shape[0] * shape[1]
        layer_scores = all_cat[idx:idx + n].view(shape)
        mask = (layer_scores > threshold).float()
        module.weight.data *= mask
        pruner.masks[name] = mask
        zeros = (mask == 0).sum().item()
        layer_info[name] = {
            "shape": tuple(shape), "total": n,
            "pruned": zeros, "sparsity_local": zeros / n,
        }
        total_pruned += zeros
        idx += n

    # Compute global percentages
    for name in layer_info:
        layer_info[name]["pct_of_total_pruned"] = (
            layer_info[name]["pruned"] / total_pruned * 100
            if total_pruned > 0 else 0
        )
        layer_info[name]["pct_of_total_model"] = (
            layer_info[name]["pruned"] / total * 100
        )

    layer_info["__total__"] = {
        "total_params": total,
        "total_pruned": total_pruned,
        "global_sparsity": total_pruned / total * 100,
    }
    return layer_info


def classify_layer(name):
    if "self_attn.linear_layers.0" in name: return "Attn Q"
    elif "self_attn.linear_layers.1" in name: return "Attn K"
    elif "self_attn.linear_layers.2" in name: return "Attn V"
    elif "self_attn.final_linear" in name: return "Attn Out"
    elif "feed_forward" in name:
        return "FFN up" if "linear_1" in name else "FFN down"
    return "Other"


def get_layer_idx(name):
    parts = name.split(".")
    for i, part in enumerate(parts):
        if part == "transformer" and i + 1 < len(parts):
            try: return int(parts[i + 1])
            except ValueError: pass
    return -1


def shorten_name(name):
    s = name.replace("encoder.transformer.", "L")
    s = s.replace(".self_attn.linear_layers.", ".attn.")
    s = s.replace(".self_attn.final_linear", ".attn.out")
    s = s.replace(".feed_forward.linear_1", ".ffn.up")
    s = s.replace(".feed_forward.linear_2", ".ffn.down")
    return s


def print_results(layer_info, label=""):
    if not layer_info:
        return

    totals = layer_info.get("__total__", {})
    total_pruned = totals.get("total_pruned", 0)
    total_params = totals.get("total_params", 0)
    global_sparsity = totals.get("global_sparsity", 0)

    # ===== TABLE 1: Per-layer =====
    print(f"\n{'='*90}")
    print(f"  PRUNING GLOBAL — {label}")
    print(f"  Sparsity globale: {global_sparsity:.2f}% "
          f"({total_pruned:,}/{total_params:,} poids prunes)")
    print(f"{'='*90}")
    print(f"  {'Couche':<30} {'Type':<10} {'Taille':>10} "
          f"{'Local%':>8} {'%modele':>8} {'%prune':>8}")
    print(f"  {'-'*78}")

    for name, info in layer_info.items():
        if name == "__total__":
            continue
        short = shorten_name(name)
        cat = classify_layer(name)
        shape_str = f"{info['shape'][0]}x{info['shape'][1]}"
        local_sp = info['sparsity_local'] * 100
        pct_model = info['pct_of_total_model']
        pct_pruned = info['pct_of_total_pruned']

        print(f"  {short:<30} {cat:<10} {shape_str:>10} "
              f"{local_sp:>7.1f}% {pct_model:>7.2f}% {pct_pruned:>7.2f}%")

    print(f"  {'-'*78}")
    print(f"  {'TOTAL':<30} {'':10} {total_params:>10,} "
          f"{global_sparsity:>7.2f}% {global_sparsity:>7.2f}% {'100.00':>7}%")

    # ===== TABLE 2: By layer type =====
    print(f"\n{'='*70}")
    print(f"  PAR TYPE DE COUCHE")
    print(f"{'='*70}")
    print(f"  {'Type':<20} {'Nb':>4} {'Params':>12} {'Prunes':>12} "
          f"{'Local%':>8} {'%modele':>8}")
    print(f"  {'-'*66}")

    type_stats = OrderedDict()
    for name, info in layer_info.items():
        if name == "__total__":
            continue
        cat = classify_layer(name)
        if cat not in type_stats:
            type_stats[cat] = {"params": 0, "pruned": 0, "count": 0}
        type_stats[cat]["params"] += info["total"]
        type_stats[cat]["pruned"] += info["pruned"]
        type_stats[cat]["count"] += 1

    for cat in ["Attn Q", "Attn K", "Attn V", "Attn Out",
                "FFN up", "FFN down", "Other"]:
        if cat not in type_stats:
            continue
        ts = type_stats[cat]
        local_sp = ts["pruned"] / ts["params"] * 100 if ts["params"] > 0 else 0
        pct_model = ts["pruned"] / total_params * 100 if total_params > 0 else 0
        print(f"  {cat:<20} {ts['count']:>4} {ts['params']:>12,} "
              f"{ts['pruned']:>12,} {local_sp:>7.1f}% {pct_model:>7.2f}%")

    # Subtotals
    attn_p = sum(ts["params"] for c, ts in type_stats.items() if "Attn" in c)
    attn_z = sum(ts["pruned"] for c, ts in type_stats.items() if "Attn" in c)
    ffn_p = sum(ts["params"] for c, ts in type_stats.items() if "FFN" in c)
    ffn_z = sum(ts["pruned"] for c, ts in type_stats.items() if "FFN" in c)
    print(f"  {'-'*66}")
    print(f"  {'Attention total':<20} {'':>4} {attn_p:>12,} {attn_z:>12,} "
          f"{attn_z/attn_p*100 if attn_p else 0:>7.1f}% "
          f"{attn_z/total_params*100 if total_params else 0:>7.2f}%")
    print(f"  {'FFN total':<20} {'':>4} {ffn_p:>12,} {ffn_z:>12,} "
          f"{ffn_z/ffn_p*100 if ffn_p else 0:>7.1f}% "
          f"{ffn_z/total_params*100 if total_params else 0:>7.2f}%")

    # ===== TABLE 3: By transformer layer =====
    print(f"\n{'='*60}")
    print(f"  PAR COUCHE TRANSFORMER (L0-L11)")
    print(f"{'='*60}")
    print(f"  {'Layer':>7} {'Params':>10} {'Prunes':>10} "
          f"{'Local%':>8} {'%modele':>8}")
    print(f"  {'-'*46}")

    lg = OrderedDict()
    for name, info in layer_info.items():
        if name == "__total__":
            continue
        idx = get_layer_idx(name)
        if idx < 0:
            continue
        if idx not in lg:
            lg[idx] = {"params": 0, "pruned": 0}
        lg[idx]["params"] += info["total"]
        lg[idx]["pruned"] += info["pruned"]

    for idx in sorted(lg.keys()):
        g = lg[idx]
        local_sp = g["pruned"] / g["params"] * 100 if g["params"] > 0 else 0
        pct_model = g["pruned"] / total_params * 100 if total_params > 0 else 0
        bar = "#" * int(local_sp / 5) + "." * (20 - int(local_sp / 5))
        print(f"  L{idx:>5} {g['params']:>10,} {g['pruned']:>10,} "
              f"{local_sp:>7.1f}% {pct_model:>7.2f}% {bar}")


def main():
    parser = argparse.ArgumentParser()
    finetune_opts(parser)
    tokenizer_opts(parser)
    adv_opts(parser)
    parser.add_argument("--soft_targets", action="store_true", default=False)
    parser.add_argument("--soft_alpha", type=float, default=0.5)
    parser.add_argument("--labels_num", type=int, default=2)
    parser.add_argument("--metric", default="pruner_zero",
                        choices=list(METRICS.keys()))
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
    print(f"\nTrain: {len(trainset)}, Dev: {len(devset)}, "
          f"Test: {len(testset) if testset else 0}")

    # 1) Dense evaluation
    print("\n[1/3] Modele DENSE...")
    model.eval()
    dense_acc_full = evaluate(args, devset)
    if testset:
        dense_test = evaluate(args, testset)

    # 2) Global pruning
    print(f"\n[2/3] GLOBAL pruning: {args.metric}, {args.sparsity*100:.0f}%")
    calib_loader = make_calibration_loader(
        trainset, args.n_calib, args.batch_size, seed=args.seed)
    model_gl = copy.deepcopy(model)
    pruner = NetGPTPruner(model_gl, metric=args.metric,
                          sparsity=args.sparsity,
                          prune_output_layers=args.prune_output)
    pruner.calibrate(calib_loader, args.device, args)
    layer_info = prune_global(pruner)

    # 3) Print results with global percentages
    print_results(layer_info,
                  f"Global {args.metric} @ {args.sparsity*100:.0f}%")

    # 4) Pruned evaluation
    print(f"\n[3/3] Modele PRUNE...")
    args.model = model_gl
    pruned_acc_full = evaluate(args, devset)
    if testset:
        pruned_test = evaluate(args, testset)

    # Summary
    print(f"\n{'='*50}")
    dense_test = dense_test[0] if isinstance(dense_test, tuple) else dense_test
    pruned_test = pruned_test[0] if isinstance(pruned_test, tuple) else pruned_test
    dense_acc = dense_acc_full[0] if isinstance(dense_acc_full, tuple) else dense_acc_full
    pruned_acc = pruned_acc_full[0] if isinstance(pruned_acc_full, tuple) else pruned_acc_full

    print(f"  DENSE  : dev={dense_acc:.4f}" +
          (f"  test={dense_test:.4f}" if testset else ""))
    print(f"  PRUNED : dev={pruned_acc:.4f}" +
          (f"  test={pruned_test:.4f}" if testset else ""))
    retention = pruned_acc / dense_acc * 100 if dense_acc > 0 else 0
    print(f"  Retention: {retention:.1f}%")
    print(f"{'='*50}")

    # CSV
    csv_path = os.path.join(args.output_dir, "global_pruning_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer", "type", "shape", "params", "pruned",
                     "local_sparsity", "pct_of_model", "pct_of_total_pruned"])
        for name, info in layer_info.items():
            if name == "__total__":
                continue
            w.writerow([
                shorten_name(name), classify_layer(name),
                f"{info['shape'][0]}x{info['shape'][1]}",
                info["total"], info["pruned"],
                f"{info['sparsity_local']:.4f}",
                f"{info['pct_of_total_model']:.4f}",
                f"{info['pct_of_total_pruned']:.4f}",
            ])
    print(f"\nCSV: {csv_path}")


if __name__ == "__main__":
    main()
