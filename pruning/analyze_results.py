"""
Analyze pruning experiment results from CSV.

Generates:
  - Summary table (mean ± std per metric × sparsity)
  - Accuracy vs sparsity plot
  - F1 vs sparsity plot
  - Retention bar chart
  - LaTeX table for publication

Usage:
    python pruning/analyze_results.py --csv results/pruning_exp1_paper_dataset.csv
"""

import argparse
import csv
import os
import sys
from collections import defaultdict


def load_csv(path):
    """Load CSV results into list of dicts."""
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        return list(reader)


def group_results(rows):
    """Group results by (metric, sparsity) and aggregate over seeds."""
    groups = defaultdict(list)
    for row in rows:
        key = (row["metric"], float(row["sparsity"]))
        groups[key].append(row)
    return groups


def mean_std(values):
    """Compute mean and std of a list of floats."""
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    mean = sum(values) / n
    if n == 1:
        return mean, 0.0
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, variance ** 0.5


def print_summary_table(groups):
    """Print a formatted summary table."""
    print("\n" + "=" * 90)
    print(f"{'Metric':<14} {'Sparsity':>8} {'Accuracy':>16} {'F1-macro':>16} "
          f"{'Retention%':>12} {'Seeds':>6}")
    print("-" * 90)
    
    # Sort by metric then sparsity
    for (metric, sparsity) in sorted(groups.keys()):
        rows = groups[(metric, sparsity)]
        
        accs = [float(r["pruned_accuracy"]) for r in rows]
        f1s = [float(r["pruned_f1_macro"]) for r in rows]
        rets = [float(r["retention_pct"]) for r in rows]
        
        acc_m, acc_s = mean_std(accs)
        f1_m, f1_s = mean_std(f1s)
        ret_m, ret_s = mean_std(rets)
        
        print(f"  {metric:<12} {sparsity*100:>6.0f}%  "
              f"{acc_m:.4f}±{acc_s:.4f}  "
              f"{f1_m:.4f}±{f1_s:.4f}  "
              f"{ret_m:>8.1f}±{ret_s:.1f}  "
              f"{len(rows):>5}")
    
    print("=" * 90)
    
    # Dense baseline (from first row)
    if groups:
        first_row = list(groups.values())[0][0]
        print(f"\n  Dense baseline: Accuracy={first_row['dense_accuracy']}, "
              f"F1-macro={first_row['dense_f1_macro']}")


def generate_latex_table(groups):
    """Generate a LaTeX table for the publication."""
    print("\n% LaTeX table for publication")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Post-training pruning results on NetGPT attack detection}")
    print("\\label{tab:pruning}")
    print("\\begin{tabular}{llccc}")
    print("\\toprule")
    print("Method & Sparsity & Accuracy & F1-macro & Retention (\\%) \\\\")
    print("\\midrule")
    
    # Dense row
    if groups:
        first_row = list(groups.values())[0][0]
        print(f"Dense & 0\\% & {float(first_row['dense_accuracy']):.4f} "
              f"& {float(first_row['dense_f1_macro']):.4f} & 100.0 \\\\")
        print("\\midrule")
    
    for (metric, sparsity) in sorted(groups.keys()):
        rows = groups[(metric, sparsity)]
        accs = [float(r["pruned_accuracy"]) for r in rows]
        f1s = [float(r["pruned_f1_macro"]) for r in rows]
        rets = [float(r["retention_pct"]) for r in rows]
        
        acc_m, acc_s = mean_std(accs)
        f1_m, f1_s = mean_std(f1s)
        ret_m, _ = mean_std(rets)
        
        metric_name = {"magnitude": "Magnitude", "wanda": "Wanda",
                       "pruner_zero": "Pruner-Zero"}.get(metric, metric)
        
        print(f"{metric_name} & {sparsity*100:.0f}\\% "
              f"& ${acc_m:.4f} \\pm {acc_s:.4f}$ "
              f"& ${f1_m:.4f} \\pm {f1_s:.4f}$ "
              f"& {ret_m:.1f} \\\\")
    
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")


def generate_plot_script(groups, output_dir):
    """Generate a matplotlib plotting script."""
    script = '''"""Auto-generated plot script for pruning results."""
import matplotlib.pyplot as plt
import numpy as np

# Data extracted from experiments
data = {
'''
    
    for (metric, sparsity) in sorted(groups.keys()):
        rows = groups[(metric, sparsity)]
        accs = [float(r["pruned_accuracy"]) for r in rows]
        f1s = [float(r["pruned_f1_macro"]) for r in rows]
        acc_m, acc_s = mean_std(accs)
        f1_m, f1_s = mean_std(f1s)
        script += f'    ("{metric}", {sparsity}): {{"acc": {acc_m:.6f}, "acc_std": {acc_s:.6f}, "f1": {f1_m:.6f}, "f1_std": {f1_s:.6f}}},\n'
    
    # Get dense baseline
    if groups:
        first_row = list(groups.values())[0][0]
        dense_acc = float(first_row["dense_accuracy"])
        dense_f1 = float(first_row["dense_f1_macro"])
    else:
        dense_acc, dense_f1 = 1.0, 1.0
    
    script += f'''}}

dense_acc = {dense_acc}
dense_f1 = {dense_f1}

metrics = ["magnitude", "wanda", "pruner_zero"]
labels = {{"magnitude": "Magnitude", "wanda": "Wanda", "pruner_zero": "Pruner-Zero"}}
colors = {{"magnitude": "#e74c3c", "wanda": "#3498db", "pruner_zero": "#2ecc71"}}
sparsities = sorted(set(s for _, s in data.keys()))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

for metric in metrics:
    sp = [s for s in sparsities]
    accs = [data[(metric, s)]["acc"] for s in sp]
    acc_stds = [data[(metric, s)]["acc_std"] for s in sp]
    f1s = [data[(metric, s)]["f1"] for s in sp]
    f1_stds = [data[(metric, s)]["f1_std"] for s in sp]
    
    sp_pct = [s * 100 for s in sp]
    
    ax1.errorbar(sp_pct, accs, yerr=acc_stds, marker="o", label=labels[metric],
                 color=colors[metric], capsize=3, linewidth=2)
    ax2.errorbar(sp_pct, f1s, yerr=f1_stds, marker="s", label=labels[metric],
                 color=colors[metric], capsize=3, linewidth=2)

# Dense baseline
ax1.axhline(y=dense_acc, color="gray", linestyle="--", alpha=0.7, label="Dense")
ax2.axhline(y=dense_f1, color="gray", linestyle="--", alpha=0.7, label="Dense")

ax1.set_xlabel("Sparsity (%)")
ax1.set_ylabel("Accuracy")
ax1.set_title("Accuracy vs Sparsity")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.set_xlabel("Sparsity (%)")
ax2.set_ylabel("F1-macro")
ax2.set_title("F1-macro vs Sparsity")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("{output_dir}/pruning_accuracy_f1.png", dpi=150, bbox_inches="tight")
plt.savefig("{output_dir}/pruning_accuracy_f1.pdf", bbox_inches="tight")
print("Plots saved to {output_dir}/")
plt.show()
'''.replace("{output_dir}", output_dir)
    
    script_path = os.path.join(output_dir, "plot_pruning.py")
    with open(script_path, "w") as f:
        f.write(script)
    print(f"Plot script saved to {script_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to results CSV")
    parser.add_argument("--output_dir", default="results",
                        help="Directory for output plots/tables")
    args = parser.parse_args()
    
    if not os.path.exists(args.csv):
        print(f"ERROR: CSV file not found: {args.csv}")
        sys.exit(1)
    
    rows = load_csv(args.csv)
    print(f"Loaded {len(rows)} experiment runs from {args.csv}")
    
    groups = group_results(rows)
    print(f"Grouped into {len(groups)} (metric, sparsity) configurations")
    
    print_summary_table(groups)
    generate_latex_table(groups)
    
    os.makedirs(args.output_dir, exist_ok=True)
    generate_plot_script(groups, args.output_dir)


if __name__ == "__main__":
    main()
