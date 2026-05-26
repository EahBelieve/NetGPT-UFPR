"""
svd_analysis.py — Analyse SVD des matrices FFN du teacher NetGPT
=================================================================
Calcule le rang effectif de chaque matrice FFN pour déterminer
la compressibilité structurelle réelle par couche.

Usage:
    python analysis/svd_analysis.py \
        --model_path models/teacher_newds_local.bin \
        --config_path models/gpt2/config.json \
        --vocab_path models/encryptd_vocab.txt
"""

import sys, os, argparse, torch, json
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from uer.utils.constants import *
from uer.utils import *
from uer.utils.config import load_hyperparam
from uer.utils.seed import set_seed
from uer.opts import tokenizer_opts, model_opts
from finetune.run_understanding import Classifier, load_or_initialize_parameters


def effective_rank_energy(S, threshold=0.99):
    """Rang effectif: nombre de valeurs singulières pour capturer `threshold` de l'énergie."""
    energy = torch.cumsum(S ** 2, dim=0) / (S ** 2).sum()
    rank = (energy < threshold).sum().item() + 1
    return rank


def effective_rank_entropy(S):
    """Rang effectif par entropie (Roy & Bhattacharya, 2007).
    erank(W) = exp(H(p)) où p_i = σ_i / Σσ_i et H = -Σ p_i log(p_i)
    Mesure combien de directions singulières contribuent significativement.
    """
    p = S / S.sum()
    p = p[p > 1e-10]  # éviter log(0)
    H = -(p * torch.log(p)).sum().item()
    return np.exp(H)


def analyze_model(model):
    """Analyse SVD de toutes les matrices FFN du modèle."""
    results = []

    for name, param in model.named_parameters():
        if "feed_forward" not in name or "weight" not in name:
            continue
        if param.dim() != 2:
            continue

        W = param.data.float()
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)

        # Rangs effectifs à différents seuils d'énergie
        rank_90 = effective_rank_energy(S, 0.90)
        rank_95 = effective_rank_energy(S, 0.95)
        rank_99 = effective_rank_energy(S, 0.99)
        rank_999 = effective_rank_energy(S, 0.999)
        erank = effective_rank_entropy(S)

        # Ratio de compressibilité
        max_rank = min(W.shape)
        compress_99 = 1.0 - (rank_99 / max_rank)

        # Top-k énergie cumulée
        total_energy = (S ** 2).sum().item()
        top10_energy = (S[:10] ** 2).sum().item() / total_energy
        top50_energy = (S[:50] ** 2).sum().item() / total_energy
        top100_energy = (S[:100] ** 2).sum().item() / total_energy

        result = {
            "name": name,
            "shape": list(W.shape),
            "max_rank": max_rank,
            "rank_90": rank_90,
            "rank_95": rank_95,
            "rank_99": rank_99,
            "rank_999": rank_999,
            "erank_entropy": round(erank, 1),
            "compress_99": round(compress_99 * 100, 1),
            "top10_energy": round(top10_energy * 100, 1),
            "top50_energy": round(top50_energy * 100, 1),
            "top100_energy": round(top100_energy * 100, 1),
            "singular_values": S.cpu().numpy(),
        }
        results.append(result)

    return results


def print_results(results):
    """Affichage formaté des résultats."""
    print("\n" + "=" * 90)
    print("  ANALYSE SVD — Matrices FFN du Teacher NetGPT")
    print("=" * 90)

    print(f"\n{'Couche':<50} {'Shape':<14} {'R90':<5} {'R95':<5} {'R99':<5} {'R999':<5} {'eRank':<7} {'Compress':<8}")
    print("-" * 90)

    for r in results:
        # Extraire le numéro de couche et le type (up/down)
        name_short = r["name"]
        name_short = name_short.replace("encoder.transformer.", "L")
        name_short = name_short.replace(".feed_forward.", " FFN ")
        name_short = name_short.replace(".weight", "")

        print(f"{name_short:<50} {str(r['shape']):<14} "
              f"{r['rank_90']:<5} {r['rank_95']:<5} {r['rank_99']:<5} {r['rank_999']:<5} "
              f"{r['erank_entropy']:<7} {r['compress_99']}%")

    # Résumé par type (linear_1 = up, linear_2 = down)
    print("\n" + "=" * 90)
    print("  RÉSUMÉ PAR TYPE")
    print("=" * 90)

    for layer_type, label in [("linear_1", "FFN Up (768→3072)"), ("linear_2", "FFN Down (3072→768)")]:
        subset = [r for r in results if layer_type in r["name"]]
        if subset:
            avg_rank99 = np.mean([r["rank_99"] for r in subset])
            avg_erank = np.mean([r["erank_entropy"] for r in subset])
            avg_compress = np.mean([r["compress_99"] for r in subset])
            print(f"\n  {label}:")
            print(f"    Rang effectif moyen (99% énergie): {avg_rank99:.0f} / {subset[0]['max_rank']}")
            print(f"    Rang effectif moyen (entropie):     {avg_erank:.0f}")
            print(f"    Compressibilité moyenne (99%):      {avg_compress:.1f}%")

    # Résumé global
    print("\n" + "=" * 90)
    print("  RECOMMANDATION d_ff")
    print("=" * 90)

    up_layers = [r for r in results if "linear_1" in r["name"]]
    if up_layers:
        # Le d_ff optimal est dicté par le rang de la matrice up (768→d_ff)
        max_rank99 = max(r["rank_99"] for r in up_layers)
        avg_rank99 = np.mean([r["rank_99"] for r in up_layers])
        min_rank99 = min(r["rank_99"] for r in up_layers)

        print(f"\n  Basé sur les matrices FFN Up (99% énergie):")
        print(f"    Rang max (couche la plus exigeante):  {max_rank99}")
        print(f"    Rang moyen:                           {avg_rank99:.0f}")
        print(f"    Rang min (couche la plus compressible): {min_rank99}")
        print(f"\n  → d_ff recommandé (sûr, basé sur max): {max_rank99}")
        print(f"  → d_ff recommandé (agressif, basé sur avg): {int(avg_rank99)}")
        print(f"\n  Rappel: d_ff original = 3072")
        print(f"  Résultat sweep empirique: d_ff=2048 → 96.88%, d_ff=1024 → 94.37%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--config_path", type=str, default="models/gpt2/config.json")
    # vocab_path already in tokenizer_opts
    tokenizer_opts(parser)
    model_opts(parser)

    args = parser.parse_args()
    args = load_hyperparam(args)
    args.labels_num = 2
    args.seq_length = 64
    args.soft_targets = False
    args.soft_alpha = 0.0
    args.dropout = 0.1
    args.tokenizer = str2tokenizer[args.tokenizer](args)

    # Charger le modèle
    print("[1/3] Construction du modèle...")
    model = Classifier(args)

    print("[2/3] Chargement des poids du teacher...")
    model.load_state_dict(
        torch.load(args.model_path, map_location="cpu"), strict=False
    )
    model.eval()

    print("[3/3] Analyse SVD des matrices FFN...")
    results = analyze_model(model)
    print_results(results)


if __name__ == "__main__":
    main()
