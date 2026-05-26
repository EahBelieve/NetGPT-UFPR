"""
activation_pca.py — Analyse PCA des activations FFN du teacher NetGPT
======================================================================
Passe des données de calibration à travers le teacher, collecte les
activations intermédiaires FFN (post-GELU, dim 3072), et fait une PCA
pour mesurer la vraie dimensionnalité effective par couche.

Usage:
    python analysis/activation_pca.py \
        --model_path models/teacher_newds_local.bin \
        --config_path models/gpt2/config.json \
        --vocab_path models/encryptd_vocab.txt \
        --calib_path finetune_dataset_newds/train_dataset.tsv \
        --n_calib 200
"""

import sys, os, argparse, torch
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from uer.utils.constants import *
from uer.utils import *
from uer.utils.config import load_hyperparam
from uer.utils.seed import set_seed
from uer.opts import tokenizer_opts, model_opts
from finetune.run_understanding import Classifier, read_dataset


def collect_activations(model, dataset, n_calib, batch_size=16, device="cpu"):
    """
    Passe n_calib échantillons à travers le modèle et collecte
    les activations post-GELU de chaque couche FFN.
    
    Hook sur linear_2 (input = post-GELU activation, dim 3072).
    """
    # Préparer les hooks
    activations = {}
    hooks = []

    def make_hook(name):
        def hook_fn(module, input, output):
            # input[0] = entrée de linear_2 = sortie de GELU(linear_1(x))
            # Shape: [batch, seq_len, d_ff]
            act = input[0].detach().cpu()
            # Mean-pool sur la dimension séquence → [batch, d_ff]
            act_pooled = act.mean(dim=1)
            if name not in activations:
                activations[name] = []
            activations[name].append(act_pooled)
        return hook_fn

    # Enregistrer les hooks sur chaque linear_2 (FFN down)
    for name, module in model.named_modules():
        if "feed_forward.linear_2" in name and isinstance(module, torch.nn.Linear):
            layer_name = name.replace("encoder.transformer.", "L").replace(".feed_forward.linear_2", "")
            h = module.register_forward_hook(make_hook(layer_name))
            hooks.append(h)
            print(f"  Hook enregistré: {layer_name} ({name})")

    # Forward pass sur les données de calibration
    model.eval()
    model.to(device)

    src = torch.LongTensor([s[0] for s in dataset[:n_calib]])
    seg = torch.LongTensor([s[2] for s in dataset[:n_calib]])

    with torch.no_grad():
        for i in range(0, min(n_calib, len(src)), batch_size):
            src_batch = src[i:i+batch_size].to(device)
            seg_batch = seg[i:i+batch_size].to(device)
            # Forward pass (on ignore la sortie, on veut juste les hooks)
            emb = model.embedding(src_batch, seg_batch)
            model.encoder(emb, seg_batch)

    # Retirer les hooks
    for h in hooks:
        h.remove()

    # Concaténer les activations par couche
    result = {}
    for name, acts in activations.items():
        result[name] = torch.cat(acts, dim=0).numpy()  # [n_samples, d_ff]

    return result


def pca_analysis(activations_dict):
    """
    Pour chaque couche, fait une PCA sur les activations et mesure
    la dimensionnalité effective.
    """
    results = []

    for layer_name, acts in sorted(activations_dict.items()):
        n_samples, d_ff = acts.shape

        # Centrer les données
        acts_centered = acts - acts.mean(axis=0, keepdims=True)

        # Matrice de covariance (d_ff × d_ff)
        # Pour d_ff=3072, c'est ~72MB → faisable
        cov = np.cov(acts_centered, rowvar=False)  # [d_ff, d_ff]

        # Valeurs propres (triées par ordre décroissant)
        eigenvalues = np.linalg.eigvalsh(cov)[::-1]
        eigenvalues = np.maximum(eigenvalues, 0)  # éliminer les négatifs numériques

        # Énergie cumulée
        total_energy = eigenvalues.sum()
        if total_energy < 1e-10:
            print(f"  {layer_name}: activations quasi-nulles, skip")
            continue

        cumulative = np.cumsum(eigenvalues) / total_energy

        # Rangs effectifs à différents seuils
        rank_90 = np.searchsorted(cumulative, 0.90) + 1
        rank_95 = np.searchsorted(cumulative, 0.95) + 1
        rank_99 = np.searchsorted(cumulative, 0.99) + 1
        rank_999 = np.searchsorted(cumulative, 0.999) + 1

        # Rang effectif par entropie
        p = eigenvalues / eigenvalues.sum()
        p = p[p > 1e-10]
        H = -(p * np.log(p)).sum()
        erank = np.exp(H)

        # Nombre de dimensions "vivantes" (eigenvalue > 1% du max)
        threshold = eigenvalues[0] * 0.01
        n_alive = (eigenvalues > threshold).sum()

        results.append({
            "layer": layer_name,
            "d_ff": d_ff,
            "n_samples": n_samples,
            "rank_90": rank_90,
            "rank_95": rank_95,
            "rank_99": rank_99,
            "rank_999": rank_999,
            "erank": round(erank, 1),
            "n_alive": n_alive,
            "compress_90": round((1 - rank_90/d_ff) * 100, 1),
            "compress_95": round((1 - rank_95/d_ff) * 100, 1),
            "compress_99": round((1 - rank_99/d_ff) * 100, 1),
            "top_eigenvalues": eigenvalues[:20],
        })

    return results


def print_results(results):
    print("\n" + "=" * 100)
    print("  ANALYSE PCA DES ACTIVATIONS FFN — Teacher NetGPT")
    print("  (dim effective des 3072 neurones intermédiaires sur données réelles)")
    print("=" * 100)

    print(f"\n{'Couche':<8} {'d_ff':<6} {'R90':<6} {'R95':<6} {'R99':<7} {'R999':<7} "
          f"{'eRank':<8} {'Alive':<7} {'Comp90':<8} {'Comp95':<8} {'Comp99':<8}")
    print("-" * 100)

    for r in results:
        print(f"{r['layer']:<8} {r['d_ff']:<6} {r['rank_90']:<6} {r['rank_95']:<6} "
              f"{r['rank_99']:<7} {r['rank_999']:<7} {r['erank']:<8} {r['n_alive']:<7} "
              f"{r['compress_90']}%{'':<4} {r['compress_95']}%{'':<4} {r['compress_99']}%")

    # Moyennes
    print("-" * 100)
    avg_r90 = np.mean([r['rank_90'] for r in results])
    avg_r95 = np.mean([r['rank_95'] for r in results])
    avg_r99 = np.mean([r['rank_99'] for r in results])
    avg_erank = np.mean([r['erank'] for r in results])
    avg_alive = np.mean([r['n_alive'] for r in results])
    avg_c90 = np.mean([r['compress_90'] for r in results])
    avg_c95 = np.mean([r['compress_95'] for r in results])
    avg_c99 = np.mean([r['compress_99'] for r in results])
    print(f"{'MOYENNE':<8} {'':<6} {avg_r90:<6.0f} {avg_r95:<6.0f} "
          f"{avg_r99:<7.0f} {'':<7} {avg_erank:<8.1f} {avg_alive:<7.0f} "
          f"{avg_c90:.1f}%{'':<4} {avg_c95:.1f}%{'':<4} {avg_c99:.1f}%")

    # Comparaison avec les résultats empiriques
    print("\n" + "=" * 100)
    print("  COMPARAISON AVEC LES RÉSULTATS EMPIRIQUES")
    print("=" * 100)
    print(f"\n  SVD des poids:     compressibilité ~5%  (rang effectif ~727/768)")
    print(f"  Pruning Wanda:     compressibilité ~85% (sparsité globale)")
    print(f"  PCA activations:   compressibilité ~{avg_c95:.0f}% à 95% énergie (rang effectif ~{avg_r95:.0f}/{results[0]['d_ff']})")
    print(f"\n  Sweep empirique:")
    print(f"    d_ff=3072 → 96.88% (baseline)")
    print(f"    d_ff=2048 → 96.88% (compression {(1-2048/3072)*100:.0f}%)")
    print(f"    d_ff=1536 → 95.63%")
    print(f"    d_ff=1024 → 94.37% (compression {(1-1024/3072)*100:.0f}%)")
    print(f"\n  → La PCA prédit-elle le bon seuil ? Rang à 95% = {avg_r95:.0f} → d_ff minimum ≈ {avg_r95:.0f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--config_path", type=str, default="models/gpt2/config.json")
    parser.add_argument("--calib_path", type=str, required=True,
                        help="TSV dataset for calibration")
    parser.add_argument("--n_calib", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=16)
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Charger le modèle
    print("\n[1/4] Construction du modèle...")
    model = Classifier(args)

    print("[2/4] Chargement des poids du teacher...")
    model.load_state_dict(
        torch.load(args.model_path, map_location="cpu"), strict=False
    )

    # Charger les données de calibration
    print(f"[3/4] Chargement de {args.n_calib} échantillons de calibration...")
    dataset = read_dataset(args, args.calib_path)
    print(f"  Dataset: {len(dataset)} samples, using {min(args.n_calib, len(dataset))}")

    # Collecter les activations
    print("[4/4] Forward pass + collecte des activations...\n")
    activations = collect_activations(
        model, dataset, args.n_calib, args.batch_size, device
    )

    # Analyse PCA
    print("\nAnalyse PCA en cours...")
    results = pca_analysis(activations)
    print_results(results)


if __name__ == "__main__":
    main()
