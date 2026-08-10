"""
activation_rate.py — Vérification empirique de la borne d_ff >= ceil(r_tau / alpha)
==================================================================================
Mesure, AU NIVEAU TOKEN (pas en pooling), pour chaque couche FFN du teacher :

  - le taux d'activation par token  alpha(x) = |S_eps(x)| / d_ff
        * version "une face"  : S_eps(x) = { j : a(x)_j > eps }   (matche la
          parcimonie type lazy-neuron ; a eps=0 c'est { pre-activation > 0 })
        * version "deux faces" : S_eps(x) = { j : |a(x)_j| > eps } (celle qu'on
          avait ecrite en Def 2.4 ; on verra qu'elle donne alpha ~ 1, donc a
          corriger)
  - la moyenne ᾱ, le minimum alpha_min, et des percentiles bas (p1, p5)
  - le rang effectif r_tau (PCA centree, comme activation_pca.py) aux seuils
    tau = 0.90 / 0.95 / 0.99, sur la MEME matrice d'activations par token
  - le minimum predit ceil(r_tau / alpha) pour alpha in {ᾱ, alpha_min, p1},
    confronte a la largeur empirique d_ff = 256 (NetGPT-Slim)

Cle de lecture : la borne du theoreme utilise alpha_min (entree la moins active).
Si l'activation est homogene entre tokens, alpha_min ~ ᾱ et 256 tient ;
sinon le minimum predit monte et 256 est a reconfronter.

Usage (identique a activation_pca.py) :
    python analysis/activation_rate.py \
        --model_path models/teacher_newds_local.bin \
        --config_path models/gpt2/config.json \
        --vocab_path models/encryptd_vocab.txt \
        --calib_path finetune_dataset_newds/train_dataset.tsv \
        --n_calib 128 --seq_length 64
"""

import sys, os, argparse, math, torch
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from uer.utils.constants import *
from uer.utils import *
from uer.utils.config import load_hyperparam
from uer.utils.seed import set_seed
from uer.opts import tokenizer_opts, model_opts
from finetune.run_understanding import Classifier, read_dataset


# ----------------------------------------------------------------------------- #
#  Collecte des activations post-GELU AU NIVEAU TOKEN (sans pooling)
# ----------------------------------------------------------------------------- #
def collect_token_activations(model, dataset, n_calib, seq_length,
                              batch_size=16, device="cpu", max_tokens=40000,
                              seed=42):
    """
    Hook sur feed_forward.linear_2 : input[0] = activation post-GELU [B, S, d_ff].
    On garde uniquement les tokens reels (seg != 0) et on empile en [n_tokens, d_ff].
    """
    activations = {}
    hooks = []

    def make_hook(name):
        def hook_fn(module, inp, out):
            act = inp[0].detach().to(torch.float32).cpu()   # [B, S, d_ff]
            activations.setdefault(name, []).append(act)
        return hook_fn

    for name, module in model.named_modules():
        if "feed_forward.linear_2" in name and isinstance(module, torch.nn.Linear):
            layer_name = (name.replace("encoder.transformer.", "L")
                              .replace(".feed_forward.linear_2", ""))
            hooks.append(module.register_forward_hook(make_hook(layer_name)))
            print(f"  Hook: {layer_name} ({name})")

    model.eval(); model.to(device)

    src = torch.LongTensor([s[0] for s in dataset[:n_calib]])
    seg = torch.LongTensor([s[2] for s in dataset[:n_calib]])
    seg_keep = []   # masque token reel, accumule batch par batch dans le meme ordre

    with torch.no_grad():
        for i in range(0, min(n_calib, len(src)), batch_size):
            src_b = src[i:i+batch_size].to(device)
            seg_b = seg[i:i+batch_size].to(device)
            emb = model.embedding(src_b, seg_b)
            model.encoder(emb, seg_b)
            seg_keep.append(seg_b.detach().cpu())

    for h in hooks:
        h.remove()

    seg_all = torch.cat(seg_keep, dim=0)            # [N, S]
    mask = (seg_all != 0).reshape(-1)               # [N*S]  tokens reels

    rng = np.random.default_rng(seed)
    result = {}
    for name, acts in activations.items():
        A = torch.cat(acts, dim=0).reshape(-1, acts[0].shape[-1])  # [N*S, d_ff]
        A = A[mask].numpy()                                        # tokens reels
        if A.shape[0] > max_tokens:                                # sous-echantillon
            idx = rng.choice(A.shape[0], size=max_tokens, replace=False)
            A = A[idx]
        result[name] = A
        print(f"  {name}: {A.shape[0]} tokens reels x {A.shape[1]} neurones")
    return result


# ----------------------------------------------------------------------------- #
#  Taux d'activation par token + rang effectif (PCA centree)
# ----------------------------------------------------------------------------- #
def activation_rates(A, eps_list):
    """Retourne, par token, les taux une-face (a>eps) et deux-faces (|a|>eps)."""
    d_ff = A.shape[1]
    out = {}
    # reference : strictement positif (a > 0), = pre-activation > 0 pour GELU
    out["pos>0"] = (A > 0).mean(axis=1)
    for eps in eps_list:
        out[f"a>{eps}"]   = (A > eps).mean(axis=1)
        out[f"|a|>{eps}"] = (np.abs(A) > eps).mean(axis=1)
    return out


def effective_rank(A, taus=(0.90, 0.95, 0.99)):
    """Rang effectif par PCA centree (energie = variance cumulee), cf Def 2.7."""
    Ac = A - A.mean(axis=0, keepdims=True)
    cov = np.cov(Ac, rowvar=False)
    ev = np.clip(np.linalg.eigvalsh(cov)[::-1], 0, None)
    tot = ev.sum()
    if tot < 1e-12:
        return {t: 0 for t in taus}
    cum = np.cumsum(ev) / tot
    return {t: int(np.searchsorted(cum, t) + 1) for t in taus}


def summarize(name, A, eps_list, taus, d_ff_target=256):
    rates = activation_rates(A, eps_list)
    ranks = effective_rank(A, taus)

    print(f"\n--- {name}  ({A.shape[0]} tokens, d_ff={A.shape[1]}) ---")
    print(f"  echelle activation : std={A.std():.3f}, "
          f"|a| median={np.median(np.abs(A)):.3f}, max={A.max():.2f}")

    print(f"  {'definition':<14}{'ᾱ (moy)':>10}{'alpha_min':>11}"
          f"{'p1':>8}{'p5':>8}")
    for key, alpha_t in rates.items():
        print(f"  {key:<14}{alpha_t.mean():>10.4f}{alpha_t.min():>11.4f}"
              f"{np.percentile(alpha_t,1):>8.4f}{np.percentile(alpha_t,5):>8.4f}")

    print(f"  rang effectif : " +
          "  ".join(f"r{int(t*100)}={ranks[t]}" for t in taus))

    # Borne predite avec la definition une-face a>0 (celle qui matche ~0.146)
    alpha = rates["pos>0"]
    r95 = ranks.get(0.95, ranks[list(ranks)[0]])
    abar, amin, ap1 = alpha.mean(), alpha.min(), np.percentile(alpha, 1)
    print(f"  >> borne (def a>0, tau=95%, r95={r95}) :")
    print(f"       ceil(r95/ᾱ)      = ceil({r95}/{abar:.4f})  = {math.ceil(r95/abar)}")
    print(f"       ceil(r95/p1)     = ceil({r95}/{ap1:.4f})  = {math.ceil(r95/ap1)}")
    print(f"       ceil(r95/a_min)  = ceil({r95}/{amin:.4f})  = {math.ceil(r95/amin)}"
          f"   {'<= 256 OK' if math.ceil(r95/amin) <= d_ff_target else '> 256 (256 a reconfronter)'}")
    return {"name": name, "r95": r95, "abar": abar, "amin": amin, "ap1": ap1}


# ----------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--config_path", type=str, default="models/gpt2/config.json")
    p.add_argument("--calib_path", type=str, required=True)
    p.add_argument("--n_calib", type=int, default=128)
    p.add_argument("--seq_length", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_tokens", type=int, default=40000)
    p.add_argument("--eps", type=float, nargs="+", default=[0.01, 0.05, 0.1])
    p.add_argument("--d_ff_target", type=int, default=256)
    tokenizer_opts(p)
    model_opts(p)

    args = p.parse_args()
    args = load_hyperparam(args)
    args.labels_num = 2
    args.soft_targets = False
    args.soft_alpha = 0.0
    args.dropout = 0.1
    args.tokenizer = str2tokenizer[args.tokenizer](args)
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("\n[1/4] Construction du modele...")
    model = Classifier(args)
    print("[2/4] Chargement des poids du teacher...")
    model.load_state_dict(torch.load(args.model_path, map_location="cpu"),
                          strict=False)

    print(f"[3/4] Chargement de {args.n_calib} echantillons (seq_length={args.seq_length})...")
    dataset = read_dataset(args, args.calib_path)
    print(f"  Dataset: {len(dataset)} samples")

    print("[4/4] Forward + collecte des activations par token...\n")
    acts = collect_token_activations(model, dataset, args.n_calib,
                                     args.seq_length, args.batch_size,
                                     device, args.max_tokens)

    taus = (0.90, 0.95, 0.99)
    print("\n" + "=" * 90)
    print("  TAUX D'ACTIVATION PAR TOKEN ET RANG EFFECTIF — par couche")
    print("=" * 90)
    rows = [summarize(name, A, args.eps, taus, args.d_ff_target)
            for name, A in sorted(acts.items())]

    # bilan global
    print("\n" + "=" * 90)
    print("  BILAN (definition a>0)")
    print("=" * 90)
    abar = np.mean([r["abar"] for r in rows])
    amin = np.min([r["amin"] for r in rows])
    r95m = np.mean([r["r95"] for r in rows])
    print(f"  ᾱ moyen toutes couches      = {abar:.4f}")
    print(f"  alpha_min global            = {amin:.4f}")
    print(f"  r95 moyen                   = {r95m:.1f}")
    print(f"  ceil(r95_moy / ᾱ)           = {math.ceil(r95m/abar)}")
    print(f"  ceil(r95_moy / alpha_min)   = {math.ceil(r95m/amin)}")
    print(f"  -> largeur empirique d_ff   = {args.d_ff_target}")
    print(f"  -> homogeneite alpha (ᾱ/alpha_min) = {abar/amin:.2f}  "
          f"({'homogene, 256 coherent' if abar/amin < 1.5 else 'heterogene, 256 a reconfronter'})")


if __name__ == "__main__":
    main()
