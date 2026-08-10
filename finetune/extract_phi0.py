#!/usr/bin/env python3
# ============================================================================
# extract_phi0.py   —   A LANCER SUR t101 (Docker netgpt:conda-gpu2)
#
# Charge le modele FINE-TUNE, hooke le FFN de la couche choisie, et sauvegarde
# dans un .npz :
#     phi0          (N, d)      = entree du FFN, poolee par flux (mean, padding exclu)
#     y             (N,)        = labels 0..3
#     act_rate_tok  (Ntok,)     = taux d'activation post-GELU par token (pour alpha)
#     act_pool      (<=K, d_ff) = sous-echantillon d'activations post-GELU (pour r_tau)
#
# Le .npz se transfere ensuite en local et s'analyse avec analyze_phi0.py
# (Tests 2/3/4). Aucune dependance lourde cote analyse (numpy/sklearn seulement).
#
# PLACER CE FICHIER DANS LE MEME DOSSIER QUE run_understanding.py (finetune/).
#
# Exemple (NetGPT-Slim, 1 couche, d_ff=256, modele fine-tune) :
#   python extract_phi0.py \
#       --config_path models/gpt2/slim_final_config.json \
#       --vocab_path models/encryptd_vocab.txt \
#       --pretrained_model_path <chemin_du_modele_finetune_slim>.bin \
#       --train_path flows_multiclass/train.tsv \
#       --tokenizer space --pooling mean --seq_length 64 \
#       --labels_num 4 --seed 42 --batch_size 32 \
#       --layer_idx 0 --save_path phi0_slim_L0.npz
#
# Pour le teacher (12 couches) : --config_path models/gpt2/config.json,
#   modele teacher fine-tune, et --layer_idx parmi 0..11 (relancer par couche).
# ============================================================================

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import numpy as np
import torch
import torch.nn as nn

from uer.utils import *
from uer.utils.constants import *
from uer.utils.config import load_hyperparam
from uer.utils.seed import set_seed
from uer.opts import finetune_opts, tokenizer_opts

# Reutilise EXACTEMENT le modele et la lecture de donnees du pipeline existant
from run_understanding import Classifier, read_dataset, batch_loader


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    finetune_opts(parser)
    tokenizer_opts(parser)
    parser.add_argument("--soft_targets", action='store_true')
    parser.add_argument("--soft_alpha", type=float, default=0.5)
    parser.add_argument("--labels_num", type=int, default=4)
    parser.add_argument("--layer_idx", type=int, default=0,
                        help="indice de la couche transformer dont on hooke le FFN")
    parser.add_argument("--act_subsample", type=int, default=6000,
                        help="nb max de tokens dont on garde l'activation complete (pour r_tau)")
    parser.add_argument("--save_path", type=str, default="phi0.npz")
    args = parser.parse_args()

    args = load_hyperparam(args)
    args.tokenizer = str2tokenizer[args.tokenizer](args)
    set_seed(args.seed)

    # --- modele + chargement du FINE-TUNE ---
    model = Classifier(args)
    state = torch.load(args.pretrained_model_path, map_location="cpu")
    missing = model.load_state_dict(state, strict=False)
    print(f"[load] missing={len(missing.missing_keys)} unexpected={len(missing.unexpected_keys)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    # --- localiser le FFN de la couche choisie ---
    enc = model.encoder
    if isinstance(enc.transformer, nn.ModuleList):
        ffn = enc.transformer[args.layer_idx].feed_forward
    else:  # parameter sharing (1 couche partagee) : un seul FFN
        ffn = enc.transformer.feed_forward
    d_ff = ffn.linear_1.out_features
    print(f"[hook] couche {args.layer_idx}, d_ff={d_ff}")

    captured = {}
    def hook(module, inp, out):
        x = inp[0].detach()                 # [B,T,d]  entree du FFN = phi_0 par token
        z = module.linear_1(x)              # [B,T,d_ff]  pre-activation
        a = module.act(z).detach()          # [B,T,d_ff]  post-GELU
        captured["x"], captured["a"] = x, a
    handle = ffn.register_forward_hook(hook)

    # --- donnees (train) ---
    dataset = read_dataset(args, args.train_path)
    src = torch.LongTensor([s[0] for s in dataset])
    tgt = torch.LongTensor([s[1] for s in dataset])
    seg = torch.LongTensor([s[2] for s in dataset])

    phi0_list, y_list, rate_list, act_pool = [], [], [], []
    n_act = 0
    with torch.no_grad():
        for src_b, tgt_b, seg_b, _ in batch_loader(args.batch_size, src, tgt, seg):
            src_b, seg_b = src_b.to(device), seg_b.to(device)
            _ = model(src_b, None, seg_b)            # tgt=None -> (None, logits) ; declenche le hook
            x, a = captured["x"], captured["a"]       # [B,T,d], [B,T,d_ff]
            mask = (seg_b > 0)                         # [B,T] tokens reels (padding exclu)
            maskf = mask.float().unsqueeze(-1)         # [B,T,1]
            denom = maskf.sum(1).clamp(min=1.0)
            phi0 = (x * maskf).sum(1) / denom          # [B,d]  mean pooling sur tokens reels
            phi0_list.append(phi0.cpu().numpy())
            y_list.append(tgt_b.numpy())
            a_real = a[mask]                           # [Ntok_reels, d_ff]
            rate_list.append((a_real > 0).float().mean(1).cpu().numpy())  # alpha par token
            if n_act < args.act_subsample:             # sous-echantillon pour r_tau
                take = min(args.act_subsample - n_act, a_real.shape[0])
                idx = torch.randperm(a_real.shape[0])[:take]
                act_pool.append(a_real[idx].cpu().numpy())
                n_act += take
    handle.remove()

    phi0 = np.concatenate(phi0_list)
    y = np.concatenate(y_list).astype(np.int64)
    act_rate_tok = np.concatenate(rate_list)
    act_pool = np.concatenate(act_pool) if act_pool else np.zeros((0, d_ff), np.float32)

    np.savez_compressed(args.save_path, phi0=phi0, y=y,
                        act_rate_tok=act_rate_tok, act_pool=act_pool,
                        layer_idx=args.layer_idx)
    print(f"[saved] {args.save_path}")
    print(f"        phi0={phi0.shape}  y={y.shape}  act_pool={act_pool.shape}")
    print(f"        alpha_mean={act_rate_tok.mean():.4f}  alpha_min={act_rate_tok.min():.4f}"
          f"  (attendu ~0.15 / ~0.017)")


if __name__ == "__main__":
    main()
