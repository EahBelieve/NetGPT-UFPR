#!/usr/bin/env python3
# ============================================================================
# analyze_phi0.py   —   Tests 2, 3, 4   (tourne PARTOUT : ne demande que numpy/sklearn)
#
# Prend le .npz produit par extract_phi0.py (sur t101) et calcule, sur le VRAI
# phi_0 de NetGPT, les grandeurs de l'axe complexite de marge :
#
#   TEST 2  — plancher de signal + largeur minimale reelle
#       (1) sonde lineaire SVM sur phi_0  -> le residuel (l'amont separe-t-il ?)
#       (2) rang LDA inter-classe r_sup (<= C-1)  vs  rang effectif r_tau (PCA des activations)
#       (3) courbe atteignable : FFN ENTRAINE etroit (MLP largeur m) -> largeur min reelle
#
#   TEST 3  — alpha mesure (lazy neurons) + borne signal x redondance
#       alpha_moyen, alpha_min, et la borne r_sup/alpha
#
#   TEST 4  — tail-utility probe : la queue de BASSE ENERGIE est-elle utile ?
#       ablation des k directions de plus basse energie  vs  k aleatoires
#       vs  k de plus basse discriminance. Si l'energie-basse fait PLUS mal que
#       l'aleatoire -> trier par energie supprime du signal (bruit utile).
#
# Usage :  python analyze_phi0.py phi0_layer0.npz
# ============================================================================

import sys
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(0)


def onehot(y, C):
    Y = np.full((len(y), C), -1.0); Y[np.arange(len(y)), y] = 1.0; return Y


def linear_probe(Phi, y):
    Phis = StandardScaler().fit_transform(Phi)
    clf = LinearSVC(C=1.0, max_iter=20000, dual='auto').fit(Phis, y)
    margins = 1.0 / np.linalg.norm(clf.coef_, axis=1)
    return clf.score(Phis, y), float(np.min(margins)), float(np.mean(margins))


def r_tau_from_activations(act_pool, tau=0.95):
    if act_pool.shape[0] < 3:
        return None
    A = act_pool - act_pool.mean(0)
    s = np.linalg.svd(A, compute_uv=False)
    e = s**2
    return int(np.searchsorted(np.cumsum(e) / e.sum(), tau) + 1)


def lda_rank(Phi, y, C):
    mu = Phi.mean(0)
    SB = np.zeros((Phi.shape[1], Phi.shape[1]))
    for c in range(C):
        Xc = Phi[y == c]; diff = (Xc.mean(0) - mu)[:, None]; SB += len(Xc) * (diff @ diff.T)
    ev = np.linalg.eigvalsh(SB)[::-1]
    ev = ev[ev > 0]
    # seuil RELATIF (le rang inter-classe est <= C-1 ; on coupe le bruit numerique)
    ev = ev[ev > ev.max() * 1e-6]
    ev = ev[:C - 1]                          # garde-fou : jamais plus de C-1
    return len(ev), (np.cumsum(ev) / ev.sum())


def achievable_trained(Phi, y, widths):
    Phis = StandardScaler().fit_transform(Phi)
    out = []
    for m in widths:
        clf = MLPClassifier(hidden_layer_sizes=(m,), activation='relu',
                            max_iter=1000, alpha=1e-4, random_state=0).fit(Phis, y)
        out.append((m, clf.score(Phis, y)))
    return out


def first_reaching(curve, target):
    for m, a in curve:
        if a >= target:
            return m
    return None


def tail_utility(Phi, y, C, fracs=(0.25, 0.5, 0.75, 0.9)):
    """Ablation d'une fraction de directions selon 3 criteres ; accuracy SVM apres ablation."""
    Phis = StandardScaler().fit_transform(Phi)
    n, d = Phis.shape
    # base PCA (min(n,d) composantes)
    U, S, Vt = np.linalg.svd(Phis - Phis.mean(0), full_matrices=False)
    scores = U * S                                    # coordonnees PCA (n x ncomp)
    ncomp = scores.shape[1]
    energy = S**2
    energy_order = np.argsort(energy)                 # croissant : [0]=plus basse energie
    # discriminance par composante (variance des labels captee)
    Yc = onehot(y, C) - onehot(y, C).mean(0)
    disc = np.empty(ncomp)
    for i in range(ncomp):
        pc = scores[:, i:i+1]; beta, *_ = np.linalg.lstsq(pc, Yc, rcond=None)
        disc[i] = np.sum((pc @ beta)**2)
    disc_order = np.argsort(disc)                     # croissant : [0]=moins discriminant

    def acc_keeping(keep_idx):
        F = scores[:, keep_idx]
        return LinearSVC(C=1.0, max_iter=20000, dual='auto').fit(F, y).score(F, y)

    rows = []
    full = acc_keeping(np.arange(ncomp))
    for f in fracs:
        k = min(int(f * ncomp), ncomp - 2)            # garde au moins 2 directions
        keep_lowE = energy_order[k:]                   # enleve k plus basse energie
        keep_rand = rng.permutation(ncomp)[k:]         # enleve k aleatoires
        keep_lowD = disc_order[k:]                      # enleve k moins discriminants
        rows.append((k, acc_keeping(keep_lowE), acc_keeping(keep_rand), acc_keeping(keep_lowD)))
    return full, rows


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "phi0.npz"
    d = np.load(path)
    Phi, y = d["phi0"], d["y"].astype(int)
    act_rate_tok = d["act_rate_tok"]
    act_pool = d["act_pool"]
    C = int(y.max() + 1)
    n, dim = Phi.shape
    print("=" * 74)
    print(f"{path}  |  n={n} flux, d={dim}, C={C} classes, "
          f"layer={int(d['layer_idx'])}")
    print("=" * 74)

    # ---- TEST 2 ----
    acc, mmin, mmean = linear_probe(Phi, y)
    print("\n[TEST 2] (1) PLANCHER LINEAIRE  (sonde SVM sur phi_0)")
    print(f"    accuracy lineaire         : {acc:.4f}")
    print(f"    marge SVM (min / moyenne) : {mmin:.4g} / {mmean:.4g}")
    print( "    -> accuracy ~1 + marge grande = l'amont separe deja (residuel faible)")

    r_tau = r_tau_from_activations(act_pool)
    rank, lda_e = lda_rank(Phi, y, C)
    print("\n[TEST 2] (2) PLANCHER DE SIGNAL")
    print(f"    r_tau (PCA des activations post-GELU, 95%) : {r_tau}")
    print(f"    rang LDA inter-classe r_sup (<= C-1)       : {rank}")
    print(f"    energie inter-classe cumulee               : {np.round(lda_e, 3)}")

    widths = [2, 3, 4, 8, 16, 32, 64, 128, 256]
    curve = achievable_trained(Phi, y, widths)
    w_min = first_reaching(curve, acc - 0.01)
    print("\n[TEST 2] (3) COURBE ATTEIGNABLE  (FFN entraine etroit sur phi_0)")
    print(f"    {'largeur m':>10} | {'accuracy':>9}")
    for m, a in curve:
        print(f"    {m:>10} | {a:>9.4f}")
    print(f"    -> LARGEUR MINIMALE reelle (acc >= sonde-1%) : m = {w_min}")
    print( "    -> a comparer a 256 (= plancher du sweep, PAS un minimum valide)")

    # ---- TEST 3 ----
    a_mean, a_min = float(act_rate_tok.mean()), float(act_rate_tok.min())
    print("\n[TEST 3] ALPHA mesure (lazy neurons)")
    print(f"    alpha_moyen : {a_mean:.4f}   alpha_min : {a_min:.4f}   "
          f"(heterogeneite {a_mean/max(a_min,1e-9):.1f}x)")
    print(f"    borne signal x redondance :  r_sup/alpha_moyen = {int(np.ceil(rank/a_mean))}"
          f"   |   r_sup/alpha_min = {int(np.ceil(rank/a_min))}")
    print( "    -> a confronter a la largeur minimale reelle (3) : la structure colle-t-elle ?")

    # ---- TEST 4 ----
    full, rows = tail_utility(Phi, y, C, fracs=(0.25, 0.5, 0.75, 0.9))
    print("\n[TEST 4] TAIL-UTILITY PROBE  (accuracy SVM apres ablation de k directions)")
    print(f"    accuracy sans ablation : {full:.4f}")
    print(f"    {'k enleve':>9} | {'basse energie':>13} | {'aleatoire':>10} | {'basse discr.':>12}")
    for k, e, r, di in rows:
        print(f"    {k:>9} | {e:>13.4f} | {r:>10.4f} | {di:>12.4f}")
    print( "    -> si 'basse energie' < 'aleatoire' : la queue d'energie porte du signal")
    print( "       (= trier par energie supprime du bruit UTILE -> justifie le /alpha)")

    print("\n" + "=" * 74)
    print("SYNTHESE")
    print(f"    residuel (acc lineaire)        : {acc:.4f}")
    print(f"    plancher de signal (r_sup)     : {rank}")
    print(f"    largeur minimale reelle        : {w_min}")
    print(f"    borne r_sup/alpha_moyen        : {int(np.ceil(rank/a_mean))}")
    print("=" * 74)


if __name__ == "__main__":
    main()
