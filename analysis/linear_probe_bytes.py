import argparse, numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import chi2

FIELD = {8:"IP ver/IHL",9:"IP total_len",10:"IP ID",11:"IP flags/frag",12:"TTL+proto",
         13:"IP cksum",14:"srcIP(0)",15:"srcIP(0)",16:"dstIP(0)",17:"dstIP(0)",
         18:"src port(0)",19:"dst port(0)",20:"seq",21:"seq",22:"ack",23:"ack",
         24:"TCP offset+flags",25:"TCP window",26:"TCP cksum",27:"urgent"}

def load(path, n):
    X, y = [], []
    with open(path) as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 2: continue
            y.append(int(p[0]))
            X.append({f"p{i}={t}": 1 for i, t in enumerate(p[1].split()[:n])})
    return X, np.array(y)

ap = argparse.ArgumentParser()
ap.add_argument("--train", default="finetune_dataset_toniot/train_dataset.tsv")
ap.add_argument("--test",  default="finetune_dataset_toniot/test_dataset.tsv")
ap.add_argument("--n_tokens", type=int, default=64)
a = ap.parse_args()

Xtr_d, ytr = load(a.train, a.n_tokens); Xte_d, yte = load(a.test, a.n_tokens)
vec = DictVectorizer(sparse=True)
Xtr = vec.fit_transform(Xtr_d); Xte = vec.transform(Xte_d)
print(f"train={Xtr.shape}  test={Xte.shape}")

lr = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
print(f"\n[1] LogReg sur bytes bruts : acc test = {lr.score(Xte, yte):.4f}")

dt = DecisionTreeClassifier(max_depth=4, random_state=42).fit(Xtr, ytr)
print(f"[2] Arbre profondeur 4    : acc test = {dt.score(Xte, yte):.4f}")

sc, _ = chi2(Xtr, ytr)
names = np.array(vec.get_feature_names_out() if hasattr(vec,"get_feature_names_out") else vec.get_feature_names())
print("\n[3] Top-25 (position=token) par chi2 — champ protocole probable :")
for i in np.argsort(sc)[::-1][:25]:
    f = names[i]; pos = int(f[1:f.index("=")])
    field = "SLL header" if pos < 8 else FIELD.get(pos, "options TCP / payload / pkt2")
    print(f"  {f:<20} chi2={sc[i]:>10.0f}   <- {field}")
