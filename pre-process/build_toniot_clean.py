#!/usr/bin/env python3
# Dataset 4-classes PROPRE depuis les captures ToN-IoT labellisees :
# chaque flow route par son label JSON -> normal + attaques de la MEME capture.
import os, re, random, argparse, subprocess

T   = "/home2/public/datasets/toniot/toniot-pcap"
OUT = "/home2/public/rtesseyre/toniot_clean/pcap_multiclass"
CAPTURES = ["normal_DDoS_1","normal_DDoS_3","normal_DDoS_5","normal_DDoS_6",
            "normal_DDoS_7","normal_DDoS_8","normal_DDoS_9","normal_DDoS_10",
            "normal_DDoS_11","normal_DDoS_12"]
LABEL2CLASS = {"normal":"normal","ddos":"ddos","dos":"dos","scanning":"scanning"}  # autres labels ignores
CLASSES = ["ddos","dos","normal","scanning"]
LABEL_RE = re.compile(r'"label"\s*:\s*"([^"]*)"')

def scan(capdir):
    j = os.path.join(capdir, "all_flows.json")
    if not os.path.isfile(j): return None, "pas de all_flows.json"
    n_pcaps = sum(1 for f in os.listdir(capdir) if f.startswith("flow_") and f.endswith(".pcap"))
    items, idx, n_lab = [], 0, 0
    with open(j, "r", errors="replace") as f:
        for line in f:
            if '"label"' not in line: continue
            m = LABEL_RE.search(line)
            if not m: continue
            cls = LABEL2CLASS.get(m.group(1).strip().lower())
            if cls: items.append((capdir, idx, cls))
            idx += 1; n_lab += 1
    if n_lab != n_pcaps:
        return None, f"DESYNC n_labels={n_lab} != n_pcaps={n_pcaps} (saute)"
    return items, f"OK n={n_lab}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_class", type=int, default=0, help="0 = min dispo (equilibrage max)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--build", action="store_true", help="cree les symlinks (sinon budget seul)")
    a = ap.parse_args(); random.seed(a.seed)

    pool = {c: [] for c in CLASSES}
    print("== Scan captures labellisees ==")
    for cap in CAPTURES:
        items, msg = scan(os.path.join(T, cap))
        if items is None: print(f"  {cap}: {msg}"); continue
        per = {c:0 for c in CLASSES}
        for cd, idx, cls in items: pool[cls].append((cd, idx)); per[cls]+=1
        print(f"  {cap}: {msg} | " + " ".join(f"{c}={per[c]}" for c in CLASSES))

    print("\n== Budget total par classe ==")
    for c in CLASSES: print(f"  {c}: {len(pool[c])}")
    avail = min(len(pool[c]) for c in CLASSES)
    per_class = min(a.per_class, avail) if a.per_class > 0 else avail
    print(f"\n  -> minoritaire={avail} ; on prend {per_class}/classe ({per_class*4} total)")
    if not a.build:
        print("\n(budget seul — relance avec --build [--per_class N] pour creer les symlinks)"); return

    print("\n== Symlinks ==")
    for c in CLASSES:
        d = os.path.join(OUT, c); os.makedirs(d, exist_ok=True)
        for f in os.listdir(d):
            if f.endswith(".pcap"): os.unlink(os.path.join(d, f))
        cand = pool[c][:]; random.shuffle(cand); cand = cand[:per_class]
        for cd, idx in cand:
            os.symlink(os.path.join(cd, f"flow_{idx}.pcap"),
                       os.path.join(d, f"{os.path.basename(cd)}__flow_{idx}.pcap"))
        print(f"  {c}: {len(cand)} -> {d}")

    print("\n== Verif 5-tuple (echantillon) ==")
    for c in CLASSES:
        d = os.path.join(OUT, c)
        for f in [x for x in os.listdir(d) if x.endswith(".pcap")][:2]:
            real = os.path.realpath(os.path.join(d, f))
            out = subprocess.run(["tshark","-r",real,"-c1","-T","fields","-e","ip.src",
                "-e","tcp.srcport","-e","ip.dst","-e","tcp.dstport"],
                capture_output=True, text=True).stdout.strip()
            print(f"  [{c}] {f} -> {out}")

if __name__ == "__main__": main()
