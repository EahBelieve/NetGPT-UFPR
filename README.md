# NetGPT-UFPR — Audit-First Compression of a Pretrained Network Traffic Transformer

> **Probe the bytes before pretraining on them.**
> How much of NetGPT does 4-class attack detection require? The answer forces
> a prior question: what does the benchmark actually measure?

## Overview

This repository contains the code, configurations, logs, and analysis scripts
for **NetGPT-Slim**, a compressed version of
[NetGPT](https://arxiv.org/abs/2304.09513) (a GPT-2 model pretrained on raw
network traffic) built for multi-class attack detection under an
**audit-first** methodology: no neural experiment is interpreted before the
task itself has been audited for information leakage with interpretable
probes (a position-indexed logistic regression, a depth-4 decision tree, and
a chi-squared feature ranking on raw byte tokens).

The audit exposed the binary benchmark inherited with the NetGPT codebase as
decided by a single capture artifact (a Marvell EDSA switch tag, EtherType
`0xDADA`, at a fixed byte position): every model, from a depth-4 tree to a
one-neuron Transformer, scores 100%. We retired it and built
**ToN-IoT-Clean**, a leakage-controlled 4-class dataset (DDoS / DoS / normal
/ scanning; 20,000 flows) by per-flow label routing within a single
[ToN-IoT](https://research.unsw.edu.au/projects/toniot-datasets) capture
session, so that no capture-environment shortcut separates the classes.

### Key results (ToN-IoT-Clean, test n=2,000)

| Model | Layers / heads / d_ff | Encoder params | Accuracy | Macro F1 | Fine-tuning |
|-------|----------------------|----------------|----------|----------|-------------|
| Baseline model | 12 / 12 / 3072 | 84.9M | 91.30% | 0.912 | 14 m 33 s |
| **NetGPT-Slim** | **1 / 12 / 256** | **2.75M (~31x less)** | **91.35%** | **0.913** | **2 m 08 s (~6.8x faster)** |

- **The task, not the model, sets the ceiling**: 23 configurations spanning a
  ~36x encoder-parameter range, pretrained or from scratch, land in a
  0.95-point band that a raw-byte logistic regression matches at 91.40%.
- **The computation lives upstream of the FFN**: a linear probe on the
  pre-FFN representation reaches 87.75% (rank-3 between-class subspace, one
  dominant axis); a one-neuron FFN recovers 90.80%. Width tracks the rank of
  the discriminative residual, not capacity estimates.
- **Two failure modes of pretrained-weight reuse** (documented with
  mechanisms): *head boundary scrambling* (changing the head count collapses
  accuracy to chance despite identical tensor shapes) and *truncation
  toxicity* (truncated fine-tuned FFN weights underperform random
  initialization).
- **Pruning cliff with a random baseline**: all informed metrics
  (Magnitude, Wanda, Pruner-Zero) hold ~91% up to 60% sparsity and collapse
  across 70–80%; random pruning is at chance by 50% sparsity.

> Note on terminology: the full fine-tuned model is called the **baseline
> model** (not "teacher"): no knowledge distillation is used in this work.

## Built With

- **[UER-py](https://github.com/dbiir/UER-py)** — PyTorch toolkit whose GPT-2
  implementation underlies NetGPT (the `uer/` directory).
- **[NetGPT](https://arxiv.org/abs/2304.09513)** — GPT-2 foundation model
  treating traffic as a language (hex tokens, `[pck]` packet delimiters).
- **[Wanda](https://github.com/locuslab/wanda)** and
  **[Pruner-Zero](https://github.com/pprp/Pruner-Zero)** — post-training
  pruning metrics evaluated here against Magnitude and a random baseline.
- **[ToN-IoT](https://research.unsw.edu.au/projects/toniot-datasets)** —
  network captures with per-flow ground-truth labels, the basis of
  ToN-IoT-Clean.

## Repository Structure

```
NetGPT-UFPR/
├── assets/
│   └── pretrained_model.bin        # Pretrained NetGPT checkpoint (not in repo)
├── configs/                        # All swept configurations (audit-first study)
│   ├── teacher.json                # Baseline model (12L/12H/3072)
│   ├── slim.json                   # NetGPT-Slim (1L/12H/256)
│   ├── depth_{1..6}.json           # Depth sweep
│   ├── w1L_{1,2,4,...,256}.json    # Sub-256 width sweep (1 layer)
│   ├── width_{512..3072}.json      # 6-layer width sweep
│   └── scramble.json               # Head boundary scrambling (h=8)
├── gen_configs.py                  # Generates the sweep configs
├── pre-process/
│   ├── build_toniot_clean.py       # ToN-IoT-Clean construction: per-flow label
│   │                               #   routing within one capture session,
│   │                               #   balanced subsampling (5,000/class, seed 42)
│   └── input_generation_understanding.py  # PCAP -> hex TSV (train/valid/test)
├── analysis/
│   ├── linear_probe_bytes.py       # Stage-0 audit probes: LogReg, depth-4 tree,
│   │                               #   chi2 ranking on raw byte tokens
│   ├── extract_phi0.py             # Hook: pre-FFN pooled representation phi_0
│   ├── analyze_phi0.py             # Linear probe on phi_0, LDA rank, residual
│   ├── activation_rate.py          # GELU activation rate (alpha)
│   ├── svd_analysis.py             # Effective rank of FFN weights
│   └── activation_pca.py           # PCA of post-GELU activations
├── finetune/
│   └── run_understanding.py        # Fine-tuning with shape-aware loading
├── pruning/
│   ├── metrics.py                  # Magnitude, Wanda, Pruner-Zero + Random baseline
│   ├── run_pruning_global.py       # Global pruning (non-uniform per-layer sparsity)
│   └── run_pruning.py              # Per-layer pruning
├── sweep_sub256.sh                 # Sub-256 width sweep driver
├── logs/toniot/                    # Logs of every run reported in the paper
└── results/                        # Pruning CSVs, benchmark JSONs
```

## Requirements

| Dependency | Version |
|------------|---------|
| Python | 3.8+ |
| PyTorch | >= 2.0 |
| CUDA | >= 11.8 |
| GPU | a single consumer GPU suffices (experiments ran on an RTX 4060) |

```bash
git clone https://github.com/EahBelieve/NetGPT-UFPR.git
cd NetGPT-UFPR
conda create -n netgpt python=3.8 -y && conda activate netgpt
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install six packaging psutil scikit-learn pandas scipy
```

The pretrained NetGPT checkpoint (`assets/pretrained_model.bin`) is required
for all experiments; it is not included (size). Refer to the
[NetGPT paper](https://arxiv.org/abs/2304.09513) for access.

## Pipeline (audit-first)

All fine-tuning runs share one protocol: **4 epochs, batch size 16,
`seq_length` 64, learning rate 2e-5 (AdamW), mean pooling, seed 42,
shape-aware loading** (tensors with matching shapes are copied from the
checkpoint, the rest are randomly initialized; omitted entirely for
from-scratch runs).

### Stage 0 — Byte-level task audit

Before any neural training, probe the task on raw byte tokens:

```bash
python analysis/linear_probe_bytes.py \
  --train finetune_dataset_toniot/train_dataset.tsv \
  --test  finetune_dataset_toniot/test_dataset.tsv
```

If a single feature saturates a probe, the benchmark is leaked and must be
retired from semantic claims. Otherwise the probe accuracy is the linear
floor that any architectural claim must exceed.

### Stage 1 — Leakage-controlled dataset (ToN-IoT-Clean)

```bash
# Route every flow of one labeled ToN-IoT capture session by its per-flow
# label (benign included), verify 5-tuple alignment, balance 5,000/class:
python pre-process/build_toniot_clean.py --build --per_class 5000

# Hex-tokenize into train/valid/test TSVs (80/10/10):
python pre-process/input_generation_understanding.py \
  --pcap_path <out>/pcap_multiclass/ \
  --dataset_dir finetune_dataset_toniot/ \
  --middle_save_path <out>/middle/ \
  --class_num 4 --random_seed 42
```

### Stage 2 — Baseline, Slim, from-scratch control, sweeps

```bash
# Baseline model (12L/12H/3072):
python finetune/run_understanding.py \
  --pretrained_model_path assets/pretrained_model.bin \
  --vocab_path models/encryptd_vocab.txt \
  --config_path configs/teacher.json \
  --train_path finetune_dataset_toniot/train_dataset.tsv \
  --dev_path   finetune_dataset_toniot/valid_dataset.tsv \
  --test_path  finetune_dataset_toniot/test_dataset.tsv \
  --epochs_num 4 --batch_size 16 --seq_length 64 --labels_num 4 \
  --learning_rate 2e-5 --pooling mean --seed 42 \
  --output_model_path models/teacher_toniot.bin

# NetGPT-Slim: same command with --config_path configs/slim.json
# From-scratch control: same as Slim but WITHOUT --pretrained_model_path
# Sweeps: loop over configs/depth_*.json, configs/w1L_*.json,
#         configs/width_*.json (see sweep_sub256.sh and logs/toniot/)
```

### Stage 3 — Localization of the discriminative computation

```bash
python analysis/extract_phi0.py \
  --pretrained_model_path models/teacher_toniot.bin \
  --config_path configs/teacher.json \
  --vocab_path models/encryptd_vocab.txt \
  --train_path finetune_dataset_toniot/train_dataset.tsv \
  --test_path  finetune_dataset_toniot/test_dataset.tsv \
  --seq_length 64 --batch_size 16 --seed 42
python analysis/analyze_phi0.py
```

### Stage 4 — Global pruning with four metrics (incl. random baseline)

```bash
for M in magnitude wanda pruner_zero random; do
  for S in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.72 0.74 0.76 0.78 0.8 0.9; do
    python pruning/run_pruning_global.py \
      --pretrained_model_path models/slim_toniot.bin \
      --config_path configs/slim.json \
      --vocab_path models/encryptd_vocab.txt \
      --train_path finetune_dataset_toniot/train_dataset.tsv \
      --dev_path   finetune_dataset_toniot/valid_dataset.tsv \
      --test_path  finetune_dataset_toniot/test_dataset.tsv \
      --labels_num 4 --pooling mean --seq_length 64 --batch_size 16 \
      --seed 42 --metric $M --sparsity $S --n_calib 128 \
      --output_dir results/pruning_toniot
  done
done
```

Calibration (Wanda activations, Pruner-Zero gradients) uses 128 samples from
the training split only.

## Important Notes

- **Never change the head count under pretrained-weight reuse.** With
  `hidden_size=768`, going from 12 to 8 heads keeps every tensor shape
  identical but re-partitions the learned 64-dim head sub-spaces into
  incoherent 96-dim slices; accuracy collapses to chance
  (`configs/scramble.json` reproduces this).
- **Never truncate a fine-tuned FFN to initialize a narrower one.**
  Truncated weights underperform random initialization (truncation
  toxicity); NetGPT-Slim re-initializes the narrowed FFN and inherits only
  attention and embeddings (shape-aware loading).
- **Always use `--pooling mean`.** The pretrained model expects mean pooling;
  the default `first` pooling collapses accuracy.
- **The legacy binary benchmark is audit material only.** Every number
  obtained on it (100%) reflects the `0xDADA` capture artifact, not traffic
  semantics; all scientific claims rest on ToN-IoT-Clean.

## Reproducibility

All experiments use fixed seeds and the single 4-epoch protocol above. The
logs behind every figure and table of the paper are in `logs/toniot/`
(training + test evaluation of each of the 23 configurations, audit runs,
pruning sweep) and `results/`. Statistical convention: on n=2,000 test flows
the 95% confidence half-width is ±1.24 points; single-run accuracies within
this margin are reported as indistinguishable.

## Authors

- **Romain Tesseyre** — Université de Technologie de Compiègne (UTC) &
  CBio Laboratory, UFPR, Curitiba, Brazil
- **Bruno Meyer** — CBio Laboratory, UFPR
- **Aurora Trinidad Ramirez Pozo** — CBio Laboratory, UFPR

## References

| Reference | Description |
|-----------|-------------|
| [NetGPT (Meng et al., 2023)](https://arxiv.org/abs/2304.09513) | GPT-2 foundation model for network traffic |
| [UER-py (Zhao et al., 2019)](https://github.com/dbiir/UER-py) | Pre-training/fine-tuning toolkit |
| [Wanda (Sun et al., ICLR 2024)](https://github.com/locuslab/wanda) | Activation-aware pruning metric |
| [Pruner-Zero (Dong et al., ICML 2024)](https://github.com/pprp/Pruner-Zero) | Evolved symbolic pruning metric |
| [ToN-IoT (Alsaedi et al., 2020)](https://research.unsw.edu.au/projects/toniot-datasets) | IoT/IIoT datasets with per-flow ground truth |
| [Arp et al., USENIX Security 2022](https://www.usenix.org/conference/usenixsecurity22/presentation/arp) | Pitfalls of ML for security (audit rationale) |
| [Geirhos et al., 2020](https://www.nature.com/articles/s42256-020-00257-z) | Shortcut learning in deep networks |
| [Blalock et al., MLSys 2020](https://arxiv.org/abs/2003.03033) | State of neural network pruning (random-baseline standard) |
| [Liu et al., ICLR 2019](https://arxiv.org/abs/1810.05270) | Rethinking the value of network pruning (from-scratch control) |

## License

This project builds upon [UER-py](https://github.com/dbiir/UER-py) and
[NetGPT](https://arxiv.org/abs/2304.09513). Please refer to the original
projects for licensing terms.
