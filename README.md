# NetGPT-UFPR: Systematic Compression of a Pretrained Network Traffic Model

> **How Much Transformer Do You Need?**
> A compression study of NetGPT for multi-class network attack detection.

## Overview

This repository contains the code, configurations, and analysis scripts for compressing [NetGPT](https://arxiv.org/abs/2304.09513), a GPT-2-based model pretrained on raw network traffic, for efficient attack detection. The project demonstrates that a **single Transformer layer** with minimal FFN (d_ff=256) matches the full 12-layer teacher at **98.06% accuracy** on 4-class classification (DDoS, DoS, scanning, normal).

### Key Findings

| Model | Layers | Heads | d_ff | Params (encoder) | Accuracy |
|-------|--------|-------|------|-----------------|----------|
| Teacher | 12 | 12 | 3072 | ~85M | 98.06% |
| **NetGPT-Slim** | **1** | **12** | **256** | **~2.75M** | **98.06%** |

- **Over-parameterization**: 1 layer = 12 layers for traffic classification
- **Head boundary scrambling**: Changing head count (12→8) destroys pretrained attention — novel finding
- **Truncation toxicity**: Truncated fine-tuned weights worse than random init — novel finding
- **Pruning cliff**: Global pruning stable to 74%, sharp cliff at 76–78% sparsity
- **Pruner-Zero superiority**: 71% accuracy at 90% sparsity vs ~50% for Magnitude/Wanda

---

## Built With

This project is built on top of the following frameworks and tools:

- **[UER-py](https://github.com/dbiir/UER-py)** (Universal Encoder Representations) — An open-source PyTorch toolkit for pre-training and fine-tuning Transformer models. UER-py provides modular components (embeddings, encoders, targets) that can be combined to implement architectures such as BERT, GPT-2, ELMo, and T5. NetGPT uses UER-py's GPT-2 implementation with custom hex tokenization and traffic-specific structural markers. The `uer/` directory in this repository contains the UER-py modules used by NetGPT.
- **[NetGPT](https://arxiv.org/abs/2304.09513)** — A GPT-2-based foundation model that treats network traffic as a language: raw bytes are hex-encoded, tokenized with WordPiece, and structured with packet delimiters (`[pck]`), flow markers (`[cls]`), and task prompts (`[tsk]`).
- **[Wanda](https://github.com/locuslab/wanda)** — Activation-aware pruning metric (Sun et al., ICLR 2024).
- **[Pruner-Zero](https://github.com/pprp/Pruner-Zero)** — Automatically discovered pruning metric via genetic programming (Dong et al., ICML 2024).

---

## Repository Structure

```
NetGPT_work/
├── assets/
│   └── pretrained_model.bin              # Pretrained NetGPT checkpoint (not in repo)
├── models/
│   ├── encryptd_vocab.txt                # Hex-encoded WordPiece vocabulary
│   └── gpt2/
│       ├── config.json                   # Teacher config (12L/12H/d_ff=3072)
│       ├── distil_config.json            # 6-layer student config
│       ├── slim_config.json              # Slim config (6L/12H/d_ff=2048)
│       ├── slim_final_config.json        # Final Slim (1L/12H/d_ff=256) ★
│       ├── distil_L{1..6}.json           # Layer sweep configs
│       └── distil_ff{256..3072}.json     # d_ff sweep configs
├── uer/                                  # UER-py framework (model architecture)
│   ├── utils/
│   │   ├── config.py                     # Hyperparameter loading from JSON
│   │   ├── constants.py                  # Special tokens, encoder/embedding maps
│   │   ├── tokenizers.py                 # CharTokenizer for hex-encoded traffic
│   │   └── vocab.py                      # Vocabulary loading
│   ├── layers/
│   │   ├── transformer.py                # TransformerLayer (attention + FFN + LN)
│   │   ├── multi_headed_attn.py          # Multi-head causal self-attention
│   │   └── position_ffn.py              # FFN with GELU activation
│   ├── encoders/
│   │   └── transformer_encoder.py        # Stacked Transformer layers
│   └── embeddings/
│       ├── word_embedding.py             # Token embeddings
│       └── pos_embedding.py              # Learned positional embeddings
├── pre-process/
│   └── input_generation_understanding.py # PCAP → TSV preprocessing pipeline
├── finetune/
│   ├── run_understanding.py              # Fine-tuning with shape-aware loading
│   ├── run_distillation.py               # Knowledge distillation (teacher→student)
│   └── run_distillation_slim.py          # Slim distillation with FFN truncation
├── pruning/
│   ├── pruner.py                         # Pruning engine (Magnitude, Wanda, Pruner-Zero)
│   ├── run_pruning.py                    # Per-layer pruning experiments
│   └── run_pruning_global.py             # Global pruning with redundancy map
├── analysis/
│   ├── svd_analysis.py                   # SVD decomposition of FFN weight matrices
│   ├── activation_pca.py                 # PCA of post-GELU FFN activations
│   └── measure_gelu_rate.py              # GELU activation sparsity measurement
└── logs/                                 # Experiment logs (reproducibility)
    ├── teacher_multiclass.txt
    ├── slim_final.txt
    ├── sweep_multiclass_layers.txt
    ├── sweep_multiclass_dff.txt
    ├── pruning_slim_global_sweep.txt
    └── pruning_slim_pz_zoom.txt
```

---

## Requirements

| Dependency | Version |
|------------|---------|
| Python | 3.8+ |
| PyTorch | ≥ 2.0 |
| CUDA | ≥ 11.8 |
| GPU VRAM | ≥ 6 GB (8+ GB recommended) |

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/EahBelieve/NetGPT-UFPR.git
cd NetGPT-UFPR

# 2. Create a virtual environment (conda or venv)
conda create -n netgpt python=3.8 -y
conda activate netgpt

# 3. Install PyTorch with CUDA support
# Adjust the CUDA version to match your system (cu118, cu121, cu124...)
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu118

# 4. Install additional dependencies
pip install six packaging psutil scikit-learn pandas scipy

# 5. Verify GPU access
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}')"
```

### Pretrained Model

The pretrained NetGPT checkpoint (`pretrained_model.bin`) is required for all experiments. It was trained by the original NetGPT authors on large-scale network traffic using autoregressive (next-token prediction) pretraining. Place it in `assets/pretrained_model.bin`.

> **Note**: This checkpoint is not included in the repository due to its size. Contact the [original authors](https://arxiv.org/abs/2304.09513) or refer to the NetGPT paper for access.

---

## Pipeline

### Step 1: Dataset Preparation

Extract TCP flows from PCAP files, organize by attack class, and preprocess into TSV format.

```bash
# Create a directory with one subfolder per class
# Labels are assigned alphabetically: 0=DDoS, 1=DoS, 2=normal, 3=scanning
mkdir -p pcap_data/{DDoS,DoS,normal,scanning}
# Place individual flow PCAP files in each subfolder

# Run preprocessing
python pre-process/input_generation_understanding.py \
  --pcap_path pcap_data/ \
  --dataset_dir finetune_dataset/ \
  --middle_save_path middle_cache/ \
  --class_num 4 --random_seed 42
```

**Output**: `finetune_dataset/` containing `train_dataset.tsv`, `valid_dataset.tsv`, `test_dataset.tsv` (80/10/10 split).

Each TSV line contains: `<label>\t<hex-encoded flow with [pck] delimiters>`

### Step 2: Fine-tune Teacher (baseline)

```bash
python finetune/run_understanding.py \
  --pretrained_model_path assets/pretrained_model.bin \
  --output_model_path models/teacher.bin \
  --vocab_path models/encryptd_vocab.txt \
  --config_path models/gpt2/config.json \
  --train_path finetune_dataset/train_dataset.tsv \
  --dev_path finetune_dataset/valid_dataset.tsv \
  --test_path finetune_dataset/test_dataset.tsv \
  --epochs_num 10 --batch_size 16 --seq_length 64 \
  --labels_num 4 --pooling mean --learning_rate 2e-5
```

### Step 3: Fine-tune NetGPT-Slim

The Slim model uses a reduced architecture (1 layer, d_ff=256). The script uses **shape-aware loading**: pretrained weights with matching shapes are loaded, while mismatched weights (FFN layers) are randomly initialized.

```bash
python finetune/run_understanding.py \
  --pretrained_model_path assets/pretrained_model.bin \
  --output_model_path models/slim_final.bin \
  --vocab_path models/encryptd_vocab.txt \
  --config_path models/gpt2/slim_final_config.json \
  --train_path finetune_dataset/train_dataset.tsv \
  --dev_path finetune_dataset/valid_dataset.tsv \
  --test_path finetune_dataset/test_dataset.tsv \
  --epochs_num 10 --batch_size 16 --seq_length 64 \
  --labels_num 4 --pooling mean --learning_rate 2e-5
```

### Step 4: Pruning Analysis

Run global pruning at multiple sparsity levels to characterize the compressibility cliff:

```bash
for SPARSITY in 0.5 0.6 0.7 0.74 0.76 0.78 0.8 0.9; do
  for METRIC in magnitude wanda pruner_zero; do
    python pruning/run_pruning_global.py \
      --pretrained_model_path models/slim_final.bin \
      --config_path models/gpt2/slim_final_config.json \
      --vocab_path models/encryptd_vocab.txt \
      --train_path finetune_dataset/train_dataset.tsv \
      --dev_path finetune_dataset/valid_dataset.tsv \
      --test_path finetune_dataset/test_dataset.tsv \
      --seq_length 64 --labels_num 4 --batch_size 32 \
      --pooling mean --metric $METRIC --sparsity $SPARSITY --seed 42
  done
done
```

### Step 5: Internal Representation Analysis

```bash
# SVD of FFN weight matrices (effective rank)
python analysis/svd_analysis.py \
  --model_path models/teacher.bin \
  --config_path models/gpt2/config.json \
  --vocab_path models/encryptd_vocab.txt --pooling mean

# PCA of post-GELU activations (effective dimensionality)
python analysis/activation_pca.py \
  --model_path models/teacher.bin \
  --config_path models/gpt2/config.json \
  --vocab_path models/encryptd_vocab.txt \
  --calib_path finetune_dataset/train_dataset.tsv \
  --n_calib 200 --pooling mean

# GELU activation rate per layer
python analysis/measure_gelu_rate.py
```

---

## Important Notes

### Pooling Strategy
Always use `--pooling mean`. The pretrained model was designed for mean pooling over all tokens. Using the default `first` pooling drops accuracy to ~50%.

### Head Count Constraint
**Never change the number of attention heads** when loading from a pretrained checkpoint. Changing from 12 to 8 heads with the same hidden_size=768 produces identical weight matrix shapes (768×768), but the internal head structure is scrambled: pretrained heads of 64 dimensions are reinterpreted as heads of 96 dimensions, mixing independently learned sub-spaces. This destroys the pretrained attention patterns and accuracy drops to chance level. Always keep `heads_num` matching the pretrained model.

### Shape-Aware Loading
When using a config with different FFN dimensions than the pretrained model (e.g., d_ff=256 vs 3072), `run_understanding.py` performs shape-aware loading: weights with matching shapes are copied from the checkpoint, while mismatched shapes are skipped and left randomly initialized. This is essential for the architectural sweep experiments.

### Sequence Length
`seq_length=64` is optimal for attack detection. Attack signatures (SYN floods, port scans, malformed headers) are detectable in early packet headers and handshakes — longer sequences add no discriminative information.

---

## Theoretical Framework

### Minimum FFN Width (Equation 3 in paper)

We derive a lower bound on the FFN intermediate dimension based on activation analysis:

```
d_ff_min = r_τ / ᾱ
```

where:
- **r_τ** = effective rank of post-GELU activations at energy threshold τ (measured by PCA)
- **ᾱ** = mean GELU activation rate (fraction of neurons producing output > 0)

**Measured values** (on the 12-layer teacher):
- r₉₅ = 32, ᾱ = 0.146 → d_ff_min ≈ 219, rounded to **256 = 2⁸**

This prediction is validated empirically: d_ff=256 achieves 98.06% accuracy (equal to the teacher with d_ff=3072).

**Interpretation**: GELU activation acts as a sparse gate, activating only ~15% of FFN neurons per input. For the active subset to span the r_τ-dimensional activation sub-space, the total number of neurons must be at least r_τ/ᾱ.

---

## Reproducing Key Results

### Architectural Sweeps

To verify that depth and FFN width have no impact on performance:

```bash
# Layer sweep: create configs for 1 to 6 layers
for NL in 1 2 3 4 5 6; do
  sed "s/\"layers_num\": 6/\"layers_num\": $NL/" models/gpt2/distil_config.json \
    > models/gpt2/distil_L${NL}.json
done

# d_ff sweep: create configs for various FFN widths
for DFF in 256 512 768 1024 1536 2048 3072; do
  sed "s/\"feedforward_size\": 3072/\"feedforward_size\": $DFF/" models/gpt2/distil_config.json \
    > models/gpt2/distil_ff${DFF}.json
done

# Run each config with the same training command as Step 3
```

### Head Boundary Scrambling Experiment

To reproduce the head scrambling failure:

```bash
# Create a config with 8 heads (same hidden_size=768)
# This will load pretrained weights (shapes match) but accuracy will be ~50%
sed 's/"heads_num": 12/"heads_num": 8/' models/gpt2/distil_config.json \
  > models/gpt2/distil_8heads.json

python finetune/run_understanding.py \
  --pretrained_model_path assets/pretrained_model.bin \
  --output_model_path models/test_8heads.bin \
  --vocab_path models/encryptd_vocab.txt \
  --config_path models/gpt2/distil_8heads.json \
  --train_path finetune_dataset/train_dataset.tsv \
  --dev_path finetune_dataset/valid_dataset.tsv \
  --test_path finetune_dataset/test_dataset.tsv \
  --epochs_num 10 --batch_size 16 --seq_length 64 \
  --labels_num 4 --pooling mean --learning_rate 2e-5
# Expected result: ~50% accuracy (chance level)
```

---

## References

| Reference | Description |
|-----------|-------------|
| [NetGPT (Wu et al., 2023)](https://arxiv.org/abs/2304.09513) | GPT-2-based foundation model for network traffic understanding and generation |
| [UER-py (Zhao et al., 2019)](https://github.com/dbiir/UER-py) | Open-source PyTorch toolkit for pre-training and fine-tuning Transformer models |
| [Wanda (Sun et al., ICLR 2024)](https://github.com/locuslab/wanda) | Pruning metric combining weight magnitude and input activation norms |
| [Pruner-Zero (Dong et al., ICML 2024)](https://github.com/pprp/Pruner-Zero) | Automatically discovered pruning metric via genetic programming |
| [DistilBERT (Sanh et al., 2019)](https://arxiv.org/abs/1910.01108) | Knowledge distillation for BERT with every-other-layer initialization |
| [Lottery Ticket Hypothesis (Frankle & Carlin, 2019)](https://arxiv.org/abs/1803.03635) | Dense networks contain sparse trainable subnetworks |
| [Focal Loss (Lin et al., ICCV 2017)](https://arxiv.org/abs/1708.02002) | Loss function addressing class imbalance in classification |
| [BoT-IoT (Koroniotis et al., 2019)](https://doi.org/10.1016/j.future.2019.05.027) | IoT botnet dataset for network forensic analytics |

---

## Authors

- **Romain Tesseyre** — Research intern, CBio Laboratory, UFPR, Curitiba, Brazil
- **Aurora Trinidad Ramirez Pozo** — Supervisor, UFPR

## License

This project builds upon [UER-py](https://github.com/dbiir/UER-py) and [NetGPT](https://arxiv.org/abs/2304.09513). Please refer to the original projects for licensing terms.
