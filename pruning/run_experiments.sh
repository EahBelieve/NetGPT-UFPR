#!/bin/bash
# ============================================================================
# NetGPT Pruning Experiments v2 — Paper Dataset
# 3 metrics x 3 sparsity x 4 seeds = 36 runs
# Fix: --pooling mean, correct paths, per-row pruning
# ============================================================================

cd ~/projects/NetGPT_work/

MODEL_PATH="models/finetuned_model.bin"
CONFIG_PATH="models/gpt2/config.json"
VOCAB_PATH="models/encryptd_vocab.txt"
TRAIN="finetune_dataset/train_dataset.tsv"
DEV="finetune_dataset/valid_dataset.tsv"
TEST="finetune_dataset/test_dataset.tsv"
OUTPUT_CSV="results/pruning_exp1_paper_dataset.csv"

SEQ_LENGTH=64
LABELS_NUM=2
BATCH_SIZE=32
N_CALIB=128
POOLING="mean"

mkdir -p results

echo "============================================================"
echo " NetGPT Pruning v2 — Paper Dataset (100k flows)"
echo " Per-row pruning (matching Wanda implementation)"
echo " 3 metrics x 3 sparsities x 4 seeds = 36 runs"
echo "============================================================"

METRICS=("magnitude" "wanda" "pruner_zero")
SPARSITIES=("0.3" "0.5" "0.7")
SEEDS=("0" "1" "2" "3")

RUN=0
TOTAL=$((${#METRICS[@]} * ${#SPARSITIES[@]} * ${#SEEDS[@]}))

for metric in "${METRICS[@]}"; do
    for sparsity in "${SPARSITIES[@]}"; do
        for seed in "${SEEDS[@]}"; do
            RUN=$((RUN + 1))
            EXP_ID="${metric}_s${sparsity}_seed${seed}"
            echo ""
            echo "-- Run ${RUN}/${TOTAL}: ${EXP_ID} --"
            python pruning/run_pruning.py \
                --pretrained_model_path "$MODEL_PATH" \
                --config_path "$CONFIG_PATH" \
                --vocab_path "$VOCAB_PATH" \
                --train_path "$TRAIN" \
                --dev_path "$DEV" \
                --test_path "$TEST" \
                --seq_length $SEQ_LENGTH \
                --labels_num $LABELS_NUM \
                --batch_size $BATCH_SIZE \
                --pooling $POOLING \
                --metric "$metric" \
                --sparsity "$sparsity" \
                --n_calib $N_CALIB \
                --seed "$seed" \
                --exp_id "$EXP_ID" \
                --output_csv "$OUTPUT_CSV"
            if [ $? -ne 0 ]; then
                echo "[ERROR] ${EXP_ID} failed!"
            fi
        done
    done
done

echo ""
echo "============================================================"
echo " Done: ${TOTAL} runs. Results: ${OUTPUT_CSV}"
echo " Next: python pruning/analyze_results.py --csv ${OUTPUT_CSV}"
echo "============================================================"
