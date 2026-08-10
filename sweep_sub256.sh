#!/usr/bin/env bash
# ============================================================================
# sweep_sub256.sh   —   TEST 1 (le test decisif)
#
# Refait le sweep de largeur du FFN EN DESSOUS de 256, avec FFN entraine
# (shape-aware init + fine-tuning, comme le run d_ff=256). Mesure ou l'accuracy
# casse vraiment. Prediction de l'axe : rupture vers ~16-20, JAMAIS a 256.
#
# Chemins du dataset multiclasse deja renseignes (train/valid/test_dataset.tsv).
# IMPORTANT : la commande d'entrainement plus bas est un TEMPLATE base sur
# run_understanding.py. Si ta commande EXACTE du run d_ff=256 differe (tokenizer,
# hyperparams), aligne-la ici ; ne change que --config_path entre les largeurs.
# ============================================================================
set -e

# --- chemins (adapte uniquement si ton arborescence locale differe) ---
BASE_CONFIG="models/gpt2/slim_final_config.json"   # ta config slim 1L/12H/d_ff=256
PRETRAINED="assets/pretrained_model.bin"
VOCAB="models/encryptd_vocab.txt"
TRAIN="finetune_dataset_multiclass/train_dataset.tsv"
DEV="finetune_dataset_multiclass/valid_dataset.tsv"
TEST="finetune_dataset_multiclass/test_dataset.tsv"
CONFIG_DIR="models/gpt2/sweep_configs"
LOG="sweep_sub256_results.txt"

mkdir -p "$CONFIG_DIR"
echo "d_ff   test_acc" > "$LOG"

# largeurs a tester (256 = controle, doit redonner ~98%)
for D in 1 2 3 4 8 16 32 64 128 256; do
  CFG="$CONFIG_DIR/slim_dff${D}.json"
  # genere une config identique a la base, avec feedforward_size = D
  python - "$BASE_CONFIG" "$CFG" "$D" << 'PYEOF'
import json, sys
base, out, d = sys.argv[1], sys.argv[2], int(sys.argv[3])
cfg = json.load(open(base))
cfg["feedforward_size"] = d
json.dump(cfg, open(out, "w"), indent=4)
print(f"  config generee : {out}  (feedforward_size={d})")
PYEOF

  echo ">>> entrainement d_ff=$D"
  OUT_MODEL="slim_dff${D}.bin"

  # ---- TEMPLATE : aligne sur ta commande EXACTE du run 256 si besoin ----
  python finetune/run_understanding.py \
      --pretrained_model_path "$PRETRAINED" \
      --config_path "$CFG" \
      --vocab_path "$VOCAB" \
      --train_path "$TRAIN" \
      --dev_path "$DEV" \
      --test_path "$TEST" \
      --output_model_path "$OUT_MODEL" \
      --tokenizer char --pooling mean --seq_length 64 \
      --labels_num 4 --learning_rate 2e-5 --batch_size 16 \
      --epochs_num 10 --seed 42 \
      2>&1 | tee "log_dff${D}.txt"
  # -----------------------------------------------------------------------

  # recupere la derniere accuracy "Acc. (Correct/Total): X" du log
  ACC=$(grep -oE "Acc\. \(Correct/Total\): [0-9.]+" "log_dff${D}.txt" | tail -1 | grep -oE "[0-9.]+$")
  echo "$D    $ACC" >> "$LOG"
  echo ">>> d_ff=$D  ->  test_acc=$ACC"
done

echo ""
echo "=== RESULTATS (rupture attendue vers ~16-20, pas 256) ==="
cat "$LOG"
