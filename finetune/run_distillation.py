"""
run_distillation.py — Knowledge Distillation for NetGPT Attack Detection
=========================================================================
Distills a fine-tuned 12-layer GPT-2 teacher into a 6-layer student using:
  L = alpha * T^2 * KL(softmax(z_t/T) || softmax(z_s/T))
    + (1 - alpha) * FocalCE(z_s, y)

Student is initialised DistilBERT-style: take one layer out of two from
the teacher (layers 1,3,5,7,9,11 → student layers 0..5).

Usage:
  python finetune/run_distillation.py \
    --teacher_model_path  models/finetuned_model.bin \
    --output_model_path   models/distilled_model.bin \
    --vocab_path          models/encryptd_vocab.txt \
    --config_path         models/gpt2/config.json \
    --student_config_path models/gpt2/distil_config.json \
    --train_path          finetune_dataset/train_dataset.tsv \
    --dev_path            finetune_dataset/valid_dataset.tsv \
    --test_path           finetune_dataset/test_dataset.tsv \
    --epochs_num 5 --batch_size 16 --seq_length 64 \
    --labels_num 2 --pooling mean \
    --temperature 3.0 --alpha 0.5 --focal_gamma 2.0 \
    --learning_rate 2e-5

References:
  [1] Hinton et al., "Distilling the Knowledge in a Neural Network", 2015
  [2] Sanh et al., "DistilBERT", arXiv:1910.01108
  [3] Lin et al., "Focal Loss for Dense Object Detection", IEEE TPAMI 2020
  [4] Wu & Zhang, "Lightweight Network Traffic Classification via KD", WISE 2021
"""
import sys
import os
import random
import argparse
import json
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tqdm
from collections import OrderedDict

uer_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(uer_dir)

from uer.embeddings import *
from uer.encoders import *
from uer.utils.vocab import Vocab
from uer.utils.constants import *
from uer.utils import *
from uer.utils.optimizers import *
from uer.utils.config import load_hyperparam
from uer.utils.seed import set_seed
from uer.utils.logging import init_logger
from uer.utils.misc import pooling
from uer.model_saver import save_model
from uer.opts import tokenizer_opts, model_opts

#  Classifier — identical to run_understanding.py

class Classifier(nn.Module):
    def __init__(self, args):
        super(Classifier, self).__init__()
        self.embedding = Embedding(args)
        for embedding_name in args.embedding:
            tmp_emb = str2embedding[embedding_name](args, len(args.tokenizer.vocab))
            self.embedding.update(tmp_emb, embedding_name)
        self.encoder = str2encoder[args.encoder](args)
        self.labels_num = args.labels_num
        self.pooling_type = args.pooling
        self.output_layer_1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.output_layer_2 = nn.Linear(args.hidden_size, self.labels_num)

    def forward(self, src, tgt, seg):
        """
        Returns (loss, logits) if tgt is not None, else (None, logits).
        For distillation we pass tgt=None to get logits only.
        """
        emb = self.embedding(src, seg)
        output = self.encoder(emb, seg)
        output = pooling(output, seg, self.pooling_type)
        output = torch.tanh(self.output_layer_1(output))
        logits = self.output_layer_2(output)
        if tgt is not None:
            loss = nn.NLLLoss()(nn.LogSoftmax(dim=-1)(logits), tgt.view(-1))
            return loss, logits
        else:
            return None, logits


#  Focal Loss

class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al., 2017).
    FL(p_t) = -(1-p_t)^gamma * log(p_t)
    Down-weights easy examples, focuses on hard attack flows.
    """
    def __init__(self, gamma=2.0, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        focal = ((1.0 - pt) ** self.gamma) * ce
        if self.reduction == "mean":
            return focal.mean()
        return focal.sum()


#  Dataset loading — same as run_understanding.py

def read_dataset(args, path):
    dataset, columns = [], {}
    with open(path, mode="r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            if line_id == 0:
                for i, column_name in enumerate(line.rstrip("\r\n").split("\t")):
                    columns[column_name] = i
                continue
            line = line.rstrip("\r\n").split("\t")
            tgt = int(line[columns["label"]])
            text_a = line[columns["text_a"]]
            src = args.tokenizer.convert_tokens_to_ids(
                [CLS_TOKEN] + args.tokenizer.tokenize(text_a) + [SEP_TOKEN]
            )
            seg = [1] * len(src)

            if len(src) > args.seq_length:
                src = src[: args.seq_length]
                seg = seg[: args.seq_length]
            PAD_ID = args.tokenizer.convert_tokens_to_ids([PAD_TOKEN])[0]
            while len(src) < args.seq_length:
                src.append(PAD_ID)
                seg.append(0)
            dataset.append((src, tgt, seg))
    return dataset


def batch_loader(batch_size, src, tgt, seg):
    instances_num = src.size()[0]
    for i in range(instances_num // batch_size):
        src_batch = src[i * batch_size: (i + 1) * batch_size, :]
        tgt_batch = tgt[i * batch_size: (i + 1) * batch_size]
        seg_batch = seg[i * batch_size: (i + 1) * batch_size, :]
        yield src_batch, tgt_batch, seg_batch
    if instances_num > instances_num // batch_size * batch_size:
        src_batch = src[instances_num // batch_size * batch_size:, :]
        tgt_batch = tgt[instances_num // batch_size * batch_size:]
        seg_batch = seg[instances_num // batch_size * batch_size:, :]
        yield src_batch, tgt_batch, seg_batch


#  Student initialisation — DistilBERT style (every other layer)

def init_student_from_teacher(teacher_state, teacher_layers=12, student_layers=6):
    """
    Copy one layer out of two from teacher to student.
    Teacher layers 1,3,5,7,9,11 (0-indexed) → student layers 0,1,2,3,4,5.
    Non-layer weights (embeddings, output heads) are copied as-is.
    """
    student_state = OrderedDict()
    step = teacher_layers // student_layers  # 2
    # Map: student_layer_idx → teacher_layer_idx
    layer_map = {i: step * i + (step - 1) for i in range(student_layers)}
    # e.g. {0:1, 1:3, 2:5, 3:7, 4:9, 5:11}

    selected_teacher = set(layer_map.values())

    for key, val in teacher_state.items():
        matched = False
        for s_idx, t_idx in layer_map.items():
            t_pattern = "transformer.{}.".format(t_idx)
            if t_pattern in key:
                s_pattern = "transformer.{}.".format(s_idx)
                new_key = key.replace(t_pattern, s_pattern)
                student_state[new_key] = val.clone()
                matched = True
                break

        if not matched:
            is_unselected_layer = False
            for layer_idx in range(teacher_layers):
                if layer_idx not in selected_teacher:
                    pattern = "transformer.{}.".format(layer_idx)
                    if pattern in key:
                        is_unselected_layer = True
                        break
            if not is_unselected_layer:
                student_state[key] = val.clone()

    return student_state



#  Evaluation — same as run_understanding.py

def evaluate(args, dataset, model, istest=False):
    src = torch.LongTensor([sample[0] for sample in dataset])
    tgt = torch.LongTensor([sample[1] for sample in dataset])
    seg = torch.LongTensor([sample[2] for sample in dataset])

    correct = 0
    confusion = torch.zeros(args.labels_num, args.labels_num, dtype=torch.long)
    model.eval()

    for src_batch, tgt_batch, seg_batch in batch_loader(args.batch_size, src, tgt, seg):
        src_batch = src_batch.to(args.device)
        tgt_batch = tgt_batch.to(args.device)
        seg_batch = seg_batch.to(args.device)
        with torch.no_grad():
            _, logits = model(src_batch, None, seg_batch)
        pred = torch.argmax(nn.Softmax(dim=1)(logits), dim=1)
        gold = tgt_batch
        for j in range(pred.size()[0]):
            confusion[pred[j], gold[j]] += 1
        correct += torch.sum(pred == gold).item()

    if istest:
        print(confusion)

    eps = 1e-9
    f1_list = []
    for i in range(confusion.size()[0]):
        p = confusion[i, i].item() / (confusion[i, :].sum().item() + eps)
        r = confusion[i, i].item() / (confusion[:, i].sum().item() + eps)
        f1 = 2 * p * r / (p + r + eps)
        f1_list.append(f1)
        args.logger.info("Label {}: P={:.3f}, R={:.3f}, F1={:.3f}".format(i, p, r, f1))

    macro_f1 = np.mean(f1_list)
    acc = correct / len(dataset)
    args.logger.info("Acc. (Correct/Total): {:.4f} ({}/{})".format(acc, correct, len(dataset)))
    args.logger.info("Macro-F1: {:.4f}".format(macro_f1))

    if istest:
        filename = args.output_model_path.replace(".bin", "_prf.csv")
        with open(filename, "w+") as f2:
            f2.write("Label,Precision,Recall,F1\n")
            for i in range(confusion.size()[0]):
                p_i = confusion[i, i].item() / (confusion[i, :].sum().item() + eps)
                r_i = confusion[i, i].item() / (confusion[:, i].sum().item() + eps)
                f1_i = 2 * p_i * r_i / (p_i + r_i + eps)
                f2.write("{},{:.6f},{:.6f},{:.6f}\n".format(i, p_i, r_i, f1_i))

    model.train()
    return acc, macro_f1, confusion


#  Distillation training step

def distill_step(args, teacher, student, optimizer, scheduler,
                 src_batch, tgt_batch, seg_batch, focal_loss_fn):
    """
    L = alpha * T^2 * KL(teacher_soft || student_soft)
      + (1 - alpha) * FocalCE(student_logits, hard_labels)
    """
    student.zero_grad()

    src_batch = src_batch.to(args.device)
    tgt_batch = tgt_batch.to(args.device)
    seg_batch = seg_batch.to(args.device)

    # Teacher forward (frozen)
    with torch.no_grad():
        _, teacher_logits = teacher(src_batch, None, seg_batch)

    # Student forward
    _, student_logits = student(src_batch, None, seg_batch)

    # Soft targets with temperature scaling
    T = args.temperature
    teacher_soft = F.log_softmax(teacher_logits / T, dim=-1)
    student_soft = F.log_softmax(student_logits / T, dim=-1)

    # KL divergence loss
    kl_loss = F.kl_div(student_soft, teacher_soft, log_target=True, reduction="batchmean")
    distill_loss = (T * T) * kl_loss

    # Hard label loss with Focal CE
    hard_loss = focal_loss_fn(student_logits, tgt_batch.view(-1))

    # Combined
    loss = args.alpha * distill_loss + (1.0 - args.alpha) * hard_loss

    loss.backward()
    optimizer.step()
    scheduler.step()

    return loss, distill_loss, hard_loss



#  Count parameters

def count_params(model):
    return sum(p.numel() for p in model.parameters())


#  Main
def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Paths
    parser.add_argument("--teacher_model_path", type=str, required=True,
                        help="Path to the fine-tuned teacher model (.bin).")
    parser.add_argument("--output_model_path", type=str,
                        default="models/distilled_model.bin")
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--dev_path", type=str, required=True)
    parser.add_argument("--test_path", type=str, default=None)
    parser.add_argument("--config_path", type=str,
                        default="models/gpt2/config.json",
                        help="Teacher config (12 layers).")
    parser.add_argument("--student_config_path", type=str,
                        default="models/gpt2/distil_config.json",
                        help="Student config (6 layers).")

    # Tokenizer
    tokenizer_opts(parser)
    model_opts(parser)

    # Model
    parser.add_argument("--labels_num", type=int, default=2)

    # Training
    parser.add_argument("--epochs_num", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seq_length", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--warmup", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--report_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)

    # Distillation hyperparameters
    parser.add_argument("--temperature", type=float, default=3.0,
                        help="Temperature for softening logits.")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Weight for distillation loss vs hard label loss.")
    parser.add_argument("--focal_gamma", type=float, default=2.0,
                        help="Focal loss gamma (0 = standard CE).")

    # Logging
    parser.add_argument("--log_path", type=str, default=None)
    parser.add_argument("--log_level", default="INFO")
    parser.add_argument("--log_file_level", default="INFO")

    args = parser.parse_args()
    set_seed(args.seed)

    # ---------------------------------------------------------------
    #  Build tokenizer
    # ---------------------------------------------------------------
    args.tokenizer = str2tokenizer[args.tokenizer](args)

    # ---------------------------------------------------------------
    #  Build TEACHER (12 layers) and load fine-tuned weights
    # ---------------------------------------------------------------
    teacher_args = copy.deepcopy(args)
    teacher_args.config_path = args.config_path
    teacher_args = load_hyperparam(teacher_args)
    teacher_args.tokenizer = args.tokenizer

    teacher = Classifier(teacher_args)
    teacher.load_state_dict(
        torch.load(args.teacher_model_path, map_location="cpu"), strict=False
    )
    teacher_layers = teacher_args.layers_num

    # ---------------------------------------------------------------
    #  Build STUDENT (6 layers) and initialise from teacher
    # ---------------------------------------------------------------
    student_args = copy.deepcopy(args)
    student_args.config_path = args.student_config_path
    student_args = load_hyperparam(student_args)
    student_args.tokenizer = args.tokenizer

    student = Classifier(student_args)
    student_layers = student_args.layers_num

    # DistilBERT-style init: copy every other layer
    teacher_state = teacher.state_dict()
    student_init = init_student_from_teacher(
        teacher_state,
        teacher_layers=teacher_layers,
        student_layers=student_layers
    )
    student.load_state_dict(student_init, strict=False)

    # ---------------------------------------------------------------
    #  Setup device, logging
    # ---------------------------------------------------------------
    args.logger = init_logger(args)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    teacher = teacher.to(args.device)
    student = student.to(args.device)

    # Freeze teacher
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    t_params = count_params(teacher)
    s_params = count_params(student)
    args.logger.info("Teacher: {:,} params ({} layers)".format(t_params, teacher_layers))
    args.logger.info("Student: {:,} params ({} layers)".format(s_params, student_layers))
    args.logger.info("Compression: {:.1f}x fewer params".format(t_params / s_params))
    args.logger.info("Distillation: T={}, alpha={}, focal_gamma={}".format(
        args.temperature, args.alpha, args.focal_gamma))
    args.logger.info("seq_length={}".format(args.seq_length))

    # ---------------------------------------------------------------
    #  Load datasets
    # ---------------------------------------------------------------
    trainset = read_dataset(args, args.train_path)
    instances_num = len(trainset)
    batch_size = args.batch_size

    args.train_steps = int(instances_num * args.epochs_num / batch_size) + 1
    args.logger.info("Batch size: {}".format(batch_size))
    args.logger.info("Training instances: {}".format(instances_num))
    args.logger.info("Total training steps: {}".format(args.train_steps))

    # ---------------------------------------------------------------
    #  Optimizer & scheduler
    # ---------------------------------------------------------------
    param_optimizer = list(student.named_parameters())
    no_decay = ["bias", "gamma", "beta"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer
                    if not any(nd in n for nd in no_decay)],
         "weight_decay": 0.01},
        {"params": [p for n, p in param_optimizer
                    if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate, correct_bias=False)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        int(args.train_steps * args.warmup),
        args.train_steps
    )

    focal_loss_fn = FocalLoss(gamma=args.focal_gamma)

    # ---------------------------------------------------------------
    #  Training loop
    # ---------------------------------------------------------------
    total_loss, total_kl, total_hard = 0.0, 0.0, 0.0
    best_result = 0.0

    args.logger.info("Start distillation training.")
    for epoch in tqdm.tqdm(range(1, args.epochs_num + 1)):
        args.logger.info("Epoch {}".format(epoch))

        random.shuffle(trainset)
        src = torch.LongTensor([ex[0] for ex in trainset])
        tgt = torch.LongTensor([ex[1] for ex in trainset])
        seg = torch.LongTensor([ex[2] for ex in trainset])

        student.train()
        for i, (src_batch, tgt_batch, seg_batch) in enumerate(
                batch_loader(batch_size, src, tgt, seg)):

            loss, kl_loss, hard_loss = distill_step(
                args, teacher, student, optimizer, scheduler,
                src_batch, tgt_batch, seg_batch, focal_loss_fn
            )

            total_loss += loss.item()
            total_kl += kl_loss.item()
            total_hard += hard_loss.item()

            if (i + 1) % args.report_steps == 0:
                avg = args.report_steps
                args.logger.info(
                    "Epoch {}, Step {}, Loss: {:.4f} "
                    "(KL: {:.4f}, Focal: {:.4f})".format(
                        epoch, i + 1,
                        total_loss / avg,
                        total_kl / avg,
                        total_hard / avg
                    ))
                total_loss, total_kl, total_hard = 0.0, 0.0, 0.0

        # Validation
        args.logger.info("--- Dev set evaluation ---")
        acc, macro_f1, _ = evaluate(
            args, read_dataset(args, args.dev_path), student
        )

        if acc > best_result:
            best_result = acc
            save_model(student, args.output_model_path)
            args.logger.info(
                "** Best model saved (Acc={:.4f}, F1={:.4f}) **".format(
                    acc, macro_f1))

    # ---------------------------------------------------------------
    #  Test evaluation
    # ---------------------------------------------------------------
    if args.test_path is not None:
        args.logger.info("=== Test set evaluation (student) ===")
        student.load_state_dict(torch.load(args.output_model_path))
        student = student.to(args.device)
        evaluate(args, read_dataset(args, args.test_path), student, istest=True)

        args.logger.info("=== Test set evaluation (teacher baseline) ===")
        evaluate(args, read_dataset(args, args.test_path), teacher, istest=False)


if __name__ == "__main__":
    main()
