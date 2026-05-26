import sys, os, torch
sys.path.insert(0, '.')
from uer.utils.constants import *
from uer.utils import *
from uer.utils.config import load_hyperparam
from uer.opts import tokenizer_opts, model_opts
from finetune.run_understanding import Classifier, read_dataset
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', default='models/teacher_newds_local.bin')
parser.add_argument('--config_path', default='models/gpt2/config.json')
tokenizer_opts(parser)
model_opts(parser)
args = parser.parse_args(['--vocab_path','models/encryptd_vocab.txt','--pooling','mean'])
args = load_hyperparam(args)
args.labels_num = 2
args.seq_length = 64
args.soft_targets = False
args.soft_alpha = 0.0
args.dropout = 0.1
args.tokenizer = str2tokenizer[args.tokenizer](args)

model = Classifier(args)
model.load_state_dict(torch.load(args.model_path, map_location='cpu'), strict=False)
model.eval()

dataset = read_dataset(args, 'finetune_dataset_newds/train_dataset.tsv')
src = torch.LongTensor([s[0] for s in dataset[:200]])
seg = torch.LongTensor([s[2] for s in dataset[:200]])

rates = {}
def make_hook(name):
    def hook_fn(m, inp, out):
        act = inp[0].detach()
        rate = (act > 0).float().mean().item()
        if name not in rates: rates[name] = []
        rates[name].append(rate)
    return hook_fn

hooks = []
for n, m in model.named_modules():
    if 'feed_forward.linear_2' in n and isinstance(m, torch.nn.Linear):
        layer_id = n.split('.')[2]
        hooks.append(m.register_forward_hook(make_hook(layer_id)))

with torch.no_grad():
    for i in range(0, 200, 16):
        emb = model.embedding(src[i:i+16], seg[i:i+16])
        model.encoder(emb, seg[i:i+16])

for h in hooks: h.remove()

print('\n' + '='*60)
print('  TAUX ACTIVATION POST-GELU PAR COUCHE')
print('='*60)
for k in sorted(rates.keys(), key=lambda x: int(x)):
    r = sum(rates[k])/len(rates[k])
    print(f'  Layer {k}: {r*100:.1f}% neurones actifs')

avg = sum(sum(v)/len(v) for v in rates.values()) / len(rates)
print(f'\n  Moyenne globale: alpha = {avg*100:.1f}%')

print('\n' + '='*60)
print('  APPLICATION DE LA FORMULE: d_ff_min = r_tau / alpha')
print('='*60)
# PCA ranks from previous analysis
r90, r95, r99 = 12, 32, 108
print(f'\n  alpha (taux activation) = {avg:.3f}')
print(f'  r_90  = {r90}  =>  d_ff_min(90%) = {r90/avg:.0f}')
print(f'  r_95  = {r95}  =>  d_ff_min(95%) = {r95/avg:.0f}')
print(f'  r_99  = {r99}  =>  d_ff_min(99%) = {r99/avg:.0f}')
print(f'\n  Sweep empirique: d_ff=1024 -> 94.4%, d_ff=2048 -> 96.9%')
