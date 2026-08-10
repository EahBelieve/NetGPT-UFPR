import sys, os, argparse, numpy as np, torch
sys.path.insert(0, os.getcwd())
from finetune.run_understanding import *

def extract(model, args, path, device):
    dataset = read_dataset(args, path)
    src = torch.LongTensor([s[0] for s in dataset])
    tgt = torch.LongTensor([s[1] for s in dataset])
    seg = torch.LongTensor([s[2] for s in dataset])
    hooked = {}
    layer0 = model.encoder.transformer[0]
    h = layer0.feed_forward.register_forward_hook(
        lambda m, inp, out: hooked.__setitem__("pre_ffn", inp[0].detach()))
    feats, labels = [], []
    with torch.no_grad():
        for i in range(0, len(dataset), args.batch_size):
            _ = model(src[i:i+args.batch_size].to(device), None,
                      seg[i:i+args.batch_size].to(device))
            feats.append(hooked["pre_ffn"].mean(dim=1).cpu().numpy())
            labels.append(tgt[i:i+args.batch_size].numpy())
    h.remove()
    return np.concatenate(feats), np.concatenate(labels)

def main():
    parser = argparse.ArgumentParser()
    finetune_opts(parser)
    tokenizer_opts(parser)
    parser.add_argument("--save_prefix", default="analysis/phi0_toniot")
    args = parser.parse_args()
    args.soft_targets = False
    args.soft_alpha = 0.0
    args = load_hyperparam(args)
    set_seed(args.seed)
    args.labels_num = count_labels_num(args.train_path)
    args.tokenizer = str2tokenizer[args.tokenizer](args)

    model = Classifier(args)
    model.load_state_dict(torch.load(args.pretrained_model_path, map_location="cpu"), strict=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    for split, p in [("train", args.train_path), ("test", args.test_path)]:
        X, y = extract(model, args, p, device)
        np.savez(f"{args.save_prefix}_{split}.npz",
                 phi0=X, y=y,
                 act_rate_tok=np.zeros(1), act_pool=np.zeros((1,1)),
                 layer_idx=np.array(0))
        print(f"{split}: phi0={X.shape} y={y.shape} -> {args.save_prefix}_{split}.npz")

if __name__ == "__main__":
    main()
