"""
STEP 10 - TCN, multi-seed, participant-level evaluation.

Architecture is reconstructed from your saved tcn_model.pt (tensor names + shapes):
  4 TemporalBlocks, channels 14 -> 32 -> 32 -> 64 -> 64, kernel 5, dilations 1,2,4,8,
  1x1 downsample where channels change, global average pool, Linear(64 -> 3). ~92k parameters.
NOT recoverable from a .pt file (so they are arguments below -- set them to what you really used and
REPORT THEM IN THE PAPER): optimizer, learning rate, epochs, batch size, dropout, early-stopping rule.

Check your checkpoint matches this definition:
    python step10_tcn_multiseed.py --verify-ckpt tcn_model.pt
Run:
    python step10_tcn_multiseed.py --data-dir . --seeds 5
"""
import argparse, os, copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score


# ----------------------------------------------------------------------------- model
class Chomp1d(nn.Module):
    def __init__(self, n):
        super().__init__(); self.n = n

    def forward(self, x):
        return x[:, :, :-self.n].contiguous() if self.n > 0 else x


class TemporalBlock(nn.Module):
    # attribute order matters: it reproduces your checkpoint's key names
    # (conv1, conv2, net.0, net.4, downsample)
    def __init__(self, cin, cout, k, dilation, dropout):
        super().__init__()
        pad = (k - 1) * dilation
        self.conv1 = nn.Conv1d(cin, cout, k, padding=pad, dilation=dilation)
        self.chomp1 = Chomp1d(pad); self.relu1 = nn.ReLU(); self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(cout, cout, k, padding=pad, dilation=dilation)
        self.chomp2 = Chomp1d(pad); self.relu2 = nn.ReLU(); self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(cin, cout, 1) if cin != cout else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCN(nn.Module):
    def __init__(self, in_ch=14, channels=(32, 32, 64, 64), k=5, dropout=0.2, n_classes=3):
        super().__init__()
        blocks, c = [], in_ch
        for i, co in enumerate(channels):
            blocks.append(TemporalBlock(c, co, k, 2 ** i, dropout)); c = co
        self.tcn = nn.Sequential(*blocks)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(c, n_classes)

    def forward(self, x):                      # x: (batch, 14, T)
        return self.fc(self.global_pool(self.tcn(x)).squeeze(-1))


# ----------------------------------------------------------------------------- helpers
def to_tensor(X, mean, std):
    """(n, T, C) numpy -> normalised (n, C, T) float tensor. Stats come from TRAIN only."""
    return torch.from_numpy(((X - mean) / std).astype(np.float32)).permute(0, 2, 1).contiguous()


@torch.no_grad()
def predict(model, X, bs=256, dev="cpu"):
    model.eval(); out = []
    for i in range(0, len(X), bs):
        out.append(model(X[i:i + bs].to(dev)).argmax(1).cpu())
    return torch.cat(out).numpy()


def vote_ba(y, p, ids):
    df = pd.DataFrame({"g": ids, "y": y, "p": p})
    v = df.groupby("g").agg(y=("y", "first"), p=("p", lambda s: s.value_counts().idxmax()))
    return balanced_accuracy_score(v["y"], v["p"])


def boot_ci(vals, n=10000, seed=0):
    vals = np.asarray(vals, float); rng = np.random.default_rng(seed)
    m = vals[rng.integers(0, len(vals), (n, len(vals)))].mean(1)
    return vals.mean(), np.quantile(m, .025), np.quantile(m, .975)


# ----------------------------------------------------------------------------- main
ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", default=".")
ap.add_argument("--seeds", type=int, default=5)
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--patience", type=int, default=8)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--wd", type=float, default=1e-4)
ap.add_argument("--bs", type=int, default=64)
ap.add_argument("--dropout", type=float, default=0.2)
ap.add_argument("--verify-ckpt", default=None)
args = ap.parse_args()

if args.verify_ckpt:
    sd = torch.load(args.verify_ckpt, map_location="cpu", weights_only=True)
    sd = sd.get("state_dict", sd)
    m = TCN(dropout=args.dropout)
    res = m.load_state_dict(sd, strict=True)
    n_par = sum(p.numel() for p in m.parameters())
    rf = 1 + 2 * (5 - 1) * sum(2 ** i for i in range(4))
    print("Checkpoint matches this architecture:", res)
    print(f"parameters: {n_par:,} | receptive field: {rf} samples = {rf / 100:.2f} s of a 10.24 s window")
    print("weight-norm keys present?", any("weight_g" in k or "parametrizations" in k for k in sd))
    raise SystemExit

dev = "cuda" if torch.cuda.is_available() else "cpu"
d = args.data_dir
tr, va, te = [np.load(os.path.join(d, f), allow_pickle=True) for f in
              ("train_tasks.npz", "val_tasks.npz", "test_tasks.npz")]
Xtr, ytr = tr["X"], tr["y"]; Xva, yva, idva = va["X"], va["y"], va["ids"]; Xte, yte, idte = te["X"], te["y"], te["ids"]
mean = Xtr.mean(axis=(0, 1), keepdims=True); std = Xtr.std(axis=(0, 1), keepdims=True) + 1e-8
Ttr, Tva, Tte = to_tensor(Xtr, mean, std), to_tensor(Xva, mean, std), to_tensor(Xte, mean, std)
ytr_t = torch.from_numpy(ytr).long()

# square-root-dampened inverse-frequency weights (as in the paper)
counts = np.bincount(ytr, minlength=3).astype(float)
cw = torch.tensor((len(ytr) / (3 * counts)) ** 0.5, dtype=torch.float32).to(dev)
print(f"device={dev} | class counts={counts.astype(int).tolist()} | class weights={cw.cpu().numpy().round(3).tolist()}")

rows = []
for seed in range(args.seeds):
    torch.manual_seed(seed); np.random.seed(seed)
    model = TCN(dropout=args.dropout).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    lossf = nn.CrossEntropyLoss(weight=cw)
    best, best_state, bad = -1, None, 0
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(Ttr))
        for i in range(0, len(perm), args.bs):
            b = perm[i:i + args.bs]
            opt.zero_grad()
            loss = lossf(model(Ttr[b].to(dev)), ytr_t[b].to(dev))
            loss.backward(); opt.step()
        vba = balanced_accuracy_score(yva, predict(model, Tva, dev=dev))   # selection on VALIDATION only
        if vba > best:
            best, best_state, bad = vba, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= args.patience:
                break
    model.load_state_dict(best_state)
    ptr, pva, pte = (predict(model, T, dev=dev) for T in (Ttr, Tva, Tte))
    r = dict(seed=seed, epochs_run=ep + 1,
             train_BA=balanced_accuracy_score(ytr, ptr), val_BA=balanced_accuracy_score(yva, pva),
             test_BA=balanced_accuracy_score(yte, pte), test_acc=accuracy_score(yte, pte),
             test_macroF1=f1_score(yte, pte, average="macro"), test_patient_vote_BA=vote_ba(yte, pte, idte))
    r["gap"] = r["train_BA"] - r["test_BA"]
    rows.append(r)
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    torch.save(best_state, f"tcn_seed{seed}.pt")

df = pd.DataFrame(rows); df.to_csv("tcn_multiseed.csv", index=False)
print("\n=========== TCN over seeds (mean +- std) ===========")
for c in ["train_BA", "val_BA", "test_BA", "gap", "test_acc", "test_macroF1", "test_patient_vote_BA"]:
    print(f"{c:22s} {df[c].mean():.3f} +- {df[c].std(ddof=1):.3f}")
print("\nIf train_BA stays ~0.5 the TCN is UNDERFITTING: its small gap then does not mean 'generalizes better'.")
