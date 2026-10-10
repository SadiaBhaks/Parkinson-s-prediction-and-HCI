"""
STEP 10b - Rebuild tcn_test_predictions.csv from the saved tcn_seed*.pt weights (NO retraining, ~1 minute).

Use this if you ran step10_tcn_multiseed.py BEFORE it started saving predictions.
Needs in the current folder: train_tasks.npz, test_tasks.npz, tcn_seed0.pt, tcn_seed1.pt, ... and tcn_multiseed.csv.
It also CHECKS itself: test balanced accuracy recomputed from the weights must match tcn_multiseed.csv.

    python step10b_make_tcn_predictions.py --data-dir .
"""
import argparse, glob, os, re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score


# ---- identical architecture to step10_tcn_multiseed.py -----------------------------------------
class Chomp1d(nn.Module):
    def __init__(self, n):
        super().__init__(); self.n = n

    def forward(self, x):
        return x[:, :, :-self.n].contiguous() if self.n > 0 else x


class TemporalBlock(nn.Module):
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

    def forward(self, x):
        return self.fc(self.global_pool(self.tcn(x)).squeeze(-1))


ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", default=".")
args = ap.parse_args()
d = args.data_dir

tr = np.load(os.path.join(d, "train_tasks.npz"), allow_pickle=True)
te = np.load(os.path.join(d, "test_tasks.npz"), allow_pickle=True)
Xtr = tr["X"]; Xte, yte, idte = te["X"], te["y"], te["ids"]
mean = Xtr.mean(axis=(0, 1), keepdims=True); std = Xtr.std(axis=(0, 1), keepdims=True) + 1e-8   # TRAIN stats only
Tte = torch.from_numpy(((Xte - mean) / std).astype(np.float32)).permute(0, 2, 1).contiguous()

ckpts = sorted(glob.glob(os.path.join(d, "tcn_seed*.pt")), key=lambda p: int(re.findall(r"seed(\d+)", p)[0]))
if not ckpts:
    raise SystemExit("No tcn_seed*.pt files found in the data folder.")
ref = pd.read_csv(os.path.join(d, "tcn_multiseed.csv")).set_index("seed")["test_BA"] \
    if os.path.exists(os.path.join(d, "tcn_multiseed.csv")) else None

rows = []
for p in ckpts:
    seed = int(re.findall(r"seed(\d+)", p)[0])
    model = TCN(); model.load_state_dict(torch.load(p, map_location="cpu", weights_only=True)); model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(Tte), 256):
            out.append(model(Tte[i:i + 256]).argmax(1))
    pred = torch.cat(out).numpy()
    ba = balanced_accuracy_score(yte, pred)
    msg = f"seed {seed}: test BA recomputed = {ba:.4f}"
    if ref is not None and seed in ref.index:
        ok = abs(ba - ref[seed]) < 5e-3
        msg += f"  | tcn_multiseed.csv = {ref[seed]:.4f}  -> {'MATCH' if ok else 'MISMATCH (check data/normalisation)'}"
    print(msg)
    rows += [dict(Model="TCN", Seed=seed, id=i, y=int(a), pred=int(b)) for i, a, b in zip(idte, yte, pred)]

pd.DataFrame(rows).to_csv(os.path.join(d, "tcn_test_predictions.csv"), index=False)
print(f"\nSaved tcn_test_predictions.csv ({len(rows)} rows, {len(ckpts)} seeds). Now run step12_model_comparison_chart.py")
