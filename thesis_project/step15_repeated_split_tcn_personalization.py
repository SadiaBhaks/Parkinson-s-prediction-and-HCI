"""
STEP 15 - TCN and few-shot personalization over REPEATED participant-level train/val/test splits.

Removes the 'one fixed split' limitation for the two experiments that still use it. For each repeat r:
  1. pool all participants and draw a NEW stratified participant-level split (70/15/15 by participant class);
  2. TCN (same architecture/hyper-parameters as step10, early stopping on that repeat's validation set);
  3. RF (100 trees) general model, then oracle-label personalization with weight chosen on VALIDATION
     participants, evaluated once on that repeat's test participants, plus the same-class control.
Outputs: repeated_split_results.csv (one row per repeat) and a printed mean +- SD summary.

Usage:  python step15_repeated_split_tcn_personalization.py --data-dir . --repeats 5
        --skip-tcn / --skip-personalization to run one part;  --fast = smoke test (2 repeats, small settings)
Time: personalization is ~700 random-forest fits per repeat; use --n-jobs and fewer --repeats if slow.
Requires common.py (and torch for the TCN) in the same folder.
"""
import argparse, copy, os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from common import extract_features, make_rf, bootstrap_ci

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", default=".")
ap.add_argument("--repeats", type=int, default=5)
ap.add_argument("--k", type=int, default=5, help="adaptation recordings per patient")
ap.add_argument("--weights", default="1,3,8,15")
ap.add_argument("--val-seeds", type=int, default=1)
ap.add_argument("--test-seeds", type=int, default=3)
ap.add_argument("--trees", type=int, default=100)
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--patience", type=int, default=8)
ap.add_argument("--skip-tcn", action="store_true")
ap.add_argument("--skip-personalization", action="store_true")
ap.add_argument("--fast", action="store_true")
args = ap.parse_args()
if args.fast:
    args.repeats, args.trees, args.val_seeds, args.test_seeds, args.epochs = 2, 30, 1, 1, 3
WEIGHTS = [float(w) for w in args.weights.split(",")]

# ---------------------------------------------------------------- pooled raw data
parts = [np.load(os.path.join(args.data_dir, f), allow_pickle=True) for f in
         ("train_tasks.npz", "val_tasks.npz", "test_tasks.npz") if os.path.exists(os.path.join(args.data_dir, f))]
Xraw = np.concatenate([p["X"] for p in parts]); y = np.concatenate([p["y"] for p in parts])
ids = np.concatenate([p["ids"] for p in parts])
uid, g = np.unique(ids, return_inverse=True)
plab = np.array([y[g == i][0] for i in range(len(uid))])
F = extract_features(Xraw)
print(f"POOLED {len(y)} recordings, {len(uid)} participants")

if not args.skip_tcn:
    import torch, torch.nn as nn

    class Chomp1d(nn.Module):
        def __init__(self, n): super().__init__(); self.n = n
        def forward(self, x): return x[:, :, :-self.n].contiguous() if self.n > 0 else x

    class Block(nn.Module):
        def __init__(self, cin, cout, k, dil, dr):
            super().__init__(); pad = (k - 1) * dil
            self.conv1 = nn.Conv1d(cin, cout, k, padding=pad, dilation=dil)
            self.conv2 = nn.Conv1d(cout, cout, k, padding=pad, dilation=dil)
            self.net = nn.Sequential(self.conv1, Chomp1d(pad), nn.ReLU(), nn.Dropout(dr),
                                     self.conv2, Chomp1d(pad), nn.ReLU(), nn.Dropout(dr))
            self.downsample = nn.Conv1d(cin, cout, 1) if cin != cout else None; self.relu = nn.ReLU()
        def forward(self, x):
            o = self.net(x); r = x if self.downsample is None else self.downsample(x); return self.relu(o + r)

    class TCN(nn.Module):
        def __init__(self, in_ch, ch=(32, 32, 64, 64), k=5, dr=0.2, nc=3):
            super().__init__(); b, c = [], in_ch
            for i, co in enumerate(ch): b.append(Block(c, co, k, 2 ** i, dr)); c = co
            self.tcn = nn.Sequential(*b); self.global_pool = nn.AdaptiveAvgPool1d(1); self.fc = nn.Linear(c, nc)
        def forward(self, x): return self.fc(self.global_pool(self.tcn(x)).squeeze(-1))

    dev = "cuda" if torch.cuda.is_available() else "cpu"

    def run_tcn(tr, va, te, seed):
        torch.manual_seed(seed); np.random.seed(seed)
        mean = Xraw[tr].mean(axis=(0, 1), keepdims=True); std = Xraw[tr].std(axis=(0, 1), keepdims=True) + 1e-8
        T = lambda idx: torch.from_numpy(((Xraw[idx] - mean) / std).astype(np.float32)).permute(0, 2, 1).contiguous()
        Ttr, Tva, Tte = T(tr), T(va), T(te); ytr = torch.from_numpy(y[tr]).long()
        counts = np.bincount(y[tr], minlength=3).astype(float)
        cw = torch.tensor((len(tr) / (3 * counts)) ** 0.5, dtype=torch.float32).to(dev)   # sqrt inverse frequency
        model = TCN(Xraw.shape[2]).to(dev); opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        lossf = nn.CrossEntropyLoss(weight=cw)

        @torch.no_grad()
        def pred(Tx):
            model.eval(); return torch.cat([model(Tx[i:i + 256].to(dev)).argmax(1).cpu() for i in range(0, len(Tx), 256)]).numpy()

        best, state, bad = -1, None, 0
        for ep in range(args.epochs):
            model.train(); perm = torch.randperm(len(Ttr))
            for i in range(0, len(perm), 64):
                b = perm[i:i + 64]; opt.zero_grad(); lossf(model(Ttr[b].to(dev)), ytr[b].to(dev)).backward(); opt.step()
            v = balanced_accuracy_score(y[va], pred(Tva))
            if v > best: best, state, bad = v, copy.deepcopy(model.state_dict()), 0
            else:
                bad += 1
                if bad >= args.patience: break
        model.load_state_dict(state)
        return balanced_accuracy_score(y[tr], pred(Ttr)), balanced_accuracy_score(y[te], pred(Tte))


def personalize(Ftr, ytr, Fev, yev, gev, weights, seeds, mode, base_seed):
    """Per-patient before/after accuracy on the patient's held-out recordings, for each weight."""
    general = make_rf(args.trees).fit(Ftr, ytr); pg = general.predict(Fev)
    out = []
    for w in weights:
        for seed in seeds:
            for pi, pid in enumerate(np.unique(gev)):
                idx = np.where(gev == pid)[0]
                if len(idx) <= args.k: continue
                rng = np.random.default_rng([base_seed, seed, args.k, pi]); sh = rng.permutation(idx)
                pick, ev = sh[:args.k], sh[args.k:]
                if mode == "personal": Xp, yp = Fev[pick], yev[pick]
                else:
                    c = rng.choice(np.where(ytr == yev[idx[0]])[0], size=args.k, replace=False); Xp, yp = Ftr[c], ytr[c]
                clf = make_rf(args.trees).fit(np.vstack([Ftr, Xp]), np.concatenate([ytr, yp]),
                                              sample_weight=np.concatenate([np.ones(len(Ftr)), np.full(args.k, w)]))
                out.append(dict(pid=pid, w=w, seed=seed, before=accuracy_score(yev[ev], pg[ev]),
                                after=accuracy_score(yev[ev], clf.predict(Fev[ev]))))
    return pd.DataFrame(out)


rows = []
for r in range(args.repeats):
    # participant-level stratified 70/15/15
    pidx = np.arange(len(uid))
    p_tr, p_rest = train_test_split(pidx, test_size=0.30, stratify=plab, random_state=500 + r)
    p_va, p_te = train_test_split(p_rest, test_size=0.50, stratify=plab[p_rest], random_state=500 + r)
    tr, va, te = (np.where(np.isin(g, p))[0] for p in (p_tr, p_va, p_te))
    row = dict(repeat=r, n_train=len(p_tr), n_val=len(p_va), n_test=len(p_te))
    rf = make_rf(args.trees, seed=r).fit(F[tr], y[tr])
    row.update(rf_train_BA=balanced_accuracy_score(y[tr], rf.predict(F[tr])),
               rf_test_BA=balanced_accuracy_score(y[te], rf.predict(F[te])))
    if not args.skip_tcn:
        row["tcn_train_BA"], row["tcn_test_BA"] = run_tcn(tr, va, te, seed=r)
    if not args.skip_personalization:
        val = personalize(F[tr], y[tr], F[va], y[va], g[va], WEIGHTS, range(1000, 1000 + args.val_seeds), "personal", r)
        val["d"] = val.after - val.before; best_w = val.groupby("w").d.mean().idxmax()
        pt = personalize(F[tr], y[tr], F[te], y[te], g[te], [best_w], range(args.test_seeds), "personal", r)
        ct = personalize(F[tr], y[tr], F[te], y[te], g[te], [best_w], range(args.test_seeds), "control", r)
        per = pt.groupby("pid")[["before", "after"]].mean(); d = (per.after - per.before).values
        dc = (lambda c: (c.after - c.before).values)(ct.groupby("pid")[["before", "after"]].mean())
        m, lo, hi = bootstrap_ci(d)
        try: p = wilcoxon(d[d != 0]).pvalue
        except ValueError: p = float("nan")
        row.update(best_w=best_w, pers_before=per.before.mean(), pers_after=per.after.mean(), pers_change=m,
                   pers_CI_lo=lo, pers_CI_hi=hi, pers_wilcoxon_p=p, frac_improved=(d > 1e-9).mean(),
                   frac_worsened=(d < -1e-9).mean(), control_change=dc.mean(), pers_minus_control=m - dc.mean())
    rows.append(row); print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    pd.DataFrame(rows).to_csv("repeated_split_results.csv", index=False)

R = pd.DataFrame(rows)
print("\n=========== mean +- SD over repeated participant-level splits ===========")
for c in R.columns:
    if c not in ("repeat", "n_train", "n_val", "n_test", "best_w"):
        print(f"{c:20s} {R[c].mean():.3f} +- {R[c].std(ddof=1):.3f}   (min {R[c].min():.3f}, max {R[c].max():.3f})")
if "pers_change" in R: print(f"\npersonalization change > 0 in {(R.pers_change > 0).sum()}/{len(R)} splits; "
                             f"beats control in {(R.pers_minus_control > 0).sum()}/{len(R)} splits")
