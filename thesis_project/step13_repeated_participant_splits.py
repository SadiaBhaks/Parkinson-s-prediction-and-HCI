"""
STEP 13 - Repeated participant-level evaluation + leakage audit for several model families.

Addresses the two main reviewer requests that need new experiments:
  (a) model comparison over MANY group-disjoint splits (not one fixed split)
  (b) the matched record-wise vs participant-wise audit for LR, RF and MLP (not RF only)

All recordings (train+val+test .npz) are pooled. For each repeat r a single set of 5 splits is drawn per
protocol and re-used for every model, so differences between models/protocols are paired.
  participant-wise : StratifiedGroupKFold  (no participant in train and test)
  record-wise      : StratifiedKFold       (same participant on both sides)

Hyper-parameters are the fixed ones from step7b / step9 (no tuning, so no validation set is needed).
Outputs
  repeated_splits_folds.csv     one row per model x protocol x repeat x fold
  repeated_splits_summary.csv   per model: mean +- SD over folds, participant-clustered bootstrap CI,
                                voted BA, record-wise minus participant-wise difference with the
                                Nadeau-Bengio corrected 95% interval
  repeated_splits_pairwise.csv  paired model differences (participant-wise) with corrected intervals
Usage:  python step13_repeated_participant_splits.py --data-dir . --repeats 5
        (add --models "Random Forest,Logistic Regression" to run fewer; the MLP is the slowest)
Requires common.py from the earlier steps in the same folder.
"""
import argparse, itertools, os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from common import load_split

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", default=".")
ap.add_argument("--repeats", type=int, default=5)
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--n-boot", type=int, default=5000)
ap.add_argument("--models", default="Logistic Regression,Random Forest,MLP")
args = ap.parse_args()
MODELS = [m.strip() for m in args.models.split(",")]


def build(name, seed):
    if name == "Logistic Regression":
        return make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed))
    if name == "MLP":
        return make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), early_stopping=True,
                                                             max_iter=500, random_state=seed))
    if name == "Random Forest":  # 100 trees, as in the leakage audit (300 in the single-split benchmark)
        return RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_leaf=5,
                                      class_weight="balanced", random_state=seed, n_jobs=-1)
    raise ValueError(name)


# ---------------- pooled data ----------------
parts = [load_split(os.path.join(args.data_dir, f)) for f in ("train_tasks.npz", "val_tasks.npz", "test_tasks.npz")
         if os.path.exists(os.path.join(args.data_dir, f))]
X = np.vstack([p[0] for p in parts]); y = np.concatenate([p[1] for p in parts])
groups = np.unique(np.concatenate([p[2] for p in parts]), return_inverse=True)[1]
classes = np.unique(y); P = groups.max() + 1
print(f"POOLED: {len(y)} recordings, {P} participants, classes {classes}")

# ---------------- run ----------------
rows = []
oof = {}   # (model, protocol, repeat) -> out-of-fold prediction for every recording (participant-wise only needed for CI)
for rep in range(args.repeats):
    seed = 200 + rep
    splits = {
        "participant-wise": list(StratifiedGroupKFold(args.folds, shuffle=True, random_state=seed).split(X, y, groups)),
        "record-wise": list(StratifiedKFold(args.folds, shuffle=True, random_state=seed).split(X, y)),
    }
    for model in MODELS:
        for proto, sp in splits.items():
            pred_all = np.full(len(y), -1)
            for fold, (tr, te) in enumerate(sp):
                clf = build(model, seed).fit(X[tr], y[tr])
                p = clf.predict(X[te]); pred_all[te] = p
                df = pd.DataFrame({"g": groups[te], "y": y[te], "p": p})
                vote = df.groupby("g").agg(y=("y", "first"), p=("p", lambda s: s.value_counts().idxmax()))
                rows.append(dict(model=model, protocol=proto, repeat=rep, fold=fold,
                                 train_BA=balanced_accuracy_score(y[tr], clf.predict(X[tr])),
                                 test_BA=balanced_accuracy_score(y[te], p),
                                 voted_BA=balanced_accuracy_score(vote["y"], vote["p"]),
                                 seen_participant_frac=np.isin(groups[te], np.unique(groups[tr])).mean()))
            oof[(model, proto, rep)] = pred_all
        print(f"repeat {rep + 1}/{args.repeats} {model} done", flush=True)

F = pd.DataFrame(rows); F.to_csv("repeated_splits_folds.csv", index=False)


# ---------------- statistics ----------------
def nb_interval(diffs, n_test_over_train):
    """Nadeau-Bengio corrected paired t interval for repeated k-fold differences (J = folds x repeats)."""
    d = np.asarray(diffs, float); J = len(d)
    var = d.var(ddof=1) * (1.0 / J + n_test_over_train)
    se = np.sqrt(var); t = stats.t.ppf(0.975, J - 1)
    return d.mean(), d.mean() - t * se, d.mean() + t * se


def cluster_boot_ba(model, proto, B, seed=0):
    """Participant-clustered bootstrap of BA computed on out-of-fold predictions, averaged over repeats."""
    rng = np.random.default_rng(seed); idx = rng.integers(0, P, size=(B, P))  # shared resamples -> paired
    bas = []
    for rep in range(args.repeats):
        pred = oof[(model, proto, rep)]
        corr = np.zeros((P, len(classes))); tot = np.zeros((P, len(classes)))
        for ci, c in enumerate(classes):
            m = y == c
            np.add.at(tot[:, ci], groups[m], 1); np.add.at(corr[:, ci], groups[m], (pred[m] == c))
        c_s, t_s = corr[idx].sum(1), tot[idx].sum(1)           # B x classes
        bas.append(np.nanmean(c_s / np.where(t_s == 0, np.nan, t_s), axis=1))
    return np.mean(bas, axis=0)


ratio = 1.0 / (args.folds - 1)
summ = []
for model in MODELS:
    pw = F[(F.model == model) & (F.protocol == "participant-wise")].sort_values(["repeat", "fold"])
    rw = F[(F.model == model) & (F.protocol == "record-wise")].sort_values(["repeat", "fold"])
    bs = cluster_boot_ba(model, "participant-wise", args.n_boot)
    d, lo, hi = nb_interval(rw.test_BA.values - pw.test_BA.values, ratio)
    summ.append(dict(model=model,
                     pw_train_BA=pw.train_BA.mean(), pw_test_BA=pw.test_BA.mean(), pw_test_SD_over_folds=pw.test_BA.std(),
                     pw_cluster_boot_CI=f"[{np.quantile(bs, .025):.3f}, {np.quantile(bs, .975):.3f}]",
                     pw_voted_BA=pw.voted_BA.mean(),
                     rw_test_BA=rw.test_BA.mean(), rw_test_SD_over_folds=rw.test_BA.std(),
                     rw_minus_pw=d, rw_minus_pw_NB95=f"[{lo:.3f}, {hi:.3f}]",
                     SD_ratio_pw_over_rw=pw.test_BA.std() / rw.test_BA.std()))
S = pd.DataFrame(summ); S.to_csv("repeated_splits_summary.csv", index=False)

pair = []
for a, b in itertools.combinations(MODELS, 2):
    da = F[(F.model == a) & (F.protocol == "participant-wise")].sort_values(["repeat", "fold"]).test_BA.values
    db = F[(F.model == b) & (F.protocol == "participant-wise")].sort_values(["repeat", "fold"]).test_BA.values
    d, lo, hi = nb_interval(da - db, ratio)
    pair.append(dict(model_a=a, model_b=b, diff_a_minus_b=d, NB95=f"[{lo:.3f}, {hi:.3f}]",
                     excludes_zero=not (lo <= 0 <= hi)))
PW = pd.DataFrame(pair); PW.to_csv("repeated_splits_pairwise.csv", index=False)

pd.set_option("display.width", 220)
print("\n=== per-model summary (participant-wise = honest estimate; rw = record-wise) ===")
print(S.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
print("\n=== paired model differences, participant-wise splits ===")
print(PW.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
print("\nNOTES: fold-level SDs and the Nadeau-Bengio interval are approximate because folds share data; "
      "the clustered bootstrap CI resamples participants of the out-of-fold predictions "
      f"({args.n_boot} resamples, averaged over {args.repeats} repeats). The TCN is not included here "
      "(far slower) - if you add it, use the same splits.")
