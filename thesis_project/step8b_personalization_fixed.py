"""
STEP 8b - Corrected personalization experiment.

Fixes vs. step8_personalization.py
  1. (k, w) is chosen on VALIDATION participants; test participants are used once, for the chosen config.
  2. Many random seeds for which k recordings are picked (paired: same recordings for before/after).
  3. Same RF hyper-parameters for general and personalized models (old code: 200 vs 100 trees).
  4. Accuracy AND balanced accuracy AND macro-F1 (pooled over held-out recordings), same recordings.
  5. Participant-level bootstrap CI, paired Wilcoxon, % improved / unchanged / worsened.
  6. CONTROL for the label issue: upweight k training recordings of the SAME CLASS as the patient
     (no patient-specific data). If this gains as much, the effect is "knowing the label", not personalization.

Usage:  python step8b_personalization_fixed.py --data-dir . --seeds 5 [--fast]
Needs train_tasks.npz, test_tasks.npz, and ideally val_tasks.npz (same format, with 'ids').
"""
import argparse, os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from common import load_split, make_rf, bootstrap_ci

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", default=".")
ap.add_argument("--seeds", type=int, default=5)
ap.add_argument("--val-seeds", type=int, default=2)
ap.add_argument("--trees", type=int, default=100)
ap.add_argument("--fast", action="store_true", help="50 trees, 2 seeds (smoke test)")
args = ap.parse_args()
if args.fast:
    args.trees, args.seeds, args.val_seeds = 50, 2, 1

K_VALUES = [1, 3, 5]
WEIGHTS = [1, 3, 8, 15]
d = args.data_dir

Xtr, ytr, idtr = load_split(os.path.join(d, "train_tasks.npz"))
Xte, yte, idte = load_split(os.path.join(d, "test_tasks.npz"))
val_path = os.path.join(d, "val_tasks.npz")
if os.path.exists(val_path):
    Xva, yva, idva = load_split(val_path)
    Xtr_sel, ytr_sel = Xtr, ytr           # general model for selection = same as final general model
    print(f"Using {val_path} for (k, w) selection.")
else:
    print("No val_tasks.npz -> carving 15% of TRAIN participants out as pseudo-validation for selection.")
    g = np.unique(idtr, return_inverse=True)[1]
    tr_i, va_i = next(GroupShuffleSplit(1, test_size=0.15, random_state=0).split(Xtr, ytr, g))
    Xva, yva, idva = Xtr[va_i], ytr[va_i], idtr[va_i]
    Xtr_sel, ytr_sel = Xtr[tr_i], ytr[tr_i]


def run(X_tr, y_tr, X_ev, y_ev, ids_ev, seeds, mode="personal", configs=None):
    """Returns long DataFrame: one row per (participant, seed, k, w) with before/after on held-out recordings.
    mode='personal': add k of the patient's own recordings.  mode='control': add k TRAIN recordings of the
    patient's class instead (no patient-specific data)."""
    general = make_rf(args.trees).fit(X_tr, y_tr)
    pred_general = general.predict(X_ev)
    rows = []
    pids = np.unique(ids_ev)
    for seed in seeds:
        for k in K_VALUES:
            for w in WEIGHTS:
                if configs is not None and (k, w) not in configs:
                    continue
                for pi, pid in enumerate(pids):
                    idx = np.where(ids_ev == pid)[0]
                    if len(idx) <= k:
                        continue
                    rng = np.random.default_rng([seed, k, pi])       # same picks for every w -> paired
                    sh = rng.permutation(idx)
                    pick, ev = sh[:k], sh[k:]
                    if mode == "personal":
                        Xp, yp = X_ev[pick], y_ev[pick]
                    else:
                        cls_idx = np.where(y_tr == y_ev[idx[0]])[0]
                        c = rng.choice(cls_idx, size=k, replace=False)
                        Xp, yp = X_tr[c], y_tr[c]
                    clf = make_rf(args.trees)
                    clf.fit(np.vstack([X_tr, Xp]), np.concatenate([y_tr, yp]),
                            sample_weight=np.concatenate([np.ones(len(X_tr)), np.full(k, float(w))]))
                    pa = clf.predict(X_ev[ev])
                    rows.append(dict(pid=pid, seed=seed, k=k, w=w, y_true=list(y_ev[ev]),
                                     pred_before=list(pred_general[ev]), pred_after=list(pa),
                                     before_acc=accuracy_score(y_ev[ev], pred_general[ev]),
                                     after_acc=accuracy_score(y_ev[ev], pa)))
        print(f"  [{mode}] seed {seed} done", flush=True)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 1) SELECT (k, w) ON VALIDATION
print("\n[1/3] Selecting (k, w) on validation participants ...")
val_df = run(Xtr_sel, ytr_sel, Xva, yva, idva, seeds=range(1000, 1000 + args.val_seeds))
val_df["delta"] = val_df["after_acc"] - val_df["before_acc"]
val_sum = val_df.groupby(["k", "w"])["delta"].mean().sort_values(ascending=False)
print(val_sum.to_string(float_format=lambda v: f"{v:+.4f}"))
best_k, best_w = val_sum.index[0]
print(f"\nSELECTED ON VALIDATION: k={best_k}, w={best_w}   (test set not touched yet)")
val_sum.to_csv("personalization_validation_grid.csv")

# ------------------------------------------------------------------ 2) TEST, chosen config only
print("\n[2/3] Evaluating chosen config on TEST participants ...")
cfg = {(best_k, best_w)}
te = run(Xtr, ytr, Xte, yte, idte, seeds=range(args.seeds), configs=cfg)
print("\n[3/3] Control: upweighting same-class TRAIN recordings (no patient data) ...")
ct = run(Xtr, ytr, Xte, yte, idte, seeds=range(args.seeds), mode="control", configs=cfg)
te.drop(columns=["y_true", "pred_before", "pred_after"]).to_csv("personalization_test_rows.csv", index=False)
ct.drop(columns=["y_true", "pred_before", "pred_after"]).to_csv("personalization_control_rows.csv", index=False)


def summarize(df, name):
    # average over seeds within participant first -> participants are the independent unit
    per = df.groupby("pid")[["before_acc", "after_acc"]].mean()
    delta = (per["after_acc"] - per["before_acc"]).values
    m, lo, hi = bootstrap_ci(delta)
    try:
        p = wilcoxon(delta[delta != 0]).pvalue if (delta != 0).any() else 1.0
    except ValueError:
        p = float("nan")
    eps = 1e-9
    imp, same, worse = (delta > eps).mean(), (np.abs(delta) <= eps).mean(), (delta < -eps).mean()
    # pooled metrics on identical held-out recordings, per seed, then averaged
    mets = []
    for s, g in df.groupby("seed"):
        yt = np.concatenate(g["y_true"].values)
        pb = np.concatenate(g["pred_before"].values)
        pa = np.concatenate(g["pred_after"].values)
        mets.append([accuracy_score(yt, pb), accuracy_score(yt, pa),
                     balanced_accuracy_score(yt, pb), balanced_accuracy_score(yt, pa),
                     f1_score(yt, pb, average="macro"), f1_score(yt, pa, average="macro")])
    mets = np.mean(mets, axis=0)
    print(f"\n--- {name}  (k={best_k}, w={best_w}, n_participants={len(delta)}, seeds={df.seed.nunique()}) ---")
    print(f"mean per-patient accuracy change : {m:+.4f}   95% bootstrap CI [{lo:+.4f}, {hi:+.4f}]")
    print(f"paired Wilcoxon p-value          : {p:.4g}")
    print(f"improved / unchanged / worsened  : {imp:.0%} / {same:.0%} / {worse:.0%}")
    print(f"pooled held-out  acc  {mets[0]:.3f} -> {mets[1]:.3f} | balAcc {mets[2]:.3f} -> {mets[3]:.3f} | "
          f"macroF1 {mets[4]:.3f} -> {mets[5]:.3f}")
    return per.assign(delta=delta)


print("\n=========== FINAL TEST RESULTS (report these) ===========")
per_te = summarize(te, "PERSONALIZED (patient's own recordings)")
per_ct = summarize(ct, "CONTROL (same-class training recordings, no patient data)")
per_te.to_csv("personalization_per_patient.csv")
print("\nInterpretation: if CONTROL gain ~= PERSONALIZED gain, the benefit is label-prior shift, "
      "not patient-specific adaptation. Say so in the paper.")