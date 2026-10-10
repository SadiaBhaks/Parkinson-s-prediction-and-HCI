"""
STEP 9 - Controlled leakage audit (fixes RQ2 / Table II).

Same recordings, same features, same Random Forest, same metric. ONLY the split differs:
  A) record-wise split      : random recordings -> the same participant appears in train AND test
  B) participant-wise split : GroupKFold-style  -> no participant in both

Reports train BA, test BA, gap, and participant overlap, repeated over several seeds x 5 folds.
Usage:  python step9_leakage_audit.py --data-dir . --repeats 3
"""
import argparse, os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score
from common import load_split, make_rf, bootstrap_ci

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", default=".")
ap.add_argument("--repeats", type=int, default=3)
ap.add_argument("--folds", type=int, default=5)
args = ap.parse_args()

# ---- pool ALL recordings (train + optional val + test) so both protocols use identical data ----
parts = []
for name in ["train_tasks.npz", "val_tasks.npz", "test_tasks.npz"]:
    p = os.path.join(args.data_dir, name)
    if os.path.exists(p):
        parts.append(load_split(p))
        print(f"loaded {name}: {len(parts[-1][1])} recordings")
X = np.vstack([p[0] for p in parts])
y = np.concatenate([p[1] for p in parts])
ids = np.concatenate([p[2] for p in parts])
groups = np.unique(ids, return_inverse=True)[1]
print(f"POOLED: {len(y)} recordings, {len(np.unique(groups))} participants\n")

rows = []
for rep in range(args.repeats):
    seed = 100 + rep
    protocols = {
        "record-wise (leaky)": StratifiedKFold(args.folds, shuffle=True, random_state=seed).split(X, y),
        "participant-wise": StratifiedGroupKFold(args.folds, shuffle=True, random_state=seed).split(X, y, groups),
    }
    for proto, splits in protocols.items():
        for fold, (tr, te) in enumerate(splits):
            clf = make_rf(seed=seed)
            clf.fit(X[tr], y[tr])
            tr_ba = balanced_accuracy_score(y[tr], clf.predict(X[tr]))
            te_pred = clf.predict(X[te])
            te_ba = balanced_accuracy_score(y[te], te_pred)
            overlap_p = len(set(groups[tr]) & set(groups[te]))
            frac_rec = np.isin(groups[te], np.unique(groups[tr])).mean()
            # patient-level majority vote on the test fold
            df = pd.DataFrame({"g": groups[te], "y": y[te], "p": te_pred})
            vote = df.groupby("g").agg(y=("y", "first"), p=("p", lambda s: s.value_counts().idxmax()))
            vote_ba = balanced_accuracy_score(vote["y"], vote["p"])
            rows.append(dict(protocol=proto, repeat=rep, fold=fold, train_BA=tr_ba, test_BA=te_ba,
                             gap=tr_ba - te_ba, patient_vote_BA=vote_ba,
                             overlap_participants=overlap_p, frac_test_recs_from_seen_participants=frac_rec))
    print(f"repeat {rep + 1}/{args.repeats} done")

df = pd.DataFrame(rows)
df.to_csv("leakage_audit_folds.csv", index=False)

print("\n=========== TABLE II (replacement) ===========")
out = []
for proto, g in df.groupby("protocol"):
    m, lo, hi = bootstrap_ci(g["test_BA"].values)
    out.append(dict(Protocol=proto,
                    Train_BA=g["train_BA"].mean(),
                    Test_BA=f"{m:.3f} [{lo:.3f}, {hi:.3f}]",
                    Gap=g["gap"].mean(),
                    PatientVote_BA=g["patient_vote_BA"].mean(),
                    Overlap_participants=g["overlap_participants"].mean(),
                    Seen_participant_frac=g["frac_test_recs_from_seen_participants"].mean()))
tab = pd.DataFrame(out)
print(tab.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
tab.to_csv("leakage_audit_table.csv", index=False)

rw = df[df.protocol.str.startswith("record")]["test_BA"].mean()
pw = df[df.protocol == "participant-wise"]["test_BA"].mean()
print(f"\nInflation from leakage: {rw:.3f} (record-wise) vs {pw:.3f} (participant-wise) = +{rw - pw:.3f} BA")
print("NOTE: CI above resamples folds (n = folds x repeats), so it is approximate; folds share data.")
