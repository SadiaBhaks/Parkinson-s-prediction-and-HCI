"""
STEP 7b - Logistic Regression / Random Forest / MLP, multi-seed, SAME participant-level split as the TCN.

Replaces model_benchmark_results.csv. Writes:
  model_benchmark_multiseed.csv   long format: Model, Seed, Split, Balanced_Accuracy
  sklearn_test_predictions.csv    Model, Seed, id, y, pred   (needed for patient-level bootstrap CIs)

Seeds only vary MODEL randomness (RF bootstrap, MLP init). The train/val/test participant split is fixed,
so seed spread is NOT split-to-split variance - say so in the paper (or use GroupKFold, see step9).
Re-run this: your paper's Table I numbers must come from the same code/settings you report.
Usage: python step7b_benchmark_multiseed.py --data-dir . --seeds 5
"""
import argparse, os
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from common import load_split

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", default=".")
ap.add_argument("--seeds", type=int, default=5)
args = ap.parse_args()

Xtr, ytr, _ = load_split(os.path.join(args.data_dir, "train_tasks.npz"))
Xva, yva, _ = load_split(os.path.join(args.data_dir, "val_tasks.npz"))
Xte, yte, idte = load_split(os.path.join(args.data_dir, "test_tasks.npz"))


def build(seed):
    return {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=seed)),
        "MLP": make_pipeline(StandardScaler(), MLPClassifier(
            hidden_layer_sizes=(64, 32), early_stopping=True, max_iter=500, random_state=seed)),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=5, class_weight="balanced",
            random_state=seed, n_jobs=-1),
    }


rows, preds = [], []
for seed in range(args.seeds):
    for name, clf in build(seed).items():
        clf.fit(Xtr, ytr)
        for split, X, y in (("Train", Xtr, ytr), ("Validation", Xva, yva), ("Test (unseen)", Xte, yte)):
            rows.append(dict(Model=name, Seed=seed, Split=split,
                             Balanced_Accuracy=balanced_accuracy_score(y, clf.predict(X))))
        p = clf.predict(Xte)
        preds += [dict(Model=name, Seed=seed, id=i, y=int(a), pred=int(b)) for i, a, b in zip(idte, yte, p)]
    print(f"seed {seed + 1}/{args.seeds} done", flush=True)

df = pd.DataFrame(rows)
df.to_csv("model_benchmark_multiseed.csv", index=False)
pd.DataFrame(preds).to_csv("sklearn_test_predictions.csv", index=False)
print(df.groupby(["Model", "Split"])["Balanced_Accuracy"].agg(["mean", "std"]).round(3))
print("\nNOTE: MLP early stopping holds out 10% of TRAIN RECORDINGS (same participants as training) - "
      "mention it, or tune the MLP on the validation participants instead.")
