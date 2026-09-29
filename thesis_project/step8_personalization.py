import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------
train = np.load("train_tasks.npz")
test = np.load("test_tasks.npz")

X_train_raw, y_train = train["X"], train["y"]
X_test_raw, y_test, test_ids = test["X"], test["y"], test["ids"]

print("Raw shapes -> train:", X_train_raw.shape, "test:", X_test_raw.shape)

# ---------------------------------------------------------
# Feature extraction (same as step7)
# ---------------------------------------------------------
SAMPLING_RATE = 100
TREMOR_BAND = (4, 6)

def extract_features(X):
    n_samples, timesteps, channels = X.shape
    feats = []
    freqs = np.fft.rfftfreq(timesteps, d=1.0 / SAMPLING_RATE)
    band_mask = (freqs >= TREMOR_BAND[0]) & (freqs <= TREMOR_BAND[1])
    for i in range(n_samples):
        sample = X[i]
        mean = sample.mean(axis=0)
        std = sample.std(axis=0)
        minimum = sample.min(axis=0)
        maximum = sample.max(axis=0)
        fft_vals = np.abs(np.fft.rfft(sample, axis=0))
        tremor_power = fft_vals[band_mask].mean(axis=0)
        feats.append(np.concatenate([mean, std, minimum, maximum, tremor_power]))
    return np.array(feats)

print("Extracting features (once, reused throughout)...")
X_train = extract_features(X_train_raw)
X_test = extract_features(X_test_raw)
print("Train features:", X_train.shape, " Test features:", X_test.shape)

# ---------------------------------------------------------
# General (non-personalized) baseline model
# Trained ONLY on original training participants -- never
# sees any test participant's data. This is the "before"
# comparison for every personalization experiment below.
# ---------------------------------------------------------
print("\nTraining general (non-personalized) baseline model...")
general_clf = RandomForestClassifier(
    n_estimators=200, max_depth=8, min_samples_leaf=5,
    class_weight="balanced", random_state=42, n_jobs=-1,
)
general_clf.fit(X_train, y_train)

# ---------------------------------------------------------
# Personalization experiment
#
# For each unseen test participant, and each combination of:
#   k          = number of that participant's own tasks used
#                for personalization
#   boost      = how much extra weight each personalization
#                sample gets relative to a normal training sample
#
# we: retrain a model on (original training set + that
# participant's k samples, weighted), then evaluate on the
# participant's REMAINING tasks, and compare against the
# general model's accuracy on those same remaining tasks.
# ---------------------------------------------------------
K_VALUES = [1, 3, 5]
WEIGHT_BOOSTS = [1, 3, 8, 15]   # 1 = no boost at all
RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

results = []  # one row per (participant, k, boost)

unique_test_ids = np.unique(test_ids)
print(f"\nRunning personalization for {len(unique_test_ids)} unseen participants, "
      f"k in {K_VALUES}, boost in {WEIGHT_BOOSTS}...")
print("(This will take a while -- retraining a model per participant per k per boost.)\n")

for k in K_VALUES:
    for boost in WEIGHT_BOOSTS:
        before_accs = []
        after_accs = []

        for pid in unique_test_ids:
            idx = np.where(test_ids == pid)[0]
            if len(idx) <= k:
                continue  # not enough of this participant's tasks to hold any out

            shuffled = rng.permutation(idx)
            personalize_idx = shuffled[:k]
            eval_idx = shuffled[k:]

            X_person = X_test[personalize_idx]
            y_person = y_test[personalize_idx]
            X_eval = X_test[eval_idx]
            y_eval = y_test[eval_idx]

            # BEFORE: general model's accuracy on this participant's held-out tasks
            before_preds = general_clf.predict(X_eval)
            before_acc = accuracy_score(y_eval, before_preds)

            # AFTER: retrain with this participant's k samples added,
            # weighted by `boost` relative to normal training samples
            X_aug = np.vstack([X_train, X_person])
            y_aug = np.concatenate([y_train, y_person])
            sample_weights = np.concatenate([
                np.ones(len(X_train)),
                np.full(len(X_person), float(boost)),
            ])

            personalized_clf = RandomForestClassifier(
                n_estimators=100, max_depth=8, min_samples_leaf=5,
                class_weight="balanced", random_state=42, n_jobs=-1,
            )
            personalized_clf.fit(X_aug, y_aug, sample_weight=sample_weights)
            after_preds = personalized_clf.predict(X_eval)
            after_acc = accuracy_score(y_eval, after_preds)

            before_accs.append(before_acc)
            after_accs.append(after_acc)
            results.append({
                "participant_id": pid,
                "k": k,
                "boost": boost,
                "before_acc": before_acc,
                "after_acc": after_acc,
            })

        print(f"k={k}, boost={boost}: BEFORE = {np.mean(before_accs):.3f}, "
              f"AFTER = {np.mean(after_accs):.3f}  (n={len(before_accs)} participants)")

# ---------------------------------------------------------
# Save and summarize
# ---------------------------------------------------------
results_df = pd.DataFrame(results)
results_df.to_csv("personalization_results.csv", index=False)

print("\n=========== SUMMARY: EFFECT OF PERSONALIZATION BY BOOST STRENGTH ===========")
summary = results_df.groupby(["k", "boost"])[["before_acc", "after_acc"]].mean()
summary["improvement"] = summary["after_acc"] - summary["before_acc"]
print(summary)

best_row = summary["improvement"].idxmax()
print(f"\nBest combination: k={best_row[0]}, boost={best_row[1]}  "
      f"(improvement = {summary.loc[best_row, 'improvement']:.3f})")

print("\nSaved full per-participant results to personalization_results.csv")