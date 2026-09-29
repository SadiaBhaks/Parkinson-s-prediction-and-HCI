import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
import pandas as pd

train = np.load("train_tasks.npz")
val = np.load("val_tasks.npz")
test = np.load("test_tasks.npz")

X_train_raw, y_train, train_ids = train["X"], train["y"], train["ids"]
X_val_raw, y_val, val_ids = val["X"], val["y"], val["ids"]
X_test_raw, y_test, test_ids = test["X"], test["y"], test["ids"]

print("Raw shapes -> train:", X_train_raw.shape, "val:", X_val_raw.shape, "test:", X_test_raw.shape)

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

print("Extracting features...")
X_train = extract_features(X_train_raw)
X_val = extract_features(X_val_raw)
X_test = extract_features(X_test_raw)
print("Feature shapes -> train:", X_train.shape, "val:", X_val.shape, "test:", X_test.shape)

clf = RandomForestClassifier(
    n_estimators=300, max_depth=8, min_samples_leaf=5,
    class_weight="balanced", random_state=42, n_jobs=-1,
)
clf.fit(X_train, y_train)

label_names = {0: "Healthy (HC)", 1: "Parkinson's (PD)", 2: "Other/DD"}

def evaluate(name, X, y):
    preds = clf.predict(X)
    acc = accuracy_score(y, preds)
    bal_acc = balanced_accuracy_score(y, preds)
    print(f"\n--- {name} (per task-recording) ---")
    print(f"Accuracy: {acc:.3f}   Balanced Accuracy: {bal_acc:.3f}")
    print(confusion_matrix(y, preds, labels=[0, 1, 2]))
    print(classification_report(y, preds, target_names=[label_names[k] for k in [0, 1, 2]], zero_division=0))
    return preds, acc, bal_acc

train_preds, train_acc, train_bal = evaluate("Training set", X_train, y_train)
val_preds, val_acc, val_bal = evaluate("Validation set", X_val, y_val)
test_preds, test_acc, test_bal = evaluate("TEST SET (unseen patients)", X_test, y_test)

# ---------------------------------------------------------
# Also report PARTICIPANT-level accuracy via majority vote
# across that participant's task recordings -- this is the
# number that's actually comparable to your earlier
# participant-level baseline (step4).
# ---------------------------------------------------------
def majority_vote_by_participant(preds, ids, true_labels):
    df = pd.DataFrame({"id": ids, "pred": preds, "true": true_labels})
    grouped = df.groupby("id").agg(
        pred_majority=("pred", lambda x: x.value_counts().idxmax()),
        true_label=("true", "first"),
    )
    acc = (grouped["pred_majority"] == grouped["true_label"]).mean()
    return acc, grouped

test_participant_acc, test_grouped = majority_vote_by_participant(test_preds, test_ids, y_test)
print(f"\n=== TEST SET participant-level accuracy (majority vote across their tasks): {test_participant_acc:.3f} ===")

print("\n=========== SUMMARY TABLE (per task-recording) ===========")
print(f"{'Split':<30}{'Accuracy':<12}{'Balanced Acc.':<15}")
print(f"{'Train (known)':<30}{train_acc:<12.3f}{train_bal:<15.3f}")
print(f"{'Validation':<30}{val_acc:<12.3f}{val_bal:<15.3f}")
print(f"{'Test (UNSEEN patients)':<30}{test_acc:<12.3f}{test_bal:<15.3f}")
print(f"\nTest participant-level accuracy (majority vote): {test_participant_acc:.3f}")