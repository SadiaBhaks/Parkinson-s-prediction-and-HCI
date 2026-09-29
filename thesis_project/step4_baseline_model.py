import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix

# ---------------------------------------------------------
# Step 1: Load the three ready datasets
# ---------------------------------------------------------
train = np.load("train_data.npz")
val = np.load("val_data.npz")
test = np.load("test_data.npz")

X_train_raw, y_train = train["X"], train["y"]
X_val_raw, y_val = val["X"], val["y"]
X_test_raw, y_test = test["X"], test["y"]

print("Raw shapes -> train:", X_train_raw.shape, "val:", X_val_raw.shape, "test:", X_test_raw.shape)

# ---------------------------------------------------------
# Step 2: Turn each participant's (976, 132) matrix into a
# small set of hand-engineered features per channel.
# This matches "Step 3 - Baseline Models" in your proposal:
# mean, variance, min/max, and tremor-band FFT power.
# ---------------------------------------------------------
SAMPLING_RATE = 100  # Hz, from the PADS documentation
TREMOR_BAND = (4, 6)  # Hz, typical parkinsonian tremor frequency range

def extract_features(X):
    """X: (n_samples, timesteps, channels) -> (n_samples, n_features)"""
    n_samples, timesteps, channels = X.shape
    feats = []

    freqs = np.fft.rfftfreq(timesteps, d=1.0 / SAMPLING_RATE)
    band_mask = (freqs >= TREMOR_BAND[0]) & (freqs <= TREMOR_BAND[1])

    for i in range(n_samples):
        sample = X[i]  # (timesteps, channels)
        mean = sample.mean(axis=0)
        std = sample.std(axis=0)
        minimum = sample.min(axis=0)
        maximum = sample.max(axis=0)

        # FFT power in the tremor band, per channel
        fft_vals = np.abs(np.fft.rfft(sample, axis=0))
        tremor_power = fft_vals[band_mask].mean(axis=0)

        row = np.concatenate([mean, std, minimum, maximum, tremor_power])
        feats.append(row)

    return np.array(feats)

print("Extracting features...")
X_train = extract_features(X_train_raw)
X_val = extract_features(X_val_raw)
X_test = extract_features(X_test_raw)
print("Feature shapes -> train:", X_train.shape, "val:", X_val.shape, "test:", X_test.shape)

# ---------------------------------------------------------
# Step 3: Train the baseline Random Forest
# ---------------------------------------------------------
clf = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,              # was None (unlimited) — this was the main cause of overfitting
    min_samples_leaf=5,       # forces each leaf to represent at least 5 samples, not 1
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
clf.fit(X_train, y_train)

# ---------------------------------------------------------
# Step 4: Evaluate on all three splits
# ---------------------------------------------------------
label_names = {0: "Healthy (HC)", 1: "Parkinson's (PD)", 2: "Other/DD"}

def evaluate(name, X, y):
    preds = clf.predict(X)
    acc = accuracy_score(y, preds)
    bal_acc = balanced_accuracy_score(y, preds)
    print(f"\n--- {name} ---")
    print(f"Accuracy: {acc:.3f}   Balanced Accuracy: {bal_acc:.3f}")
    print("Confusion matrix (rows=true, cols=predicted), order [0,1,2]:")
    print(confusion_matrix(y, preds, labels=[0, 1, 2]))
    print(classification_report(y, preds, target_names=[label_names[k] for k in [0, 1, 2]], zero_division=0))
    return acc, bal_acc

print("\n=========== RESULTS ===========")
train_acc, train_bal = evaluate("Training set (known participants)", X_train, y_train)
val_acc, val_bal = evaluate("Validation set", X_val, y_val)
test_acc, test_bal = evaluate("TEST SET (unseen patients)", X_test, y_test)

print("\n=========== SUMMARY TABLE ===========")
print(f"{'Split':<30}{'Accuracy':<12}{'Balanced Acc.':<15}")
print(f"{'Train (known)':<30}{train_acc:<12.3f}{train_bal:<15.3f}")
print(f"{'Validation':<30}{val_acc:<12.3f}{val_bal:<15.3f}")
print(f"{'Test (UNSEEN patients)':<30}{test_acc:<12.3f}{test_bal:<15.3f}")