import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import pandas as pd

train = np.load("train_tasks.npz")
val = np.load("val_tasks.npz")
test = np.load("test_tasks.npz")

X_train_raw, y_train = train["X"], train["y"]
X_val_raw, y_val = val["X"], val["y"]
X_test_raw, y_test = test["X"], test["y"]

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

# Logistic Regression and MLP need scaled features; Random Forest doesn't,
# but scaling doesn't hurt it either, so we use the same scaled data for all three
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000, class_weight="balanced", random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=5,
        class_weight="balanced", random_state=42, n_jobs=-1,
    ),
    "Neural Network (MLP)": MLPClassifier(
        hidden_layer_sizes=(64, 32), max_iter=1000,
        early_stopping=True, random_state=42,
    ),
}

results = []
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_scaled, y_train)

    for split_name, X_split, y_split in [
        ("Train", X_train_scaled, y_train),
        ("Validation", X_val_scaled, y_val),
        ("Test (unseen)", X_test_scaled, y_test),
    ]:
        preds = model.predict(X_split)
        acc = accuracy_score(y_split, preds)
        bal_acc = balanced_accuracy_score(y_split, preds)
        results.append({"Model": name, "Split": split_name, "Accuracy": acc, "Balanced_Accuracy": bal_acc})

results_df = pd.DataFrame(results)
results_df.to_csv("model_benchmark_results.csv", index=False)

print("\n=========== MODEL BENCHMARK COMPARISON ===========")
pivot = results_df.pivot(index="Model", columns="Split", values="Balanced_Accuracy")
pivot = pivot[["Train", "Validation", "Test (unseen)"]]  # keep a sensible column order
print(pivot.round(3))

print("\nSaved full results to model_benchmark_results.csv")