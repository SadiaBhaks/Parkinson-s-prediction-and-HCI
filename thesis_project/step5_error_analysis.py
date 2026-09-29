import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# ---------------------------------------------------------
# Reload data and retrain the same baseline (quick, ~seconds)
# ---------------------------------------------------------
train = np.load("train_data.npz")
test = np.load("test_data.npz")

X_train_raw, y_train = train["X"], train["y"]
X_test_raw, y_test, test_ids = test["X"], test["y"], test["ids"]

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

X_train = extract_features(X_train_raw)
X_test = extract_features(X_test_raw)

clf = RandomForestClassifier(
    n_estimators=300, max_depth=8, min_samples_leaf=5,
    class_weight="balanced", random_state=42, n_jobs=-1,
)
clf.fit(X_train, y_train)
preds = clf.predict(X_test)

# ---------------------------------------------------------
# Merge predictions with participant metadata
# ---------------------------------------------------------
meta = pd.read_csv("split_test.csv")
meta = meta.set_index("id").loc[test_ids].reset_index()  # match order to test_ids

results = pd.DataFrame({
    "id": test_ids,
    "true_label": y_test,
    "predicted_label": preds,
    "correct": y_test == preds,
})
results = results.merge(meta[["id", "condition", "age", "gender", "handedness"]], on="id")

label_names = {0: "Healthy", 1: "Parkinson's", 2: "Other/DD"}
results["true_name"] = results["true_label"].map(label_names)
results["predicted_name"] = results["predicted_label"].map(label_names)

print("=========== OVERALL TEST ACCURACY ===========")
print(f"{results['correct'].mean():.3f}  ({results['correct'].sum()}/{len(results)} correct)\n")

# ---------------------------------------------------------
# Break down accuracy by different participant factors
# ---------------------------------------------------------
print("=========== ACCURACY BY DIAGNOSIS GROUP (condition) ===========")
print(results.groupby("condition")["correct"].agg(["mean", "count"]).sort_values("mean"))

print("\n=========== ACCURACY BY GENDER ===========")
print(results.groupby("gender")["correct"].agg(["mean", "count"]))

print("\n=========== ACCURACY BY HANDEDNESS ===========")
print(results.groupby("handedness")["correct"].agg(["mean", "count"]))

print("\n=========== ACCURACY BY AGE GROUP ===========")
results["age_group"] = pd.cut(results["age"], bins=[0, 50, 65, 100], labels=["Under 50", "50-65", "Over 65"])
print(results.groupby("age_group", observed=True)["correct"].agg(["mean", "count"]))

print("\n=========== WORST-PERFORMING CONDITIONS (detail) ===========")
worst = results[~results["correct"]].groupby("condition").size().sort_values(ascending=False)
print(worst)

# Save full results for your thesis appendix / supervisor
results.to_csv("error_analysis_results.csv", index=False)
print("\nSaved full breakdown to error_analysis_results.csv")