import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

train = np.load("train_tasks.npz")
X_train_raw, y_train = train["X"], train["y"]

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

print("Extracting training features...")
X_train = extract_features(X_train_raw)

print("Training final general model...")
general_clf = RandomForestClassifier(
    n_estimators=300, max_depth=8, min_samples_leaf=5,
    class_weight="balanced", random_state=42, n_jobs=-1,
)
general_clf.fit(X_train, y_train)

# Save everything the app will need: the trained model, plus the
# training features themselves (needed to retrain a personalized
# version on the fly for a chosen patient)
joblib.dump(general_clf, "general_model.pkl")
np.savez_compressed("train_features.npz", X=X_train, y=y_train)

print("Saved general_model.pkl and train_features.npz")