"""Shared helpers: feature extraction (identical to your step7), loading, RF factory, bootstrap CI."""
import numpy as np
from sklearn.ensemble import RandomForestClassifier

SAMPLING_RATE = 100
TREMOR_BAND = (4, 6)


def extract_features(X, chunk=256):
    """Same 5 stats x C channels as your original code (mean, std, min, max, 4-6 Hz FFT mag).
    Vectorised + chunked so it is fast and does not blow up memory."""
    n, T, C = X.shape
    freqs = np.fft.rfftfreq(T, d=1.0 / SAMPLING_RATE)
    band = (freqs >= TREMOR_BAND[0]) & (freqs <= TREMOR_BAND[1])
    out = []
    for s in range(0, n, chunk):
        xb = X[s:s + chunk]
        fft = np.abs(np.fft.rfft(xb, axis=1))
        out.append(np.concatenate([
            xb.mean(axis=1), xb.std(axis=1), xb.min(axis=1), xb.max(axis=1),
            fft[:, band, :].mean(axis=1),
        ], axis=1))
    return np.vstack(out)


def load_split(path):
    """Returns features, labels, participant ids. The .npz MUST contain 'ids'."""
    d = np.load(path, allow_pickle=True)
    if "ids" not in d.files:
        raise KeyError(f"{path} has no 'ids' array (found {d.files}). "
                       "Re-save it with participant ids per recording.")
    return extract_features(d["X"]), d["y"], d["ids"]


def make_rf(n_estimators=100, seed=42):
    # ONE configuration used for general AND personalized models (your old code used 200 vs 100 trees).
    return RandomForestClassifier(
        n_estimators=n_estimators, max_depth=8, min_samples_leaf=5,
        class_weight="balanced", random_state=seed, n_jobs=-1,
    )


def bootstrap_ci(values, n_boot=10000, seed=0, alpha=0.05):
    """Percentile bootstrap CI of the mean, resampling PARTICIPANTS (not recordings)."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    return values.mean(), np.quantile(means, alpha / 2), np.quantile(means, 1 - alpha / 2)
