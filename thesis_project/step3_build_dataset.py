import pandas as pd
import numpy as np
import os

# Auto-locate file_list.csv, same trick as before
target = None
for root, dirs, files in os.walk("."):
    if "file_list.csv" in files:
        target = os.path.join(root, "file_list.csv")
        break

if target is None:
    raise FileNotFoundError("Could not find file_list.csv anywhere under the current folder.")

base = os.path.dirname(target)
movement_dir = os.path.join(base, "movement")
print("Using movement folder:", movement_dir)

TIMESTEPS = 976
CHANNELS = 132

def load_one(participant_id):
    """Load one participant's .bin file as a (976, 132) array."""
    pid = str(participant_id).zfill(3)
    path = os.path.join(movement_dir, f"{pid}_ml.bin")
    arr = np.fromfile(path, dtype=np.float32)
    if arr.size != TIMESTEPS * CHANNELS:
        raise ValueError(f"Unexpected size for participant {pid}: got {arr.size} values")
    return arr.reshape(TIMESTEPS, CHANNELS)

def build_split(csv_path):
    df = pd.read_csv(csv_path)
    X = []
    y = []
    ids = []
    skipped = []
    for _, row in df.iterrows():
        try:
            arr = load_one(row["id"])
            X.append(arr)
            y.append(row["label"])
            ids.append(row["id"])
        except Exception as e:
            skipped.append((row["id"], str(e)))
    X = np.stack(X)          # shape: (n_samples, 976, 132)
    y = np.array(y)
    ids = np.array(ids)
    print(f"{csv_path}: loaded {len(y)} samples, skipped {len(skipped)}")
    if skipped:
        print("  Skipped:", skipped[:5], "..." if len(skipped) > 5 else "")
    return X, y, ids

for split_name in ["split_train.csv", "split_val.csv", "split_test.csv"]:
    X, y, ids = build_split(split_name)
    out_name = split_name.replace("split_", "").replace(".csv", "_data.npz")
    np.savez_compressed(out_name, X=X, y=y, ids=ids)
    print(f"Saved {out_name}  ->  X shape: {X.shape}, y shape: {y.shape}\n")

print("Done. You now have train_data.npz, val_data.npz, test_data.npz")