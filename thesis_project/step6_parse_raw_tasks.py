import os
import json
import traceback
import numpy as np
import pandas as pd

# ---------------------------------------------------------
# Step 1: Locate the raw "movement" folder
# ---------------------------------------------------------
movement_dir = None
for root, dirs, files in os.walk("."):
    if "timeseries" in dirs and any(f.startswith("observation_") for f in files):
        movement_dir = root
        break

if movement_dir is None:
    raise FileNotFoundError("Could not find the raw 'movement' folder (with observation_*.json files).")

timeseries_dir = os.path.join(movement_dir, "timeseries")
print("Using raw movement folder:", movement_dir)
print("Using timeseries folder:", timeseries_dir)

# ---------------------------------------------------------
# Step 2: Robust loader for one sensor .txt file
# ---------------------------------------------------------
def load_sensor_file(txt_path, expected_channels):
    df = pd.read_csv(txt_path, sep=None, engine="python", header=None)

    first_row_numeric = pd.to_numeric(df.iloc[0], errors="coerce").notna().all()
    if not first_row_numeric:
        df = df.iloc[1:].reset_index(drop=True)

    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=1, how="all")

    arr = df.to_numpy(dtype=np.float32)

    if arr.shape[1] > expected_channels:
        arr = arr[:, -expected_channels:]
    elif arr.shape[1] < expected_channels:
        raise ValueError(f"File has fewer columns ({arr.shape[1]}) than expected ({expected_channels}): {txt_path}")

    return arr

# ---------------------------------------------------------
# Step 3: Parser for one participant, with an optional debug mode
# that prints every session's shapes instead of silently continuing
# ---------------------------------------------------------
def parse_participant(pid, debug=False):
    pid_str = str(pid).zfill(3)
    json_path = os.path.join(movement_dir, f"observation_{pid_str}.json")
    with open(json_path) as f:
        obs = json.load(f)

    steps = []
    for session in obs["session"]:
        task_name = session["record_name"]
        rows = session["rows"]

        wrist_arrays = {}
        for record in session["records"]:
            location = record["device_location"]
            n_channels = len(record["channels"])
            file_name = os.path.basename(record["file_name"])
            txt_path = os.path.join(timeseries_dir, file_name)
            arr = load_sensor_file(txt_path, n_channels)
            wrist_arrays[location] = arr

        if debug:
            print(f"  Task: {task_name:20s} rows(json)={rows:6d}  "
                  f"LeftWrist shape={wrist_arrays.get('LeftWrist', np.array([])).shape}  "
                  f"RightWrist shape={wrist_arrays.get('RightWrist', np.array([])).shape}")

        if "LeftWrist" not in wrist_arrays or "RightWrist" not in wrist_arrays:
            if debug:
                print("    -> SKIPPED: missing a wrist")
            continue

        left = wrist_arrays["LeftWrist"]
        right = wrist_arrays["RightWrist"]
        min_len = min(len(left), len(right))
        combined = np.hstack([left[:min_len], right[:min_len]])

        if debug:
            print(f"    -> combined shape={combined.shape}")

        if rows >= 2048:
            half = min_len // 2
            steps.append((f"{task_name}1", combined[:half]))
            steps.append((f"{task_name}2", combined[half:half * 2]))
        else:
            steps.append((task_name, combined))

    return steps

# ---------------------------------------------------------
# DEBUG RUN: parse participant #1 verbosely before doing anything else
# ---------------------------------------------------------
print("\n=========== DEBUG: parsing participant 001 verbosely ===========")
debug_steps = parse_participant(1, debug=True)
print(f"\nTotal steps returned for participant 001: {len(debug_steps)}")
for name, arr in debug_steps:
    print(f"  Step '{name}': shape={arr.shape}")
print("=========== END DEBUG ===========\n")

# ---------------------------------------------------------
# Step 4: Build dataset for a split file
# ---------------------------------------------------------
def build_task_split(csv_path, target_len=1024, show_first_error=True):
    df = pd.read_csv(csv_path)
    X, y, ids, task_names = [], [], [], []
    skipped = []
    zero_step_participants = 0
    first_error_shown = not show_first_error

    for _, row in df.iterrows():
        try:
            steps = parse_participant(row["id"])
            if len(steps) == 0:
                zero_step_participants += 1
            for task_name, arr in steps:
                if arr.shape[0] < target_len:
                    continue
                arr = arr[:target_len]
                if arr.shape != (target_len, 14):
                    continue
                X.append(arr)
                y.append(row["label"])
                ids.append(row["id"])
                task_names.append(task_name)
        except Exception as e:
            skipped.append((row["id"], str(e)))
            if not first_error_shown:
                print(f"\n=== FULL ERROR for participant {row['id']} ===")
                traceback.print_exc()
                first_error_shown = True

    print(f"{csv_path}: participants with zero usable steps: {zero_step_participants} / {len(df)}")

    if len(X) == 0:
        raise RuntimeError(f"No samples loaded from {csv_path}. See debug output above.")

    X = np.stack(X).astype(np.float32)
    y = np.array(y)
    ids = np.array(ids)
    task_names = np.array(task_names)

    print(f"{csv_path}: {len(y)} task-level samples from {df['id'].nunique()} participants, skipped {len(skipped)} with errors")
    return X, y, ids, task_names

for split_name in ["split_train.csv", "split_val.csv", "split_test.csv"]:
    X, y, ids, task_names = build_task_split(split_name)
    out_name = split_name.replace("split_", "").replace(".csv", "_tasks.npz")
    np.savez_compressed(out_name, X=X, y=y, ids=ids, task_names=task_names)
    print(f"Saved {out_name} -> X shape: {X.shape}\n")

print("Done. You now have train_tasks.npz, val_tasks.npz, test_tasks.npz")