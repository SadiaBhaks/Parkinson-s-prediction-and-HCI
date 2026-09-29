import pandas as pd
import os

target = None
for root, dirs, files in os.walk("."):
    if "file_list.csv" in files:
        target = os.path.join(root, "file_list.csv")
        break

if target is None:
    print("Could not find file_list.csv anywhere under the current folder.")
    print("Current folder is:", os.getcwd())
    print("Folders found here:", os.listdir("."))
else:
    print("Found it at:", target)
    base = os.path.dirname(target)
    df = pd.read_csv(target)
    print("Rows in file_list.csv:", len(df))
    print(df["condition"].value_counts())

    sample_id = str(df["id"].iloc[0]).zfill(3)
    bin_path = os.path.join(base, "movement", f"{sample_id}_ml.bin")
    if os.path.exists(bin_path):
        size_bytes = os.path.getsize(bin_path)
        print(f"\nFile: {bin_path}")
        print("Exact size in bytes:", size_bytes)
        n_floats = size_bytes // 4
        print("Total float32 values:", n_floats)
        print("If 1024 timesteps, channels =", n_floats / 1024)
    else:
        print("Found file_list.csv but not the matching .bin file at:", bin_path)