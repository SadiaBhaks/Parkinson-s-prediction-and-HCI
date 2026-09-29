import pandas as pd
import os
from sklearn.model_selection import train_test_split

# Auto-locate file_list.csv anywhere under the current folder
target = None
for root, dirs, files in os.walk("."):
    if "file_list.csv" in files:
        target = os.path.join(root, "file_list.csv")
        break

if target is None:
    raise FileNotFoundError("Could not find file_list.csv anywhere under the current folder.")

print("Using file_list.csv at:", target)
df = pd.read_csv(target)

train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df["label"], random_state=42)

print(f"Train: {len(train_df)}  Val: {len(val_df)}  Test (unseen): {len(test_df)}")

train_df.to_csv("split_train.csv", index=False)
val_df.to_csv("split_val.csv", index=False)
test_df.to_csv("split_test.csv", index=False)
print("Saved split_train.csv, split_val.csv, split_test.csv in:", os.getcwd())