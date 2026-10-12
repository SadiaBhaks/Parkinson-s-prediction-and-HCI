"""
STEP 14 - Prints what the paper's reproducibility section needs: software versions and a fingerprint of
your existing train/val/test split. Usage: python step14_env_and_split_info.py --data-dir .
NOTE: the random seed that originally produced your split is NOT stored in the .npz files. Look for it in
the script that created train_tasks/val_tasks/test_tasks (e.g. random_state=...). If you cannot find it, say in
the paper that the split is fixed and provided as participant-ID lists rather than quoting a seed.
"""
import argparse, hashlib, os, platform, sys
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("--data-dir", default="."); args = ap.parse_args()
print("Python", sys.version.split()[0], "|", platform.platform())
for mod in ("numpy", "pandas", "scipy", "sklearn", "torch", "matplotlib"):
    try:
        m = __import__(mod); print(f"{mod:11s}", m.__version__)
    except Exception:
        print(f"{mod:11s} not installed")
print()
tot = 0
for name in ("train", "val", "test"):
    p = os.path.join(args.data_dir, f"{name}_tasks.npz")
    if not os.path.exists(p):
        continue
    d = np.load(p, allow_pickle=True); ids, y = d["ids"], d["y"]
    uid = np.unique(ids)
    lab = {u: y[ids == u][0] for u in uid}
    cls = np.bincount([lab[u] for u in uid], minlength=3)
    h = hashlib.md5(",".join(sorted(map(str, uid))).encode()).hexdigest()[:10]
    print(f"{name:5s}: {len(uid)} participants (class 0/1/2 = {cls.tolist()}), {len(y)} recordings, "
          f"X shape {d['X'].shape}, id-list md5 {h}")
    tot += len(uid)
print("total participants:", tot)
