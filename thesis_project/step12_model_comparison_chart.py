"""
STEP 12 - Model comparison table + figures (replaces the old chart_full_model_comparison script).

Reads:  model_benchmark_multiseed.csv , sklearn_test_predictions.csv   (step7b_benchmark_multiseed.py)
        tcn_multiseed.csv           , tcn_test_predictions.csv         (step10_tcn_multiseed.py)
Writes: full_model_comparison.csv          mean/SD per model x split  (Table I)
        model_gap_table.csv                train-test gap per model
        fig_model_comparison.png           balanced accuracy by split, whiskers = +-1 SD over seeds
        fig_model_test_and_gap.png         (a) test BA with 95% CI bootstrapped over PATIENTS, (b) train-test gap

Honest-uncertainty rules built in:
  * Fig 1 whiskers are SEED spread under ONE fixed split (model randomness only) - labelled as such.
  * The CI that actually reflects "new patients" is the patient bootstrap in Fig 2a (71 test patients is
    the effective sample size, not the ~1000 recordings).
  * A chance line (1/3) is drawn; models that don't clear it are visible at a glance.
Style: IEEE two-column width, greys + hatching (B/W-safe, colour-blind safe), one axis per chart.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

INK, MUTED, GRID = "#1a1a1a", "#6b6b6b", "#d9d9d9"
SPLITS = ["Train", "Validation", "Test (unseen)"]
SHADES = ["#d0d0d0", "#8c8c8c", "#2b2b2b"]
HATCH = ["", "", ""]
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False,
                     "axes.spines.right": False})
W2 = 7.16  # inches, IEEE two-column

# ------------------------------------------------------------------ load + combine
bench = pd.read_csv("model_benchmark_multiseed.csv")
tw = pd.read_csv("tcn_multiseed.csv")
tcn = (tw.melt(id_vars="seed", value_vars=["train_BA", "val_BA", "test_BA"], var_name="Split",
               value_name="Balanced_Accuracy").rename(columns={"seed": "Seed"}))
tcn["Split"] = tcn["Split"].map({"train_BA": "Train", "val_BA": "Validation", "test_BA": "Test (unseen)"})
tcn["Model"] = "TCN"
long = pd.concat([bench, tcn[bench.columns]], ignore_index=True)
preds = pd.concat([pd.read_csv("sklearn_test_predictions.csv"), pd.read_csv("tcn_test_predictions.csv")],
                  ignore_index=True)

ORDER = [m for m in ["Logistic Regression", "MLP", "Random Forest", "TCN"] if m in set(long["Model"])]
n_seeds = long.groupby("Model")["Seed"].nunique().to_dict()

stats = (long.groupby(["Model", "Split"])["Balanced_Accuracy"].agg(["mean", "std"]).fillna(0.0)
         .reindex(pd.MultiIndex.from_product([ORDER, SPLITS])))
stats.reset_index().rename(columns={"level_0": "Model", "level_1": "Split"}).to_csv(
    "full_model_comparison.csv", index=False)

wide = long.pivot_table(index=["Model", "Seed"], columns="Split", values="Balanced_Accuracy")[SPLITS]
wide["Gap"] = wide["Train"] - wide["Test (unseen)"]
gap = wide.groupby("Model")["Gap"].agg(["mean", "std"]).fillna(0.0).reindex(ORDER)
gap.to_csv("model_gap_table.csv")

# ------------------------------------------------------------------ patient-level bootstrap of test BA
def patient_bootstrap_ba(df, B=5000, seed=0):
    """BA = mean per-class recall. Resample PATIENTS with replacement; average BA over seeds."""
    rng = np.random.default_rng(seed)
    pids = np.sort(df["id"].unique()); P = len(pids)
    seeds = np.sort(df["Seed"].unique()); classes = np.sort(df["y"].unique())
    pidx = {p: i for i, p in enumerate(pids)}
    tot = np.zeros((P, len(seeds), len(classes))); cor = np.zeros_like(tot)
    for si, s in enumerate(seeds):
        d = df[df["Seed"] == s]
        for ci, c in enumerate(classes):
            dc = d[d["y"] == c]
            np.add.at(tot[:, si, ci], dc["id"].map(pidx).values, 1)
            np.add.at(cor[:, si, ci], dc[dc["pred"] == c]["id"].map(pidx).values, 1)
    point = np.nanmean(cor.sum(0) / np.where(tot.sum(0) == 0, np.nan, tot.sum(0)), axis=1).mean()
    Wt = rng.multinomial(P, np.full(P, 1 / P), size=B).astype(float)       # (B, P) resampling multiplicities
    num = np.einsum("bp,psc->bsc", Wt, cor); den = np.einsum("bp,psc->bsc", Wt, tot)
    with np.errstate(invalid="ignore", divide="ignore"):
        rec = np.where(den > 0, num / den, np.nan)
    ba = np.nanmean(rec, axis=2).mean(axis=1)
    return point, np.quantile(ba, .025), np.quantile(ba, .975), P


ci = {m: patient_bootstrap_ba(preds[preds["Model"] == m]) for m in ORDER}

# ------------------------------------------------------------------ console tables
print("=========== TABLE I (balanced accuracy, mean +- SD over seeds; fixed participant split) ===========")
tab = stats["mean"].unstack()[SPLITS].round(3).astype(str) + " +- " + stats["std"].unstack()[SPLITS].round(3).astype(str)
print(tab.to_string())
print("\n=========== TEST BA, 95% CI bootstrapped over test PATIENTS ===========")
for m in ORDER:
    p, lo, hi, P = ci[m]
    print(f"{m:20s} {p:.3f} [{lo:.3f}, {hi:.3f}]   (n_patients={P}, seeds={n_seeds[m]})")
print("\n=========== TRAIN-TEST GAP (mean +- SD over seeds) ===========")
print(gap.round(3).sort_values("mean", ascending=False).to_string())
low = [m for m in ORDER if stats.loc[(m, "Train"), "mean"] < 0.6 and stats.loc[(m, "Test (unseen)"), "mean"] < 0.6]
if low:
    print(f"\nWARNING: train BA < 0.6 for {low}: a small gap here can mean UNDERFITTING, not good generalization.")

# ------------------------------------------------------------------ Fig 1: grouped bars
x = np.arange(len(ORDER)); width = 0.26
fig, ax = plt.subplots(figsize=(W2, 3.1))
for i, split in enumerate(SPLITS):
    mu = np.array([stats.loc[(m, split), "mean"] for m in ORDER])
    sd = np.array([stats.loc[(m, split), "std"] for m in ORDER])
    ax.bar(x + (i - 1) * width, mu, width * 0.92, color=SHADES[i], edgecolor=INK, linewidth=0.7,
           label=split, zorder=2)
    ax.errorbar(x + (i - 1) * width, mu, yerr=sd, fmt="none", ecolor=INK, capsize=2, lw=0.9, zorder=3)
    for xi, m_, s_ in zip(x + (i - 1) * width, mu, sd):
        ax.text(xi, m_ + s_ + 0.012, f"{m_:.2f}", ha="center", fontsize=7, color=INK)
ax.axhline(1 / 3, color=MUTED, ls="--", lw=0.9, zorder=1, label="Chance (0.33)")
ax.set_xticks(x); ax.set_xticklabels(ORDER)
ax.set_ylabel("Balanced accuracy"); ax.set_ylim(0, 1.05)
ax.set_yticks(np.arange(0, 1.01, 0.2))
ax.yaxis.grid(True, color=GRID, lw=0.6, zorder=0); ax.set_axisbelow(True)
ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.13), frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig("fig_model_comparison.png", dpi=300, facecolor="white"); plt.close(fig)

# ------------------------------------------------------------------ Fig 2: (a) test BA + patient CI, (b) gap
fig, (a1, a2) = plt.subplots(1, 2, figsize=(W2, 2.5), gridspec_kw={"width_ratios": [1.15, 1]})
y = np.arange(len(ORDER))[::-1]
for yi, m in zip(y, ORDER):
    p, lo, hi, _ = ci[m]
    a1.plot([lo, hi], [yi, yi], color=INK, lw=1.4, solid_capstyle="butt", zorder=2)
    a1.plot(p, yi, "o", ms=6, mfc=INK, mec="white", mew=1.2, zorder=3)
    a1.text(hi + 0.012, yi, f"{p:.2f}", va="center", fontsize=7.5)
a1.axvline(1 / 3, color=MUTED, ls="--", lw=0.9, zorder=1)
a1.text(1 / 3, len(ORDER) - 0.45, "chance", ha="center", va="bottom", fontsize=7, color=MUTED)
a1.set_yticks(y); a1.set_yticklabels(ORDER)
a1.set_xlim(min(0.2, min(c[1] for c in ci.values()) - 0.04), max(0.7, max(c[2] for c in ci.values()) + 0.1)); a1.set_ylim(-0.6, len(ORDER) - 0.2)
a1.set_xlabel("Test balanced accuracy"); a1.set_title("(a) 95% CI over test patients", fontsize=8, loc="left")
a1.xaxis.grid(True, color=GRID, lw=0.6, zorder=0); a1.set_axisbelow(True)

gm = gap["mean"].values; gs = gap["std"].values
a2.barh(y, gm, 0.55, color="#2b2b2b", edgecolor=INK, zorder=2)
a2.errorbar(gm, y, xerr=gs, fmt="none", ecolor=MUTED, capsize=2, lw=0.9, zorder=3)
for yi, g_, s_ in zip(y, gm, gs):
    a2.text(max(g_ + s_, 0) + 0.01, yi, f"{g_:.2f}", va="center", fontsize=7.5)
a2.axvline(0, color=MUTED, lw=0.8)
a2.set_yticks(y); a2.set_yticklabels([])
a2.set_ylim(-0.6, len(ORDER) - 0.2)
a2.set_xlim(min(0, (gm - gs).min()) - 0.02, max(0.5, (gm + gs).max() + 0.1))
a2.set_xlabel("Train - test balanced accuracy"); a2.set_title("(b) Gap, $\\pm$1 SD over seeds", fontsize=8, loc="left")
a2.xaxis.grid(True, color=GRID, lw=0.6, zorder=0); a2.set_axisbelow(True)
fig.tight_layout(); fig.savefig("fig_model_test_and_gap.png", dpi=300, facecolor="white"); plt.close(fig)

print("\nSaved: full_model_comparison.csv, model_gap_table.csv, fig_model_comparison.png, fig_model_test_and_gap.png")
