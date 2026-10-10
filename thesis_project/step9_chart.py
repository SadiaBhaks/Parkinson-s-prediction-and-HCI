"""
STEP 11 - Paper figures for the CORRECTED personalization experiment (step8b_personalization_fixed.py).

Reads:  personalization_validation_grid.csv   (k, w, delta)   <- where (k, w) was chosen (validation only)
        personalization_test_rows.csv         (pid, seed, k, w, before_acc, after_acc)
        personalization_control_rows.csv      (same columns; same-class training-sample control)
Writes: fig_personalization_validation_grid.png   (replaces old heatmap, Fig. 2)
        fig_personalization_effect.png            (replaces old before/after bars, Fig. 3: now with CIs + control)
        fig_personalization_distribution.png      (per-patient change; shows the non-uniformity finding)

Why the old chart script can't be reused: the test file now holds ONLY the validation-chosen (k, w), so a
test-set k x boost heatmap / "best boost across k" bars no longer exist, by design. The grid heatmap now
comes from VALIDATION participants. Put details (n, CI definition, "outlined = selected") in the LaTeX caption.
Style: IEEE single-column width, greys + hatching (prints in B/W, colour-blind safe), one axis per chart.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

INK, MUTED, GRID = "#1a1a1a", "#6b6b6b", "#d9d9d9"
DARK, MID, LIGHT = "#2b2b2b", "#8c8c8c", "#d0d0d0"
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False, "hatch.linewidth": 0.8})
W = 3.5  # inches, IEEE single column


def boot_ci(v, n=10000, seed=0):
    v = np.asarray(v, float); rng = np.random.default_rng(seed)
    m = v[rng.integers(0, len(v), (n, len(v)))].mean(1)
    return v.mean(), np.quantile(m, .025), np.quantile(m, .975)


def per_patient(df):  # average over seeds first: participants are the independent unit
    return df.groupby("pid")[["before_acc", "after_acc"]].mean()


val = pd.read_csv("personalization_validation_grid.csv")
te = pd.read_csv("personalization_test_rows.csv")
ct = pd.read_csv("personalization_control_rows.csv")
K, WGT = int(te["k"].iloc[0]), int(te["w"].iloc[0])           # the validation-chosen config
pt, pc = per_patient(te), per_patient(ct)
d_te = (pt["after_acc"] - pt["before_acc"]).values
n = len(pt)

# ------------------------------------------------------------------ Fig A: validation grid heatmap
pivot = val.pivot(index="k", columns="w", values="delta").sort_index()
lo_v, hi_v = min(0.0, pivot.values.min()), max(pivot.values.max(), 1e-6)
fig, ax = plt.subplots(figsize=(W, 2.3))
im = ax.imshow(pivot.values, cmap="Greys", vmin=lo_v, vmax=hi_v, aspect="auto")
ax.set_xticks(range(pivot.shape[1])); ax.set_xticklabels(pivot.columns)
ax.set_yticks(range(pivot.shape[0])); ax.set_yticklabels(pivot.index)
ax.set_xlabel("Sample weight $w$"); ax.set_ylabel("Samples $k$")
for i in range(pivot.shape[0]):
    for j in range(pivot.shape[1]):
        v = pivot.values[i, j]
        ax.text(j, i, f"{v:+.3f}", ha="center", va="center", fontsize=7.5,
                color="white" if (v - lo_v) > 0.55 * (hi_v - lo_v) else INK)
bi, bj = list(pivot.index).index(K), list(pivot.columns).index(WGT)
ax.add_patch(Rectangle((bj - .5, bi - .5), 1, 1, fill=False, ec=INK, lw=3.2))      # visible on light AND dark cells
ax.add_patch(Rectangle((bj - .5, bi - .5), 1, 1, fill=False, ec="white", lw=1.2))
ax.set_title("Validation: mean accuracy change", fontsize=8.5)
for s in ax.spines.values(): s.set_visible(False)
cb = fig.colorbar(im, ax=ax, pad=0.02); cb.outline.set_visible(False); cb.ax.tick_params(labelsize=7)
fig.tight_layout(); fig.savefig("fig_personalization_validation_grid.png", dpi=300, facecolor="white")
plt.close(fig)

# ------------------------------------------------------------------ Fig B: general vs control vs personalized
gen_m, gen_lo, gen_hi = boot_ci(pt["before_acc"])
per_m, per_lo, per_hi = boot_ci(pt["after_acc"])
ctl_m, ctl_lo, ctl_hi = boot_ci(pc["after_acc"])
bars = [("General", gen_m, gen_lo, gen_hi, LIGHT, ""),
        ("Control\n(same class)", ctl_m, ctl_lo, ctl_hi, "white", "////"),
        ("Personalized\n(own data)", per_m, per_lo, per_hi, DARK, "")]
top = max(b[3] for b in bars)
fig, ax = plt.subplots(figsize=(W, 2.5))
for i, (lab, m, lo, hi, fc, ht) in enumerate(bars):
    ax.bar(i, m, 0.55, color=fc, edgecolor=INK, hatch=ht, linewidth=0.8, zorder=2)
    ax.errorbar(i, m, yerr=[[m - lo], [hi - m]], color=INK, capsize=3, lw=1, zorder=3)
    ax.text(i, 0.025, f"{m:.3f}", ha="center", va="bottom", fontsize=8,
            color="white" if fc == DARK else INK,
            bbox=dict(fc="white", ec="none", pad=1.5) if ht else None, zorder=4)
ax.set_xticks(range(3)); ax.set_xticklabels([b[0] for b in bars], fontsize=7.5)
ax.set_xlim(-0.6, 2.6)
ax.set_ylim(0, min(1.05, top + 0.08)); ax.set_yticks(np.arange(0, ax.get_ylim()[1], 0.2))
ax.set_ylabel("Accuracy (held-out)")
ax.yaxis.grid(True, color=GRID, lw=0.6, zorder=0); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig("fig_personalization_effect.png", dpi=300, facecolor="white"); plt.close(fig)

# ------------------------------------------------------------------ Fig C: per-patient change distribution
eps = 1e-9
imp, same, worse = (d_te > eps).mean(), (np.abs(d_te) <= eps).mean(), (d_te < -eps).mean()
m, lo, hi = boot_ci(d_te)
step = 0.05
half = np.ceil((np.abs(d_te).max() + 1e-9) / step) * step + step
edges = np.arange(-half - step / 2, half + step, step)            # a bin is centred on 0
cnt, e = np.histogram(d_te, bins=edges)
ctr = (e[:-1] + e[1:]) / 2
fig, ax = plt.subplots(figsize=(W, 2.5))
col = [DARK if c > eps else (LIGHT if c < -eps else MID) for c in ctr]
ax.bar(ctr, cnt, width=step * 0.9, color=col, edgecolor=INK, linewidth=0.6, zorder=2)
ax.axvspan(lo, hi, color=GRID, alpha=0.7, zorder=0)
ax.axvline(0, color=MUTED, lw=0.8, zorder=1)
ax.axvline(m, color=INK, lw=1.4, zorder=3)
nz = np.nonzero(cnt)[0]
ax.set_xlim(ctr[nz[0]] - 3 * step, ctr[nz[-1]] + 3 * step)
ax.set_ylim(0, cnt.max() * 1.35)
ax.text(0.985, 0.97, f"mean {m:+.3f}\n95% CI [{lo:+.3f}, {hi:+.3f}]", transform=ax.transAxes,
        va="top", ha="right", fontsize=7, color=INK)
ax.set_xlabel("Per-patient accuracy change")
ax.set_ylabel("Patients")
ax.yaxis.grid(True, color=GRID, lw=0.6, zorder=0); ax.set_axisbelow(True)
ax.set_title(f"Improved {imp:.0%}  |  unchanged {same:.0%}  |  worsened {worse:.0%}", fontsize=8)
fig.tight_layout(); fig.savefig("fig_personalization_distribution.png", dpi=300, facecolor="white"); plt.close(fig)

print(f"chosen config: k={K}, w={WGT}, n_patients={n}")
print(f"general {gen_m:.3f} [{gen_lo:.3f},{gen_hi:.3f}] | personalized {per_m:.3f} [{per_lo:.3f},{per_hi:.3f}] | "
      f"control {ctl_m:.3f} [{ctl_lo:.3f},{ctl_hi:.3f}]")
print(f"change: mean {m:+.3f} [{lo:+.3f},{hi:+.3f}] | improved {imp:.0%} unchanged {same:.0%} worsened {worse:.0%}")
print("Saved 3 PNGs: fig_personalization_validation_grid / _effect / _distribution")