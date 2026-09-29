import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

results = pd.read_csv("personalization_results.csv")

# ---------------------------------------------------------
# Chart 1: Before vs After accuracy, for the best boost value,
# across different amounts of personalization data (k)
# ---------------------------------------------------------
BEST_BOOST = 8  # from your results: k=5, boost=8 was the best combination

subset = results[results["boost"] == BEST_BOOST]
summary = subset.groupby("k")[["before_acc", "after_acc"]].mean()

fig, ax = plt.subplots(figsize=(6.5, 4.2))
x = np.arange(len(summary.index))
width = 0.35

ax.bar(x - width/2, summary["before_acc"], width, label="Before personalization (general model)",
       color="#888888", edgecolor="black")
ax.bar(x + width/2, summary["after_acc"], width, label=f"After personalization (boost={BEST_BOOST})",
       color="#333333", edgecolor="black")

ax.set_xticks(x)
ax.set_xticklabels([f"k={k}" for k in summary.index])
ax.set_ylabel("Accuracy on held-out tasks")
ax.set_title("Effect of Personalization on Unseen-Patient Accuracy")
ax.set_ylim(0, 0.8)
ax.legend()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

for i, k in enumerate(summary.index):
    ax.text(i - width/2, summary["before_acc"].iloc[i] + 0.01, f"{summary['before_acc'].iloc[i]:.2f}",
            ha="center", fontsize=9)
    ax.text(i + width/2, summary["after_acc"].iloc[i] + 0.01, f"{summary['after_acc'].iloc[i]:.2f}",
            ha="center", fontsize=9, fontweight="bold")

plt.tight_layout()
plt.savefig("chart_personalization_before_after.png", dpi=200, facecolor="white")
print("Saved chart_personalization_before_after.png")
plt.close()

# ---------------------------------------------------------
# Chart 2: Heatmap of improvement across all (k, boost) combos
# ---------------------------------------------------------
pivot = results.groupby(["k", "boost"])["after_acc"].mean().unstack() - \
        results.groupby(["k", "boost"])["before_acc"].mean().unstack()

fig, ax = plt.subplots(figsize=(6, 4))
im = ax.imshow(pivot.values, cmap="Greys", aspect="auto")

ax.set_xticks(range(len(pivot.columns)))
ax.set_xticklabels([f"boost={b}" for b in pivot.columns])
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels([f"k={k}" for k in pivot.index])
ax.set_title("Personalization Improvement\n(after_acc \u2212 before_acc)")

for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        val = pivot.values[i, j]
        color = "white" if val > pivot.values.max() * 0.6 else "black"
        ax.text(j, i, f"{val:+.3f}", ha="center", va="center", color=color, fontsize=10)

plt.colorbar(im, ax=ax, label="Accuracy improvement")
plt.tight_layout()
plt.savefig("chart_personalization_heatmap.png", dpi=200, facecolor="white")
print("Saved chart_personalization_heatmap.png")
plt.close()

print("\nBoth charts saved in your thesis_project folder.")