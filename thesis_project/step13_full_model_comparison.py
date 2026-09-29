import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------
# Load and combine both result files
# ---------------------------------------------------------
benchmark = pd.read_csv("model_benchmark_results.csv")   # Logistic Regression, Random Forest, MLP
tcn = pd.read_csv("tcn_results.csv")                       # TCN

combined = pd.concat([benchmark, tcn], ignore_index=True)
combined.to_csv("full_model_comparison.csv", index=False)

print("=========== COMBINED COMPARISON TABLE ===========")
pivot = combined.pivot(index="Model", columns="Split", values="Balanced_Accuracy")
pivot = pivot[["Train", "Validation", "Test (unseen)"]]
print(pivot.round(3))

# ---------------------------------------------------------
# Grouped bar chart: one group per model, one bar per split
# ---------------------------------------------------------
models = pivot.index.tolist()
splits = ["Train", "Validation", "Test (unseen)"]
shades = ["#cccccc", "#888888", "#222222"]  # light -> dark, grayscale only

x = np.arange(len(models))
width = 0.25

fig, ax = plt.subplots(figsize=(8, 5))
for i, split in enumerate(splits):
    values = pivot[split].values
    bars = ax.bar(x + (i - 1) * width, values, width, label=split, color=shades[i], edgecolor="black")
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, f"{v:.2f}",
                ha="center", fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels(models, rotation=10)
ax.set_ylabel("Balanced Accuracy")
ax.set_ylim(0, 1.0)
ax.set_title("Model Comparison: Balanced Accuracy by Split")
ax.legend()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("chart_full_model_comparison.png", dpi=200, facecolor="white")
print("\nSaved chart_full_model_comparison.png")

# ---------------------------------------------------------
# Also compute and highlight the train-test GAP per model --
# this is your key "which model overfits vs underfits" evidence
# ---------------------------------------------------------
pivot["Train-Test Gap"] = pivot["Train"] - pivot["Test (unseen)"]
print("\n=========== GENERALIZATION GAP BY MODEL ===========")
print(pivot[["Train", "Test (unseen)", "Train-Test Gap"]].round(3).sort_values("Train-Test Gap", ascending=False))