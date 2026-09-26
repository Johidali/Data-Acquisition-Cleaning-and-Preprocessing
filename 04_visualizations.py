import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
})
PALETTE = ["#2E4374", "#C06C3E", "#4B8B6F", "#A63D40", "#7A6FA6"]

raw = pd.read_csv("data/titanic_raw.csv")
clean = pd.read_csv("data/titanic_cleaned.csv")

# 1. Missingness bar chart -------------------------------------------------
miss = raw.isnull().mean().sort_values(ascending=False)
miss = miss[miss > 0] * 100
fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.barh(miss.index[::-1], miss.values[::-1], color=PALETTE[0])
ax.set_xlabel("% missing")
ax.set_title("Missing Values by Column (Raw Dataset)")
for b, v in zip(bars, miss.values[::-1]):
    ax.text(v + 1, b.get_y() + b.get_height()/2, f"{v:.1f}%", va="center", fontsize=10)
ax.set_xlim(0, 90)
plt.tight_layout()
plt.savefig("figures/01_missingness.png")
plt.close()

# 2. Age distribution before/after imputation ------------------------------
fig, axes = plt.subplots(1, 2, figsize=(9.5, 4), sharey=True)
axes[0].hist(raw["age"].dropna(), bins=30, color=PALETTE[0], edgecolor="white")
axes[0].set_title("Age Distribution — Raw\n(missing values excluded)")
axes[0].set_xlabel("Age")
axes[0].set_ylabel("Count")
axes[1].hist(clean["age"], bins=30, color=PALETTE[1], edgecolor="white")
axes[1].set_title("Age Distribution — After\nGroup-wise Median Imputation")
axes[1].set_xlabel("Age")
plt.tight_layout()
plt.savefig("figures/02_age_before_after.png")
plt.close()

# 3. Fare boxplot with outliers --------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4))
bp = ax.boxplot(raw["fare"], vert=False, patch_artist=True, widths=0.5,
                 flierprops=dict(marker="o", markerfacecolor=PALETTE[3], markersize=4, alpha=0.6))
for patch in bp["boxes"]:
    patch.set_facecolor(PALETTE[0])
    patch.set_alpha(0.6)
ax.set_xlabel("Fare ($)")
ax.set_title("Fare Distribution — Boxplot (IQR Outlier View)")
ax.set_yticks([])
plt.tight_layout()
plt.savefig("figures/03_fare_boxplot.png")
plt.close()

# 4. Fare distribution: raw vs log1p ---------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(9.5, 4))
axes[0].hist(raw["fare"], bins=40, color=PALETTE[0], edgecolor="white")
axes[0].set_title("Fare — Raw (right-skewed)")
axes[0].set_xlabel("Fare ($)")
axes[0].set_ylabel("Count")
axes[1].hist(clean["fare_log"], bins=40, color=PALETTE[2], edgecolor="white")
axes[1].set_title("Fare — log1p Transformed")
axes[1].set_xlabel("log(1 + Fare)")
plt.tight_layout()
plt.savefig("figures/04_fare_log_transform.png")
plt.close()

# 5. Survival rate by class & sex (post-cleaning sanity/insight chart) -----
pivot = clean.pivot_table(index="pclass", columns="sex", values="survived", aggfunc="mean")
fig, ax = plt.subplots(figsize=(6.5, 4.2))
pivot.plot(kind="bar", ax=ax, color=[PALETTE[0], PALETTE[1]])
ax.set_ylabel("Survival Rate")
ax.set_xlabel("Passenger Class")
ax.set_title("Survival Rate by Class and Sex (Cleaned Data)")
ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
ax.set_xticklabels(["1st", "2nd", "3rd"], rotation=0)
ax.legend(title="Sex")
plt.tight_layout()
plt.savefig("figures/05_survival_by_class_sex.png")
plt.close()

# 6. Correlation heatmap (numeric, cleaned) --------------------------------
num_cols = ["survived", "pclass", "age", "sibsp", "parch", "fare",
            "family_size", "fare_per_person", "deck_recorded"]
corr = clean[num_cols].corr()
fig, ax = plt.subplots(figsize=(7.5, 6.5))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(num_cols)))
ax.set_yticks(range(len(num_cols)))
ax.set_xticklabels(num_cols, rotation=45, ha="right")
ax.set_yticklabels(num_cols)
for i in range(len(num_cols)):
    for j in range(len(num_cols)):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center",
                fontsize=8, color="white" if abs(corr.iloc[i, j]) > 0.5 else "black")
fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
ax.set_title("Correlation Matrix — Cleaned Numeric Features")
plt.tight_layout()
plt.savefig("figures/06_correlation_heatmap.png")
plt.close()

# 7. Missingness heatmap style overview ------------------------------------
fig, ax = plt.subplots(figsize=(8, 4))
missing_matrix = raw[["age", "deck", "embarked", "embark_town"]].isnull().astype(int).T
ax.imshow(missing_matrix, aspect="auto", cmap="Greys", interpolation="nearest")
ax.set_yticks(range(4))
ax.set_yticklabels(["age", "deck", "embarked", "embark_town"])
ax.set_xlabel("Row index (passenger)")
ax.set_title("Missing-Value Pattern Across Rows (dark = missing)")
plt.tight_layout()
plt.savefig("figures/07_missingness_pattern.png")
plt.close()

print("All 7 figures saved to figures/")
