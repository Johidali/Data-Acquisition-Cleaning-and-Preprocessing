import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({"figure.dpi": 150, "font.size": 11})
PAL = {"survived_no": "#A63D40", "survived_yes": "#2E7D5B", "main": "#2E4374", "accent": "#C06C3E"}
SEX_PAL = {"male": "#2E4374", "female": "#C06C3E", "Male": "#2E4374", "Female": "#C06C3E"}
SURV_PAL = {0: "#A63D40", 1: "#2E7D5B"}

df = pd.read_csv("titanic_cleaned.csv")
df["Survived"] = df["survived"].map({0: "No", 1: "Yes"})
df["Sex"] = df["sex"].str.capitalize()

def savefig(name):
    plt.tight_layout()
    plt.savefig(f"figures/{name}", bbox_inches="tight")
    plt.close()

# ---------------------------------------------------------------
# 1. Target variable distribution
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 4.2))
counts = df["Survived"].value_counts().reindex(["No", "Yes"])
bars = ax.bar(counts.index, counts.values, color=[SURV_PAL[0], SURV_PAL[1]])
for b, v in zip(bars, counts.values):
    ax.text(b.get_x() + b.get_width()/2, v + 8, f"{v}\n({v/len(df)*100:.1f}%)",
            ha="center", fontsize=10)
ax.set_title("Figure 1. Survival Outcome Distribution (n = 891)", fontsize=12, fontweight="bold")
ax.set_xlabel("Survived")
ax.set_ylabel("Number of Passengers")
ax.set_ylim(0, 620)
savefig("01_target_distribution.png")

# ---------------------------------------------------------------
# 2. Age & Fare univariate distributions
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
sns.histplot(df["age"], bins=30, kde=True, color=PAL["main"], ax=axes[0])
axes[0].set_title("Age Distribution", fontweight="bold")
axes[0].set_xlabel("Age (years)")
axes[0].set_ylabel("Count")
sns.histplot(df["fare"], bins=40, kde=True, color=PAL["accent"], ax=axes[1])
axes[1].set_title("Fare Distribution (right-skewed)", fontweight="bold")
axes[1].set_xlabel("Fare ($)")
axes[1].set_ylabel("Count")
fig.suptitle("Figure 2. Univariate Distributions of Age and Fare", fontsize=12, fontweight="bold", y=1.03)
savefig("02_age_fare_distributions.png")

# ---------------------------------------------------------------
# 3. Categorical feature counts: pclass, sex, embark_town
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
sns.countplot(x="pclass", data=df, ax=axes[0], color=PAL["main"])
axes[0].set_title("Passenger Class")
axes[0].set_xlabel("Class")
axes[0].set_ylabel("Count")
sns.countplot(x="Sex", data=df, ax=axes[1], hue="Sex", palette=SEX_PAL, legend=False)
axes[1].set_title("Sex")
axes[1].set_xlabel("")
axes[1].set_ylabel("")
sns.countplot(x="embark_town", data=df, ax=axes[2], color=PAL["accent"], order=["Southampton", "Cherbourg", "Queenstown"])
axes[2].set_title("Embarkation Port")
axes[2].set_xlabel("")
axes[2].set_ylabel("")
axes[2].tick_params(axis="x", rotation=20)
fig.suptitle("Figure 3. Categorical Feature Distributions", fontsize=12, fontweight="bold", y=1.04)
savefig("03_categorical_counts.png")

# ---------------------------------------------------------------
# 4. Survival rate by Sex (annotated bar)
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 4.3))
rates = df.groupby("Sex")["survived"].mean().reindex(["Male", "Female"])
bars = ax.bar(rates.index, rates.values, color=[SEX_PAL["male"], SEX_PAL["female"]])
for b, v in zip(bars, rates.values):
    ax.text(b.get_x() + b.get_width()/2, v + 0.02, f"{v*100:.1f}%", ha="center", fontweight="bold")
ax.set_ylim(0, 1)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
ax.set_title("Figure 4. Survival Rate by Sex", fontsize=12, fontweight="bold")
ax.set_ylabel("Survival Rate")
ax.set_xlabel("Sex")
savefig("04_survival_by_sex.png")

# ---------------------------------------------------------------
# 5. Survival rate by class and sex (grouped bar)
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.5))
pivot = df.pivot_table(index="pclass", columns="Sex", values="survived", aggfunc="mean")
pivot.plot(kind="bar", ax=ax, color=[SEX_PAL["female"], SEX_PAL["male"]])
ax.set_title("Figure 5. Survival Rate by Passenger Class and Sex", fontsize=12, fontweight="bold")
ax.set_xlabel("Passenger Class")
ax.set_ylabel("Survival Rate")
ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
ax.set_xticklabels(["1st", "2nd", "3rd"], rotation=0)
ax.legend(title="Sex")
for container in ax.containers:
    ax.bar_label(container, fmt=lambda x: f"{x*100:.0f}%", fontsize=9)
savefig("05_survival_class_sex.png")

# ---------------------------------------------------------------
# 6. Age distribution by survival, split by sex (violin)
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4.8))
sns.violinplot(data=df, x="Sex", y="age", hue="Survived", split=True,
                palette={"No": SURV_PAL[0], "Yes": SURV_PAL[1]}, ax=ax, inner="quartile")
ax.set_title("Figure 6. Age Distribution by Sex and Survival Outcome", fontsize=12, fontweight="bold")
ax.set_xlabel("Sex")
ax.set_ylabel("Age (years)")
ax.legend(title="Survived", loc="upper right")
savefig("06_age_violin_sex_survival.png")

# ---------------------------------------------------------------
# 7. FacetGrid: age histograms by class, colored by survival
# ---------------------------------------------------------------
g = sns.FacetGrid(df, col="pclass", hue="Survived", palette={"No": SURV_PAL[0], "Yes": SURV_PAL[1]},
                   height=3.6, aspect=0.95, col_order=[1, 2, 3])
g.map(sns.histplot, "age", bins=20, alpha=0.6, kde=False)
g.add_legend(title="Survived")
g.set_titles("Class {col_name}")
g.set_axis_labels("Age (years)", "Count")
g.figure.suptitle("Figure 7. Age Distribution by Passenger Class and Survival", fontsize=12, fontweight="bold", y=1.06)
g.figure.savefig("figures/07_facet_age_class_survival.png", bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------
# 8. Fare vs Age scatter, colored by survival, sized by class
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5.2))
size_map = {1: 90, 2: 45, 3: 20}
sizes = df["pclass"].map(size_map)
scatter = ax.scatter(df["age"], df["fare"], c=df["survived"].map(SURV_PAL), s=sizes, alpha=0.55, edgecolor="white", linewidth=0.3)
ax.set_yscale("symlog")
ax.set_title("Figure 8. Fare vs. Age, Colored by Survival, Sized by Class", fontsize=12, fontweight="bold")
ax.set_xlabel("Age (years)")
ax.set_ylabel("Fare ($, symlog scale)")
from matplotlib.lines import Line2D
legend_elems = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor=SURV_PAL[0], markersize=9, label="Did not survive"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=SURV_PAL[1], markersize=9, label="Survived"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markersize=11, label="1st class (larger)"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markersize=6, label="3rd class (smaller)"),
]
ax.legend(handles=legend_elems, loc="upper right", fontsize=9)
savefig("08_fare_age_scatter.png")

# ---------------------------------------------------------------
# 9. Survival rate by family size (line + bar hybrid)
# ---------------------------------------------------------------
fam = df.groupby("family_size")["survived"].agg(["mean", "count"]).reset_index()
fig, ax1 = plt.subplots(figsize=(8, 4.8))
ax2 = ax1.twinx()
ax2.bar(fam["family_size"], fam["count"], color="#D9D9D9", alpha=0.7, label="Passenger count")
ax1.plot(fam["family_size"], fam["mean"], color=PAL["accent"], marker="o", linewidth=2.5, label="Survival rate")
ax1.set_zorder(ax2.get_zorder() + 1)
ax1.patch.set_visible(False)
ax1.set_xlabel("Family Size (self + siblings/spouses + parents/children)")
ax1.set_ylabel("Survival Rate", color=PAL["accent"])
ax2.set_ylabel("Number of Passengers", color="#888888")
ax1.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
ax1.set_title("Figure 9. Survival Rate vs. Family Size", fontsize=12, fontweight="bold")
lines, labels = ax1.get_legend_handles_labels()
bars_, bar_labels = ax2.get_legend_handles_labels()
ax1.legend(lines + bars_, labels + bar_labels, loc="upper right", fontsize=9)
savefig("09_family_size_survival.png")

# ---------------------------------------------------------------
# 10. Correlation heatmap (seaborn)
# ---------------------------------------------------------------
num_cols = ["survived", "pclass", "age", "sibsp", "parch", "fare",
            "family_size", "fare_per_person", "deck_recorded", "is_alone"]
corr = df[num_cols].corr()
fig, ax = plt.subplots(figsize=(8.5, 7))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1,
            square=True, linewidths=0.5, cbar_kws={"label": "Pearson r"}, ax=ax)
ax.set_title("Figure 10. Correlation Matrix of Numeric Features", fontsize=12, fontweight="bold")
savefig("10_correlation_heatmap.png")

# ---------------------------------------------------------------
# 11. Embarkation port vs class vs survival (heatmap of survival rate)
# ---------------------------------------------------------------
pivot2 = df.pivot_table(index="embark_town", columns="pclass", values="survived", aggfunc="mean")
pivot2 = pivot2.reindex(["Southampton", "Cherbourg", "Queenstown"])
fig, ax = plt.subplots(figsize=(6.5, 4.3))
sns.heatmap(pivot2, annot=True, fmt=".0%", cmap="YlGnBu", cbar_kws={"label": "Survival Rate"}, ax=ax)
ax.set_title("Figure 11. Survival Rate by Embarkation Port and Class", fontsize=12, fontweight="bold")
ax.set_xlabel("Passenger Class")
ax.set_ylabel("Embarkation Port")
savefig("11_embark_class_heatmap.png")

# ---------------------------------------------------------------
# 12. Deck recorded proxy vs fare (boxplot) - anomaly/explanation chart
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.6))
df["Cabin Recorded"] = df["deck_recorded"].map({0: "No", 1: "Yes"})
sns.boxplot(data=df, x="Cabin Recorded", y="fare", hue="Cabin Recorded",
            palette={"No": "#A6A6A6", "Yes": PAL["main"]}, ax=ax, legend=False)
ax.set_yscale("symlog")
ax.set_title("Figure 12. Fare Paid vs. Whether Cabin/Deck Was Recorded", fontsize=12, fontweight="bold")
ax.set_xlabel("Was a cabin/deck recorded for this passenger?")
ax.set_ylabel("Fare ($, symlog scale)")
savefig("12_deck_recorded_fare_box.png")

# ---------------------------------------------------------------
# 13. Pairplot of key numeric variables colored by survival
# ---------------------------------------------------------------
pp_cols = ["age", "fare_log", "family_size", "pclass"]
g2 = sns.pairplot(df, vars=pp_cols, hue="Survived", palette={"No": SURV_PAL[0], "Yes": SURV_PAL[1]},
                   diag_kind="kde", plot_kws={"alpha": 0.5, "s": 18}, height=2.1)
g2.figure.suptitle("Figure 13. Pairwise Relationships Among Key Numeric Features", fontsize=12, fontweight="bold", y=1.02)
g2.figure.savefig("figures/13_pairplot.png", bbox_inches="tight")
plt.close()

print("All visualizations saved to figures/")
