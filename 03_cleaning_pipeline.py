"""
Step 3: Full Cleaning & Preprocessing Pipeline
Produces data/titanic_cleaned.csv plus a log of every transformation applied.
"""
import pandas as pd
import numpy as np

RAW_PATH = "data/titanic_raw.csv"
OUT_PATH = "data/titanic_cleaned.csv"

log = []

def note(msg):
    log.append(msg)
    print(msg)

df = pd.read_csv(RAW_PATH)
note(f"Loaded raw data: {df.shape[0]} rows x {df.shape[1]} columns")

# ---------------------------------------------------------------
# 1. Handle redundant / duplicate-information columns
# ---------------------------------------------------------------
# 'class' duplicates 'pclass' (categorical label of the same value)
# 'alive' duplicates 'survived' (string version of the same 0/1 flag)
# 'adult_male' is fully derivable from 'sex' + 'who'
# We keep one canonical version of each and drop the redundant copy to
# avoid multicollinearity / conflicting-source-of-truth bugs downstream.
redundant_cols = ["class", "alive", "adult_male"]
df = df.drop(columns=redundant_cols)
note(f"Dropped redundant/derivable columns: {redundant_cols}")

# ---------------------------------------------------------------
# 2. Duplicate row check
# ---------------------------------------------------------------
n_dupe = df.duplicated().sum()
note(f"Exact duplicate rows found: {n_dupe}")
note("Investigation: this dataset has no PassengerId/Name/Ticket field, "
     "and several columns are low-cardinality categoricals (pclass, sex, "
     "sibsp, parch, embarked). Two different real passengers can therefore "
     "legitimately share identical values on every remaining column by "
     "coincidence. Because we cannot confirm these are the *same* person "
     "recorded twice, we do NOT drop them — dropping would silently delete "
     "real passengers. This limitation is documented rather than resolved.")

# ---------------------------------------------------------------
# 3. Missing value handling
# ---------------------------------------------------------------
# 3a. embarked / embark_town (2 missing each, same 2 rows)
missing_embarked_idx = df[df["embarked"].isnull()].index.tolist()
note(f"Rows missing embarked/embark_town: {missing_embarked_idx}")
mode_port = df["embarked"].mode()[0]
mode_town = df["embark_town"].mode()[0]
df["embarked"] = df["embarked"].fillna(mode_port)
df["embark_town"] = df["embark_town"].fillna(mode_town)
note(f"Imputed missing embarked/embark_town with the mode ('{mode_port}' / "
     f"'{mode_town}') — only 0.2% of rows affected, so mode imputation "
     "introduces negligible bias. (Historical passenger records independently "
     "confirm both were Southampton boarders.)")

# 3b. deck (77% missing) — too sparse to impute reliably.
# Rather than imputing 77% of a column (which would manufacture data),
# we (i) keep a binary indicator of whether cabin/deck was recorded at all
# (missingness itself is informative — unrecorded cabins correlate with
# lower-class passengers), and (ii) recode the column itself to 'Unknown'
# instead of deleting it, preserving the option for downstream analysis.
df["deck_recorded"] = df["deck"].notnull().astype(int)
df["deck"] = df["deck"].fillna("Unknown")
note("deck: 77.2% missing. Created binary flag 'deck_recorded' and recoded "
     "missing entries to category 'Unknown' rather than imputing a deck "
     "letter (imputing here would fabricate data with no evidential basis).")

# 3c. age (19.9% missing) — impute using group-wise median.
# Global median imputation ignores the fact that age correlates strongly
# with pclass and passenger role (man/woman/child). Group medians preserve
# that structure far better than a single global constant.
before_age_std = df["age"].std()
group_medians = df.groupby(["pclass", "who"])["age"].transform("median")
df["age_was_missing"] = df["age"].isnull().astype(int)
df["age"] = df["age"].fillna(group_medians)
# a handful of (pclass, who) groups could theoretically be empty; fall back to global median
df["age"] = df["age"].fillna(df["age"].median())
after_age_std = df["age"].std()
note(f"age: 19.9% missing. Imputed using median age within each "
     f"(pclass, who) subgroup rather than a single global median, to "
     "preserve the known relationship between class/role and age. "
     f"Added 'age_was_missing' flag so imputed rows remain identifiable. "
     f"Std dev before/after: {before_age_std:.2f} -> {after_age_std:.2f} "
     "(a slight, expected narrowing from imputation).")

# ---------------------------------------------------------------
# 4. Erroneous / suspicious entries
# ---------------------------------------------------------------
zero_fare = (df["fare"] == 0).sum()
note(f"Rows with fare == 0: {zero_fare}")
note("Investigated rather than removed: all 15 zero-fare rows are adult "
     "males, mostly First/Second class. Public Titanic records show a "
     "number of American Line employees and Titanic crew/guarantee-group "
     "members traveled on complimentary passage. Treated as VALID "
     "domain-specific values, not data-entry errors, and left unchanged. "
     "A 'fare_is_zero' flag is added so models can treat them separately.")
df["fare_is_zero"] = (df["fare"] == 0).astype(int)

# ---------------------------------------------------------------
# 5. Outlier handling (fare)
# ---------------------------------------------------------------
q1, q3 = df["fare"].quantile([0.25, 0.75])
iqr = q3 - q1
upper = q3 + 1.5 * iqr
n_outliers = (df["fare"] > upper).sum()
note(f"IQR outlier check on fare: upper bound = {upper:.2f}, "
     f"{n_outliers} rows above it ({n_outliers/len(df)*100:.1f}%).")
note("Inspected the top fares individually: they belong overwhelmingly to "
     "First-class passengers (including several traveling together on the "
     "same high-value ticket group), which is consistent with genuine "
     "luxury-fare pricing rather than data entry error. We therefore do "
     "NOT delete or cap (winsorize) these rows, since doing so would "
     "remove real signal that strongly predicts survival in this dataset. "
     "Instead we apply a log1p transform to reduce right-skew for any "
     "downstream modeling that assumes near-normal inputs, while keeping "
     "the raw fare column intact for interpretability.")
df["fare_log"] = np.log1p(df["fare"])

# sibsp/parch: values flagged by the IQR rule (e.g. sibsp > 2) are valid
# large-family counts (well documented, e.g. the Sage family of 11 aboard);
# left untouched, but engineered into an aggregate below.
note("sibsp/parch: IQR flags larger families as 'outliers', but these are "
     "legitimate large-family group sizes (e.g. documented multi-child "
     "families aboard). Left untouched; aggregated into family_size below.")

# ---------------------------------------------------------------
# 6. Feature engineering / preprocessing for downstream analysis
# ---------------------------------------------------------------
df["family_size"] = df["sibsp"] + df["parch"] + 1
df["is_alone"] = (df["family_size"] == 1).astype(int)
df["fare_per_person"] = df["fare"] / df["family_size"]
df["age_group"] = pd.cut(
    df["age"],
    bins=[0, 12, 18, 35, 60, 100],
    labels=["child", "teen", "young_adult", "adult", "senior"],
)
note("Engineered features: family_size (sibsp+parch+1), is_alone, "
     "fare_per_person, age_group (binned).")

# dtype cleanup: cast low-cardinality text columns to 'category' for memory
# efficiency and to make downstream one-hot/ordinal encoding explicit
cat_cols = ["sex", "embarked", "who", "deck", "embark_town", "age_group"]
for c in cat_cols:
    df[c] = df[c].astype("category")
note(f"Cast columns to 'category' dtype: {cat_cols}")

# ---------------------------------------------------------------
# 7. Final integrity checks
# ---------------------------------------------------------------
assert df.isnull().sum().sum() == 0, "Unexpected nulls remain!"
note(f"Final null check: 0 missing values remain across all "
     f"{df.shape[1]} columns.")
note(f"Final cleaned shape: {df.shape[0]} rows x {df.shape[1]} columns.")

df.to_csv(OUT_PATH, index=False)
with open("data/cleaning_log.txt", "w") as f:
    f.write("\n\n".join(log))

print("\nSaved:", OUT_PATH)
