"""
Step 1: Initial Data Exploration
Dataset: Titanic passenger data (seaborn-data mirror of the classic Kaggle Titanic dataset)
Source: https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv
"""
import pandas as pd
import numpy as np

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 140)

df = pd.read_csv("data/titanic_raw.csv")

print("=" * 70)
print("SHAPE:", df.shape)
print("=" * 70)
print("\nDTYPES:\n", df.dtypes)

print("\n" + "=" * 70)
print("HEAD:")
print(df.head())

print("\n" + "=" * 70)
print("DESCRIBE (numeric):")
print(df.describe())

print("\n" + "=" * 70)
print("DESCRIBE (categorical):")
print(df.describe(include="object"))

print("\n" + "=" * 70)
print("MISSING VALUES (count / percent):")
miss = df.isnull().sum()
pct = (miss / len(df) * 100).round(2)
missing_report = pd.DataFrame({"missing_count": miss, "missing_pct": pct})
missing_report = missing_report[missing_report.missing_count > 0].sort_values("missing_count", ascending=False)
print(missing_report)

print("\n" + "=" * 70)
print("DUPLICATE ROWS:", df.duplicated().sum())

print("\n" + "=" * 70)
print("UNIQUE VALUES PER CATEGORICAL COLUMN:")
for col in df.select_dtypes(include="object").columns:
    print(f"  {col}: {df[col].unique()}")

print("\n" + "=" * 70)
print("CONSISTENCY CHECK: 'class' vs 'pclass'")
print(pd.crosstab(df["pclass"], df["class"]))

print("\nCONSISTENCY CHECK: 'sex' vs 'who' vs 'adult_male'")
print(df.groupby(["sex", "who", "adult_male"]).size())

print("\nCONSISTENCY CHECK: 'survived' vs 'alive'")
print(pd.crosstab(df["survived"], df["alive"]))

missing_report.to_csv("data/missing_report.csv")
