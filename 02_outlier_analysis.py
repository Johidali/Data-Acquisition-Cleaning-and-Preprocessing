"""
Step 2: Outlier Detection (IQR method + z-score cross-check)
"""
import pandas as pd
import numpy as np

df = pd.read_csv("data/titanic_raw.csv")

def iqr_bounds(series, k=1.5):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr

for col in ["fare", "age", "sibsp", "parch"]:
    s = df[col].dropna()
    low, high = iqr_bounds(s)
    n_out = ((s < low) | (s > high)).sum()
    print(f"{col:8s}  Q1={s.quantile(.25):8.2f}  Q3={s.quantile(.75):8.2f}  "
          f"IQR bounds=({low:8.2f}, {high:8.2f})  outliers={n_out}  ({n_out/len(s)*100:.1f}%)")

print("\nTop 10 highest fares:")
print(df.nlargest(10, "fare")[["pclass", "sex", "age", "fare", "class", "embark_town"]])

print("\nRows with fare == 0:")
print(df[df["fare"] == 0][["pclass", "sex", "age", "fare", "class", "who"]])

print("\nAge outlier candidates (very young / very old):")
print(df.nsmallest(5, "age")[["age", "sibsp", "parch", "who"]])
print(df.nlargest(5, "age")[["age", "sibsp", "parch", "who"]])
