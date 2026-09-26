"""
Week 2 - Step 1: Initial EDA summary statistics
"""
import pandas as pd
import numpy as np

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 140)

df = pd.read_csv("titanic_cleaned.csv")

print("=" * 70)
print("SHAPE:", df.shape)

print("\n" + "=" * 70)
print("NUMERIC SUMMARY:")
print(df[["age", "fare", "family_size", "fare_per_person"]].describe().T)

print("\n" + "=" * 70)
print("OVERALL SURVIVAL RATE: {:.1f}%".format(df["survived"].mean() * 100))

print("\n" + "=" * 70)
print("SURVIVAL RATE BY SEX:")
print(df.groupby("sex")["survived"].agg(["mean", "count"]))

print("\n" + "=" * 70)
print("SURVIVAL RATE BY PCLASS:")
print(df.groupby("pclass")["survived"].agg(["mean", "count"]))

print("\n" + "=" * 70)
print("SURVIVAL RATE BY PCLASS x SEX:")
print(df.groupby(["pclass", "sex"])["survived"].agg(["mean", "count"]))

print("\n" + "=" * 70)
print("SURVIVAL RATE BY AGE GROUP:")
print(df.groupby("age_group", observed=True)["survived"].agg(["mean", "count"]))

print("\n" + "=" * 70)
print("SURVIVAL RATE BY FAMILY SIZE:")
print(df.groupby("family_size")["survived"].agg(["mean", "count"]))

print("\n" + "=" * 70)
print("SURVIVAL RATE BY EMBARKATION PORT:")
print(df.groupby("embark_town")["survived"].agg(["mean", "count"]))

print("\n" + "=" * 70)
print("SURVIVAL RATE BY DECK RECORDED (proxy for cabin class):")
print(df.groupby("deck_recorded")["survived"].agg(["mean", "count"]))

print("\n" + "=" * 70)
print("FARE STATS BY PCLASS:")
print(df.groupby("pclass")["fare"].describe())

print("\n" + "=" * 70)
print("SKEWNESS:")
print("fare skew:", df["fare"].skew())
print("fare_log skew:", df["fare_log"].skew())
print("age skew:", df["age"].skew())

print("\n" + "=" * 70)
print("CORRELATION WITH SURVIVED (numeric features):")
num_cols = ["survived", "pclass", "age", "sibsp", "parch", "fare",
            "family_size", "fare_per_person", "deck_recorded", "is_alone"]
print(df[num_cols].corr()["survived"].sort_values(ascending=False))

print("\n" + "=" * 70)
print("WOMEN & CHILDREN FIRST CHECK (by 'who'):")
print(df.groupby("who")["survived"].agg(["mean", "count"]))

print("\n" + "=" * 70)
print("CROSSTAB: pclass x embark_town (counts):")
print(pd.crosstab(df["pclass"], df["embark_town"]))
