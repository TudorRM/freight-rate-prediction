import numpy as np
import pandas as pd

# data read
df = pd.read_csv("data/train_test.csv")

print("--- Shape ---")
print(df.shape)

print("\n--- Info ---")
print(df.info())

print("\n--- Describe ---")
print(df.describe().T)

print("\n--- Anomaly Checks ---")
# <= 0 checks
print("posted rate <= 0:", len(df[df["posted_rate"] <= 0]))
print("distance <= 0:", len(df[df["distance"] <= 0]))
print("weight <= 0:", len(df[df["weight"] <= 0]))

print("\n--- Categorical ---")
print("Equipment:", df["equipment"].unique())
print("Unique pickup:", len(df["pickup"].unique()))
print("Unique delivery:", len(df["delivery"].unique()))

print("\n--- Dates ---")
df["date"] = pd.to_datetime(df["date"])
print("Min date:", df["date"].min())
print("Max date:", df["date"].max())