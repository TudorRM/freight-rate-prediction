import numpy as np
import pandas as pd

# Load dataset
df = pd.read_csv("data/train_test.csv")

# Fix negative values for weights
df["weight"] = df["weight"].abs()

# Fill missing values for weights using equipment medians
df["weight"] = df["weight"].fillna(df.groupby("equipment")["weight"].transform("median"))

# Fill missing values for market index using global median
df["market_index"] = df["market_index"].fillna(df["market_index"].median())

# Extract date features
# Note: 'year' is omitted since all records are from 2025 (zero variance)
df["date"] = pd.to_datetime(df["date"])
df["month"] = df["date"].dt.month
df["day_of_week"] = df["date"].dt.dayofweek
df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)


# Converting equipment to numbers
equipment_cols = pd.get_dummies(df["equipment"], prefix="equip").astype(int)
df = pd.concat([df, equipment_cols], axis=1)

# Save processed dataset
df.to_csv("data/processed_train_test.csv", index=False)

print("Feature engineering completed!")
print("New shape:", df.shape)
print("Remaining missing values:", df.isnull().sum().sum())
