import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

# Load processed dataset from Day 2
data_path = "data/processed_train_test.csv"
df = pd.read_csv(data_path)

# Separate target variable and unused columns
drop_cols = ["load_id", "date", "equipment", "pickup", "delivery", "posted_rate"]
x = df.drop(columns=drop_cols, errors="ignore")
y = df["posted_rate"]

# Train/Validation split (80-20)
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

print("Done data split)")

# Initialize and train the Random Forest model
print("Training ...")
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(x_train, y_train)

# Evaluate model predictions
preds = rf.predict(x_val)

rmse = np.sqrt(mean_squared_error(y_val, preds))
mae = mean_absolute_error(y_val, preds)

print(f"Validation RMSE: {rmse:.4f}")
print(f"Validation MAE: {mae:.4f}")