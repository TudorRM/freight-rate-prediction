import numpy as np
import pandas as pd
import xgboost as xgb

# Load proccessed trainind data and fit the model
train_df = pd.read_csv("data/processed_train_test.csv")

drop_cols = ["load_id", "date", "equipment", "pickup", "delivery", "posted_rate"]
x_train = train_df.drop(columns=drop_cols, errors="ignore")
y_train = train_df["posted_rate"]

print("Training XGBoost model on full dataset ...")
model = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1)

model.fit(x_train, y_train)

# Process validation data and make predictions
print("Processing validation data ...")
var_raw = pd.read_csv("data/validation.csv")

val_df = var_raw.copy()
val_df["date"] = pd.to_datetime(val_df["date"])
val_df["month"] = val_df["date"].dt.month
val_df["day_of_week"] = val_df["date"].dt.dayofweek
val_df["is_weekend"] = val_df["day_of_week"].isin([5, 6]).astype(int)

val_df = pd.get_dummies(val_df, columns=["equipment"], prefix="equip", dtype=int)

for col in x_train.columns:
    if col not in val_df.columns:
        val_df[col] = 0

x_val = val_df[x_train.columns]

val_preds_df = pd.DataFrame({
    "load_id": var_raw["load_id"],
    "predicted_rate": np.round(model.predict(x_val), 2)
})

val_preds_df.to_csv("data/validation_predictions.csv", index=False)
print("Predictions saved to data/validation_predictions.csv")

# Process december_chart_inputs.csv and make predictions
print("Processing december_chart_inputs.csv ...")
dec_raw = pd.read_csv("data/december_chart_inputs.csv")

dec_df = dec_raw.copy()
dec_df["date"] = pd.to_datetime(dec_df["date"])
dec_df["month"] = dec_df["date"].dt.month
dec_df["day_of_week"] = dec_df["date"].dt.dayofweek
dec_df["is_weekend"] = dec_df["day_of_week"].isin([5, 6]).astype(int)

# Extracting real coordinates from the pickup and delivery columns
lex_coords = train_df[train_df["pickup"] == "Lexington"][["pickup_lat", "pickup_lon"]].mean()
fw_coords = train_df[train_df["delivery"] == "Fort Wayne"][["delivery_lat", "delivery_lon"]].mean()

# Atributing the coordinates
dec_df["pickup_lat"] = lex_coords["pickup_lat"]
dec_df["pickup_lon"] = lex_coords["pickup_lon"]
dec_df["delivery_lat"] = fw_coords["delivery_lat"]
dec_df["delivery_lon"] = fw_coords["delivery_lon"]

# Filling missing values with the median of the training dataset
dec_df["market_index"] = train_df["market_index"].median()
dec_df["quote_signal"] = train_df["quote_signal"].median()

dec_df = pd.get_dummies(dec_df, columns=["equipment"], prefix="equip", dtype=int)

for col in x_train.columns:
    if col not in dec_df.columns:
        dec_df[col] = 0

x_dec = dec_df[x_train.columns]

dec_raw["predicted_rate"] = np.round(model.predict(x_dec), 2)
dec_raw.to_csv("data/december_chart_inputs.csv", index=False)

print("Predictions saved to data/december_chart_inputs.csv")