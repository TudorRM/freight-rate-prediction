# Freight Rate Prediction Challenge

See `Freight_Rate_ML_Assessment.pdf` for the assessment instructions.

## What to do

1. Train and validate your model using `data/train_test.csv`.
2. Predict every load in `data/validation.csv`. Each load has a unique `load_id`.
3. Fill the matching `predicted_rate` values in `data/validation_predictions_template.csv` and save it as `validation_predictions.csv`.
4. Predict every row in `data/december_chart_inputs.csv` by filling its `predicted_rate` column.
5. Install the scorer requirements and run:

```bash
python -m pip install -r requirements.txt
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

The scorer validates both files and creates `scorer_results/candidate_december.png`.

## Submit

- GitHub repository containing your code, dependencies, and run instructions
- `validation_predictions.csv`
- PDF or DOCX report containing your validation, data split approach and `candidate_december.png`
- 2-3 minute Loom link





## Progress Log

### Day 1: Project Setup & EDA

**Task Understanding**
* Goal: Predict freight rate (`posted_rate`) for December loads.
* Strategy: EDA -> Data Cleaning & Feature Engineering -> Baseline Model.

**EDA Observations**
* Dataset: 48,000 rows, 14 columns (Jan – Oct 2025).
* Missing Data: 300 in `weight`, 374 in `market_index`.
* Anomalies: 292 negative values in `weight` (min -47,500 lbs).
* Categoricals: 3 equipment types, 64 unique pickup & delivery locations.

### Day 2: Data Cleaning & Feature Engineering

**Tasks Completed**
* Fixed 292 negative `weight` values using absolute values (`abs()`).
* Imputed missing `weight` values using the median weight grouped by `equipment` type.
* Imputed missing `market_index` values using the global median.
* Extracted temporal features: `month`, `day_of_week`, and `is_weekend`. Omitted `year` due to zero variance (all data is from 2025).
* Applied One-Hot Encoding to categorical `equipment` types (`equip_*`).
* Saved the cleaned dataset to `data/processed_train_test.csv`.

**Pipeline Output**
* Processed Dataset: 48,000 rows, 20 columns.
* Missing Values: 0 across all features.

### Day 3: Baseline Model (Random Forest)

**Tasks Completed**
* Built baseline model using `RandomForestRegressor` (`src/03_train_rf.py`).
* Split data 80/20 into train and validation sets.
* Trained on distance, weight, GPS coordinates, market index, and equipment features.

**Model Performance**
* **Validation MAE:** $130.00
* **Validation RMSE:** $568.21