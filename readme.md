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