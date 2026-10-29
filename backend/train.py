import os
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

# Define directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Locate dataset (check current dir or parent dir)
csv_candidates = [
    os.path.join(BASE_DIR, "train.csv"),
    os.path.join(BASE_DIR, "..", "train.csv")
]
dataset_path = None
for candidate in csv_candidates:
    if os.path.exists(candidate):
        dataset_path = candidate
        break

if not dataset_path:
    raise FileNotFoundError("train.csv could not be found.")

print(f"Loading data from: {dataset_path}")
df = pd.read_csv(dataset_path)

# Features used by the house price predictor web app
FEATURES = ['OverallQual', 'GrLivArea', 'TotalBsmtSF', 'GarageCars', 'FullBath', 'YearBuilt']
TARGET = 'SalePrice'

X = df[FEATURES].copy()
y = df[TARGET]

# Impute missing values with column median
medians = X.median().to_dict()
X = X.fillna(medians)

# 80/20 train/validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluation
val_preds = model.predict(X_val)
r2 = float(r2_score(y_val, val_preds))
rmse = float(np.sqrt(mean_squared_error(y_val, val_preds)))

print("--- House Price Model Trained Successfully ---")
print(f"Validation R2 Score : {r2:.4f}")
print(f"Validation RMSE     : {rmse:.2f}")

# Model weights
coefficients = {feat: float(coef) for feat, coef in zip(FEATURES, model.coef_)}
intercept = float(model.intercept_)

# Save model artifact
model_path = os.path.join(MODELS_DIR, "house_price_model.joblib")
joblib.dump(model, model_path)
print(f"Model saved to: {model_path}")

# Save metadata and metrics
metrics_data = {
    "model_name": "LinearRegression",
    "features": FEATURES,
    "medians": medians,
    "intercept": intercept,
    "coefficients": coefficients,
    "metrics": {
        "r2_score": round(r2, 4),
        "rmse": round(rmse, 2)
    }
}
metrics_path = os.path.join(MODELS_DIR, "model_metrics.json")
with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(metrics_data, f, indent=2)
print(f"Metrics saved to: {metrics_path}")
