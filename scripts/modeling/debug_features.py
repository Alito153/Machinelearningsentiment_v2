import json
import pandas as pd
import joblib
from pathlib import Path

project_dir = Path('/app/forex_macro_sentiment_1329')

print("=== CHECKING FEATURE CONFIGURATION ===\n")

with open(project_dir / 'models' / 'feature_engineering_config_cleaned.json', 'r') as f:
    config = json.load(f)
    print("Config keys:", list(config.keys()))
    print("\nFull config:")
    print(json.dumps(config, indent=2))

print("\n=== CHECKING DATA COLUMNS ===\n")

df = pd.read_csv(project_dir / 'data' / 'macro_events_labeled_cleaned.csv')
print(f"Dataset shape: {df.shape}")
print(f"Columns ({len(df.columns)}):")
for col in df.columns:
    print(f"  - {col}")

print("\n=== CHECKING MODEL FEATURE COUNT ===\n")

rf_model = joblib.load(project_dir / 'models' / 'random_forest_spike_model_cleaned.pkl')
xgb_model = joblib.load(project_dir / 'models' / 'xgboost_directional_model_cleaned.pkl')

print(f"RF model n_features_in_: {rf_model.n_features_in_}")
print(f"XGB model n_features_in_: {xgb_model.n_features_in_}")

print("\n=== CHECKING SCALER ===\n")

scaler = joblib.load(project_dir / 'models' / 'feature_scaler_cleaned.pkl')
print(f"Scaler n_features_in_: {scaler.n_features_in_}")
print(f"Scaler mean shape: {scaler.mean_.shape}")
print(f"Scaler scale shape: {scaler.scale_.shape}")