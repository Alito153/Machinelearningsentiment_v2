import pandas as pd
import json
import sys
import os
from pathlib import Path

sys.stdout.flush()

data_path = '/app/forex_macro_sentiment_1329/data/macro_events_labeled.csv'
config_path = '/app/forex_macro_sentiment_1329/models/feature_engineering_config.json'

print("=" * 80)
print("LEAKAGE AUDIT: Identifying Outcome-Encoding Features")
print("=" * 80)

df = pd.read_csv(data_path)
print(f"\nDataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"\nColumn names:")
for i, col in enumerate(df.columns):
    print(f"  {i}: {col}")

print(f"\nData types:")
for col in df.columns:
    print(f"  {col}: {df[col].dtype}")

print(f"\nFirst few rows:")
print(df.head(2))

print(f"\n" + "=" * 80)
print("IDENTIFYING LEAKY COLUMNS")
print("=" * 80)

leaky_patterns = [
    'return', 'move', 'pips', 'drawdown', 'adverse', 'max_', 'min_',
    'directional', 'spike', 'outcome', 'winner', 'loser', 'profit'
]

leaky_cols = []
for col in df.columns:
    col_lower = col.lower()
    if any(pattern in col_lower for pattern in leaky_patterns):
        if col not in ['binary_label', 'directional_label']:
            leaky_cols.append(col)

print(f"\nPotentially leaky columns identified: {len(leaky_cols)}")
for col in leaky_cols:
    print(f"  - {col}")

print(f"\n" + "=" * 80)
print("CLEAN FEATURE CANDIDATES (pre-announcement info only)")
print("=" * 80)

clean_cols = [col for col in df.columns if col not in leaky_cols 
              and col not in ['binary_label', 'directional_label']]
print(f"\nClean features: {len(clean_cols)}")
for col in clean_cols:
    print(f"  - {col}")

with open(config_path, 'r') as f:
    config = json.load(f)

print(f"\n" + "=" * 80)
print("FEATURE ENGINEERING CONFIG ANALYSIS")
print("=" * 80)
print(json.dumps(config, indent=2)[:1500])

print("\n" + "=" * 80)
print("DATA QUALITY CHECK")
print("=" * 80)
print(f"\nMissing values per column:")
missing = df.isnull().sum()
if missing.sum() > 0:
    print(missing[missing > 0])
else:
    print("  No missing values")

print(f"\nLabel distribution:")
if 'binary_label' in df.columns:
    print(df['binary_label'].value_counts())
if 'directional_label' in df.columns:
    print(df['directional_label'].value_counts())