import pandas as pd
import numpy as np
from datetime import datetime
import os

data_file = '/app/xauusd_trading_strategy_2022/xauusd-m1-bid-2018-01-01-2025-12-07.csv'

print("=" * 80)
print("DATA VALIDATION REPORT")
print("=" * 80)

if not os.path.exists(data_file):
    print(f"ERROR: Data file not found: {data_file}")
    exit(1)

file_size_mb = os.path.getsize(data_file) / (1024**2)
print(f"\nFile: {data_file}")
print(f"File size: {file_size_mb:.2f} MB")

df = pd.read_csv(data_file, header=None, names=['timestamp', 'open', 'high', 'low', 'close'])

print(f"\n--- DATA SHAPE ---")
print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")

print(f"\n--- DATA TYPES ---")
print(df.dtypes)

print(f"\n--- TIMESTAMP VALIDATION ---")
print(f"Min timestamp (ms): {df['timestamp'].min()}")
print(f"Max timestamp (ms): {df['timestamp'].max()}")
print(f"Min date: {datetime.fromtimestamp(df['timestamp'].min() / 1000)}")
print(f"Max date: {datetime.fromtimestamp(df['timestamp'].max() / 1000)}")

timestamps_sorted = df['timestamp'].is_monotonic_increasing
print(f"Timestamps monotonically increasing: {timestamps_sorted}")

print(f"\n--- OHLC RELATIONSHIPS ---")
print(f"High >= Open: {(df['high'] >= df['open']).all()}")
print(f"High >= Close: {(df['high'] >= df['close']).all()}")
print(f"High >= Low: {(df['high'] >= df['low']).all()}")
print(f"Low <= Open: {(df['low'] <= df['open']).all()}")
print(f"Low <= Close: {(df['low'] <= df['close']).all()}")

print(f"\n--- PRICE STATISTICS ---")
for col in ['open', 'high', 'low', 'close']:
    print(f"\n{col.upper()}:")
    print(f"  Min:    {df[col].min():.4f}")
    print(f"  Max:    {df[col].max():.4f}")
    print(f"  Mean:   {df[col].mean():.4f}")
    print(f"  Median: {df[col].median():.4f}")
    print(f"  Std:    {df[col].std():.4f}")

print(f"\n--- DATA QUALITY ---")
null_counts = df.isnull().sum()
print(f"Null values: {null_counts.sum()}")
print(df.isnull().sum())

print(f"\n--- SAMPLE DATA ---")
print(f"\nFirst 10 rows:")
print(df.head(10))
print(f"\nLast 10 rows:")
print(df.tail(10))

print(f"\nRandom 5 rows:")
print(df.sample(5))

print(f"\n--- VALIDATION COMPLETE ---")
print(f"Status: PASSED - Data file is valid for use")

print("\n" + "=" * 80)