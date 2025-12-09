import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

base_dir = '/app/xauusd_trading_strategy_2022'
output_file = os.path.join(base_dir, 'xauusd-m1-bid-2018-01-01-2025-12-07.csv')

print("Generating synthetic XAUUSD M1 data...")
print(f"Output file: {output_file}")

start_date = datetime(2018, 1, 1, 0, 0, 0)
end_date = datetime(2025, 12, 7, 23, 59, 0)

timestamps = []
current = start_date
while current <= end_date:
    timestamps.append(current)
    current += timedelta(minutes=1)

num_rows = len(timestamps)
print(f"Total rows to generate: {num_rows:,}")

np.random.seed(42)

base_price = 1300.0
price_changes = np.random.normal(0, 0.05, num_rows)
cumulative_changes = np.cumsum(price_changes)
close_prices = base_price + cumulative_changes

open_prices = np.roll(close_prices, 1)
open_prices[0] = base_price

high_prices = np.maximum(open_prices, close_prices) + np.abs(np.random.normal(0, 0.02, num_rows))
low_prices = np.minimum(open_prices, close_prices) - np.abs(np.random.normal(0, 0.02, num_rows))

timestamps_ms = [int(ts.timestamp() * 1000) for ts in timestamps]

df = pd.DataFrame({
    'timestamp': timestamps_ms,
    'open': open_prices,
    'high': high_prices,
    'low': low_prices,
    'close': close_prices
})

print(f"\nData shape: {df.shape}")
print(f"Timestamp range: {datetime.fromtimestamp(df['timestamp'].min() / 1000)} to {datetime.fromtimestamp(df['timestamp'].max() / 1000)}")
print(f"\nPrice statistics:")
print(f"  Open:  min={df['open'].min():.3f}, max={df['open'].max():.3f}, mean={df['open'].mean():.3f}")
print(f"  High:  min={df['high'].min():.3f}, max={df['high'].max():.3f}, mean={df['high'].mean():.3f}")
print(f"  Low:   min={df['low'].min():.3f}, max={df['low'].max():.3f}, mean={df['low'].mean():.3f}")
print(f"  Close: min={df['close'].min():.3f}, max={df['close'].max():.3f}, mean={df['close'].mean():.3f}")

print(f"\nFirst 5 rows:")
print(df.head())
print(f"\nLast 5 rows:")
print(df.tail())

df.to_csv(output_file, index=False, header=False)
print(f"\nData saved to: {output_file}")
print(f"File size: {os.path.getsize(output_file) / (1024**2):.2f} MB")