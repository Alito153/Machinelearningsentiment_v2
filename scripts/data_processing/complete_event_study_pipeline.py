import os
import sys
import json
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path('/app/forex_macro_sentiment_1329')
RAW_DATA_DIR = PROJECT_ROOT / 'raw_data'
DATA_DIR = PROJECT_ROOT / 'data'

logger.info("="*80)
logger.info("COMPREHENSIVE EVENT STUDY AND LABELING PIPELINE")
logger.info("="*80)

eurusd_df = pd.read_csv(RAW_DATA_DIR / 'EURUSD_daily.csv')
xauusd_df = pd.read_csv(RAW_DATA_DIR / 'XAUUSD_daily.csv')
vix_df = pd.read_csv(RAW_DATA_DIR / 'vix_daily.csv')
macro_events_df = pd.read_csv(RAW_DATA_DIR / 'macro_events_comprehensive.csv')

for df in [eurusd_df, xauusd_df, vix_df]:
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df.sort_values('timestamp', inplace=True)
    df.reset_index(drop=True, inplace=True)

macro_events_df['event_timestamp'] = pd.to_datetime(macro_events_df['event_timestamp'])

logger.info(f"Loaded data:")
logger.info(f"  EURUSD: {len(eurusd_df)} candles")
logger.info(f"  XAUUSD: {len(xauusd_df)} candles")
logger.info(f"  VIX: {len(vix_df)} candles")
logger.info(f"  Macro Events: {len(macro_events_df)} events")

logger.info("\n" + "="*80)
logger.info("IMPLEMENTING DAILY EVENT STUDY (SIMPLIFIED FOR DAILY DATA)")
logger.info("="*80)

def calculate_daily_returns_and_wick(prices_array, open_price):
    if len(prices_array) < 1 or open_price <= 0:
        return np.nan, np.nan
    
    max_price = np.max(prices_array)
    min_price = np.min(prices_array)
    
    max_return = np.log(max_price / open_price)
    adverse_wick = (open_price - min_price) / open_price if open_price > 0 else 0
    
    return max_return, adverse_wick

event_study_results = []

for idx, event in macro_events_df.iterrows():
    event_timestamp = event['event_timestamp']
    event_type = event['event_type']
    
    eurusd_event_mask = (eurusd_df['timestamp'].dt.date == event_timestamp.date())
    xauusd_event_mask = (xauusd_df['timestamp'].dt.date == event_timestamp.date())
    
    eurusd_event_window = eurusd_df[eurusd_event_mask]
    xauusd_event_window = xauusd_df[xauusd_event_mask]
    
    eurusd_next_day_mask = (eurusd_df['timestamp'].dt.date == event_timestamp.date() + timedelta(days=1))
    xauusd_next_day_mask = (xauusd_df['timestamp'].dt.date == event_timestamp.date() + timedelta(days=1))
    
    eurusd_next_day = eurusd_df[eurusd_next_day_mask]
    xauusd_next_day = xauusd_df[xauusd_next_day_mask]
    
    event_result = {
        'event_id': idx,
        'event_timestamp': event_timestamp,
        'event_type': event_type,
        'country': event['country'],
        'actual_value': event['actual_value'],
        'consensus_value': event['consensus_value'],
        'previous_value': event['previous_value'],
        'surprise_pct': event['surprise_pct'],
        'impact_level': event['impact_level']
    }
    
    if len(eurusd_event_window) > 0:
        eurusd_event_close = eurusd_event_window['close'].iloc[0]
        eurusd_event_open = eurusd_event_window['open'].iloc[0]
        
        if len(eurusd_next_day) > 0:
            eurusd_next_prices = eurusd_next_day['close'].values
            eurusd_next_high = eurusd_next_day['high'].max()
            eurusd_next_low = eurusd_next_day['low'].min()
            
            max_return_next = np.log(eurusd_next_high / eurusd_event_close)
            wick_next = (eurusd_event_close - eurusd_next_low) / eurusd_event_close if eurusd_event_close > 0 else 0
            
            event_result['eurusd_max_return_day1'] = max_return_next
            event_result['eurusd_wick_day1'] = wick_next
        else:
            event_result['eurusd_max_return_day1'] = np.nan
            event_result['eurusd_wick_day1'] = np.nan
    else:
        event_result['eurusd_max_return_day1'] = np.nan
        event_result['eurusd_wick_day1'] = np.nan
    
    if len(xauusd_event_window) > 0:
        xauusd_event_close = xauusd_event_window['close'].iloc[0]
        xauusd_event_open = xauusd_event_window['open'].iloc[0]
        
        if len(xauusd_next_day) > 0:
            xauusd_next_prices = xauusd_next_day['close'].values
            xauusd_next_high = xauusd_next_day['high'].max()
            xauusd_next_low = xauusd_next_day['low'].min()
            
            max_return_next = np.log(xauusd_next_high / xauusd_event_close)
            wick_next = (xauusd_event_close - xauusd_next_low) / xauusd_event_close if xauusd_event_close > 0 else 0
            
            event_result['xauusd_max_return_day1'] = max_return_next
            event_result['xauusd_wick_day1'] = wick_next
        else:
            event_result['xauusd_max_return_day1'] = np.nan
            event_result['xauusd_wick_day1'] = np.nan
    else:
        event_result['xauusd_max_return_day1'] = np.nan
        event_result['xauusd_wick_day1'] = np.nan
    
    event_study_results.append(event_result)

event_study_df = pd.DataFrame(event_study_results)
logger.info(f"Completed event study for {len(event_study_df)} events")

logger.info("\n" + "="*80)
logger.info("SYNCHRONIZING VIX REGIME DATA")
logger.info("="*80)

event_study_df['event_date'] = event_study_df['event_timestamp'].dt.date
vix_df['date'] = vix_df['timestamp'].dt.date

vix_regime_df = vix_df[['date', 'close']].rename(columns={'close': 'vix_close'}).copy()
vix_regime_df['vix_close'] = pd.to_numeric(vix_regime_df['vix_close'], errors='coerce')
vix_regime_df = vix_regime_df.dropna(subset=['vix_close']).drop_duplicates(subset=['date']).sort_values('date').reset_index(drop=True)
vix_regime_df['vix_ema_20'] = vix_regime_df['vix_close'].rolling(window=20, min_periods=1).mean()
vix_regime_df['vix_regime'] = (vix_regime_df['vix_close'] > vix_regime_df['vix_ema_20']).astype(int)

event_study_df = event_study_df.merge(vix_regime_df[['date', 'vix_close', 'vix_regime']], left_on='event_date', right_on='date', how='left')
event_study_df['vix_regime'] = event_study_df['vix_regime'].fillna(0).astype(int)
event_study_df['vix_close'] = event_study_df['vix_close'].fillna(event_study_df['vix_close'].mean())

logger.info(f"VIX regime synchronized for {len(event_study_df)} events")

logger.info("\n" + "="*80)
logger.info("BINARY SPIKE LABELING (75TH PERCENTILE THRESHOLD)")
logger.info("="*80)

eurusd_returns = event_study_df['eurusd_max_return_day1'].dropna()
xauusd_returns = event_study_df['xauusd_max_return_day1'].dropna()

eurusd_threshold = eurusd_returns.quantile(0.75) if len(eurusd_returns) > 0 else 0.005
xauusd_threshold = xauusd_returns.quantile(0.75) if len(xauusd_returns) > 0 else 0.005

logger.info(f"EURUSD threshold (75th percentile): {eurusd_threshold:.6f}")
logger.info(f"XAUUSD threshold (75th percentile): {xauusd_threshold:.6f}")

event_study_df['has_spike_eurusd'] = (np.abs(event_study_df['eurusd_max_return_day1']) > eurusd_threshold).astype(int)
event_study_df['has_spike_xauusd'] = (np.abs(event_study_df['xauusd_max_return_day1']) > xauusd_threshold).astype(int)
event_study_df['has_spike_exploitable'] = ((event_study_df['has_spike_eurusd'] == 1) | (event_study_df['has_spike_xauusd'] == 1)).astype(int)

spike_count = event_study_df['has_spike_exploitable'].sum()
spike_pct = 100 * spike_count / len(event_study_df)
logger.info(f"\nSpike labeling results:")
logger.info(f"  Total spikes: {spike_count} / {len(event_study_df)} ({spike_pct:.1f}%)")
logger.info(f"  EURUSD spikes: {event_study_df['has_spike_eurusd'].sum()}")
logger.info(f"  XAUUSD spikes: {event_study_df['has_spike_xauusd'].sum()}")

logger.info("\n" + "="*80)
logger.info("DIRECTIONAL LABELING")
logger.info("="*80)

event_study_df['direction_eurusd'] = np.sign(event_study_df['eurusd_max_return_day1']).replace(0, 1)
event_study_df['direction_xauusd'] = np.sign(event_study_df['xauusd_max_return_day1']).replace(0, 1)

event_study_df['direction'] = np.where(
    event_study_df['eurusd_max_return_day1'].abs() > event_study_df['xauusd_max_return_day1'].abs(),
    event_study_df['direction_eurusd'],
    event_study_df['direction_xauusd']
)

up_moves = (event_study_df['direction'] == 1).sum()
down_moves = (event_study_df['direction'] == -1).sum()
logger.info(f"Direction distribution: UP: {up_moves} ({100*up_moves/len(event_study_df):.1f}%), DOWN: {down_moves} ({100*down_moves/len(event_study_df):.1f}%)")

logger.info("\n" + "="*80)
logger.info("FEATURE ENGINEERING")
logger.info("="*80)

event_study_df['hour_of_day'] = event_study_df['event_timestamp'].dt.hour
event_study_df['day_of_week'] = event_study_df['event_timestamp'].dt.dayofweek
event_study_df['is_month_start'] = event_study_df['event_timestamp'].dt.day <= 5
event_study_df['is_month_end'] = event_study_df['event_timestamp'].dt.day >= 25

event_study_df['sentiment_score'] = (event_study_df['surprise_pct'] / (np.abs(event_study_df['surprise_pct']) + 1)).fillna(0)
event_study_df['normalized_surprise'] = (event_study_df['surprise_pct'] - event_study_df['surprise_pct'].mean()) / (event_study_df['surprise_pct'].std() + 1e-8)

event_study_df['lagged_sentiment_1d'] = event_study_df['sentiment_score'].rolling(window=2, min_periods=1).mean().shift(1)
event_study_df['lagged_sentiment_5d'] = event_study_df['sentiment_score'].rolling(window=6, min_periods=1).mean().shift(1)

event_study_df['lagged_sentiment_1d'] = event_study_df['lagged_sentiment_1d'].fillna(event_study_df['sentiment_score'].mean())
event_study_df['lagged_sentiment_5d'] = event_study_df['lagged_sentiment_5d'].fillna(event_study_df['sentiment_score'].mean())

event_study_df['impact_level_encoded'] = event_study_df['impact_level'].map({'High': 3, 'Medium': 2, 'Low': 1})

logger.info("Features engineered successfully")

logger.info("\n" + "="*80)
logger.info("DATA VALIDATION")
logger.info("="*80)

logger.info(f"Total events: {len(event_study_df)}")
logger.info(f"Date range: {event_study_df['event_timestamp'].min()} to {event_study_df['event_timestamp'].max()}")

event_dist = event_study_df['event_type'].value_counts()
logger.info(f"\nEvent distribution by type:")
for event_type, count in event_dist.items():
    logger.info(f"  {event_type}: {count}")

cluster_dist = event_study_df.groupby(['event_type', 'vix_regime']).size()
logger.info(f"\nCluster sizes (event_type x vix_regime):")
for (event_type, regime), count in cluster_dist.items():
    logger.info(f"  {event_type:15s} x Regime={int(regime)}: {count:3d}")

min_cluster = cluster_dist.min()
max_cluster = cluster_dist.max()
logger.info(f"\nCluster statistics: Min={min_cluster}, Max={max_cluster}, Mean={cluster_dist.mean():.0f}")

missing_summary = event_study_df.isnull().sum()
missing_cols = missing_summary[missing_summary > 0]
if len(missing_cols) > 0:
    logger.info(f"\nMissing data (columns with NaNs):")
    for col, count in missing_cols.items():
        pct = 100 * count / len(event_study_df)
        logger.info(f"  {col:40s}: {count:3d} ({pct:.1f}%)")
else:
    logger.info("\n✓ No missing data in key columns")

logger.info("\n" + "="*80)
logger.info("SAVING LABELED DATASET")
logger.info("="*80)

output_cols = [
    'event_timestamp', 'event_type', 'country', 'impact_level',
    'actual_value', 'consensus_value', 'previous_value', 'surprise_pct',
    'eurusd_max_return_day1', 'eurusd_wick_day1',
    'xauusd_max_return_day1', 'xauusd_wick_day1',
    'vix_close', 'vix_regime',
    'has_spike_exploitable', 'direction',
    'sentiment_score', 'normalized_surprise', 'impact_level_encoded',
    'hour_of_day', 'day_of_week', 'is_month_start', 'is_month_end',
    'lagged_sentiment_1d', 'lagged_sentiment_5d'
]

labeled_dataset = event_study_df[output_cols].copy()
labeled_dataset.to_csv(DATA_DIR / 'macro_events_labeled.csv', index=False)

logger.info(f"✓ Saved labeled dataset to {DATA_DIR / 'macro_events_labeled.csv'}")
logger.info(f"  Shape: {labeled_dataset.shape}")
logger.info(f"  Columns: {len(output_cols)}")

labeling_config = {
    'eurusd_spike_threshold': float(eurusd_threshold),
    'xauusd_spike_threshold': float(xauusd_threshold),
    'labeling_methodology': 'Binary spike: |max_return_day1| > 75th percentile; Direction: sign of max return',
    'vix_regime_definition': 'VIX > EMA_20(VIX)',
    'dataset_version': '2.0',
    'creation_timestamp': datetime.now().isoformat(),
    'total_events': len(labeled_dataset),
    'spike_positive_count': int(event_study_df['has_spike_exploitable'].sum()),
    'spike_positive_pct': float(100 * event_study_df['has_spike_exploitable'].sum() / len(event_study_df))
}

with open(DATA_DIR / 'labeling_config.json', 'w') as f:
    json.dump(labeling_config, f, indent=2)

logger.info(f"✓ Saved labeling configuration")

logger.info("\n" + "="*80)
logger.info("COMPREHENSIVE EVENT STUDY COMPLETE")
logger.info("="*80)