"""
Event Study Labeling Pipeline - Real Data Only
Performs event study analysis on real news events using real FX price data.
NO SYNTHETIC DATA - Only uses real data from processed/ directory.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime, timedelta
import sys
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / 'data' / 'processed'
OUTPUT_DIR = PROCESSED_DATA_DIR
CONFIG_DIR = PROJECT_ROOT / 'data' / 'config'

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

logger.info("="*80)
logger.info("EVENT STUDY LABELING PIPELINE - REAL DATA ONLY")
logger.info("="*80)

# ============================================================================
# 1. LOAD PROCESSED DATA
# ============================================================================
logger.info("\n" + "="*80)
logger.info("STEP 1: Loading Processed Data")
logger.info("="*80)

# Load news events
news_file = PROCESSED_DATA_DIR / 'news_events_processed.csv'
if not news_file.exists():
    raise FileNotFoundError(f"Processed news file not found: {news_file}. Run load_real_data_pipeline.py first.")

news_df = pd.read_csv(news_file)
news_df['event_timestamp'] = pd.to_datetime(news_df['event_timestamp'], utc=True)
news_df = news_df.sort_values('event_timestamp').reset_index(drop=True)
logger.info(f"Loaded {len(news_df)} news events")

# Load FX M1 data
eurusd_m1_file = PROCESSED_DATA_DIR / 'eurusd_m1_processed.csv'
xauusd_m1_file = PROCESSED_DATA_DIR / 'xauusd_m1_processed.csv'
gbpusd_m1_file = PROCESSED_DATA_DIR / 'gbpusd_m1_processed.csv'

if not eurusd_m1_file.exists():
    raise FileNotFoundError(f"EURUSD M1 file not found: {eurusd_m1_file}")

eurusd_m1 = pd.read_csv(eurusd_m1_file)
eurusd_m1['timestamp'] = pd.to_datetime(eurusd_m1['timestamp'], utc=True)
eurusd_m1 = eurusd_m1.sort_values('timestamp').reset_index(drop=True)
logger.info(f"Loaded {len(eurusd_m1):,} EURUSD M1 candles")

xauusd_m1 = None
if xauusd_m1_file.exists():
    xauusd_m1 = pd.read_csv(xauusd_m1_file)
    xauusd_m1['timestamp'] = pd.to_datetime(xauusd_m1['timestamp'], utc=True)
    xauusd_m1 = xauusd_m1.sort_values('timestamp').reset_index(drop=True)
    logger.info(f"Loaded {len(xauusd_m1):,} XAUUSD M1 candles")

gbpusd_m1 = None
if gbpusd_m1_file.exists():
    gbpusd_m1 = pd.read_csv(gbpusd_m1_file)
    gbpusd_m1['timestamp'] = pd.to_datetime(gbpusd_m1['timestamp'], utc=True)
    gbpusd_m1 = gbpusd_m1.sort_values('timestamp').reset_index(drop=True)
    logger.info(f"Loaded {len(gbpusd_m1):,} GBPUSD M1 candles")

# Load VIX data
vix_file = PROCESSED_DATA_DIR / 'vix_processed.csv'
vix_df = None
if vix_file.exists():
    vix_df = pd.read_csv(vix_file)
    vix_df['timestamp'] = pd.to_datetime(vix_df['timestamp'], utc=True)
    vix_df = vix_df.sort_values('timestamp').reset_index(drop=True)
    logger.info(f"Loaded {len(vix_df):,} VIX data points")

# ============================================================================
# 2. EVENT STUDY ANALYSIS
# ============================================================================
logger.info("\n" + "="*80)
logger.info("STEP 2: Performing Event Study Analysis")
logger.info("="*80)

def calculate_max_return_and_wick(prices_df, event_time, window_minutes=15):
    """
    Calculate max return and adverse wick in a window after event.
    
    Args:
        prices_df: DataFrame with timestamp, open, high, low, close
        event_time: Event timestamp
        window_minutes: Minutes after event to analyze
    
    Returns:
        max_return: Maximum log return in window
        adverse_wick: Maximum adverse wick (slippage risk)
        direction: 1 if positive, -1 if negative, 0 if neutral
    """
    window_start = event_time
    window_end = event_time + timedelta(minutes=window_minutes)
    
    mask = (prices_df['timestamp'] >= window_start) & (prices_df['timestamp'] <= window_end)
    window_data = prices_df[mask].copy()
    
    if len(window_data) < 2:
        return np.nan, np.nan, 0
    
    # Get open price at event time (or first candle in window)
    open_price = window_data.iloc[0]['open']
    
    # Calculate max positive and negative moves
    max_price = window_data['high'].max()
    min_price = window_data['low'].min()
    
    # Log returns
    max_return_up = np.log(max_price / open_price) if open_price > 0 else 0
    max_return_down = np.log(min_price / open_price) if open_price > 0 else 0
    
    # Max return is the larger absolute move
    if abs(max_return_up) > abs(max_return_down):
        max_return = max_return_up
        direction = 1
    else:
        max_return = max_return_down
        direction = -1
    
    # Adverse wick: maximum move against the direction
    if direction == 1:
        adverse_wick = abs(max_return_down)  # How much it went down before going up
    else:
        adverse_wick = abs(max_return_up)  # How much it went up before going down
    
    return max_return, adverse_wick, direction

def get_vix_at_time(vix_df, event_time):
    """Get VIX value closest to event time."""
    if vix_df is None or len(vix_df) == 0:
        return np.nan
    
    # Find closest VIX timestamp
    vix_df['time_diff'] = abs((vix_df['timestamp'] - event_time).dt.total_seconds())
    closest_idx = vix_df['time_diff'].idxmin()
    
    if vix_df.loc[closest_idx, 'time_diff'] > 86400:  # More than 1 day away
        return np.nan
    
    return vix_df.loc[closest_idx, 'vix_close']

# Process each news event
event_study_results = []

logger.info(f"Processing {len(news_df)} events...")

for idx, event in news_df.iterrows():
    if idx % 100 == 0:
        logger.info(f"  Processed {idx}/{len(news_df)} events...")
    
    event_time = event['event_timestamp']
    event_result = {
        'event_timestamp': event_time,
        'event_type': event['event_type'],
        'country': event['country'],
        'impact_level': event['impact_level'],
        'actual_value': event['actual_value'],
        'consensus_value': event['consensus_value'],
        'previous_value': event['previous_value'],
        'surprise_pct': event.get('surprise_pct', 0),
    }
    
    # Analyze EURUSD
    eurusd_max_return, eurusd_wick, eurusd_direction = calculate_max_return_and_wick(
        eurusd_m1, event_time, window_minutes=15
    )
    event_result['eurusd_max_return'] = eurusd_max_return
    event_result['eurusd_wick'] = eurusd_wick
    event_result['eurusd_direction'] = eurusd_direction
    
    # Analyze XAUUSD if available
    if xauusd_m1 is not None:
        xauusd_max_return, xauusd_wick, xauusd_direction = calculate_max_return_and_wick(
            xauusd_m1, event_time, window_minutes=15
        )
        event_result['xauusd_max_return'] = xauusd_max_return
        event_result['xauusd_wick'] = xauusd_wick
        event_result['xauusd_direction'] = xauusd_direction
    else:
        event_result['xauusd_max_return'] = np.nan
        event_result['xauusd_wick'] = np.nan
        event_result['xauusd_direction'] = 0
    
    # Analyze GBPUSD if available
    if gbpusd_m1 is not None:
        gbpusd_max_return, gbpusd_wick, gbpusd_direction = calculate_max_return_and_wick(
            gbpusd_m1, event_time, window_minutes=15
        )
        event_result['gbpusd_max_return'] = gbpusd_max_return
        event_result['gbpusd_wick'] = gbpusd_wick
        event_result['gbpusd_direction'] = gbpusd_direction
    else:
        event_result['gbpusd_max_return'] = np.nan
        event_result['gbpusd_wick'] = np.nan
        event_result['gbpusd_direction'] = 0
    
    # Get VIX at event time
    vix_value = get_vix_at_time(vix_df, event_time) if vix_df is not None else np.nan
    event_result['vix_close'] = vix_value
    
    event_study_results.append(event_result)

study_df = pd.DataFrame(event_study_results)
logger.info(f"✓ Completed event study for {len(study_df)} events")

# ============================================================================
# 3. LABELING: SPIKE DETECTION
# ============================================================================
logger.info("\n" + "="*80)
logger.info("STEP 3: Labeling Spike Events")
logger.info("="*80)

# Calculate percentiles for spike detection
eurusd_returns = study_df['eurusd_max_return'].abs().dropna()
xauusd_returns = study_df['xauusd_max_return'].abs().dropna()

if len(eurusd_returns) > 0:
    eurusd_threshold = eurusd_returns.quantile(0.75)
    logger.info(f"EURUSD spike threshold (75th percentile): {eurusd_threshold:.6f}")
else:
    eurusd_threshold = 0.005
    logger.warning(f"No EURUSD returns, using default threshold: {eurusd_threshold}")

if len(xauusd_returns) > 0:
    xauusd_threshold = xauusd_returns.quantile(0.75)
    logger.info(f"XAUUSD spike threshold (75th percentile): {xauusd_threshold:.6f}")
else:
    xauusd_threshold = 0.01
    logger.warning(f"No XAUUSD returns, using default threshold: {xauusd_threshold}")

# Label spikes: event has exploitable spike if max_return exceeds threshold
study_df['has_spike_exploitable'] = (
    (study_df['eurusd_max_return'].abs() >= eurusd_threshold) |
    (study_df['xauusd_max_return'].abs() >= xauusd_threshold)
).astype(int)

# Determine direction: use the instrument with larger absolute return
def determine_direction(row):
    eur_abs = abs(row.get('eurusd_max_return', 0)) if not pd.isna(row.get('eurusd_max_return')) else 0
    xau_abs = abs(row.get('xauusd_max_return', 0)) if not pd.isna(row.get('xauusd_max_return')) else 0
    
    if eur_abs >= xau_abs:
        return row.get('eurusd_direction', 0)
    else:
        return row.get('xauusd_direction', 0)

study_df['direction'] = study_df.apply(determine_direction, axis=1)

# Map direction: -1 (down) -> 0, 0 (neutral) -> 1, 1 (up) -> 2 for ML
study_df['direction_encoded'] = study_df['direction'].map({-1: 0, 0: 1, 1: 2})

spike_count = study_df['has_spike_exploitable'].sum()
spike_pct = 100 * spike_count / len(study_df)
logger.info(f"Spike events: {spike_count} ({spike_pct:.2f}%)")
logger.info(f"Direction distribution: {study_df['direction'].value_counts().to_dict()}")

# ============================================================================
# 4. FEATURE ENGINEERING
# ============================================================================
logger.info("\n" + "="*80)
logger.info("STEP 4: Feature Engineering")
logger.info("="*80)

# VIX regime (low < 15, medium 15-25, high > 25)
study_df['vix_regime'] = pd.cut(
    study_df['vix_close'],
    bins=[0, 15, 25, 100],
    labels=[0, 1, 2]
).astype(float).fillna(1)

# Temporal features
study_df['hour_of_day'] = study_df['event_timestamp'].dt.hour
study_df['day_of_week'] = study_df['event_timestamp'].dt.dayofweek
study_df['is_month_start'] = (study_df['event_timestamp'].dt.day <= 5).astype(int)
study_df['is_month_end'] = (study_df['event_timestamp'].dt.day >= 25).astype(int)

# Sentiment score from surprise
study_df['sentiment_score'] = (
    np.sign(study_df['surprise_pct']) * 
    np.minimum(np.abs(study_df['surprise_pct']) / 100, 1.0)
).fillna(0)

# Normalized surprise
surprise_mean = study_df['surprise_pct'].mean()
surprise_std = study_df['surprise_pct'].std()
study_df['normalized_surprise'] = (
    (study_df['surprise_pct'] - surprise_mean) / (surprise_std + 1e-8)
).fillna(0)

# Impact encoding
study_df['impact_encoded'] = study_df['impact_level'].map({'High': 3, 'Medium': 2, 'Low': 1}).fillna(2)

# Lagged sentiment (rolling windows)
study_df = study_df.sort_values('event_timestamp').reset_index(drop=True)
study_df['lagged_sentiment_1d'] = study_df['sentiment_score'].rolling(2, min_periods=1).mean().shift(1).fillna(0)
study_df['lagged_sentiment_5d'] = study_df['sentiment_score'].rolling(6, min_periods=1).mean().shift(1).fillna(0)

logger.info("✓ Feature engineering complete")

# ============================================================================
# 5. SAVE LABELED DATASET
# ============================================================================
logger.info("\n" + "="*80)
logger.info("STEP 5: Saving Labeled Dataset")
logger.info("="*80)

output_file = OUTPUT_DIR / 'macro_events_labeled_real.csv'
study_df.to_csv(output_file, index=False)
logger.info(f"✓ Saved labeled dataset to {output_file}")
logger.info(f"  Total events: {len(study_df)}")
logger.info(f"  Spike events: {spike_count} ({spike_pct:.2f}%)")
logger.info(f"  Date range: {study_df['event_timestamp'].min()} to {study_df['event_timestamp'].max()}")

# Save config
config = {
    'dataset_version': '4.0_real_data',
    'creation_timestamp': datetime.now().isoformat(),
    'total_events': len(study_df),
    'date_range': {
        'start': str(study_df['event_timestamp'].min()),
        'end': str(study_df['event_timestamp'].max())
    },
    'spike_positive_count': int(spike_count),
    'spike_positive_pct': float(spike_pct),
    'eurusd_threshold': float(eurusd_threshold),
    'xauusd_threshold': float(xauusd_threshold),
    'labeling_methodology': 'Binary: |max_return| > 75th percentile; Direction: sign of dominant instrument return',
    'data_source': 'real_data_only'
}

config_file = CONFIG_DIR / 'labeling_config_real.json'
with open(config_file, 'w') as f:
    json.dump(config, f, indent=2)
logger.info(f"✓ Saved labeling config to {config_file}")

logger.info("\n" + "="*80)
logger.info("EVENT STUDY LABELING COMPLETE")
logger.info("="*80)

