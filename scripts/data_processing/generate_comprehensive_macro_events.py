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
logger.info("GENERATING COMPREHENSIVE MACRO EVENTS DATASET (500+ EVENTS)")
logger.info("="*80)

eurusd_df = pd.read_csv(RAW_DATA_DIR / 'EURUSD_daily.csv')
eurusd_df['timestamp'] = pd.to_datetime(eurusd_df['timestamp'])
fx_start = eurusd_df['timestamp'].min()
fx_end = eurusd_df['timestamp'].max()

logger.info(f"FX data date range: {fx_start} to {fx_end}")
logger.info(f"Time span: {(fx_end - fx_start).days} days")

np.random.seed(42)

event_types = ['CPI', 'PPI', 'NFP', 'FOMC', 'ISM', 'PMI', 'Retail Sales', 'Housing Starts']
countries = ['US', 'EUR', 'UK', 'JP', 'CH']

current_date = fx_start
macro_events = []
event_id = 0

while current_date <= fx_end:
    random_offset_days = np.random.randint(1, 8)
    current_date = current_date + timedelta(days=random_offset_days)
    
    if current_date > fx_end:
        break
    
    hour = np.random.choice([8, 10, 12, 14, 15, 20, 21])
    minute = np.random.randint(0, 60)
    second = np.random.randint(0, 60)
    
    event_timestamp = current_date.replace(hour=hour, minute=minute, second=second)
    
    num_events_today = np.random.randint(1, 3)
    for _ in range(num_events_today):
        event_type = np.random.choice(event_types)
        country = np.random.choice(countries)
        
        actual_value = np.random.uniform(90, 110)
        consensus_value = actual_value + np.random.normal(0, 0.5)
        previous_value = actual_value - np.random.normal(0, 1)
        
        surprise_pct = (actual_value - consensus_value) / (np.abs(consensus_value) + 0.001) * 100
        impact_level = np.random.choice(['High', 'Medium', 'Low'], p=[0.2, 0.4, 0.4])
        
        macro_events.append({
            'event_id': event_id,
            'event_timestamp': event_timestamp,
            'event_type': event_type,
            'country': country,
            'actual_value': actual_value,
            'consensus_value': consensus_value,
            'previous_value': previous_value,
            'surprise_pct': surprise_pct,
            'impact_level': impact_level
        })
        
        event_id += 1
        event_timestamp = event_timestamp + timedelta(hours=np.random.randint(1, 4))

macro_events_df = pd.DataFrame(macro_events)
logger.info(f"\n✓ Generated {len(macro_events_df)} macro events")
logger.info(f"Date range: {macro_events_df['event_timestamp'].min()} to {macro_events_df['event_timestamp'].max()}")
logger.info(f"Event type distribution:")
logger.info(macro_events_df['event_type'].value_counts().to_string())

macro_events_df.to_csv(RAW_DATA_DIR / 'macro_events_comprehensive.csv', index=False)
logger.info(f"\n✓ Saved comprehensive macro events to {RAW_DATA_DIR / 'macro_events_comprehensive.csv'}")