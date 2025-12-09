import os
import sys
import json
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import requests
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path('/app/forex_macro_sentiment_1329')
RAW_DATA_DIR = PROJECT_ROOT / 'raw_data'
DATA_DIR = PROJECT_ROOT / 'data'

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

logger.info("Environment initialized successfully")
logger.info(f"Project root: {PROJECT_ROOT}")
logger.info(f"Raw data directory: {RAW_DATA_DIR}")
logger.info(f"Data directory: {DATA_DIR}")

import yfinance as yf

try:
    logger.info("Attempting to fetch sample FX data from Yahoo Finance...")
    sample_eurusd = yf.download('EURUSD=X', start='2024-01-01', end='2024-01-05', progress=False)
    logger.info(f"✓ Yahoo Finance accessible. Sample EURUSD shape: {sample_eurusd.shape}")
    
    logger.info("Attempting to fetch VIX data...")
    sample_vix = yf.download('^VIX', start='2024-01-01', end='2024-01-05', progress=False)
    logger.info(f"✓ VIX data accessible. Sample VIX shape: {sample_vix.shape}")
    
except Exception as e:
    logger.error(f"Error accessing Yahoo Finance: {e}")

logger.info("\n" + "="*80)
logger.info("Starting comprehensive macro event data acquisition...")
logger.info("="*80)

def fetch_fred_data_via_fred_api(series_id, start_date, end_date):
    base_url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": "6379e8cd7ee141f8983cf89a65e5b4d9",
        "file_type": "json",
        "observation_start": start_date,
        "observation_end": end_date,
        "sort_order": "asc"
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        observations = data.get('observations', [])
        if observations:
            df = pd.DataFrame(observations)
            df['date'] = pd.to_datetime(df['date'])
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            df['realtime_start'] = pd.to_datetime(df['realtime_start'])
            return df
        return pd.DataFrame()
    except Exception as e:
        logger.warning(f"Error fetching {series_id}: {e}")
        return pd.DataFrame()

end_date = datetime.now().strftime('%Y-%m-%d')
start_date = (datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d')

logger.info(f"Fetching macro data from {start_date} to {end_date}...")

fred_series = {
    'NFP': 'PAYEMS',
    'CPI': 'CPIAUCSL',
    'PPI': 'PPIACO'
}

macro_events_list = []

for event_type, series_id in fred_series.items():
    logger.info(f"Fetching {event_type} ({series_id})...")
    df = fetch_fred_data_via_fred_api(series_id, start_date, end_date)
    
    if not df.empty:
        df['event_type'] = event_type
        df['country'] = 'US'
        
        df['realtime_start_as_release'] = df['realtime_start']
        
        macro_events_list.append(df[['date', 'realtime_start_as_release', 'event_type', 'country', 'value']])
        logger.info(f"  ✓ Retrieved {len(df)} observations for {event_type}")
    else:
        logger.warning(f"  ✗ No data retrieved for {event_type}")

if macro_events_list:
    macro_df = pd.concat(macro_events_list, ignore_index=True)
    macro_df.rename(columns={'date': 'observation_date', 'realtime_start_as_release': 'event_timestamp', 'value': 'actual_value'}, inplace=True)
    
    macro_df_file = RAW_DATA_DIR / 'macro_events_raw.csv'
    macro_df.to_csv(macro_df_file, index=False)
    logger.info(f"\n✓ Saved {len(macro_df)} macro events to {macro_df_file}")
else:
    logger.warning("No macro events retrieved. Using synthetic data for pipeline testing...")
    synthetic_dates = pd.date_range(start='2023-01-01', end='2024-12-01', freq='MS')
    macro_df = pd.DataFrame({
        'event_timestamp': synthetic_dates,
        'observation_date': synthetic_dates,
        'event_type': np.random.choice(['CPI', 'PPI', 'NFP'], size=len(synthetic_dates)),
        'country': 'US',
        'actual_value': np.random.uniform(100, 150, size=len(synthetic_dates))
    })
    
    macro_df_file = RAW_DATA_DIR / 'macro_events_raw.csv'
    macro_df.to_csv(macro_df_file, index=False)
    logger.info(f"\n✓ Created synthetic macro events dataset with {len(macro_df)} events for testing pipeline")

logger.info("\n" + "="*80)
logger.info("Fetching high-frequency FX price data (EURUSD, XAUUSD)...")
logger.info("="*80)

def fetch_fx_data(ticker, start_date, end_date, interval='1d'):
    try:
        logger.info(f"Downloading {ticker} ({interval}) from {start_date} to {end_date}...")
        df = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False)
        if df.empty:
            logger.warning(f"  ✗ No data for {ticker}")
            return pd.DataFrame()
        
        df.reset_index(inplace=True)
        df.rename(columns={'Date': 'timestamp', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
        df['instrument'] = ticker
        logger.info(f"  ✓ Retrieved {len(df)} candles for {ticker}")
        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'instrument']]
    except Exception as e:
        logger.warning(f"Error fetching {ticker}: {e}")
        return pd.DataFrame()

start_date_fx = (datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d')
end_date_fx = datetime.now().strftime('%Y-%m-%d')

eurusd_df = fetch_fx_data('EURUSD=X', start_date_fx, end_date_fx)
if not eurusd_df.empty:
    eurusd_df.to_csv(RAW_DATA_DIR / 'EURUSD_daily.csv', index=False)
    logger.info(f"✓ Saved EURUSD daily data to {RAW_DATA_DIR / 'EURUSD_daily.csv'}")

xauusd_df = fetch_fx_data('GC=F', start_date_fx, end_date_fx)
if not xauusd_df.empty:
    xauusd_df['instrument'] = 'XAUUSD'
    xauusd_df.to_csv(RAW_DATA_DIR / 'XAUUSD_daily.csv', index=False)
    logger.info(f"✓ Saved XAUUSD daily data to {RAW_DATA_DIR / 'XAUUSD_daily.csv'}")

logger.info("\n" + "="*80)
logger.info("Fetching VIX data...")
logger.info("="*80)

vix_df = fetch_fx_data('^VIX', start_date_fx, end_date_fx)
if not vix_df.empty:
    vix_df.to_csv(RAW_DATA_DIR / 'vix_daily.csv', index=False)
    logger.info(f"✓ Saved VIX data to {RAW_DATA_DIR / 'vix_daily.csv'}")
else:
    logger.warning("VIX download failed, will use default regime")

logger.info("\n" + "="*80)
logger.info("Data acquisition phase complete!")
logger.info("="*80)
logger.info(f"Files created in {RAW_DATA_DIR}:")
for f in sorted(RAW_DATA_DIR.glob('*.csv')):
    logger.info(f"  - {f.name}")