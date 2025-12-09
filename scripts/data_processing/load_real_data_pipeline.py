"""
Load Real Data Pipeline
Loads all real data from raw/ directory and prepares it for processing.
NO SYNTHETIC DATA - Only uses real files from raw/ directory.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
RAW_DATA_DIR = PROJECT_ROOT / 'data' / 'raw'
PROCESSED_DATA_DIR = PROJECT_ROOT / 'data' / 'processed'

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

logger.info("="*80)
logger.info("LOADING REAL DATA FROM raw/ DIRECTORY")
logger.info("="*80)
logger.info(f"Project root: {PROJECT_ROOT}")
logger.info(f"Raw data directory: {RAW_DATA_DIR}")
logger.info(f"Processed data directory: {PROCESSED_DATA_DIR}")

# ============================================================================
# 1. LOAD NEWS DATA
# ============================================================================
logger.info("\n" + "="*80)
logger.info("STEP 1: Loading News Data")
logger.info("="*80)

news_file = RAW_DATA_DIR / 'high-medium impact news 2008-2025.csv'
if not news_file.exists():
    raise FileNotFoundError(f"News file not found: {news_file}")

news_df = pd.read_csv(news_file)
logger.info(f"Loaded {len(news_df)} news events from {news_file.name}")

# Parse DateTime
news_df['DateTime'] = pd.to_datetime(news_df['DateTime'], utc=True)
news_df = news_df.sort_values('DateTime').reset_index(drop=True)

# Standardize column names
news_df = news_df.rename(columns={
    'DateTime': 'event_timestamp',
    'Currency': 'country',
    'Impact': 'impact_level',
    'Event': 'event_type',
    'Actual': 'actual_value',
    'Forecast': 'consensus_value',
    'Previous': 'previous_value'
})

# Clean impact level
news_df['impact_level'] = news_df['impact_level'].str.replace(' Impact Expected', '').str.strip()

# Filter to High and Medium impact only
news_df = news_df[news_df['impact_level'].isin(['High', 'Medium'])].copy()
logger.info(f"Filtered to {len(news_df)} High/Medium impact events")

# Parse numeric values (handle percentage strings)
def parse_numeric_value(val):
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float)):
        return float(val)
    val_str = str(val).strip()
    if val_str == '' or val_str.lower() == 'nan':
        return np.nan
    # Remove percentage signs and convert
    if '%' in val_str:
        try:
            return float(val_str.replace('%', ''))
        except:
            return np.nan
    try:
        return float(val_str)
    except:
        return np.nan

news_df['actual_value'] = news_df['actual_value'].apply(parse_numeric_value)
news_df['consensus_value'] = news_df['consensus_value'].apply(parse_numeric_value)
news_df['previous_value'] = news_df['previous_value'].apply(parse_numeric_value)

# Calculate surprise percentage
mask_valid = (~news_df['actual_value'].isna()) & (~news_df['consensus_value'].isna())
news_df.loc[mask_valid, 'surprise_pct'] = (
    (news_df.loc[mask_valid, 'actual_value'] - news_df.loc[mask_valid, 'consensus_value']) /
    (news_df.loc[mask_valid, 'consensus_value'].abs() + 1e-8) * 100
)
news_df['surprise_pct'] = news_df['surprise_pct'].fillna(0)

logger.info(f"News data date range: {news_df['event_timestamp'].min()} to {news_df['event_timestamp'].max()}")
logger.info(f"Event types: {news_df['event_type'].nunique()} unique types")
logger.info(f"Countries: {sorted(news_df['country'].unique())}")

# Save processed news
news_output = PROCESSED_DATA_DIR / 'news_events_processed.csv'
news_df.to_csv(news_output, index=False)
logger.info(f"✓ Saved processed news data to {news_output}")

# ============================================================================
# 2. LOAD FX DATA (EURUSD, XAUUSD, GBPUSD, etc.)
# ============================================================================
logger.info("\n" + "="*80)
logger.info("STEP 2: Loading FX Price Data")
logger.info("="*80)

fx_data = {}

# Load EURUSD M1 data
eurusd_m1_file = RAW_DATA_DIR / 'eurusd-m1-bid-2008-01-01-2025-11-26.csv'
if eurusd_m1_file.exists():
    logger.info(f"Loading EURUSD M1 from {eurusd_m1_file.name}...")
    eurusd_m1 = pd.read_csv(eurusd_m1_file, header=0)
    eurusd_m1.columns = ['timestamp', 'open', 'high', 'low', 'close']
    eurusd_m1['timestamp'] = pd.to_datetime(eurusd_m1['timestamp'], unit='ms', utc=True)
    eurusd_m1 = eurusd_m1.sort_values('timestamp').reset_index(drop=True)
    eurusd_m1['instrument'] = 'EURUSD'
    fx_data['EURUSD_M1'] = eurusd_m1
    logger.info(f"  ✓ Loaded {len(eurusd_m1):,} EURUSD M1 candles")
    logger.info(f"    Date range: {eurusd_m1['timestamp'].min()} to {eurusd_m1['timestamp'].max()}")

# Load XAUUSD M1 data
xauusd_m1_file = RAW_DATA_DIR / 'xauusd-m1-bid-2008-01-01-2025-11-26.csv'
if xauusd_m1_file.exists():
    logger.info(f"Loading XAUUSD M1 from {xauusd_m1_file.name}...")
    xauusd_m1 = pd.read_csv(xauusd_m1_file, header=0)
    xauusd_m1.columns = ['timestamp', 'open', 'high', 'low', 'close']
    xauusd_m1['timestamp'] = pd.to_datetime(xauusd_m1['timestamp'], unit='ms', utc=True)
    xauusd_m1 = xauusd_m1.sort_values('timestamp').reset_index(drop=True)
    xauusd_m1['instrument'] = 'XAUUSD'
    fx_data['XAUUSD_M1'] = xauusd_m1
    logger.info(f"  ✓ Loaded {len(xauusd_m1):,} XAUUSD M1 candles")
    logger.info(f"    Date range: {xauusd_m1['timestamp'].min()} to {xauusd_m1['timestamp'].max()}")

# Load GBPUSD M1 data
gbpusd_m1_file = RAW_DATA_DIR / 'gbpusd-m1-bid-2008-01-01-2025-11-26.csv'
if gbpusd_m1_file.exists():
    logger.info(f"Loading GBPUSD M1 from {gbpusd_m1_file.name}...")
    gbpusd_m1 = pd.read_csv(gbpusd_m1_file, header=0)
    gbpusd_m1.columns = ['timestamp', 'open', 'high', 'low', 'close']
    gbpusd_m1['timestamp'] = pd.to_datetime(gbpusd_m1['timestamp'], unit='ms', utc=True)
    gbpusd_m1 = gbpusd_m1.sort_values('timestamp').reset_index(drop=True)
    gbpusd_m1['instrument'] = 'GBPUSD'
    fx_data['GBPUSD_M1'] = gbpusd_m1
    logger.info(f"  ✓ Loaded {len(gbpusd_m1):,} GBPUSD M1 candles")
    logger.info(f"    Date range: {gbpusd_m1['timestamp'].min()} to {gbpusd_m1['timestamp'].max()}")

# Load EURUSD H1 for daily aggregation
eurusd_h1_file = RAW_DATA_DIR / 'eurusd-h1-bid-2008-01-01-2025-11-26.csv'
if eurusd_h1_file.exists():
    logger.info(f"Loading EURUSD H1 from {eurusd_h1_file.name}...")
    eurusd_h1 = pd.read_csv(eurusd_h1_file, header=0)
    eurusd_h1.columns = ['timestamp', 'open', 'high', 'low', 'close']
    eurusd_h1['timestamp'] = pd.to_datetime(eurusd_h1['timestamp'], unit='ms', utc=True)
    eurusd_h1 = eurusd_h1.sort_values('timestamp').reset_index(drop=True)
    eurusd_h1['instrument'] = 'EURUSD'
    fx_data['EURUSD_H1'] = eurusd_h1
    logger.info(f"  ✓ Loaded {len(eurusd_h1):,} EURUSD H1 candles")

# Load XAUUSD H1
xauusd_h1_file = RAW_DATA_DIR / 'xauusd-h1-bid-2008-01-01-2025-11-26.csv'
if xauusd_h1_file.exists():
    logger.info(f"Loading XAUUSD H1 from {xauusd_h1_file.name}...")
    xauusd_h1 = pd.read_csv(xauusd_h1_file, header=0)
    xauusd_h1.columns = ['timestamp', 'open', 'high', 'low', 'close']
    xauusd_h1['timestamp'] = pd.to_datetime(xauusd_h1['timestamp'], unit='ms', utc=True)
    xauusd_h1 = xauusd_h1.sort_values('timestamp').reset_index(drop=True)
    xauusd_h1['instrument'] = 'XAUUSD'
    fx_data['XAUUSD_H1'] = xauusd_h1
    logger.info(f"  ✓ Loaded {len(xauusd_h1):,} XAUUSD H1 candles")

# ============================================================================
# 3. LOAD VIX DATA
# ============================================================================
logger.info("\n" + "="*80)
logger.info("STEP 3: Loading VIX Data")
logger.info("="*80)

vix_file = RAW_DATA_DIR / 'vix_ohlc_2008_2025.csv'
if vix_file.exists():
    logger.info(f"Loading VIX from {vix_file.name}...")
    # VIX file has header rows that need to be skipped, columns are: Date, Open, High, Low, Close, Volume, logrange
    vix_df = pd.read_csv(vix_file, skiprows=2, names=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'logrange'])
    
    # Parse date
    vix_df['timestamp'] = pd.to_datetime(vix_df['Date'], utc=True)
    
    # Get close price
    vix_df['vix_close'] = pd.to_numeric(vix_df['Close'], errors='coerce')
    
    vix_df = vix_df[['timestamp', 'vix_close']].copy()
    vix_df = vix_df.dropna(subset=['timestamp', 'vix_close'])
    vix_df = vix_df.sort_values('timestamp').reset_index(drop=True)
    
    fx_data['VIX'] = vix_df
    logger.info(f"  ✓ Loaded {len(vix_df):,} VIX data points")
    logger.info(f"    Date range: {vix_df['timestamp'].min()} to {vix_df['timestamp'].max()}")

# ============================================================================
# 4. SAVE PROCESSED DATA
# ============================================================================
logger.info("\n" + "="*80)
logger.info("STEP 4: Saving Processed Data")
logger.info("="*80)

# Save FX data
for key, df in fx_data.items():
    output_file = PROCESSED_DATA_DIR / f'{key.lower()}_processed.csv'
    df.to_csv(output_file, index=False)
    logger.info(f"✓ Saved {key} to {output_file}")

logger.info("\n" + "="*80)
logger.info("DATA LOADING COMPLETE")
logger.info("="*80)
logger.info(f"Total instruments loaded: {len(fx_data)}")
logger.info(f"News events: {len(news_df)}")
logger.info(f"All processed data saved to: {PROCESSED_DATA_DIR}")

