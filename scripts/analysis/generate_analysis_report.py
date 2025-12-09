import os
import sys
import json
import pandas as pd
import numpy as np
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

BASE_DIR = Path('/app/forex_trading_strategy_1723/data/extracted/forex_macro_sentiment_1329')
OUTPUT_DIR = BASE_DIR / 'analysis'

os.makedirs(OUTPUT_DIR, exist_ok=True)

logger.info("=" * 80)
logger.info("COMPREHENSIVE SENTIMENT DATA ANALYSIS REPORT")
logger.info("=" * 80)

macro_events_labeled = pd.read_csv(BASE_DIR / 'data' / 'macro_events_labeled.csv')
macro_events_cleaned = pd.read_csv(BASE_DIR / 'data' / 'macro_events_labeled_cleaned.csv')
eurusd_data = pd.read_csv(BASE_DIR / 'raw_data' / 'EURUSD_daily.csv')
xauusd_data = pd.read_csv(BASE_DIR / 'raw_data' / 'XAUUSD_daily.csv')
vix_data = pd.read_csv(BASE_DIR / 'raw_data' / 'vix_daily.csv')

logger.info("\n" + "=" * 80)
logger.info("1. DATA STRUCTURE OVERVIEW")
logger.info("=" * 80)
logger.info(f"Macro Events Labeled: {macro_events_labeled.shape[0]} rows × {macro_events_labeled.shape[1]} columns")
logger.info(f"Macro Events Cleaned: {macro_events_cleaned.shape[0]} rows × {macro_events_cleaned.shape[1]} columns")
logger.info(f"EURUSD Daily: {eurusd_data.shape[0]} rows × {eurusd_data.shape[1]} columns")
logger.info(f"XAUUSD Daily: {xauusd_data.shape[0]} rows × {xauusd_data.shape[1]} columns")
logger.info(f"VIX Daily: {vix_data.shape[0]} rows × {vix_data.shape[1]} columns")

logger.info("\nMacro Events Columns:")
logger.info(f"{list(macro_events_labeled.columns)}")

logger.info("\n" + "=" * 80)
logger.info("2. MACRO EVENTS DATASET SUMMARY")
logger.info("=" * 80)
logger.info(f"Total Events: {len(macro_events_labeled)}")
logger.info(f"Event Types: {macro_events_labeled['event_type'].nunique()} unique types")
logger.info(f"Countries: {macro_events_labeled['country'].nunique()} unique countries")
logger.info(f"Impact Levels: {sorted(macro_events_labeled['impact_level'].unique())}")
logger.info(f"Date Range: {macro_events_labeled['event_timestamp'].min()} to {macro_events_labeled['event_timestamp'].max()}")

logger.info("\n" + "=" * 80)
logger.info("3. SENTIMENT METRICS ANALYSIS")
logger.info("=" * 80)
logger.info(f"Sentiment Score - Mean: {macro_events_labeled['sentiment_score'].mean():.6f}, Std: {macro_events_labeled['sentiment_score'].std():.6f}")
logger.info(f"Sentiment Score - Range: [{macro_events_labeled['sentiment_score'].min():.6f}, {macro_events_labeled['sentiment_score'].max():.6f}]")
logger.info(f"Normalized Surprise - Mean: {macro_events_labeled['normalized_surprise'].mean():.6f}, Std: {macro_events_labeled['normalized_surprise'].std():.6f}")
logger.info(f"Surprise % - Mean: {macro_events_labeled['surprise_pct'].mean():.6f}, Std: {macro_events_labeled['surprise_pct'].std():.6f}")

logger.info("\n" + "=" * 80)
logger.info("4. SPIKE DETECTION ANALYSIS")
logger.info("=" * 80)
spike_count = macro_events_labeled['has_spike_exploitable'].sum()
spike_rate = 100 * macro_events_labeled['has_spike_exploitable'].mean()
logger.info(f"Total Exploitable Spikes: {spike_count} out of {len(macro_events_labeled)} ({spike_rate:.2f}%)")

logger.info("\nSpike Rate by Event Type:")
spike_by_type = macro_events_labeled.groupby('event_type').agg({
    'has_spike_exploitable': ['sum', 'count', 'mean']
}).round(4)
spike_by_type.columns = ['Spikes', 'Total', 'Rate']
logger.info(spike_by_type.to_string())

logger.info("\nSpike Rate by Impact Level:")
spike_by_impact = macro_events_labeled.groupby('impact_level').agg({
    'has_spike_exploitable': ['sum', 'count', 'mean']
}).round(4)
spike_by_impact.columns = ['Spikes', 'Total', 'Rate']
logger.info(spike_by_impact.to_string())

logger.info("\nSpike Rate by VIX Regime:")
spike_by_vix = macro_events_labeled.groupby('vix_regime').agg({
    'has_spike_exploitable': ['sum', 'count', 'mean']
}).round(4)
spike_by_vix.columns = ['Spikes', 'Total', 'Rate']
logger.info(spike_by_vix.to_string())

logger.info("\n" + "=" * 80)
logger.info("5. PRICE ACTION ANALYSIS")
logger.info("=" * 80)
logger.info(f"EURUSD Max Return: Mean={macro_events_labeled['eurusd_max_return'].mean():.6f}, Std={macro_events_labeled['eurusd_max_return'].std():.6f}")
logger.info(f"EURUSD Wick: Mean={macro_events_labeled['eurusd_wick'].mean():.6f}, Std={macro_events_labeled['eurusd_wick'].std():.6f}")
logger.info(f"XAUUSD Max Return: Mean={macro_events_labeled['xauusd_max_return'].mean():.6f}, Std={macro_events_labeled['xauusd_max_return'].std():.6f}")
logger.info(f"XAUUSD Wick: Mean={macro_events_labeled['xauusd_wick'].mean():.6f}, Std={macro_events_labeled['xauusd_wick'].std():.6f}")

logger.info("\n" + "=" * 80)
logger.info("6. VIX REGIME ANALYSIS")
logger.info("=" * 80)
vix_regimes = macro_events_labeled['vix_regime'].value_counts().sort_index()
logger.info(f"VIX Regime Distribution:\n{vix_regimes.to_string()}")
logger.info(f"Regime 0 (Low Vol): {vix_regimes[0]} events ({100*vix_regimes[0]/len(macro_events_labeled):.1f}%)")
logger.info(f"Regime 1 (High Vol): {vix_regimes[1]} events ({100*vix_regimes[1]/len(macro_events_labeled):.1f}%)")

logger.info("\n" + "=" * 80)
logger.info("7. TEMPORAL ANALYSIS")
logger.info("=" * 80)
macro_events_labeled['event_timestamp'] = pd.to_datetime(macro_events_labeled['event_timestamp'])
events_by_year = macro_events_labeled.groupby(macro_events_labeled['event_timestamp'].dt.year).size()
logger.info(f"Events by Year:\n{events_by_year.to_string()}")

logger.info("\n" + "=" * 80)
logger.info("8. DATA QUALITY ASSESSMENT")
logger.info("=" * 80)
missing = macro_events_labeled.isnull().sum()
total_cells = len(macro_events_labeled) * len(macro_events_labeled.columns)
missing_pct = 100 * missing.sum() / total_cells
logger.info(f"Total Missing Values: {missing.sum()} out of {total_cells} ({missing_pct:.2f}%)")
if (missing > 0).any():
    logger.info(f"Missing Values by Column:\n{missing[missing > 0].to_string()}")
else:
    logger.info("No missing values in dataset")

logger.info("\n" + "=" * 80)
logger.info("9. FEATURE ENGINEERING CONFIGURATION")
logger.info("=" * 80)
try:
    with open(BASE_DIR / 'models' / 'feature_engineering_config_cleaned.json', 'r') as f:
        feat_config = json.load(f)
    logger.info(f"Number of Features: {feat_config.get('num_features', 'N/A')}")
    logger.info(f"Train/Test Split Index: {feat_config.get('split_index', 'N/A')}")
    logger.info(f"Features: {feat_config.get('feature_names', [])}")
except Exception as e:
    logger.error(f"Could not read feature config: {e}")

logger.info("\n" + "=" * 80)
logger.info("10. MODEL PERFORMANCE SUMMARY")
logger.info("=" * 80)
try:
    with open(BASE_DIR / 'models' / 'directional_model_fix_metadata.json', 'r') as f:
        model_meta = json.load(f)
    logger.info(f"Directional Model - Test Accuracy: {model_meta.get('test_accuracy', 'N/A')}")
    logger.info(f"Directional Model - Test F1 (Weighted): {model_meta.get('test_f1_weighted', 'N/A')}")
    logger.info(f"Directional Model - Test Precision: {model_meta.get('test_precision_weighted', 'N/A')}")
except Exception as e:
    logger.warning(f"Model metadata not available: {e}")

try:
    with open(BASE_DIR / 'outputs' / 'backtesting_summary_statistics.json', 'r') as f:
        backtest_stats = json.load(f)
    logger.info(f"Spike Detection - Recall: {backtest_stats['spike_detection'].get('recall', 'N/A')}")
    logger.info(f"Spike Detection - Precision: {backtest_stats['spike_detection'].get('precision', 'N/A')}")
    logger.info(f"Spike Detection - F1 Score: {backtest_stats['spike_detection'].get('f1_score', 'N/A')}")
except Exception as e:
    logger.warning(f"Backtesting stats not available: {e}")

logger.info("\n" + "=" * 80)
logger.info("ANALYSIS COMPLETE")
logger.info("=" * 80)