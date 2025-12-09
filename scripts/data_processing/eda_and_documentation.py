import os
import sys
import json
import logging
from datetime import datetime
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path('/app/forex_macro_sentiment_1329')
DATA_DIR = PROJECT_ROOT / 'data'
ANALYSIS_DIR = PROJECT_ROOT / 'analysis'
ANALYSIS_DIR.mkdir(exist_ok=True)

logger.info("="*80)
logger.info("EXPLORATORY DATA ANALYSIS AND DOCUMENTATION")
logger.info("="*80)

df = pd.read_csv(DATA_DIR / 'macro_events_labeled.csv')
df['event_timestamp'] = pd.to_datetime(df['event_timestamp'])

logger.info(f"\nLoaded dataset: {len(df)} events, {len(df.columns)} columns")
logger.info(f"Date range: {df['event_timestamp'].min()} to {df['event_timestamp'].max()}")

logger.info("\n" + "="*80)
logger.info("GENERATING EDA STATISTICS AND VISUALIZATIONS")
logger.info("="*80)

plt.style.use('seaborn-v0_8-darkgrid')
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('FX Macro Events Dataset - Exploratory Data Analysis', fontsize=14, fontweight='bold')

ax = axes[0, 0]
event_counts = df['event_type'].value_counts()
event_counts.plot(kind='barh', ax=ax, color='steelblue')
ax.set_title('Event Type Distribution')
ax.set_xlabel('Count')

ax = axes[0, 1]
spike_dist = df['has_spike_exploitable'].value_counts()
labels = ['No Spike', 'Spike']
ax.pie(spike_dist.values, labels=labels, autopct='%1.1f%%', colors=['#ff7f0e', '#2ca02c'])
ax.set_title('Binary Spike Distribution')

ax = axes[0, 2]
direction_dist = df['direction'].value_counts()
labels = ['UP', 'DOWN']
colors_map = {1: '#2ca02c', -1: '#d62728'}
colors = [colors_map.get(k, 'gray') for k in direction_dist.index]
ax.bar(labels, direction_dist.values, color=colors)
ax.set_title('Directional Movement Distribution')
ax.set_ylabel('Count')

ax = axes[1, 0]
eurusd_valid = df['eurusd_max_return'].dropna()
xauusd_valid = df['xauusd_max_return'].dropna()
ax.hist([eurusd_valid, xauusd_valid], bins=30, label=['EURUSD', 'XAUUSD'], alpha=0.7)
ax.set_title('Max Return Distribution (Daily)')
ax.set_xlabel('Log Return')
ax.set_ylabel('Frequency')
ax.legend()

ax = axes[1, 1]
impact_dist = df['impact_level'].value_counts()
impact_order = {'High': 3, 'Medium': 2, 'Low': 1}
impact_dist = impact_dist.reindex(sorted(impact_dist.index, key=lambda x: impact_order.get(x, 0)))
impact_dist.plot(kind='bar', ax=ax, color=['#d62728', '#ff7f0e', '#2ca02c'])
ax.set_title('Impact Level Distribution')
ax.set_xlabel('')

ax = axes[1, 2]
sentiment_by_spike = df.groupby('has_spike_exploitable')['sentiment_score'].apply(list)
ax.boxplot([sentiment_by_spike[0], sentiment_by_spike[1]], labels=['No Spike', 'Spike'])
ax.set_title('Sentiment Score by Spike Status')
ax.set_ylabel('Sentiment Score')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(ANALYSIS_DIR / 'eda_analysis.png', dpi=150, bbox_inches='tight')
logger.info(f"✓ Saved EDA plots to {ANALYSIS_DIR / 'eda_analysis.png'}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Time-Based Patterns', fontsize=14, fontweight='bold')

ax = axes[0]
hour_spike = df.groupby('hour_of_day')['has_spike_exploitable'].mean()
ax.plot(hour_spike.index, hour_spike.values, marker='o', linewidth=2, color='steelblue')
ax.set_title('Spike Frequency by Hour of Day')
ax.set_xlabel('Hour of Day (UTC)')
ax.set_ylabel('Spike Rate')

ax = axes[1]
dow_spike = df.groupby('day_of_week')['has_spike_exploitable'].mean()
dow_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
ax.bar(range(7), dow_spike.values, color='coral')
ax.set_xticks(range(7))
ax.set_xticklabels(dow_labels)
ax.set_title('Spike Frequency by Day of Week')
ax.grid(True, alpha=0.3, axis='y')

plt.savefig(ANALYSIS_DIR / 'temporal_patterns.png', dpi=150, bbox_inches='tight')
logger.info(f"✓ Saved temporal patterns to {ANALYSIS_DIR / 'temporal_patterns.png'}")

fig, ax = plt.subplots(figsize=(10, 8))
corr_cols = ['eurusd_max_return', 'xauusd_max_return', 'sentiment_score', 'normalized_surprise', 'vix_regime', 'has_spike_exploitable']
corr_data = df[corr_cols].dropna().corr()
sns.heatmap(corr_data, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax, cbar_kws={'label': 'Correlation'})
ax.set_title('Feature Correlation Matrix')
plt.savefig(ANALYSIS_DIR / 'correlation_matrix.png', dpi=150, bbox_inches='tight')
logger.info(f"✓ Saved correlation matrix to {ANALYSIS_DIR / 'correlation_matrix.png'}")

plt.close('all')

logger.info("GENERATING SUMMARY STATISTICS")

stats = {
    'dataset': {
        'total_events': len(df),
        'date_range_start': str(df['event_timestamp'].min()),
        'date_range_end': str(df['event_timestamp'].max()),
        'columns': len(df.columns)
    },
    'labels': {
        'spike_positive_count': int(df['has_spike_exploitable'].sum()),
        'spike_positive_pct': float(100 * df['has_spike_exploitable'].sum() / len(df)),
        'direction_up_count': int((df['direction'] == 1).sum()),
        'direction_up_pct': float(100 * (df['direction'] == 1).sum() / len(df)),
        'direction_down_count': int((df['direction'] == -1).sum()),
        'direction_down_pct': float(100 * (df['direction'] == -1).sum() / len(df))
    },
    'event_types': dict(df['event_type'].value_counts()),
    'impact_levels': dict(df['impact_level'].value_counts()),
    'countries': dict(df['country'].value_counts()),
    'returns_statistics': {
        'eurusd_max_return_mean': float(df['eurusd_max_return'].mean()),
        'eurusd_max_return_std': float(df['eurusd_max_return'].std()),
        'eurusd_max_return_min': float(df['eurusd_max_return'].min()),
        'eurusd_max_return_max': float(df['eurusd_max_return'].max()),
        'xauusd_max_return_mean': float(df['xauusd_max_return'].mean()),
        'xauusd_max_return_std': float(df['xauusd_max_return'].std()),
        'xauusd_max_return_min': float(df['xauusd_max_return'].min()),
        'xauusd_max_return_max': float(df['xauusd_max_return'].max())
    },
    'vix_regime': {
        'low_regime_count': int((df['vix_regime'] == 0).sum()),
        'high_regime_count': int((df['vix_regime'] == 1).sum())
    }
}

with open(ANALYSIS_DIR / 'summary_statistics.json', 'w') as f:
    json.dump(stats, f, indent=2)

logger.info(f"✓ Summary statistics saved to {ANALYSIS_DIR / 'summary_statistics.json'}")

logger.info("CREATING COMPREHENSIVE DOCUMENTATION")

doc = """# FX MACRO SENTIMENT TRADING SYSTEM - DATASET DOCUMENTATION

## Executive Summary
This document describes the labeled dataset of macroeconomic events and their associated FX price movements, designed for training ML-based trading models to predict exploitable price spikes and directional bias following major announcements.

## Dataset Overview

### Basic Statistics
- **Total Events**: 898 macro news announcements
- **Date Range**: 2023-12-08 to 2025-12-04 (729 days)
- **Event Types**: 12 (CPI, PPI, NFP, FOMC, ISM, PMI, Retail Sales, Housing Starts, Unemployment, Inflation, Earnings, Fed Rate)
- **Countries**: 7 (US, EUR, UK, JP, CH, CA, AU)
- **Data Format**: CSV with 25 columns

### Data Quality
- **Missing Data**: 45-48% of FX return data (due to forward-looking next-day requirement at series end)
- **VIX Missing**: 33.6% (aligned with FX data availability)
- **Label Distribution**: Balanced across high/low VIX regimes and event types

## Data Acquisition Sources

### Macro News Events
**Source**: Synthetic generation with realistic statistical properties
**Rationale**: Financial Juice API requires paid credentials; synthetic data maintains statistical integrity while respecting temporal alignment with FX data.
**Coverage**: 898 events across 12 event types from Dec 2023 - Dec 2024

### FX Price Data (Daily Candles)
**Sources**: Yahoo Finance (yfinance library)
**Tickers**: EURUSD=X, GC=F (Gold/XAUUSD proxy)
**Coverage**: 2023-12-07 to 2025-12-05 (520 EURUSD, 504 Gold candles)
**Granularity**: Daily OHLC bars
**Note**: Daily data used due to M1 data availability constraints; event study adapted to daily windows

### Volatility Index (VIX)
**Source**: Yahoo Finance (^VIX)
**Coverage**: 2023-12-07 to 2025-12-05 (502 candles)
**Granularity**: Daily close prices
**Usage**: Regime filtering (VIX > EMA_20 = high volatility regime)

## Labeling Methodology

### 1. Event Study Framework
For each macro announcement, the event study window captures FX price reaction in the trading day following the event:

**Window Definition**:
- Anchor: Close price on event date
- Forward window: Next trading day (opening through closing)
- Metrics calculated:
  - **Max return**: max(ln(high/close_event)) - upside potential
  - **Adverse wick**: (close_event - min_low) / close_event - downside risk

### 2. Binary Spike Labeling ("has_spike_exploitable")
**Definition**: Whether the absolute maximum price move exceeds a statistical threshold in either instrument.

**Threshold Calculation**:
- Computed as 75th percentile of observed max returns across all events
- EURUSD threshold: 0.005795 (0.58% log return)
- XAUUSD threshold: 0.010296 (1.03% log return)
- Different thresholds account for different volatility regimes of these instruments
**Label Assignment**:
"""
# Save documentation to markdown
with open(ANALYSIS_DIR / 'dataset_documentation.md', 'w') as f:
    f.write(doc)
    f.write(doc)