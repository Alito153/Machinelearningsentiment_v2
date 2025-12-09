import pandas as pd
import numpy as np
import json
import pickle
import sys
import logging
from pathlib import Path
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)

project_dir = Path('/app/forex_macro_sentiment_1329')
data_dir = project_dir / 'data'
models_dir = project_dir / 'models'
outputs_dir = project_dir / 'outputs'

logger.info("=" * 80)
logger.info("DEEP DEBUG: Feature Setup and Label Encoding Analysis")
logger.info("=" * 80)

df_cleaned = pd.read_csv(data_dir / 'macro_events_labeled_cleaned.csv')
logger.info(f"\nCleaned dataset shape: {df_cleaned.shape}")

df_preds = pd.read_csv(outputs_dir / 'walk_forward_predictions.csv')
logger.info(f"Walk-forward predictions shape: {df_preds.shape}")
logger.info(f"\nPredictions columns: {df_preds.columns.tolist()}")

logger.info("\n[PROBLEM 1] Direction Encoding Mismatch:")
logger.info(f"  Cleaned dataset direction values: {sorted(df_cleaned['direction'].unique())}")
logger.info(f"  Expected by model (classes [0,1,2]): Direction should be 0, 1, 2")
logger.info(f"  Actual in data: -1, 0, 1")
logger.info(f"  Interpretation: -1=DOWN, 0=NEUTRAL, 1=UP (or similar)")

logger.info(f"\n  Direction distribution in cleaned dataset:")
for val in sorted(df_cleaned['direction'].unique()):
    count = (df_cleaned['direction'] == val).sum()
    pct = count / len(df_cleaned) * 100
    logger.info(f"    {val}: {count} ({pct:.1f}%)")

logger.info("\n[PROBLEM 2] Walk-forward Predictions Column Mismatch:")
logger.info(f"  Expected: 'direction_pred', 'direction_label'")
logger.info(f"  Actual: 'xgb_dir_pred', 'true_direction'")

logger.info(f"\n[PROBLEM 3] Analyzing walk-forward predictions accuracy:")
if 'xgb_dir_pred' in df_preds.columns and 'true_direction' in df_preds.columns:
    acc = accuracy_score(df_preds['true_direction'], df_preds['xgb_dir_pred'])
    logger.info(f"  Accuracy (xgb_dir_pred vs true_direction): {acc:.4f}")
    
    logger.info(f"\n  XGBoost predicted classes distribution:")
    logger.info(f"    {df_preds['xgb_dir_pred'].value_counts().sort_index().to_dict()}")
    
    logger.info(f"\n  True direction classes distribution:")
    logger.info(f"    {df_preds['true_direction'].value_counts().sort_index().to_dict()}")
    
    cm = confusion_matrix(df_preds['true_direction'], df_preds['xgb_dir_pred'])
    logger.info(f"\n  Confusion matrix (true_direction vs xgb_dir_pred):")
    logger.info(f"    {cm}")
    
    logger.info(f"\n  First 20 predictions:")
    logger.info(df_preds[['true_direction', 'xgb_dir_pred']].head(20).to_string())

logger.info("\n[FEATURE ANALYSIS] Features in cleaned dataset:")
feature_candidates = [
    'vix_close', 'sentiment_score', 'normalized_surprise', 'impact_encoded',
    'hour_of_day', 'day_of_week', 'is_month_start', 'is_month_end',
    'lagged_sentiment_1d', 'lagged_sentiment_5d', 'surprise_pct', 'vix_regime'
]

logger.info(f"  Total candidate features: {len(feature_candidates)}")
for feat in feature_candidates:
    if feat in df_cleaned.columns:
        null_count = df_cleaned[feat].isna().sum()
        dtype = df_cleaned[feat].dtype
        logger.info(f"    ✓ {feat}: dtype={dtype}, nulls={null_count}")
    else:
        logger.info(f"    ✗ {feat}: MISSING")

logger.info("\n[ENCODING INTERPRETATION]")
logger.info("  Based on context, likely mapping:")
logger.info("    -1 → DOWN (decrease in price)")
logger.info("     0 → NEUTRAL/NO_SPIKE (no significant move)")
logger.info("     1 → UP (increase in price)")
logger.info("  Model expects: 0, 1, 2 (multiclass softmax)")
logger.info("  SOLUTION: Remap -1→0, 0→1, 1→2 OR Remap -1→1, 0→0, 1→2")

logger.info("\n" + "=" * 80)