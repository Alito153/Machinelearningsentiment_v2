import pandas as pd
import numpy as np
import pickle
import sys
import logging
from pathlib import Path
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)

project_dir = Path('/app/forex_macro_sentiment_1329')
data_dir = project_dir / 'data'
models_dir = project_dir / 'models'
outputs_dir = project_dir / 'outputs'

logger.info("=" * 80)
logger.info("Walk-Forward Validation with Fixed XGBoost Model")
logger.info("=" * 80)

df_cleaned = pd.read_csv(data_dir / 'macro_events_labeled_cleaned.csv')
logger.info(f"\nLoaded cleaned dataset: {df_cleaned.shape}")

with open(models_dir / 'xgboost_directional_model_fixed.pkl', 'rb') as f:
    model_fixed = pickle.load(f)
logger.info(f"Loaded fixed model: {type(model_fixed)}")

feature_cols = [
    'vix_close', 'sentiment_score', 'normalized_surprise', 'impact_encoded',
    'hour_of_day', 'day_of_week', 'is_month_start', 'is_month_end',
    'lagged_sentiment_1d', 'lagged_sentiment_5d', 'surprise_pct', 'vix_regime'
]

logger.info(f"\nFeatures: {len(feature_cols)}")

X = df_cleaned[feature_cols].copy()
y_original = df_cleaned['direction'].copy()
y_fixed = df_cleaned['direction'].map({-1: 0, 0: 1, 1: 2}).copy()

X['vix_close'].fillna(X['vix_close'].median(), inplace=True)

logger.info(f"\n[STEP 1] Chronological Walk-Forward Split (60/20/20)...")
split_60 = int(len(df_cleaned) * 0.6)
split_80 = int(len(df_cleaned) * 0.8)

X_train_wf = X.iloc[:split_60]
X_val_wf = X.iloc[split_60:split_80]
X_test_wf = X.iloc[split_80:]

y_train_wf = y_fixed.iloc[:split_60]
y_val_wf = y_fixed.iloc[split_60:split_80]
y_test_wf = y_fixed.iloc[split_80:]

logger.info(f"  Train fold: {len(X_train_wf)} samples")
logger.info(f"  Val fold: {len(X_val_wf)} samples")
logger.info(f"  Test fold: {len(X_test_wf)} samples")
logger.info(f"  Test fold label distribution: {y_test_wf.value_counts().sort_index().to_dict()}")

logger.info(f"\n[STEP 2] Generate predictions on walk-forward test fold...")
y_test_pred_fixed = model_fixed.predict(X_test_wf)
y_test_proba_fixed = model_fixed.predict_proba(X_test_wf)

logger.info(f"  Predictions shape: {y_test_pred_fixed.shape}")
logger.info(f"  Probabilities shape: {y_test_proba_fixed.shape}")

logger.info(f"\n[STEP 3] Evaluate on test fold...")
test_acc_wf = accuracy_score(y_test_wf, y_test_pred_fixed)
test_precision = precision_score(y_test_wf, y_test_pred_fixed, average='weighted', zero_division=0)
test_recall = recall_score(y_test_wf, y_test_pred_fixed, average='weighted', zero_division=0)
test_f1 = f1_score(y_test_wf, y_test_pred_fixed, average='weighted', zero_division=0)

logger.info(f"  Accuracy: {test_acc_wf:.4f}")
logger.info(f"  Precision (weighted): {test_precision:.4f}")
logger.info(f"  Recall (weighted): {test_recall:.4f}")
logger.info(f"  F1-score (weighted): {test_f1:.4f}")

logger.info(f"\n  Classification Report:")
report = classification_report(y_test_wf, y_test_pred_fixed, target_names=['DOWN(0)', 'NEUTRAL(1)', 'UP(2)'], zero_division=0)
logger.info(report)

cm = confusion_matrix(y_test_wf, y_test_pred_fixed)
logger.info(f"\n  Confusion Matrix:")
logger.info(f"    {cm}")

logger.info(f"\n[STEP 4] Save walk-forward predictions with fixed model...")
df_test_indices = df_cleaned.iloc[split_80:].copy()
df_test_indices['direction_original'] = y_original.iloc[split_80:].values
df_test_indices['direction_fixed'] = y_test_wf.values
df_test_indices['direction_pred_fixed'] = y_test_pred_fixed
df_test_indices['direction_pred_proba_0'] = y_test_proba_fixed[:, 0]
df_test_indices['direction_pred_proba_1'] = y_test_proba_fixed[:, 1]
df_test_indices['direction_pred_proba_2'] = y_test_proba_fixed[:, 2]

wf_pred_path = outputs_dir / 'walk_forward_predictions_fixed.csv'
df_test_indices[['event_timestamp', 'event_type', 'direction_original', 'direction_fixed', 
                  'direction_pred_fixed', 'direction_pred_proba_0', 'direction_pred_proba_1', 
                  'direction_pred_proba_2']].to_csv(wf_pred_path, index=False)
logger.info(f"  Saved to: {wf_pred_path}")

logger.info(f"\n[STEP 5] Per-class metrics...")
for class_idx, class_name in enumerate(['DOWN(0)', 'NEUTRAL(1)', 'UP(2)']):
    mask = y_test_wf == class_idx
    if mask.sum() > 0:
        acc = accuracy_score(y_test_wf[mask], y_test_pred_fixed[mask])
        logger.info(f"  {class_name}: {mask.sum()} samples, accuracy={acc:.4f}")

logger.info("\n" + "=" * 80)
logger.info(f"WALK-FORWARD VALIDATION COMPLETE")
logger.info(f"Test Accuracy: {test_acc_wf:.4f} (Target: >0.55) - {'✓ PASS' if test_acc_wf > 0.55 else '✗ FAIL'}")
logger.info("=" * 80)