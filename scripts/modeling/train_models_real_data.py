"""
Train ML Models - Real Data Only
Trains Random Forest (spike detection) and XGBoost (direction prediction) models
using only real data from the labeled dataset.
NO SYNTHETIC DATA - Only uses real processed data.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys
import json
import joblib
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / 'data' / 'processed'
MODELS_DIR = PROJECT_ROOT / 'models'
CONFIG_DIR = PROJECT_ROOT / 'data' / 'config'

MODELS_DIR.mkdir(parents=True, exist_ok=True)

logger.info("="*80)
logger.info("MODEL TRAINING - REAL DATA ONLY")
logger.info("="*80)

# ============================================================================
# 1. LOAD LABELED DATA
# ============================================================================
logger.info("\n" + "="*80)
logger.info("STEP 1: Loading Labeled Dataset")
logger.info("="*80)

labeled_file = PROCESSED_DATA_DIR / 'macro_events_labeled_real.csv'
if not labeled_file.exists():
    raise FileNotFoundError(
        f"Labeled dataset not found: {labeled_file}. "
        f"Run event_study_labeling_real_data.py first."
    )

df = pd.read_csv(labeled_file)
df['event_timestamp'] = pd.to_datetime(df['event_timestamp'], utc=True)
df = df.sort_values('event_timestamp').reset_index(drop=True)

logger.info(f"Loaded {len(df)} labeled events")
logger.info(f"Date range: {df['event_timestamp'].min()} to {df['event_timestamp'].max()}")
logger.info(f"Spike events: {df['has_spike_exploitable'].sum()} ({100*df['has_spike_exploitable'].mean():.2f}%)")
logger.info(f"Direction distribution: {df['direction'].value_counts().to_dict()}")

# ============================================================================
# 2. FEATURE ENGINEERING
# ============================================================================
logger.info("\n" + "="*80)
logger.info("STEP 2: Feature Engineering")
logger.info("="*80)

# Select features (excluding leaky features like max_return, wick)
clean_features = [
    'sentiment_score', 'normalized_surprise', 'actual_value', 'consensus_value',
    'surprise_pct', 'vix_close', 'vix_regime', 'impact_encoded',
    'hour_of_day', 'day_of_week', 'is_month_start', 'is_month_end',
    'lagged_sentiment_1d', 'lagged_sentiment_5d'
]

# Build feature matrix
features_list = []
feature_names = []

# Numeric features
for col in clean_features:
    if col in df.columns:
        feature_values = df[col].fillna(df[col].median())
        features_list.append(feature_values.values)
        feature_names.append(col)
    else:
        logger.warning(f"Feature {col} not found in dataset")

# Categorical features
le_event_type = LabelEncoder()
if 'event_type' in df.columns:
    event_type_encoded = le_event_type.fit_transform(df['event_type'].fillna('Unknown'))
    features_list.append(event_type_encoded)
    feature_names.append('event_type_encoded')
    joblib.dump(le_event_type, MODELS_DIR / 'label_encoder_event_type.pkl')

le_country = LabelEncoder()
if 'country' in df.columns:
    country_encoded = le_country.fit_transform(df['country'].fillna('Unknown'))
    features_list.append(country_encoded)
    feature_names.append('country_encoded')
    joblib.dump(le_country, MODELS_DIR / 'label_encoder_country.pkl')

X = np.column_stack(features_list)
logger.info(f"Feature matrix shape: {X.shape}")
logger.info(f"Number of features: {len(feature_names)}")
logger.info(f"Features: {feature_names}")

# ============================================================================
# 3. TRAIN/VALIDATION SPLIT
# ============================================================================
logger.info("\n" + "="*80)
logger.info("STEP 3: Train/Validation Split")
logger.info("="*80)

# Temporal split: 70% train, 30% validation
split_idx = int(len(df) * 0.7)
X_train, X_val = X[:split_idx], X[split_idx:]
logger.info(f"Train: {X_train.shape[0]} samples, Validation: {X_val.shape[0]} samples")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
joblib.dump(scaler, MODELS_DIR / 'feature_scaler_real.pkl')
logger.info("✓ Feature scaler saved")

# Prepare targets
y_binary = df['has_spike_exploitable'].values.astype(int)
y_direction = df['direction'].values.astype(int)

# Map direction: -1 -> 0, 0 -> 1, 1 -> 2
direction_mapping = {-1: 0, 0: 1, 1: 2}
y_direction_mapped = np.array([direction_mapping.get(d, 1) for d in y_direction])

y_binary_train, y_binary_val = y_binary[:split_idx], y_binary[split_idx:]
y_direction_train, y_direction_val = y_direction_mapped[:split_idx], y_direction_mapped[split_idx:]

logger.info(f"Binary label distribution (train): {np.bincount(y_binary_train, minlength=2)}")
logger.info(f"Direction label distribution (train): {np.bincount(y_direction_train, minlength=3)}")

# ============================================================================
# 4. TRAIN RANDOM FOREST (SPIKE DETECTION)
# ============================================================================
logger.info("\n" + "="*80)
logger.info("STEP 4: Training Random Forest (Spike Detection)")
logger.info("="*80)

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train_scaled, y_binary_train)

# Predictions
rf_pred_train = rf_model.predict(X_train_scaled)
rf_pred_val = rf_model.predict(X_val_scaled)

# Metrics
rf_train_acc = accuracy_score(y_binary_train, rf_pred_train)
rf_val_acc = accuracy_score(y_binary_val, rf_pred_val)
rf_val_prec = precision_score(y_binary_val, rf_pred_val, zero_division=0)
rf_val_rec = recall_score(y_binary_val, rf_pred_val, zero_division=0)
rf_val_f1 = f1_score(y_binary_val, rf_pred_val, zero_division=0)

logger.info(f"RF Binary - Train Acc: {rf_train_acc:.4f}, Val Acc: {rf_val_acc:.4f}")
logger.info(f"RF Validation - Precision: {rf_val_prec:.4f}, Recall: {rf_val_rec:.4f}, F1: {rf_val_f1:.4f}")
logger.info(f"RF Confusion Matrix (Validation):\n{confusion_matrix(y_binary_val, rf_pred_val)}")

# Save model
joblib.dump(rf_model, MODELS_DIR / 'random_forest_spike_model_real.pkl')
logger.info("✓ Random Forest model saved")

# Feature importance
rf_importance = rf_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': rf_importance
}).sort_values('importance', ascending=False)
logger.info(f"\nTop 10 RF Features:\n{feature_importance_df.head(10)}")

# ============================================================================
# 5. TRAIN XGBOOST (DIRECTION PREDICTION)
# ============================================================================
logger.info("\n" + "="*80)
logger.info("STEP 5: Training XGBoost (Direction Prediction)")
logger.info("="*80)

xgb_model = XGBClassifier(
    learning_rate=0.1,
    max_depth=7,
    n_estimators=300,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='multi:softmax',
    num_class=3,
    random_state=42,
    eval_metric='mlogloss',
    verbosity=0,
    n_jobs=-1
)

xgb_model.fit(X_train_scaled, y_direction_train)

# Predictions
xgb_pred_train = xgb_model.predict(X_train_scaled)
xgb_pred_val = xgb_model.predict(X_val_scaled)

# Metrics
xgb_train_acc = accuracy_score(y_direction_train, xgb_pred_train)
xgb_val_acc = accuracy_score(y_direction_val, xgb_pred_val)
xgb_val_prec = precision_score(y_direction_val, xgb_pred_val, average='weighted', zero_division=0)
xgb_val_rec = recall_score(y_direction_val, xgb_pred_val, average='weighted', zero_division=0)
xgb_val_f1 = f1_score(y_direction_val, xgb_pred_val, average='weighted', zero_division=0)

logger.info(f"XGB Directional - Train Acc: {xgb_train_acc:.4f}, Val Acc: {xgb_val_acc:.4f}")
logger.info(f"XGB Validation - Precision: {xgb_val_prec:.4f}, Recall: {xgb_val_rec:.4f}, F1: {xgb_val_f1:.4f}")
logger.info(f"XGB Confusion Matrix (Validation):\n{confusion_matrix(y_direction_val, xgb_pred_val)}")

# Save model
joblib.dump(xgb_model, MODELS_DIR / 'xgboost_directional_model_real.pkl')
logger.info("✓ XGBoost model saved")

# Feature importance
xgb_importance = xgb_model.feature_importances_
xgb_feature_importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': xgb_importance
}).sort_values('importance', ascending=False)
logger.info(f"\nTop 10 XGB Features:\n{xgb_feature_importance_df.head(10)}")

# ============================================================================
# 6. SAVE CONFIGURATION
# ============================================================================
logger.info("\n" + "="*80)
logger.info("STEP 6: Saving Model Configuration")
logger.info("="*80)

config = {
    'model_version': '4.0_real_data',
    'creation_timestamp': datetime.now().isoformat(),
    'training_samples': int(len(X_train)),
    'validation_samples': int(len(X_val)),
    'features': feature_names,
    'random_forest': {
        'n_estimators': 200,
        'max_depth': 15,
        'train_accuracy': float(rf_train_acc),
        'val_accuracy': float(rf_val_acc),
        'val_precision': float(rf_val_prec),
        'val_recall': float(rf_val_rec),
        'val_f1': float(rf_val_f1)
    },
    'xgboost': {
        'learning_rate': 0.1,
        'max_depth': 7,
        'n_estimators': 300,
        'train_accuracy': float(xgb_train_acc),
        'val_accuracy': float(xgb_val_acc),
        'val_precision': float(xgb_val_prec),
        'val_recall': float(xgb_val_rec),
        'val_f1': float(xgb_val_f1)
    },
    'data_source': 'real_data_only'
}

config_file = MODELS_DIR / 'model_config_real.json'
with open(config_file, 'w') as f:
    json.dump(config, f, indent=2)
logger.info(f"✓ Model configuration saved to {config_file}")

# Save feature importance
feature_importance_df.to_csv(MODELS_DIR / 'rf_feature_importance_real.csv', index=False)
xgb_feature_importance_df.to_csv(MODELS_DIR / 'xgb_feature_importance_real.csv', index=False)

logger.info("\n" + "="*80)
logger.info("MODEL TRAINING COMPLETE")
logger.info("="*80)

