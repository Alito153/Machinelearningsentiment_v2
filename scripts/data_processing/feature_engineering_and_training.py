import pandas as pd
import numpy as np
import json
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')
import logging
import sys

logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

base_path = '/app/forex_macro_sentiment_1329'
os.makedirs(f'{base_path}/models', exist_ok=True)
os.makedirs(f'{base_path}/analysis', exist_ok=True)

logger.info("Loading macro_events_labeled.csv...")
df = pd.read_csv(f'{base_path}/data/macro_events_labeled.csv')
logger.info(f"Dataset shape: {df.shape}")

df['event_timestamp'] = pd.to_datetime(df['event_timestamp'])
df = df.sort_values('event_timestamp').reset_index(drop=True)
logger.info(f"Temporal ordering verified. Date range: {df['event_timestamp'].min()} to {df['event_timestamp'].max()}")

direction_mapping = {-1: 0, 0: 1, 1: 2}
df['direction_remapped'] = df['direction'].map(direction_mapping)
logger.info(f"Direction remapped: {-1} -> 0, {0} -> 1, {1} -> 2")
logger.info(f"Remapped direction unique values: {sorted(df['direction_remapped'].unique())}")

features_to_use = [
    'sentiment_score', 'normalized_surprise', 'actual_value', 'consensus_value',
    'surprise_pct', 'vix_close', 'vix_regime', 'impact_encoded', 'hour_of_day',
    'day_of_week', 'lagged_sentiment_1d', 'lagged_sentiment_5d', 'eurusd_max_return',
    'eurusd_wick', 'xauusd_max_return', 'xauusd_wick'
]

features_list = []
feature_names = []

for col in features_to_use:
    if col in df.columns:
        feature_values = df[col].fillna(df[col].median())
        features_list.append(feature_values.values)
        feature_names.append(col)

le_event_type = LabelEncoder()
event_type_encoded = le_event_type.fit_transform(df['event_type'].fillna('Unknown'))
features_list.append(event_type_encoded)
feature_names.append('event_type_encoded')

le_country = LabelEncoder()
country_encoded = le_country.fit_transform(df['country'].fillna('Unknown'))
features_list.append(country_encoded)
feature_names.append('country_encoded')

le_impact = LabelEncoder()
impact_encoded = le_impact.fit_transform(df['impact_level'].fillna('Unknown'))
features_list.append(impact_encoded)
feature_names.append('impact_level_encoded')

X = np.column_stack(features_list)
logger.info(f"Feature matrix shape: {X.shape}")
logger.info(f"Number of features: {len(feature_names)}")

split_idx = int(len(df) * 0.7)
X_train, X_val = X[:split_idx], X[split_idx:]
logger.info(f"Train-test split: {X_train.shape[0]} training, {X_val.shape[0]} validation")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
joblib.dump(scaler, f'{base_path}/models/feature_scaler.pkl')
logger.info("Feature scaler saved")

y_binary = df['has_spike_exploitable'].values.astype(int)
y_direction = df['direction_remapped'].values.astype(int)

y_binary_train, y_binary_val = y_binary[:split_idx], y_binary[split_idx:]
y_direction_train, y_direction_val = y_direction[:split_idx], y_direction[split_idx:]

logger.info(f"Binary label distribution (train): {np.bincount(y_binary_train, minlength=2)}")
logger.info(f"Direction label distribution (train): {np.bincount(y_direction_train, minlength=3)}")

logger.info("\n=== TRAINING RANDOM FOREST FOR BINARY SPIKE PREDICTION ===")
rf_model = RandomForestClassifier(n_estimators=150, max_depth=12, min_samples_split=5,
                                   min_samples_leaf=2, class_weight='balanced', random_state=42, n_jobs=-1)
rf_model.fit(X_train_scaled, y_binary_train)
rf_pred_train = rf_model.predict(X_train_scaled)
rf_pred_val = rf_model.predict(X_val_scaled)
rf_train_acc = accuracy_score(y_binary_train, rf_pred_train)
rf_val_acc = accuracy_score(y_binary_val, rf_pred_val)
logger.info(f"RF Binary - Train Acc: {rf_train_acc:.4f}, Val Acc: {rf_val_acc:.4f}")
rf_prec = precision_score(y_binary_val, rf_pred_val, zero_division=0)
rf_rec = recall_score(y_binary_val, rf_pred_val, zero_division=0)
rf_f1 = f1_score(y_binary_val, rf_pred_val, zero_division=0)
logger.info(f"RF Validation Precision: {rf_prec:.4f}, Recall: {rf_rec:.4f}, F1: {rf_f1:.4f}")
logger.info(f"RF Confusion Matrix:\n{confusion_matrix(y_binary_val, rf_pred_val)}")
joblib.dump(rf_model, f'{base_path}/models/random_forest_spike_model.pkl')
logger.info("RF model saved")

logger.info("\n=== TRAINING XGBOOST FOR DIRECTIONAL PREDICTION ===")
xgb_model = XGBClassifier(learning_rate=0.08, max_depth=6, n_estimators=300, subsample=0.8,
                          colsample_bytree=0.8, random_state=42, eval_metric='mlogloss', verbosity=0)
xgb_model.fit(X_train_scaled, y_direction_train)
xgb_pred_train = xgb_model.predict(X_train_scaled)
xgb_pred_val = xgb_model.predict(X_val_scaled)
xgb_train_acc = accuracy_score(y_direction_train, xgb_pred_train)
xgb_val_acc = accuracy_score(y_direction_val, xgb_pred_val)
logger.info(f"XGB Directional - Train Acc: {xgb_train_acc:.4f}, Val Acc: {xgb_val_acc:.4f}")
xgb_prec = precision_score(y_direction_val, xgb_pred_val, average='weighted', zero_division=0)
xgb_rec = recall_score(y_direction_val, xgb_pred_val, average='weighted', zero_division=0)
xgb_f1 = f1_score(y_direction_val, xgb_pred_val, average='weighted', zero_division=0)
logger.info(f"XGB Validation Precision (weighted): {xgb_prec:.4f}, Recall (weighted): {xgb_rec:.4f}, F1 (weighted): {xgb_f1:.4f}")
logger.info(f"XGB Confusion Matrix:\n{confusion_matrix(y_direction_val, xgb_pred_val)}")
joblib.dump(xgb_model, f'{base_path}/models/xgboost_directional_model.pkl')
logger.info("XGB model saved")

logger.info("\n=== FEATURE IMPORTANCE ANALYSIS ===")
rf_importance = rf_model.feature_importances_
xgb_importance = xgb_model.feature_importances_

logger.info("Top 15 RF features by importance:")
rf_top = sorted(zip(feature_names, rf_importance), key=lambda x: x[1], reverse=True)[:15]
for i, (fname, imp) in enumerate(rf_top, 1):
    logger.info(f"  {i}. {fname}: {imp:.4f}")

logger.info("\nTop 15 XGB features by importance:")
xgb_top = sorted(zip(feature_names, xgb_importance), key=lambda x: x[1], reverse=True)[:15]
for i, (fname, imp) in enumerate(xgb_top, 1):
    logger.info(f"  {i}. {fname}: {imp:.4f}")

config = {
    'feature_names': feature_names,
    'num_features': len(feature_names),
    'split_index': int(split_idx),
    'train_size': int(X_train_scaled.shape[0]),
    'val_size': int(X_val_scaled.shape[0]),
    'direction_mapping': {str(k): v for k, v in direction_mapping.items()},
    'rf_train_accuracy': float(rf_train_acc),
    'rf_val_accuracy': float(rf_val_acc),
    'rf_val_precision': float(rf_prec),
    'rf_val_recall': float(rf_rec),
    'rf_val_f1': float(rf_f1),
    'xgb_train_accuracy': float(xgb_train_acc),
    'xgb_val_accuracy': float(xgb_val_acc),
    'xgb_val_precision': float(xgb_prec),
    'xgb_val_recall': float(xgb_rec),
    'xgb_val_f1': float(xgb_f1),
    'feature_importance_rf': rf_importance.tolist(),
    'feature_importance_xgb': xgb_importance.tolist()
}
with open(f'{base_path}/models/feature_engineering_config.json', 'w') as f:
    json.dump(config, f, indent=2)

logger.info("\n===== PIPELINE COMPLETED SUCCESSFULLY =====")
logger.info(f"Models saved to {base_path}/models/")
logger.info(f"Configuration saved to {base_path}/models/feature_engineering_config.json")