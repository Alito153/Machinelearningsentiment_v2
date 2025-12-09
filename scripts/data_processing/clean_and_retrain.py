import pandas as pd
import numpy as np
import json
import sys
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

sys.stdout.flush()

data_path = '/app/forex_macro_sentiment_1329/data/macro_events_labeled.csv'
config_path = '/app/forex_macro_sentiment_1329/models/feature_engineering_config.json'

print("=" * 80)
print("DATASET CLEANING AND MODEL RETRAINING")
print("=" * 80)

df = pd.read_csv(data_path)
print(f"\nOriginal dataset: {df.shape[0]} rows, {df.shape[1]} columns")

leaky_features = ['eurusd_max_return', 'xauusd_max_return', 'eurusd_wick', 'xauusd_wick']
all_features = [col for col in df.columns if col not in ['has_spike_exploitable', 'direction']]
clean_features = [col for col in all_features if col not in leaky_features]

print(f"\nRemoved leaky columns: {leaky_features}")
print(f"Retained clean features: {len(clean_features)}")

df_clean = df[clean_features + ['has_spike_exploitable', 'direction']].copy()
print(f"Cleaned dataset: {df_clean.shape[0]} rows, {df_clean.shape[1]} columns")

df_clean.to_csv('/app/forex_macro_sentiment_1329/data/macro_events_labeled_cleaned.csv', index=False)
print(f"✓ Cleaned dataset saved")

print(f"\n" + "=" * 80)
print("DATA QUALITY CHECK")
print("=" * 80)

missing = df_clean.isnull().sum()
missing_clean = missing[missing > 0]
if len(missing_clean) > 0:
    print(f"Missing values: {dict(missing_clean)}")
else:
    print("No missing values in clean features")

print(f"\nTarget distributions:")
print(f"  has_spike_exploitable: {dict(df_clean['has_spike_exploitable'].value_counts())}")
print(f"  direction (raw): {dict(df_clean['direction'].value_counts())}")

categorical_cols = df_clean.select_dtypes(include=['object']).columns.tolist()
numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [col for col in numeric_cols if col not in ['has_spike_exploitable', 'direction']]

print(f"Numeric features: {len(numeric_cols)}")

df_encoded = df_clean.copy()
for col in categorical_cols:
    df_encoded[col] = pd.factorize(df_encoded[col])[0]

print(f"\n" + "=" * 80)
print("WALK-FORWARD VALIDATION: RANDOM FOREST (BINARY SPIKE)")
print("=" * 80)

X = df_encoded[numeric_cols].values
y_binary = df_clean['has_spike_exploitable'].values

split_idx = 628
X_train, X_val = X[:split_idx], X[split_idx:]
y_train, y_val = y_binary[:split_idx], y_binary[split_idx:]

scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_val_scaled = scaler.transform(X_val)

rf_model = RandomForestClassifier(n_estimators=150, max_depth=12, 
                                   class_weight='balanced', random_state=42, n_jobs=-1)
rf_model.fit(X_train_scaled, y_train)

rf_train_pred = rf_model.predict(X_train_scaled)
rf_val_pred = rf_model.predict(X_val_scaled)

rf_train_acc = accuracy_score(y_train, rf_train_pred)
rf_val_acc = accuracy_score(y_val, rf_val_pred)
rf_val_prec = precision_score(y_val, rf_val_pred, average='weighted', zero_division=0)
rf_val_recall = recall_score(y_val, rf_val_pred, average='weighted', zero_division=0)
rf_val_f1 = f1_score(y_val, rf_val_pred, average='weighted', zero_division=0)

print(f"\nRandom Forest Results (Binary Spike):")
print(f"  Train Accuracy: {rf_train_acc:.4f}")
print(f"  Val Accuracy:   {rf_val_acc:.4f}")
print(f"  Val Precision:  {rf_val_prec:.4f}")
print(f"  Val Recall:     {rf_val_recall:.4f}")
print(f"  Val F1:         {rf_val_f1:.4f}")
print(f"\nConfusion Matrix (Validation):")
cm_rf = confusion_matrix(y_val, rf_val_pred)
print(cm_rf)

rf_feature_importance = rf_model.feature_importances_
top_rf_features = sorted(zip(numeric_cols, rf_feature_importance), 
                         key=lambda x: x[1], reverse=True)[:10]
print(f"\nTop 10 Features (RF):")
for feat, imp in top_rf_features:
    print(f"  {feat}: {imp:.4f}")

pickle.dump(rf_model, open('/app/forex_macro_sentiment_1329/models/random_forest_spike_model_cleaned.pkl', 'wb'))
print(f"✓ RF model saved")

print(f"\n" + "=" * 80)
print("WALK-FORWARD VALIDATION: XGBOOST (DIRECTIONAL)")
print("=" * 80)

y_direction = df_clean['direction'].values
y_train_dir, y_val_dir = y_direction[:split_idx], y_direction[split_idx:]

y_direction_mapped = np.where(y_direction == -1, 0, np.where(y_direction == 0, 1, 2))
y_train_dir_mapped = y_direction_mapped[:split_idx]
y_val_dir_mapped = y_direction_mapped[split_idx:]

print(f"Direction mapping: -1→0 (DOWN), 0→1 (NEUTRAL), 1→2 (UP)")
print(f"Mapped train distribution: {np.bincount(y_train_dir_mapped)}")
print(f"Mapped val distribution: {np.bincount(y_val_dir_mapped)}")

xgb_model = XGBClassifier(learning_rate=0.08, max_depth=6, n_estimators=300,
                          subsample=0.8, objective='multi:softmax', num_class=3,
                          random_state=42, n_jobs=-1, verbosity=0)
xgb_model.fit(X_train_scaled, y_train_dir_mapped)

xgb_train_pred = xgb_model.predict(X_train_scaled)
xgb_val_pred = xgb_model.predict(X_val_scaled)

xgb_train_acc = accuracy_score(y_train_dir_mapped, xgb_train_pred)
xgb_val_acc = accuracy_score(y_val_dir_mapped, xgb_val_pred)
xgb_val_prec = precision_score(y_val_dir_mapped, xgb_val_pred, average='weighted', zero_division=0)
xgb_val_recall = recall_score(y_val_dir_mapped, xgb_val_pred, average='weighted', zero_division=0)
xgb_val_f1 = f1_score(y_val_dir_mapped, xgb_val_pred, average='weighted', zero_division=0)

print(f"\nXGBoost Results (Directional):")
print(f"  Train Accuracy: {xgb_train_acc:.4f}")
print(f"  Val Accuracy:   {xgb_val_acc:.4f}")
print(f"  Val Precision:  {xgb_val_prec:.4f}")
print(f"  Val Recall:     {xgb_val_recall:.4f}")
print(f"  Val F1:         {xgb_val_f1:.4f}")
print(f"\nConfusion Matrix (Validation):")
cm_xgb = confusion_matrix(y_val_dir_mapped, xgb_val_pred)
print(cm_xgb)

xgb_feature_importance = xgb_model.feature_importances_
top_xgb_features = sorted(zip(numeric_cols, xgb_feature_importance),
                          key=lambda x: x[1], reverse=True)[:10]
print(f"\nTop 10 Features (XGB):")
for feat, imp in top_xgb_features:
    print(f"  {feat}: {imp:.4f}")

pickle.dump(xgb_model, open('/app/forex_macro_sentiment_1329/models/xgboost_directional_model_cleaned.pkl', 'wb'))
print(f"✓ XGB model saved")

pickle.dump(scaler, open('/app/forex_macro_sentiment_1329/models/feature_scaler_cleaned.pkl', 'wb'))
print(f"✓ Feature scaler saved")

print(f"\n" + "=" * 80)
print("UPDATED FEATURE ENGINEERING CONFIG")
print("=" * 80)

with open(config_path, 'r') as f:
    config = json.load(f)

config['feature_names_cleaned'] = numeric_cols
config['num_features_cleaned'] = len(numeric_cols)
config['leaky_features_removed'] = leaky_features
config['rf_train_accuracy_cleaned'] = float(rf_train_acc)
config['rf_val_accuracy_cleaned'] = float(rf_val_acc)
config['rf_val_precision_cleaned'] = float(rf_val_prec)
config['rf_val_recall_cleaned'] = float(rf_val_recall)
config['rf_val_f1_cleaned'] = float(rf_val_f1)
config['xgb_train_accuracy_cleaned'] = float(xgb_train_acc)
config['xgb_val_accuracy_cleaned'] = float(xgb_val_acc)
config['xgb_val_precision_cleaned'] = float(xgb_val_prec)
config['xgb_val_recall_cleaned'] = float(xgb_val_recall)
config['xgb_val_f1_cleaned'] = float(xgb_val_f1)
config['feature_importance_rf_cleaned'] = [float(x) for x in rf_feature_importance]
config['feature_importance_xgb_cleaned'] = [float(x) for x in xgb_feature_importance]
config['cycle'] = 3
config['status'] = 'LEAKAGE_FIXED'

with open('/app/forex_macro_sentiment_1329/models/feature_engineering_config_cleaned.json', 'w') as f:
    json.dump(config, f, indent=2)

print(f"✓ Updated config saved")

print(f"\n" + "=" * 80)
print("CYCLE 3 SUMMARY: REALISTIC BASELINE ESTABLISHED")
print("=" * 80)
print(f"\nBEFORE (with leakage):")
print(f"  RF Accuracy:  99.26% (INVALID - data leakage)")
print(f"  XGB Accuracy: 97.40% (INVALID - data leakage)")
print(f"\nAFTER (leakage removed):")
print(f"  RF Accuracy:  {rf_val_acc:.2%} (REALISTIC)")
print(f"  XGB Accuracy: {xgb_val_acc:.2%} (REALISTIC)")
print(f"\n✓ All artifacts saved to /app/forex_macro_sentiment_1329/models/")
print("=" * 80)