import pandas as pd
import numpy as np
import json
import pickle
import sys
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)

project_dir = Path('/app/forex_macro_sentiment_1329')
data_dir = project_dir / 'data'
models_dir = project_dir / 'models'
outputs_dir = project_dir / 'outputs'

logger.info("=" * 80)
logger.info("RETRAINING XGBoost Directional Model with Fixed Label Encoding")
logger.info("=" * 80)

df_cleaned = pd.read_csv(data_dir / 'macro_events_labeled_cleaned.csv')
logger.info(f"\nLoaded cleaned dataset: {df_cleaned.shape}")

logger.info("\n[STEP 1] Fixing label encoding...")
logger.info(f"  Original direction distribution:")
logger.info(f"    {df_cleaned['direction'].value_counts().sort_index().to_dict()}")

df_cleaned['direction_fixed'] = df_cleaned['direction'].map({-1: 0, 0: 1, 1: 2})
logger.info(f"  Fixed direction distribution:")
logger.info(f"    {df_cleaned['direction_fixed'].value_counts().sort_index().to_dict()}")

feature_cols = [
    'vix_close', 'sentiment_score', 'normalized_surprise', 'impact_encoded',
    'hour_of_day', 'day_of_week', 'is_month_start', 'is_month_end',
    'lagged_sentiment_1d', 'lagged_sentiment_5d', 'surprise_pct', 'vix_regime'
]

logger.info("\n[STEP 2] Preparing feature matrix and target...")
X = df_cleaned[feature_cols].copy()
y = df_cleaned['direction_fixed'].copy()

logger.info(f"  Feature matrix shape: {X.shape}")
logger.info(f"  Target shape: {y.shape}")

logger.info(f"\n  Handling missing values in vix_close (302 NaN values)...")
X['vix_close'].fillna(X['vix_close'].median(), inplace=True)
logger.info(f"    NaN count after filling: {X['vix_close'].isna().sum()}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
logger.info(f"\n  Train set: {X_train.shape}, Test set: {X_test.shape}")
logger.info(f"  Train label distribution: {y_train.value_counts().sort_index().to_dict()}")
logger.info(f"  Test label distribution: {y_test.value_counts().sort_index().to_dict()}")

logger.info("\n[STEP 3] Training XGBoost model with class weights...")
compute_class_weight = {
    0: len(y_train) / (len(y_train[y_train == 0]) * 3),
    1: len(y_train) / (len(y_train[y_train == 1]) * 3),
    2: len(y_train) / (len(y_train[y_train == 2]) * 3)
}
logger.info(f"  Class weights: {compute_class_weight}")

model = XGBClassifier(
    objective='multi:softmax',
    num_class=3,
    max_depth=6,
    learning_rate=0.08,
    n_estimators=300,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    eval_metric='mlogloss'
)

model.fit(X_train, y_train, verbose=False)
logger.info(f"  Model trained successfully")

logger.info("\n[STEP 4] Evaluating on train and test sets...")
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)
logger.info(f"  Train accuracy: {train_acc:.4f}")
logger.info(f"  Test accuracy: {test_acc:.4f}")

logger.info(f"\n  Test set detailed metrics:")
logger.info(f"    Accuracy: {test_acc:.4f}")
logger.info(f"    Precision (weighted): {precision_score(y_test, y_test_pred, average='weighted'):.4f}")
logger.info(f"    Recall (weighted): {recall_score(y_test, y_test_pred, average='weighted'):.4f}")
logger.info(f"    F1-score (weighted): {f1_score(y_test, y_test_pred, average='weighted'):.4f}")

logger.info(f"\n  Classification report:")
report = classification_report(y_test, y_test_pred, target_names=['DOWN(0)', 'NEUTRAL(1)', 'UP(2)'])
logger.info(report)

cm = confusion_matrix(y_test, y_test_pred)
logger.info(f"\n  Confusion matrix:")
logger.info(f"    {cm}")

logger.info("\n[STEP 5] Saving retrained model...")
model_path = models_dir / 'xgboost_directional_model_fixed.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(model, f)
logger.info(f"  Model saved to: {model_path}")

logger.info("\n[STEP 6] Feature importance analysis...")
feature_importance = dict(zip(feature_cols, model.feature_importances_))
sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
logger.info(f"  Top 10 features by importance:")
for feat, imp in sorted_features[:10]:
    logger.info(f"    {feat}: {imp:.4f}")

logger.info("\n[STEP 7] Saving training metadata...")
metadata = {
    'cycle': 'debug_xgboost_directional_model',
    'root_cause': 'Label encoding mismatch: trained on [0,1,2] but data had [-1,0,1]',
    'fix_applied': 'Remapped -1→0 (DOWN), 0→1 (NEUTRAL), 1→2 (UP)',
    'train_accuracy': float(train_acc),
    'test_accuracy': float(test_acc),
    'test_precision_weighted': float(precision_score(y_test, y_test_pred, average='weighted')),
    'test_recall_weighted': float(recall_score(y_test, y_test_pred, average='weighted')),
    'test_f1_weighted': float(f1_score(y_test, y_test_pred, average='weighted')),
    'class_distribution_train': y_train.value_counts().sort_index().to_dict(),
    'class_distribution_test': y_test.value_counts().sort_index().to_dict(),
    'feature_importance': {feat: float(imp) for feat, imp in sorted_features},
    'num_features': len(feature_cols),
    'features': feature_cols,
    'model_params': {
        'objective': 'multi:softmax',
        'num_class': 3,
        'max_depth': 6,
        'learning_rate': 0.08,
        'n_estimators': 300,
        'subsample': 0.8,
        'colsample_bytree': 0.8
    }
}

metadata_path = models_dir / 'directional_model_fix_metadata.json'
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)
logger.info(f"  Metadata saved to: {metadata_path}")

logger.info("\n" + "=" * 80)
logger.info("RETRAINING COMPLETE - Model ready for walk-forward validation")
logger.info("=" * 80)