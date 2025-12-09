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
logger.info("COMPREHENSIVE DEBUGGING: XGBoost Directional Model (0% Accuracy)")
logger.info("=" * 80)

try:
    logger.info("\n[STEP 1] Loading cleaned dataset...")
    df_cleaned = pd.read_csv(data_dir / 'macro_events_labeled_cleaned.csv')
    logger.info(f"  Dataset shape: {df_cleaned.shape}")
    logger.info(f"  Columns: {df_cleaned.columns.tolist()}")
    
    if 'direction' in df_cleaned.columns:
        logger.info(f"\n  Direction column statistics:")
        logger.info(f"    Value counts:\n{df_cleaned['direction'].value_counts()}")
        logger.info(f"    Unique values: {df_cleaned['direction'].unique()}")
        logger.info(f"    Data type: {df_cleaned['direction'].dtype}")
        logger.info(f"    Null count: {df_cleaned['direction'].isna().sum()}")
    
    logger.info(f"\n  First 5 rows of direction column:")
    logger.info(df_cleaned[['direction']].head(10).to_string())
    
except Exception as e:
    logger.error(f"  ERROR loading dataset: {e}")
    sys.exit(1)

try:
    logger.info("\n[STEP 2] Loading feature engineering config...")
    with open(models_dir / 'feature_engineering_config_cleaned.json', 'r') as f:
        config = json.load(f)
    logger.info(f"  Config keys: {config.keys()}")
    
    if 'direction_encoding' in config:
        logger.info(f"  Direction encoding: {config['direction_encoding']}")
    if 'label_encoding' in config:
        logger.info(f"  Label encoding: {config['label_encoding']}")
    if 'features' in config:
        logger.info(f"  Number of features: {len(config['features'])}")
        logger.info(f"  Features: {config['features'][:10]}...")
        
except Exception as e:
    logger.error(f"  ERROR loading config: {e}")

try:
    logger.info("\n[STEP 3] Loading walk-forward predictions...")
    df_preds = pd.read_csv(outputs_dir / 'walk_forward_predictions.csv')
    logger.info(f"  Predictions shape: {df_preds.shape}")
    logger.info(f"  Columns: {df_preds.columns.tolist()}")
    
    if 'direction_pred' in df_preds.columns and 'direction_label' in df_preds.columns:
        logger.info(f"\n  Prediction analysis:")
        logger.info(f"    Unique predicted classes: {df_preds['direction_pred'].unique()}")
        logger.info(f"    Predicted class counts:\n{df_preds['direction_pred'].value_counts()}")
        logger.info(f"    Unique label classes: {df_preds['direction_label'].unique()}")
        logger.info(f"    Label class counts:\n{df_preds['direction_label'].value_counts()}")
        
        acc = accuracy_score(df_preds['direction_label'], df_preds['direction_pred'])
        logger.info(f"\n  Overall accuracy: {acc:.4f}")
        
        cm = confusion_matrix(df_preds['direction_label'], df_preds['direction_pred'])
        logger.info(f"  Confusion matrix:\n{cm}")
        
        logger.info(f"\n  First 10 predictions:")
        logger.info(df_preds[['direction_label', 'direction_pred']].head(10).to_string())
        
except Exception as e:
    logger.error(f"  ERROR loading predictions: {e}")

try:
    logger.info("\n[STEP 4] Loading and analyzing XGBoost directional model...")
    with open(models_dir / 'xgboost_directional_model_cleaned.pkl', 'rb') as f:
        model = pickle.load(f)
    logger.info(f"  Model type: {type(model)}")
    logger.info(f"  Model params: {model.get_params() if hasattr(model, 'get_params') else 'N/A'}")
    
    if hasattr(model, 'n_classes_'):
        logger.info(f"  Number of classes: {model.n_classes_}")
    if hasattr(model, 'classes_'):
        logger.info(f"  Classes: {model.classes_}")
        
except Exception as e:
    logger.error(f"  ERROR loading model: {e}")

try:
    logger.info("\n[STEP 5] Feature column analysis...")
    feature_cols = config.get('features', []) if 'config' in locals() else []
    logger.info(f"  Expected features from config: {len(feature_cols)}")
    
    for col in feature_cols:
        if col in df_cleaned.columns:
            null_count = df_cleaned[col].isna().sum()
            if null_count > 0:
                logger.warning(f"    {col}: {null_count} NaN values")
        else:
            logger.warning(f"    {col}: MISSING from cleaned dataset")
    
    logger.info(f"\n  Actual columns in cleaned dataset: {len(df_cleaned.columns)}")
    logger.info(f"  {df_cleaned.columns.tolist()}")
    
except Exception as e:
    logger.error(f"  ERROR analyzing features: {e}")

logger.info("\n" + "=" * 80)
logger.info("DEBUG ANALYSIS COMPLETE")
logger.info("=" * 80)