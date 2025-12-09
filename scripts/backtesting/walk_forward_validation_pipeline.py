import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import joblib
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, precision_recall_fscore_support
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

project_dir = Path('/app/forex_macro_sentiment_1329')
data_dir = project_dir / 'data'
model_dir = project_dir / 'models'
output_dir = project_dir / 'outputs'
output_dir.mkdir(exist_ok=True)

logger.info("=" * 80)
logger.info("WALK-FORWARD VALIDATION AND RISK CALIBRATION PIPELINE")
logger.info("=" * 80)

logger.info("\n[STEP 1] Loading cleaned dataset and trained models...")
df = pd.read_csv(data_dir / 'macro_events_labeled_cleaned.csv')
logger.info(f"Dataset shape: {df.shape}")

rf_model = joblib.load(model_dir / 'random_forest_spike_model_cleaned.pkl')
xgb_model = joblib.load(model_dir / 'xgboost_directional_model_cleaned.pkl')
scaler = joblib.load(model_dir / 'feature_scaler_cleaned.pkl')

with open(model_dir / 'feature_engineering_config_cleaned.json', 'r') as f:
    config = json.load(f)

feature_cols = config.get('feature_names_cleaned', [])
logger.info(f"Loaded RF model with {rf_model.n_estimators} estimators")
logger.info(f"Loaded XGBoost model with {xgb_model.n_estimators} estimators")
logger.info(f"Feature set: {len(feature_cols)} features - {feature_cols}")

logger.info("\n[STEP 2] Implementing walk-forward validation framework...")

def create_walk_forward_folds(df_len, train_pct=0.6, val_pct=0.2, stride_pct=0.2):
    """Create rolling walk-forward folds with temporal ordering."""
    total_size = df_len
    train_size = int(total_size * train_pct)
    val_size = int(total_size * val_pct)
    test_size = int(total_size * (1 - train_pct - val_pct))
    stride = int(total_size * stride_pct)
    
    folds = []
    start_idx = 0
    
    while start_idx + train_size + val_size + test_size <= total_size:
        train_end = start_idx + train_size
        val_end = train_end + val_size
        test_end = val_end + test_size
        
        fold = {
            'fold_id': len(folds),
            'train_idx': list(range(start_idx, train_end)),
            'val_idx': list(range(train_end, val_end)),
            'test_idx': list(range(val_end, test_end))
        }
        folds.append(fold)
        start_idx += stride
    
    return folds

folds = create_walk_forward_folds(len(df))
logger.info(f"Created {len(folds)} walk-forward folds")
for fold in folds:
    logger.info(f"  Fold {fold['fold_id']}: Train={len(fold['train_idx'])}, Val={len(fold['val_idx'])}, Test={len(fold['test_idx'])}")

logger.info("\n[STEP 3] Generating walk-forward predictions...")

all_predictions = []
feature_importance_list = []

for fold_id, fold in enumerate(folds):
    logger.info(f"\nProcessing fold {fold_id}...")
    
    train_idx, val_idx, test_idx = fold['train_idx'], fold['val_idx'], fold['test_idx']
    
    X_train = df.iloc[train_idx][feature_cols].fillna(df.iloc[train_idx][feature_cols].mean())
    X_val = df.iloc[val_idx][feature_cols].fillna(df.iloc[train_idx][feature_cols].mean())
    X_test = df.iloc[test_idx][feature_cols].fillna(df.iloc[train_idx][feature_cols].mean())
    
    y_spike_val = df.iloc[val_idx]['has_spike_exploitable'].values
    y_spike_test = df.iloc[test_idx]['has_spike_exploitable'].values
    y_dir_val = df.iloc[val_idx]['direction'].values
    y_dir_test = df.iloc[test_idx]['direction'].values
    
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    rf_proba_val = rf_model.predict_proba(X_val_scaled)[:, 1]
    rf_proba_test = rf_model.predict_proba(X_test_scaled)[:, 1]
    
    xgb_proba_val = xgb_model.predict_proba(X_val_scaled)
    xgb_proba_test = xgb_model.predict_proba(X_test_scaled)
    
    xgb_pred_val = xgb_model.predict(X_val_scaled)
    xgb_pred_test = xgb_model.predict(X_test_scaled)
    
    for idx_list, rf_proba, xgb_proba, xgb_pred, y_spike, y_dir, set_type in [
        (val_idx, rf_proba_val, xgb_proba_val, xgb_pred_val, y_spike_val, y_dir_val, 'val'),
        (test_idx, rf_proba_test, xgb_proba_test, xgb_pred_test, y_spike_test, y_dir_test, 'test')
    ]:
        for i, orig_idx in enumerate(idx_list):
            row_data = df.iloc[orig_idx]
            pred_record = {
                'fold_id': fold_id,
                'set_type': set_type,
                'idx': orig_idx,
                'event_timestamp': row_data.get('event_timestamp', ''),
                'event_type': row_data.get('event_type', 'Unknown'),
                'vix_close': row_data.get('vix_close', np.nan),
                'sentiment_score': row_data.get('sentiment_score', np.nan),
                'rf_spike_prob': float(rf_proba[i]),
                'xgb_dir_pred': int(xgb_pred[i]),
                'xgb_dir_prob_max': float(xgb_proba[i].max()) if hasattr(xgb_proba[i], 'max') else float(xgb_proba[i]),
                'true_spike': int(y_spike[i]),
                'true_direction': int(y_dir[i]),
                'surprise_pct': row_data.get('surprise_pct', np.nan),
                'vix_regime': row_data.get('vix_regime', np.nan)
            }
            all_predictions.append(pred_record)
    
    feature_importance_list.append({
        'fold_id': fold_id,
        'importances': dict(zip(feature_cols, rf_model.feature_importances_))
    })
    logger.info(f"  Fold {fold_id} - Val: {len(val_idx)} samples, Test: {len(test_idx)} samples")

predictions_df = pd.DataFrame(all_predictions)
logger.info(f"\nGenerated {len(predictions_df)} predictions across {len(folds)} folds")

logger.info("\n[STEP 4] Computing return and drawdown metrics...")

if 'vix_close' in df.columns:
    df['vix_ema_20'] = df['vix_close'].rolling(window=20, min_periods=1).mean()
    df['vix_regime'] = (df['vix_close'] > df['vix_ema_20']).astype(int)
elif 'vix_regime' not in df.columns:
    logger.warning("VIX regime not in dataset, creating from vix_value if available")
    if 'vix_value' in df.columns:
        df['vix_ema_20'] = df['vix_value'].rolling(window=20, min_periods=1).mean()
        df['vix_regime'] = (df['vix_value'] > df['vix_ema_20']).astype(int)
    else:
        df['vix_regime'] = 0

if 'sentiment_score' in df.columns:
    df['sentiment_sign'] = pd.cut(df['sentiment_score'], bins=[-np.inf, -0.1, 0.1, np.inf], labels=['negative', 'neutral', 'positive'])
elif 'sentiment_normalized' in df.columns:
    df['sentiment_sign'] = pd.cut(df['sentiment_normalized'], bins=[-np.inf, -0.1, 0.1, np.inf], labels=['negative', 'neutral', 'positive'])
else:
    df['sentiment_sign'] = 'neutral'

logger.info(f"VIX regime indicator created")
logger.info(f"Sentiment sign classification created")

if 'has_spike_exploitable' in df.columns:
    logger.info(f"Target variable (has_spike_exploitable) distribution:\n{df['has_spike_exploitable'].value_counts()}")

logger.info("\n[STEP 5] Creating clustering by event_type × vix_regime × sentiment_sign...")

clusters = []
for event_type in df['event_type'].unique():
    if pd.isna(event_type):
        continue
    for vix_regime_val in sorted(df['vix_regime'].unique()):
        if pd.isna(vix_regime_val):
            continue
        for sentiment_sign in ['negative', 'neutral', 'positive']:
            cluster_mask = (df['event_type'] == event_type) & (df['vix_regime'] == vix_regime_val) & (df['sentiment_sign'] == sentiment_sign)
            cluster_data = df[cluster_mask]
            
            if len(cluster_data) > 0:
                cluster_info = {
                    'cluster_id': f"{event_type}_{int(vix_regime_val)}_{sentiment_sign}",
                    'event_type': str(event_type),
                    'vix_regime': int(vix_regime_val),
                    'sentiment_sign': sentiment_sign,
                    'sample_count': len(cluster_data),
                    'indices': cluster_data.index.tolist()
                }
                clusters.append(cluster_info)

logger.info(f"Created {len(clusters)} clusters")
logger.info(f"\nCluster summary (top 10 by sample size):")
sorted_clusters = sorted(clusters, key=lambda x: x['sample_count'], reverse=True)
for c in sorted_clusters[:10]:
    status = "SUFFICIENT" if c['sample_count'] >= 50 else "INSUFFICIENT"
    logger.info(f"  {c['cluster_id']}: {c['sample_count']} samples [{status}]")

logger.info("\n[STEP 6] Extracting empirical distributions and calibrating TP/SL/horizon...")

calibration_tables = {}

for cluster in clusters:
    cluster_id = cluster['cluster_id']
    cluster_indices = cluster['indices']
    cluster_data = df.loc[cluster_indices]
    
    if cluster['sample_count'] < 50:
        calibration_tables[cluster_id] = {
            'status': 'insufficient_data',
            'sample_count': cluster['sample_count'],
            'tp_50th': None,
            'tp_60th': None,
            'sl_80th': None,
            'sl_90th': None,
            'horizon_median': None
        }
        continue
    
    max_moves = cluster_data['surprise_pct'].dropna().abs().values
    time_to_spike = cluster_data['has_spike_exploitable'].dropna().values
    
    if len(max_moves) > 0 and len(time_to_spike) > 0:
        tp_50 = np.percentile(max_moves, 50)
        tp_60 = np.percentile(max_moves, 60)
        sl_80 = np.percentile(max_moves, 80)
        sl_90 = np.percentile(max_moves, 90)
        horizon_med = np.median(time_to_spike)
        
        calibration_tables[cluster_id] = {
            'status': 'calibrated',
            'sample_count': cluster['sample_count'],
            'tp_50th': float(tp_50),
            'tp_60th': float(tp_60),
            'sl_80th': float(sl_80),
            'sl_90th': float(sl_90),
            'horizon_median': float(horizon_med),
            'surprise_pct_mean': float(max_moves.mean()),
            'surprise_pct_std': float(max_moves.std()),
            'spike_rate': float((cluster_data['has_spike_exploitable'] == 1).sum() / len(cluster_data))
        }
    else:
        calibration_tables[cluster_id] = {
            'status': 'missing_metrics',
            'sample_count': cluster['sample_count']
        }

calibrated_count = sum(1 for c in calibration_tables.values() if c.get('status') == 'calibrated')
logger.info(f"Calibrated {calibrated_count} clusters out of {len(clusters)}")

logger.info("\n[STEP 7] Validating calibration against historical outcomes...")

validation_stats = {}

for cluster_id, calib in calibration_tables.items():
    if calib.get('status') != 'calibrated':
        continue
    
    matching_cluster = [c for c in clusters if c['cluster_id'] == cluster_id]
    if not matching_cluster:
        continue
    
    cluster_indices = matching_cluster[0]['indices']
    cluster_data = df.loc[cluster_indices]
    
    tp_level = calib['tp_50th']
    sl_level = calib['sl_80th']
    spike_threshold = 0.5
    
    spike_pred = predictions_df[predictions_df['idx'].isin(cluster_indices)]['rf_spike_prob'].values
    spike_actual = cluster_data['has_spike_exploitable'].values
    
    if len(spike_pred) > 0:
        hit_tp = (spike_pred > spike_threshold).sum()
        hit_sl = (spike_actual == 0).sum()
        timeouts = len(cluster_data) - hit_tp
        
        total = len(cluster_data)
        
        validation_stats[cluster_id] = {
            'total_events': total,
            'predicted_spike_count': int(hit_tp),
            'actual_spike_count': int((cluster_data['has_spike_exploitable'] == 1).sum()),
            'timeout_count': int(timeouts),
            'predicted_spike_pct': float(hit_tp / total * 100) if total > 0 else 0,
            'actual_spike_pct': float((cluster_data['has_spike_exploitable'] == 1).sum() / total * 100) if total > 0 else 0,
            'timeout_pct': float(timeouts / total * 100) if total > 0 else 0,
            'tp_level': tp_level,
            'sl_level': sl_level,
            'spike_threshold': spike_threshold
        }

logger.info(f"\nValidation results (sample size >= 50 clusters):")
for cluster_id, stats in sorted(validation_stats.items())[:5]:
    logger.info(f"  {cluster_id}:")
    logger.info(f"    Predicted Spike: {stats['predicted_spike_pct']:.1f}% | Actual Spike: {stats['actual_spike_pct']:.1f}%")

logger.info("\n[STEP 8] Computing walk-forward metrics...")

spike_actuals = predictions_df['true_spike'].values
spike_predictions = (predictions_df['rf_spike_prob'] > 0.5).astype(int).values
spike_probs = predictions_df['rf_spike_prob'].values

if len(np.unique(spike_actuals)) > 1:
    tn, fp, fn, tp = confusion_matrix(spike_actuals, spike_predictions, labels=[0, 1]).ravel()
    spike_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    spike_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    spike_f1 = 2 * (spike_precision * spike_recall) / (spike_precision + spike_recall) if (spike_precision + spike_recall) > 0 else 0
    
    try:
        spike_auc = roc_auc_score(spike_actuals, spike_probs)
    except:
        spike_auc = 0
else:
    spike_recall = spike_precision = spike_f1 = spike_auc = 0

logger.info(f"\nSpike Detection Performance (RF):")
logger.info(f"  Recall: {spike_recall:.4f}")
logger.info(f"  Precision: {spike_precision:.4f}")
logger.info(f"  F1-Score: {spike_f1:.4f}")
logger.info(f"  ROC-AUC: {spike_auc:.4f}")

dir_correct = (predictions_df['true_direction'] == predictions_df['xgb_dir_pred']).sum()
dir_accuracy_overall = dir_correct / len(predictions_df) if len(predictions_df) > 0 else 0
logger.info(f"\nDirectional Accuracy (XGBoost): {dir_accuracy_overall:.4f}")

logger.info("\n[STEP 9] Computing feature importance rankings...")

all_importances = {}
for feat_import in feature_importance_list:
    for feat, imp in feat_import['importances'].items():
        if feat not in all_importances:
            all_importances[feat] = []
        all_importances[feat].append(imp)

avg_importances = {feat: np.mean(imps) for feat, imps in all_importances.items()}
sorted_features = sorted(avg_importances.items(), key=lambda x: x[1], reverse=True)

logger.info(f"\nTop 15 Feature Importances (averaged across folds):")
for i, (feat, imp) in enumerate(sorted_features[:15], 1):
    logger.info(f"  {i}. {feat}: {imp:.6f}")

feature_importance_df = pd.DataFrame([
    {'rank': i, 'feature': feat, 'importance': imp, 'normalized_importance': imp/sorted_features[0][1] if sorted_features else 0}
    for i, (feat, imp) in enumerate(sorted_features, 1)
])

logger.info("\n[STEP 10] Saving artifacts...")

calibration_tables['metadata'] = {
    'timestamp': datetime.now().isoformat(),
    'total_events': len(df),
    'clusters_total': len(clusters),
    'clusters_calibrated': sum(1 for c in calibration_tables.values() if isinstance(c, dict) and c.get('status') == 'calibrated'),
    'walk_forward_folds': len(folds),
    'feature_set': feature_cols
}

with open(output_dir / 'calibration_tables.json', 'w') as f:
    json.dump(calibration_tables, f, indent=2)

with open(output_dir / 'validation_statistics.json', 'w') as f:
    json.dump(validation_stats, f, indent=2)

predictions_df.to_csv(output_dir / 'walk_forward_predictions.csv', index=False)
feature_importance_df.to_csv(output_dir / 'feature_importance_rankings.csv', index=False)

cluster_analysis = {
    'metadata': {
        'total_clusters': len(clusters),
        'sufficient_data_clusters': sum(1 for c in clusters if c['sample_count'] >= 50),
        'insufficient_data_clusters': sum(1 for c in clusters if c['sample_count'] < 50)
    },
    'clusters': [
        {
            'cluster_id': c['cluster_id'],
            'event_type': c['event_type'],
            'vix_regime': int(c['vix_regime']),
            'sentiment_sign': c['sentiment_sign'],
            'sample_count': c['sample_count']
        }
        for c in sorted(clusters, key=lambda x: x['sample_count'], reverse=True)
    ]
}

with open(output_dir / 'cluster_analysis.json', 'w') as f:
    json.dump(cluster_analysis, f, indent=2)

backtesting_summary = {
    'metadata': {
        'timestamp': datetime.now().isoformat(),
        'walk_forward_folds': len(folds),
        'total_predictions': len(predictions_df)
    },
    'spike_detection': {
        'recall': float(spike_recall),
        'precision': float(spike_precision),
        'f1_score': float(spike_f1),
        'roc_auc': float(spike_auc)
    },
    'directional_accuracy': float(dir_accuracy_overall),
    'validation_statistics': validation_stats
}

with open(output_dir / 'backtesting_summary_statistics.json', 'w') as f:
    json.dump(backtesting_summary, f, indent=2)

logger.info(f"\n✓ Calibration tables saved to {output_dir / 'calibration_tables.json'}")
logger.info(f"✓ Predictions saved to {output_dir / 'walk_forward_predictions.csv'}")
logger.info(f"✓ Feature importance saved to {output_dir / 'feature_importance_rankings.csv'}")
logger.info(f"✓ Cluster analysis saved to {output_dir / 'cluster_analysis.json'}")
logger.info(f"✓ Validation stats saved to {output_dir / 'validation_statistics.json'}")
logger.info(f"✓ Summary statistics saved to {output_dir / 'backtesting_summary_statistics.json'}")

logger.info("\n" + "=" * 80)
logger.info("WALK-FORWARD VALIDATION AND RISK CALIBRATION COMPLETE")
logger.info("=" * 80)