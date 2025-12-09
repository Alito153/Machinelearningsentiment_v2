import pandas as pd
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt
from pathlib import Path

print("=" * 80)
print("FEATURE IMPORTANCE ANALYSIS - CLEANED MODELS")
print("=" * 80)

rf_model = pickle.load(open('/app/forex_macro_sentiment_1329/models/random_forest_spike_model_cleaned.pkl', 'rb'))
xgb_model = pickle.load(open('/app/forex_macro_sentiment_1329/models/xgboost_directional_model_cleaned.pkl', 'rb'))

config = json.load(open('/app/forex_macro_sentiment_1329/models/feature_engineering_config_cleaned.json', 'r'))

numeric_cols = config['feature_names_cleaned']
print(f"\nAnalyzing {len(numeric_cols)} clean features:")
for col in numeric_cols:
    print(f"  - {col}")

rf_importance = rf_model.feature_importances_
xgb_importance = xgb_model.feature_importances_

rf_features_sorted = sorted(zip(numeric_cols, rf_importance), key=lambda x: x[1], reverse=True)
xgb_features_sorted = sorted(zip(numeric_cols, xgb_importance), key=lambda x: x[1], reverse=True)

print(f"\n" + "=" * 80)
print("RANDOM FOREST FEATURE IMPORTANCE (Binary Spike Classification)")
print("=" * 80)
print(f"\nTop 15 features:")
for i, (feat, imp) in enumerate(rf_features_sorted[:15], 1):
    pct = imp * 100
    print(f"{i:2d}. {feat:30s} {imp:.6f} ({pct:5.2f}%)")

print(f"\n" + "=" * 80)
print("XGBOOST FEATURE IMPORTANCE (Directional Classification)")
print("=" * 80)
print(f"\nTop 15 features:")
for i, (feat, imp) in enumerate(xgb_features_sorted[:15], 1):
    pct = imp * 100
    print(f"{i:2d}. {feat:30s} {imp:.6f} ({pct:5.2f}%)")

df_rf_imp = pd.DataFrame({'feature': [f[0] for f in rf_features_sorted], 
                          'importance': [f[1] for f in rf_features_sorted],
                          'model': 'Random Forest'})
df_xgb_imp = pd.DataFrame({'feature': [f[0] for f in xgb_features_sorted],
                           'importance': [f[1] for f in xgb_features_sorted],
                           'model': 'XGBoost'})

df_combined = pd.concat([df_rf_imp, df_xgb_imp], ignore_index=True)
df_combined.to_csv('/app/forex_macro_sentiment_1329/analysis/feature_importance_cleaned.csv', index=False)
print(f"\n✓ Feature importance saved to feature_importance_cleaned.csv")

fig, axes = plt.subplots(1, 2, figsize=(14, 8))

top_n = 12
rf_top = rf_features_sorted[:top_n]
xgb_top = xgb_features_sorted[:top_n]

axes[0].barh([f[0] for f in reversed(rf_top)], [f[1] for f in reversed(rf_top)], color='steelblue')
axes[0].set_xlabel('Importance')
axes[0].set_title('Random Forest - Binary Spike\n(Top 12 Features)')
axes[0].grid(axis='x', alpha=0.3)

axes[1].barh([f[0] for f in reversed(xgb_top)], [f[1] for f in reversed(xgb_top)], color='coral')
axes[1].set_xlabel('Importance')
axes[1].set_title('XGBoost - Directional\n(Top 12 Features)')
axes[1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('/app/forex_macro_sentiment_1329/analysis/feature_importance_comparison_cleaned.png', dpi=150, bbox_inches='tight')
print(f"✓ Feature importance visualization saved")

print(f"\n" + "=" * 80)
print("KEY INSIGHTS FROM CLEANED MODELS")
print("=" * 80)

sentiment_features = [f for f in numeric_cols if 'sentiment' in f.lower() or 'surprise' in f.lower()]
time_features = [f for f in numeric_cols if 'hour' in f.lower() or 'day' in f.lower()]
vix_features = [f for f in numeric_cols if 'vix' in f.lower()]
economic_features = [f for f in numeric_cols if any(x in f.lower() for x in ['actual', 'consensus', 'previous'])]

print(f"\n1. SENTIMENT & SURPRISE FEATURES:")
for feat in sentiment_features:
    rf_imp = next((imp for f, imp in rf_features_sorted if f == feat), 0)
    xgb_imp = next((imp for f, imp in xgb_features_sorted if f == feat), 0)
    print(f"   {feat:30s} RF: {rf_imp:.4f} | XGB: {xgb_imp:.4f}")

print(f"\n2. VIX & VOLATILITY FEATURES:")
for feat in vix_features:
    rf_imp = next((imp for f, imp in rf_features_sorted if f == feat), 0)
    xgb_imp = next((imp for f, imp in xgb_features_sorted if f == feat), 0)
    print(f"   {feat:30s} RF: {rf_imp:.4f} | XGB: {xgb_imp:.4f}")

print(f"\n3. TEMPORAL FEATURES:")
for feat in time_features:
    rf_imp = next((imp for f, imp in rf_features_sorted if f == feat), 0)
    xgb_imp = next((imp for f, imp in xgb_features_sorted if f == feat), 0)
    print(f"   {feat:30s} RF: {rf_imp:.4f} | XGB: {xgb_imp:.4f}")

print(f"\n4. ECONOMIC FEATURES:")
for feat in economic_features:
    rf_imp = next((imp for f, imp in rf_features_sorted if f == feat), 0)
    xgb_imp = next((imp for f, imp in xgb_features_sorted if f == feat), 0)
    print(f"   {feat:30s} RF: {rf_imp:.4f} | XGB: {xgb_imp:.4f}")

print(f"\n" + "=" * 80)
print("FEATURE CATEGORY IMPORTANCE SUMMARY")
print("=" * 80)

def category_total(feats, sorted_list):
    return sum(imp for f, imp in sorted_list if f in feats)

print(f"\nRandom Forest:")
print(f"  Sentiment/Surprise:  {category_total(sentiment_features, rf_features_sorted):.4f}")
print(f"  VIX/Volatility:      {category_total(vix_features, rf_features_sorted):.4f}")
print(f"  Temporal:            {category_total(time_features, rf_features_sorted):.4f}")
print(f"  Economic Data:       {category_total(economic_features, rf_features_sorted):.4f}")

print(f"\nXGBoost:")
print(f"  Sentiment/Surprise:  {category_total(sentiment_features, xgb_features_sorted):.4f}")
print(f"  VIX/Volatility:      {category_total(vix_features, xgb_features_sorted):.4f}")
print(f"  Temporal:            {category_total(time_features, xgb_features_sorted):.4f}")
print(f"  Economic Data:       {category_total(economic_features, xgb_features_sorted):.4f}")

print(f"\n" + "=" * 80)
print("✓ Feature importance analysis complete")
print("=" * 80)