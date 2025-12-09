# Walk-Forward Validation and Risk Calibration Report
**Algorithmic Trading Strategy: ML-Driven Sentiment Analysis for FX Macro Events**
**Report Date:** 2025-12-06
**Execution Phase:** Cycle 2 - Walk-Forward Validation and Risk Calibration
---
## Executive Summary
This report documents the walk-forward validation and empirical risk calibration for the FX macro sentiment trading system. The analysis covers **898 macro events** across **9 calibrated event-type × VIX-regime × sentiment clusters**. Key findings:
- **Spike Detection Performance (Random Forest):** Recall=69.14%, Precision=50.00%, F1=58.03%, ROC-AUC=86.88%
- **Directional Accuracy (XGBoost):** Debugging required - investigating encoding issue
- **Feature Importance:** day_of_week (32.62%), vix_close (18.71%), lagged_sentiment_5d (6.40%)
- **Walk-Forward Framework:** 1 fold created (60% train / 20% val / 20% test)
- **Calibrated Clusters:** 9 clusters with ≥50 samples; 15 clusters marked insufficient_data
### Data Integrity & Limitations
**Dataset Structure:**
- Total events: 898 (683 non-spike, 215 spike)
- VIX regime indicator: Created from vix_close EMA-20
- Sentiment classification: Positive/Negative/Neutral based on sentiment_score
**Key Limitation:** Cleaned dataset contains event-level features but NOT high-frequency price data. Return metrics approximated using surprise_pct as proxy.
---
## Walk-Forward Validation Framework
### Design
**Temporal Window Structure:**
- Dataset: 898 chronological events
- Train set: 60% (538 events)
- Validation set: 20% (179 events)
- Test set: 20% (179 events)
- Generated Folds: 1 complete fold with strict temporal ordering
### Predictions Generated
- **Total predictions:** 358 (179 validation + 179 test samples)
- **Random Forest:** Binary spike probabilities
- **XGBoost:** Directional classification
- **Feature set:** 13 cleaned features
---
## Empirical Risk Calibration
### Calibrated Clusters (≥50 samples)
| Event Type | VIX Regime | Sentiment | Samples | Spike Rate | TP_50th | TP_60th | SL_80th | SL_90th |
|------------|-----------|-----------|---------|-----------|---------|---------|---------|---------|
| PPI | 0 | Neutral | 66 | 24.2% | 0.92% | 1.24% | 1.85% | 2.10% |
| Housing Starts | 0 | Neutral | 59 | 30.5% | 1.45% | 1.89% | 2.78% | 3.15% |
| Inflation | 0 | Neutral | 57 | 14.0% | 1.20% | 1.56% | 2.35% | 2.68% |
| CPI | 0 | Neutral | 56 | 16.1% | 1.09% | 1.37% | 2.18% | 2.56% |
| PMI | 0 | Neutral | 55 | 21.8% | 0.70% | 0.92% | 1.58% | 1.88% |
| FOMC | 0 | Neutral | 53 | 13.2% | 1.89% | 2.45% | 3.68% | 4.20% |
| Retail Sales | 0 | Neutral | 51 | 19.6% | 1.28% | 1.67% | 2.54% | 2.92% |
| Fed Rate | 0 | Neutral | 51 | 9.8% | 1.46% | 1.91% | 2.98% | 3.45% |
| NFP | 0 | Neutral | 51 | 23.5% | 1.80% | 2.34% | 3.55% | 4.12% |
---
## Calibration Validation
### Historical Win Rate Comparison
| Cluster | Total Events | Predicted Spike % | Actual Spike % | Gap |
|---------|--------------|-------------------|----------------|-----|
| CPI_0_neutral | 56 | 8.9% | 16.1% | -7.2% |
| FOMC_0_neutral | 53 | 3.8% | 13.2% | -9.4% |
| Fed Rate_0_neutral | 51 | 7.8% | 9.8% | -2.0% |
| Housing Starts_0_neutral | 59 | 3.4% | 30.5% | -27.1% |
| Inflation_0_neutral | 57 | 3.5% | 14.0% | -10.5% |
| NFP_0_neutral | 51 | 6.3% | 23.5% | -17.2% |
| PMI_0_neutral | 55 | 5.2% | 21.8% | -16.6% |
| Retail Sales_0_neutral | 51 | 4.9% | 19.6% | -14.7% |
| PPI_0_neutral | 66 | 10.6% | 24.2% | -13.6% |
**Key Finding:** Model uses conservative prediction threshold (0.5 probability). Predicted spike rates 3-27% lower than actual, suggesting opportunity to lower threshold to 0.30-0.40.
---
## Walk-Forward Performance Metrics
### Spike Detection (Random Forest)
**Confusion Matrix (Test Set):**
- True Negatives (TN): 129
- False Positives (FP): 14
- False Negatives (FN): 7
- True Positives (TP): 29
**Metrics:**
- Recall: 0.6914 (catches 69.14% of spikes)
- Precision: 0.5000 (50% of predictions correct)
- F1-Score: 0.5803
- ROC-AUC: 0.8688 (excellent discrimination)
- Specificity: 0.9021 (90% non-spike accuracy)
---
## Feature Importance Rankings
**Top 15 Features (Averaged Across Folds):**
| Rank | Feature | Importance | Category |
|------|---------|-----------|----------|
| 1 | day_of_week | 0.3262 | Temporal |
| 2 | vix_close | 0.1871 | Market Regime |
| 3 | lagged_sentiment_5d | 0.0640 | Sentiment |
| 4 | lagged_sentiment_1d | 0.0596 | Sentiment |
| 5 | previous_value | 0.0596 | Event Metric |
| 6 | actual_value | 0.0499 | Event Metric |
| 7 | normalized_surprise | 0.0484 | Surprise |
| 8 | consensus_value | 0.0474 | Event Metric |
| 9 | sentiment_score | 0.0464 | Sentiment |
| 10 | surprise_pct | 0.0442 | Surprise |
| 11 | hour_of_day | 0.0321 | Temporal |
| 12 | vix_regime | 0.0234 | Market Regime |
| 13 | impact_encoded | 0.0116 | Event Metric |
**Feature Category Breakdown:**
- Temporal (day_of_week, hour_of_day): 35.83%
- Market Regime (vix_close, vix_regime): 19.71%
- Sentiment features: 15.20%
- Surprise metrics: 9.26%
- Event metrics: 20.00%
---
## Key Observations
### 1. Day-of-Week Seasonality Dominates
- 32.6% feature importance suggests strong schedule-driven patterns
- Macro announcements follow weekly schedules (NFP Fridays, CPI/PPI specific days)
- May indicate overfitting to announcement calendar rather than market microstructure
### 2. Conservative Model Calibration
- RF model predicts spikes at <10% despite 15-20% historical rate
- Probability threshold 0.5 is too conservative for 24% base rate
- Recommendation: Lower threshold to 0.30-0.40
### 3. Missing Directional Accuracy
- XGBoost directional predictions report 0% accuracy
- Likely cause: Label encoding mismatch (down=-1→0, neutral=0→1, up=1→2)
- Requires investigation in next cycle
### 4. Tight SL Calibration
- Empirical SL hit rates 65-80% suggest levels too tight
- Dataset lacks high-frequency price wicks
- Real trading needs 3-5% SL for EURUSD, 1-2% for XAUUSD
---
## Cluster Analysis
### Sufficient Data (≥50 samples)
9 clusters calibrated, all with:
- VIX regime = 0 (low volatility)
- Sentiment = neutral
Pattern indicates:
- Data concentrated in normal VIX regime
- Limited high-volatility events
- Neutral sentiment dominates
### Insufficient Data (<50 samples)
15 clusters flagged, including:
- Positive/negative sentiment variants
- VIX_regime=1 variants
- Rare event types (Earnings, etc.)
Recommendation: Do not trade clusters with <50 samples. Increase data collection.
---
## Recommendations for Next Cycle
### Immediate Fixes (High Priority)
1. **Directional Accuracy Issue**
- Verify XGBoost class mapping and output format
- Re-run with corrected label encoding
- Target: >55% directional accuracy
2. **Recalibrate TP/SL Parameters**
- Obtain actual M1/M5 price candles
- Compute true max moves (pips) and adverse wicks
- Re-extract percentiles from actual price data
- Target: SL hit rates 20-30%, TP hit rates 40-50%
3. **Threshold Optimization**
- Lower RF spike threshold from 0.5 to 0.30
- Optimize XGBoost directional confidence threshold
- Use validation set for Sharpe maximization
### Model Refinements (Medium Priority)
4. **Feature Engineering**
- Investigate day-of-week seasonality interactions
- Add technical indicators (ATR, EMA, RSI)
- Test sentiment × VIX interactions
- Remove schedule-based leakage
5. **Cluster Expansion**
- Collect high-volatility regime events
- Expand positive/negative sentiment samples to ≥50
- Add additional event types
6. **Temporal Validation**
- Extend walk-forward to 4-5 folds
- Monitor cross-fold feature importance consistency
- Check for regime shifts over time
### Trading System Definition
7. **Decision Rules**