# OPTIMIZATION AND MONTE CARLO RECOMMENDATIONS

## Executive Summary
Based on comprehensive backtesting of the FX macro sentiment trading strategy, this document outlines recommended stress-test scenarios, parameter optimization ranges, and Monte Carlo robustness testing approach.

### Current Baseline Performance
- **Total Return**: 18.76% (11.22% annualized)
- **Sharpe Ratio**: 1.43
- **Max Drawdown**: 12.34%
- **Win Rate**: 59.15% (84/142 trades)
- **Profit Factor**: 2.31
- **Total Trades Executed**: 142
- **Consecutive Wins**: 5
- **Consecutive Losses**: 4

---

## STRESS-TEST SCENARIOS

### 1. VOLATILITY SHOCK SCENARIOS

Simulate market volatility multipliers to test strategy robustness to extreme volatility conditions.

**Multipliers**:
- 1.5x Historical Volatility: Mild volatility increase
- 2.0x Historical Volatility: Moderate volatility increase  
- 3.0x Historical Volatility: Severe volatility shock

**Implementation**:
- Apply multipliers to daily returns while preserving direction
- Maintain correlation structure with sentiment features
- Test both directional (XGBoost) and spike (Random Forest) models

**Expected Impact**:
- Increased drawdown and volatility
- Potential reduction in win rate due to whipsaws
- Sharpe ratio degradation

---

### 2. REGIME CHANGE SCENARIOS

Simulate different market regime conditions to test regime sensitivity.

**Regime Types**:

#### Trending Regime
- **Characteristics**: Strong directional bias, positive or negative drift
- **Implementation**: Add persistent trend component to prices
- **Expected**: Directional model should perform well

#### Mean-Reverting Regime
- **Characteristics**: Prices oscillate around moving average with high reversion
- **Implementation**: Apply Ornstein-Uhlenbeck process or similar
- **Expected**: Mean-reversion strategies outperform trend followers

#### Ranging Regime
- **Characteristics**: Prices oscillate within bounds without directional bias
- **Implementation**: Sine-wave-like oscillations within defined bands
- **Expected**: Strategy struggles, whipsaw losses increase

---

### 3. DATA QUALITY DEGRADATION SCENARIOS

Test model robustness to missing and corrupted data.

**Degradation Levels**:
- 5% Missing Data: Light data corruption
- 10% Missing Data: Moderate corruption
- 25% Missing Data: Significant data issues
- 50% Missing Data: Severe data quality problems

**Outlier Injection**:
- 1% Extreme Values: Mild outlier pollution
- 5% Extreme Values: Moderate outlier pollution
- 10% Extreme Values: Severe outlier pollution

**Handling Strategy**:
- Forward fill for temporal data
- Median imputation for missing features
- Outlier capping at 3-sigma bounds

---

### 4. CORRELATION BREAKDOWN SCENARIOS

Test signal quality degradation when feature-return relationships weaken.

**Breakdown Types**:
- 20% Decorrelation: Features 80% shuffled
- 50% Decorrelation: Features 50% shuffled
- 80% Decorrelation: Features 20% shuffled (mostly noise)

**Structural Breaks**:
- Simulate regime shifts where correlations change
- Test recovery time from structural breaks
- Measure prediction accuracy degradation

---

## PARAMETER OPTIMIZATION RANGES

### Primary Variables

#### Sentiment Threshold (Spike Detection)
- **Current Value**: 0.5 (from validation statistics)
- **Optimization Range**: 0.005 - 0.020
- **Impact**: Controls sensitivity of spike detection model
- **Trade-off**: Higher threshold = fewer false signals, lower sensitivity

#### VIX Regime Threshold
- **Current Range**: 15 - 28
- **Impact**: Volatility regime classification
- **Rationale**: Low VIX (< 15) = calm, Mid VIX (15-28) = normal, High VIX (> 28) = stressed

#### Impact Level Weighting
- **Current**: Equal weighting across impact levels
- **Options**:
  - Equal Weight: All events treated the same
  - High Impact Emphasis: 1.5x weight for high-impact events
  - Low Impact Filter: Exclude low-impact events

#### Holding Period (Trade Duration)
- **Current**: Variable based on event
- **Optimization Range**: 2 - 6 hours
- **Rationale**: Shorter = less exposure to adverse moves, Longer = capture larger moves

#### Event Filtering
- **Current**: All macro events included
- **Options**:
  - High Impact Only: Filter to high and medium events
  - Currency Relevant: Filter to currency-relevant events only
  - Consensus Surprise**: Only trade when actual vs consensus > threshold

---

## MONTE CARLO ROBUSTNESS TESTING

### Simulation Parameters
- **Iterations**: 1000 Monte Carlo simulations
- **Resampling Method**: Block bootstrap (50-observation blocks)
- **Confidence Level**: 95%

### Robustness Metrics

#### Performance Metrics
- **Sharpe Ratio**: Target ≥ 1.0 (99% of simulations)
- **Maximum Drawdown**: Target ≤ 20% (99% of simulations)
- **Win Rate**: Target ≥ 50% (95% of simulations)
- **Profit Factor**: Target ≥ 1.5 (95% of simulations)

#### Risk Metrics
- **Value at Risk (VaR 95%)**: Measure worst-case daily loss
- **Conditional VaR (CVaR)**: Average loss in worst 5% of days
- **Recovery Time**: Average time to recover from drawdown
- **Consecutive Loss Limit**: Max consecutive losses threshold

---

## FAILURE MODE DEFINITIONS

### Critical Failures (Sharpe < 0.5)
- Model performs worse than baseline
- Requires immediate recalibration

### High-Severity Failures (Sharpe 0.5-1.0)
- Acceptable but degraded performance
- Monitor closely in live trading

### Medium-Severity Failures (Sharpe 1.0-1.2)
- Minor performance degradation
- Within acceptable tolerance

### Low-Severity Failures (Sharpe > 1.2)
- Robust performance maintained
- Model resilient to stress scenario

---

## RECOMMENDATION SUMMARY

### Key Findings
1. **Strategy Baseline**: Strong performance with 1.43 Sharpe ratio
2. **Win Rate**: 59.15% indicates good directional bias
3. **Profit Factor**: 2.31 shows favorable risk-reward
4. **Volatility Sensitivity**: Moderate volatility tolerance expected

### Next Steps
1. Execute comprehensive stress tests on all 4 scenario types
2. Identify performance thresholds for parameter optimization
3. Run 1000-iteration Monte Carlo robustness tests
4. Document robust parameter ranges for walk-forward optimization
5. Validate results before deploying to production

---

## ACCEPTANCE CRITERIA

**Stress-Test Success Criteria**:
- ✓ Sharpe Ratio ≥ 0.8 under mild stress (1.5x volatility)
- ✓ Sharpe Ratio ≥ 0.6 under moderate stress (2.0x volatility)  
- ✓ Max Drawdown ≤ 25% under 3x volatility shock
- ✓ Win Rate ≥ 40% under severe stress scenarios
- ✓ Model executes without errors on all degraded data

**Robustness Threshold**:
- 95% of Monte Carlo simulations maintain Sharpe ≥ 1.0
- 99% of simulations maintain max drawdown ≤ 25%
- 95% maintain win rate ≥ 50%

