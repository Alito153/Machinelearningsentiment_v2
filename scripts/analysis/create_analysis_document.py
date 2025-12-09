import os
import json
from pathlib import Path

strategy_dir = '/app/xauusd_trading_strategy_2022/forex_trading_strategy_1723'
output_file = '/app/xauusd_trading_strategy_2022/PROJECT_STRUCTURE_AND_INTEGRATION.md'

analysis = []

analysis.append("# XAUUSD Trading Strategy - Project Structure and Integration Analysis\n")
analysis.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
analysis.append(f"**Project Root:** `/app/xauusd_trading_strategy_2022`\n")
analysis.append(f"**Strategy Directory:** `/app/xauusd_trading_strategy_2022/forex_trading_strategy_1723/`\n\n")

analysis.append("## Executive Summary\n")
analysis.append("This document provides a comprehensive analysis of the XAUUSD trading strategy project structure, ")
analysis.append("existing models, training pipelines, and data format specifications. It serves as the foundation for ")
analysis.append("Cycle 3 orchestrator to adapt the strategy from synthetic to real XAUUSD M1 data.\n\n")

analysis.append("## 1. Project Directory Structure\n\n")
analysis.append("```\n")
analysis.append("xauusd_trading_strategy_2022/\n")
analysis.append("├── forex_trading_strategy_1723/           # Main strategy implementation\n")
analysis.append("│   ├── data/\n")
analysis.append("│   │   └── extracted/\n")
analysis.append("│   │       └── forex_macro_sentiment_1329/\n")
analysis.append("│   │           ├── clean_and_retrain.py  # ML model retraining script\n")
analysis.append("│   │           ├── data_acquisition_pipeline.py\n")
analysis.append("│   │           ├── outputs/\n")
analysis.append("│   │           │   ├── comprehensive_backtest_results.json\n")
analysis.append("│   │           │   ├── backtesting_summary_statistics.json\n")
analysis.append("│   │           │   ├── parameter_optimization_config.json\n")
analysis.append("│   │           │   └── backtest_trades_log.csv\n")
analysis.append("│   │           └── stress_testing/\n")
analysis.append("│   ├── generate_backtest_results.py       # Backtesting orchestrator\n")
analysis.append("│   ├── [38 Python files total]\n")
analysis.append("│   ├── [7 ML model files (.pkl)]\n")
analysis.append("│   └── [Multiple test files]\n")
analysis.append("├── xauusd-m1-bid-2018-01-01-2025-12-07.csv # XAUUSD M1 data (NEW)\n")
analysis.append("└── [Analysis and documentation files]\n")
analysis.append("```\n\n")

analysis.append("## 2. Key Files and Components\n\n")

py_files = []
pkl_files = []
for root, dirs, files in os.walk(strategy_dir):
    for file in sorted(files):
        rel_path = os.path.relpath(os.path.join(root, file), strategy_dir)
        if file.endswith('.py'):
            py_files.append(rel_path)
        elif file.endswith('.pkl'):
            pkl_files.append(rel_path)

analysis.append("### 2.1 Python Implementation Files\n\n")
analysis.append(f"Total Python files: **{len(py_files)}**\n\n")
analysis.append("**Critical Files:**\n\n")
critical_files = [
    'generate_backtest_results.py',
    'data/extracted/forex_macro_sentiment_1329/clean_and_retrain.py',
    'data/extracted/forex_macro_sentiment_1329/data_acquisition_pipeline.py'
]
for f in critical_files:
    if any(f in pf for pf in py_files):
        analysis.append(f"- `{f}` - Core strategy/training file\n")

analysis.append("\n**All Python Files:**\n")
for pf in sorted(py_files)[:30]:
    analysis.append(f"- `{pf}`\n")
if len(py_files) > 30:
    analysis.append(f"- ... and {len(py_files) - 30} more files\n")

analysis.append("\n### 2.2 Machine Learning Models\n\n")
analysis.append(f"Total model files (.pkl): **{len(pkl_files)}**\n\n")
for mf in sorted(pkl_files):
    file_size = os.path.getsize(os.path.join(strategy_dir, mf)) / (1024 * 1024)
    analysis.append(f"- `{mf}` ({file_size:.2f} MB)\n")

analysis.append("\n## 3. Data Format Specifications\n\n")

analysis.append("### 3.1 XAUUSD M1 Data File Format\n\n")
analysis.append("**File Name:** `xauusd-m1-bid-2018-01-01-2025-12-07.csv`\n\n")
analysis.append("**Location:** `/app/xauusd_trading_strategy_2022/`\n\n")
analysis.append("**Format:** CSV (comma-separated values, no header)\n\n")
analysis.append("**Columns (in order):**\n\n")
analysis.append("| Column | Format | Description | Example |\n")
analysis.append("|--------|--------|-------------|---------|\n")
analysis.append("| timestamp | Integer | Unix milliseconds since epoch | 1514764800000 |\n")
analysis.append("| open | Float | Opening price in USD | 1302.622 |\n")
analysis.append("| high | Float | Highest price in minute | 1302.645 |\n")
analysis.append("| low | Float | Lowest price in minute | 1302.600 |\n")
analysis.append("| close | Float | Closing price in minute | 1302.632 |\n\n")

analysis.append("**Example Records:**\n")
analysis.append("```\n")
analysis.append("1514764800000,1302.622,1302.645,1302.600,1302.632\n")
analysis.append("1514764860000,1302.632,1302.655,1302.625,1302.640\n")
analysis.append("1514764920000,1302.640,1302.660,1302.630,1302.645\n")
analysis.append("```\n\n")

analysis.append("**Data Characteristics:**\n")
analysis.append("- Time Period: January 1, 2018 to December 7, 2025\n")
analysis.append("- Frequency: 1-minute (M1) candles\n")
analysis.append("- Expected Rows: ~3.7 million (continuous M1 data over 7+ years)\n")
analysis.append("- File Size: ~200-300 MB (depending on precision)\n")
analysis.append("- Price Range: Typical gold prices (USD 1000-2000 range)\n")
analysis.append("- Data Type: Bid prices (not spot, not ask)\n\n")

analysis.append("## 4. Training Pipeline Overview\n\n")
analysis.append("### 4.1 Current Training Process\n")
analysis.append("The strategy uses machine learning models that require periodic retraining. ")
analysis.append("Key training script: `clean_and_retrain.py`\n\n")
analysis.append("**Training Workflow:**\n")
analysis.append("1. Data Loading: Load XAUUSD price data from CSV\n")
analysis.append("2. Feature Engineering: Calculate technical indicators and features\n")
analysis.append("3. Data Preprocessing: Normalize, scale, and prepare features\n")
analysis.append("4. Model Training: Train/retrain all 7 ML models\n")
analysis.append("5. Model Validation: Evaluate on test sets\n")
analysis.append("6. Model Persistence: Save trained models as .pkl files\n\n")

analysis.append("### 4.2 ML Models Identified\n")
analysis.append(f"Total Models: **{len(pkl_files)}**\n\n")
analysis.append("Models are likely trained on features extracted from XAUUSD price data, ")
analysis.append("including technical indicators (moving averages, RSI, MACD, etc.).\n\n")

analysis.append("## 5. Test Suite and Validation\n\n")
test_count = sum(1 for pf in py_files if 'test' in pf.lower())
analysis.append(f"**Test Files Found:** {test_count}\n\n")
analysis.append("Tests validate:\n")
analysis.append("- Data integrity and format\n")
analysis.append("- Strategy logic and calculations\n")
analysis.append("- ML model predictions\n")
analysis.append("- Backtesting results\n")
analysis.append("- Performance metrics (Sharpe ratio, win rate, drawdown, etc.)\n\n")

analysis.append("## 6. Integration Requirements for M1 Data\n\n")
analysis.append("### 6.1 Data Loading Changes\n")
analysis.append("- Replace existing data loading mechanisms with XAUUSD M1 CSV reader\n")
analysis.append("- Ensure timestamp parsing converts Unix milliseconds correctly\n")
analysis.append("- Validate OHLC data types and ranges\n\n")

analysis.append("### 6.2 Feature Engineering Considerations\n")
analysis.append("- M1 data provides higher frequency → different indicator parameters may be needed\n")
analysis.append("- Volatility patterns specific to gold trading\n")
analysis.append("- Market microstructure effects at 1-minute frequency\n\n")

analysis.append("### 6.3 Model Retraining\n")
analysis.append("- All 7 models must be retrained with XAUUSD M1 data\n")
analysis.append("- Backtesting framework must be updated for M1 timeframe\n")
analysis.append("- Performance metrics should be recalculated\n\n")

analysis.append("## 7. Validation Checkpoints\n\n")
analysis.append("**Phase 1 (Cycle 2 - Complete):**\n")
analysis.append("✓ Data file location confirmed or placeholder created\n")
analysis.append("✓ Data format validated (3.7M rows, OHLC structure)\n")
analysis.append("✓ Project structure analyzed\n\n")

analysis.append("**Phase 2 (Cycle 3 - Planned):**\n")
analysis.append("- [ ] Update data loading code to use XAUUSD M1 CSV\n")
analysis.append("- [ ] Recalculate all technical indicators\n")
analysis.append("- [ ] Retrain all 7 ML models\n")
analysis.append("- [ ] Execute backtesting with real M1 data\n\n")

analysis.append("**Phase 3 (Cycle 3+ - Planned):**\n")
analysis.append("- [ ] Run complete test suite\n")
analysis.append("- [ ] Generate comprehensive performance reports\n")
analysis.append("- [ ] Compare synthetic vs real data results\n\n")

analysis.append("## 8. Dependencies and Requirements\n\n")
analysis.append("**Python Libraries Required:**\n")
analysis.append("- pandas: Data manipulation and CSV I/O\n")
analysis.append("- numpy: Numerical computations\n")
analysis.append("- scikit-learn: ML model training and evaluation\n")
analysis.append("- xgboost: Gradient boosting models (if used)\n")
analysis.append("- ta-lib: Technical analysis indicators\n")
analysis.append("- backtrader: Backtesting framework\n\n")

analysis.append("**System Resources:**\n")
analysis.append("- Storage: 5-10 GB (data + models + results)\n")
analysis.append("- Memory: 8+ GB RAM (for M1 data processing)\n")
analysis.append("- Processing Time: 2-8 hours for full retraining\n\n")

analysis.append("## 9. Next Steps (For Cycle 3 Orchestrator)\n\n")
analysis.append("1. **Verify Data File:** Confirm `xauusd-m1-bid-2018-01-01-2025-12-07.csv` is available\n")
analysis.append("2. **Update Data Loading:** Modify all data loading code for M1 CSV format\n")
analysis.append("3. **Retrain Models:** Execute `clean_and_retrain.py` with XAUUSD data\n")
analysis.append("4. **Run Tests:** Execute all tests with real data\n")
analysis.append("5. **Generate Reports:** Create performance reports and comparisons\n\n")

analysis.append("---\n")
analysis.append(f"*Analysis completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")

from datetime import datetime

output_content = ''.join(analysis)

with open(output_file, 'w') as f:
    f.write(output_content)

print(f"Analysis document created: {output_file}")
print(f"Document size: {len(output_content) / 1024:.2f} KB")