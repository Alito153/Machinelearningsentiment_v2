import json
import csv
import os
from pathlib import Path
from datetime import datetime, timedelta
import logging
import sys

logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

base_dir = Path("/app/forex_trading_strategy_1723/data/extracted/forex_macro_sentiment_1329/outputs")

logger.info("=== LOADING BACKTEST AND STRESS TEST RESULTS ===\n")

backtest_file = base_dir / "comprehensive_backtest_results.json"
with open(backtest_file) as f:
    backtest_results = json.load(f)

logger.info("BASELINE BACKTEST METRICS:")
logger.info(f"  Total Return: {backtest_results.get('total_return', 'N/A')}")
logger.info(f"  Annualized Return: {backtest_results.get('annualized_return', 'N/A')}")
logger.info(f"  Sharpe Ratio: {backtest_results.get('sharpe_ratio', 'N/A')}")
logger.info(f"  Max Drawdown: {backtest_results.get('max_drawdown', 'N/A')}")
logger.info(f"  Win Rate: {backtest_results.get('win_rate', 'N/A')}")
logger.info(f"  Profit Factor: {backtest_results.get('profit_factor', 'N/A')}")
logger.info(f"  Recovery Factor: {backtest_results.get('recovery_factor', 'N/A')}")
logger.info(f"  Number of Trades: {backtest_results.get('number_of_trades', 'N/A')}\n")

stress_test_summary_file = base_dir / "stress_testing" / "stress_test_summary.json"
with open(stress_test_summary_file) as f:
    stress_summary = json.load(f)

logger.info("STRESS TEST SCENARIOS:")
logger.info(f"  Total Scenarios: {len(stress_summary.get('scenarios', []))}")
for scenario in stress_summary.get('scenarios', [])[:5]:
    logger.info(f"  - {scenario.get('name')}: Sharpe={scenario.get('sharpe_ratio')}, Win Rate={scenario.get('win_rate')}")

stress_test_detailed_file = base_dir / "stress_testing" / "stress_test_metrics_detailed.csv"
stress_metrics = []
with open(stress_test_detailed_file) as f:
    reader = csv.DictReader(f)
    for row in reader:
        stress_metrics.append(row)

logger.info(f"\nSTRESS TEST DETAILED METRICS: {len(stress_metrics)} scenarios\n")

sentiment_report = Path("/app/forex_trading_strategy_1723/data/extracted/forex_macro_sentiment_1329/SENTIMENT_DATA_ANALYSIS_REPORT.md")
with open(sentiment_report) as f:
    report_content = f.read()
    
for line in report_content.split('\n'):
    if 'Date Range' in line or 'Total Records' in line or 'Event Types' in line:
        logger.info(f"  {line.strip()}")

logger.info("\n=== ANALYSIS COMPLETE ===")
logger.info(f"Backtest period identified from data")
logger.info(f"Stress tests cover {len(stress_metrics)} scenarios")
logger.info(f"Ready to design walk-forward calendar and parameter optimization")