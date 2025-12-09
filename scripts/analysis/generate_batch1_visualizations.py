import json
import os
import sys
import logging
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)

base_dir = '/app/forex_trading_strategy_1723/data/extracted/forex_macro_sentiment_1329'
outputs_dir = os.path.join(base_dir, 'outputs')

with open(os.path.join(outputs_dir, 'walk_forward_batch1_results.json')) as f:
    batch1_results = json.load(f)

with open(os.path.join(outputs_dir, 'walk_forward_batch1_aggregated_results.json')) as f:
    aggregated_results = json.load(f)

logger.info("=" * 100)
logger.info("GENERATING BATCH 1 VISUALIZATIONS")
logger.info("=" * 100)

baseline_metrics = batch1_results.get('baseline_metrics', {})
baseline_sharpe = baseline_metrics.get('sharpe_ratio', 1.43)
baseline_win_rate = baseline_metrics.get('win_rate', 59.15)

periods_data = batch1_results.get('periods', {})

in_sample_sharpes = []
out_sample_sharpes = []
win_rates = []
profit_factors = []
degradations = []
period_nums = []

for period_num in range(1, 8):
    key = f'period_{period_num}'
    if key in periods_data:
        period = periods_data[key]
        in_sample_sharpes.append(period.get('in_sample_metrics', {}).get('sharpe_ratio', 0))
        out_sample_sharpes.append(period.get('out_of_sample_metrics', {}).get('sharpe_ratio', 0))
        win_rates.append(period.get('out_of_sample_metrics', {}).get('win_rate', 0))
        profit_factors.append(period.get('out_of_sample_metrics', {}).get('profit_factor', 0))
        degradations.append(period.get('out_of_sample_metrics', {}).get('degradation_percent', 0))
        period_nums.append(period_num)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Walk-Forward Optimization Batch 1 Analysis (Periods 1-7)', fontsize=16, fontweight='bold')

ax1.plot(period_nums, in_sample_sharpes, marker='o', linewidth=2, markersize=8, label='In-Sample (Train)', color='#2E86AB')
ax1.plot(period_nums, out_sample_sharpes, marker='s', linewidth=2, markersize=8, label='Out-of-Sample (Test)', color='#A23B72')
ax1.axhline(y=baseline_sharpe, color='#F18F01', linestyle='--', linewidth=2, label=f'Baseline ({baseline_sharpe:.3f})')
ax1.set_xlabel('Period', fontweight='bold')
ax1.set_ylabel('Sharpe Ratio', fontweight='bold')
ax1.set_title('Performance Convergence: Train vs Test Sharpe Ratios', fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(loc='best')
ax1.set_xticks(period_nums)

ax2.bar(period_nums, degradations, color='#06A77D', alpha=0.8, edgecolor='black', linewidth=1.5)
ax2.axhline(y=30, color='red', linestyle='--', linewidth=2, label='30% Threshold')
ax2.set_xlabel('Period', fontweight='bold')
ax2.set_ylabel('Degradation (%)', fontweight='bold')
ax2.set_title('Train-to-Test Degradation by Period', fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')
ax2.legend()
ax2.set_xticks(period_nums)

ax3.bar(period_nums, win_rates, color='#D62246', alpha=0.8, edgecolor='black', linewidth=1.5)
ax3.axhline(y=baseline_win_rate, color='#F18F01', linestyle='--', linewidth=2, label=f'Baseline ({baseline_win_rate:.2f}%)')
ax3.axhline(y=50, color='gray', linestyle=':', linewidth=1.5, label='50% Min Target')
ax3.set_xlabel('Period', fontweight='bold')
ax3.set_ylabel('Win Rate (%)', fontweight='bold')
ax3.set_title('Out-of-Sample Win Rate by Period', fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')
ax3.legend()
ax3.set_xticks(period_nums)
ax3.set_ylim([0, 100])

ax4.plot(period_nums, profit_factors, marker='D', linewidth=2, markersize=8, color='#4ECDC4', label='Profit Factor')
ax4.axhline(y=1.5, color='green', linestyle='--', linewidth=2, label='1.5 Target')
ax4.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5, label='1.0 Min')
ax4.set_xlabel('Period', fontweight='bold')
ax4.set_ylabel('Profit Factor', fontweight='bold')
ax4.set_title('Out-of-Sample Profit Factor by Period', fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')
ax4.legend()
ax4.set_xticks(period_nums)

plt.tight_layout()
viz_path = os.path.join(outputs_dir, 'batch1_parameter_convergence_analysis.png')
plt.savefig(viz_path, dpi=300, bbox_inches='tight')
logger.info(f"Visualization saved: {viz_path}")
plt.close()

logger.info("\n✓ BATCH 1 VISUALIZATIONS COMPLETE")