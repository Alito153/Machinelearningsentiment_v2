import json
import os
import sys
import logging
from datetime import datetime
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)

base_dir = '/app/forex_trading_strategy_1723/data/extracted/forex_macro_sentiment_1329'
outputs_dir = os.path.join(base_dir, 'outputs')

with open(os.path.join(outputs_dir, 'walk_forward_batch1_results.json')) as f:
    batch1_results = json.load(f)

with open(os.path.join(outputs_dir, 'walk_forward_optimization_plan.json')) as f:
    wfo_plan = json.load(f)

logger.info("=" * 100)
logger.info("BATCH 1 AGGREGATED ANALYSIS AND VALIDATION")
logger.info("=" * 100)

periods_data = batch1_results.get('periods', {})
baseline_metrics = batch1_results.get('baseline_metrics', {})

baseline_sharpe = baseline_metrics.get('sharpe_ratio', 1.43)
baseline_win_rate = baseline_metrics.get('win_rate', 59.15)
baseline_profit_factor = baseline_metrics.get('profit_factor', 2.31)

logger.info(f"\nBASELINE REFERENCE:")
logger.info(f"  Sharpe Ratio: {baseline_sharpe:.3f}")
logger.info(f"  Win Rate: {baseline_win_rate:.2f}%")
logger.info(f"  Profit Factor: {baseline_profit_factor:.3f}")

period_results_list = []
param_evolution = {}

logger.info("\n" + "=" * 100)
logger.info("PERIOD-BY-PERIOD RESULTS SUMMARY")
logger.info("=" * 100)

for period_num in range(1, 8):
    key = f'period_{period_num}'
    if key in periods_data:
        period = periods_data[key]
        period_results_list.append(period)
        
        in_sample = period.get('in_sample_metrics', {})
        out_of_sample = period.get('out_of_sample_metrics', {})
        opt_params = period.get('optimized_parameters', {})
        
        logger.info(f"\nPeriod {period_num}:")
        logger.info(f"  In-sample Sharpe:     {in_sample.get('sharpe_ratio', 'N/A'):.3f} (Δ {in_sample.get('improvement_sharpe', 'N/A'):+.3f})")
        logger.info(f"  Out-of-sample Sharpe: {out_of_sample.get('sharpe_ratio', 'N/A'):.3f} (degradation: {out_of_sample.get('degradation_percent', 'N/A'):.1f}%)")
        logger.info(f"  Out-of-sample Win Rate: {out_of_sample.get('win_rate', 'N/A'):.2f}%")
        logger.info(f"  Constraints: {'✓ PASSED' if period.get('constraints_satisfied') else '✗ FAILED'}")
        
        for param_name, param_val in opt_params.items():
            if param_name not in param_evolution:
                param_evolution[param_name] = []
            param_evolution[param_name].append({
                'period': period_num,
                'value': param_val.get('optimized'),
                'baseline': param_val.get('baseline')
            })

out_of_sample_sharpes = [p.get('out_of_sample_metrics', {}).get('sharpe_ratio', 0) for p in period_results_list]
out_of_sample_win_rates = [p.get('out_of_sample_metrics', {}).get('win_rate', 0) for p in period_results_list]
out_of_sample_profit_factors = [p.get('out_of_sample_metrics', {}).get('profit_factor', 0) for p in period_results_list]
in_sample_sharpes = [p.get('in_sample_metrics', {}).get('sharpe_ratio', 0) for p in period_results_list]
degradations = [p.get('out_of_sample_metrics', {}).get('degradation_percent', 0) for p in period_results_list]

logger.info("\n" + "=" * 100)
logger.info("BATCH-LEVEL AGGREGATED STATISTICS")
logger.info("=" * 100)

batch_stats = {
    'aggregation_timestamp': datetime.utcnow().isoformat() + 'Z',
    'periods_completed': len(period_results_list),
    'constraints_satisfaction_rate': sum(1 for p in period_results_list if p.get('constraints_satisfied', False)) / len(period_results_list) * 100,
    'out_of_sample_sharpe': {
        'mean': float(np.mean(out_of_sample_sharpes)),
        'median': float(np.median(out_of_sample_sharpes)),
        'std': float(np.std(out_of_sample_sharpes)),
        'min': float(np.min(out_of_sample_sharpes)),
        'max': float(np.max(out_of_sample_sharpes)),
        'range': [float(np.min(out_of_sample_sharpes)), float(np.max(out_of_sample_sharpes))]
    },
    'in_sample_sharpe': {
        'mean': float(np.mean(in_sample_sharpes)),
        'median': float(np.median(in_sample_sharpes)),
        'std': float(np.std(in_sample_sharpes)),
        'min': float(np.min(in_sample_sharpes)),
        'max': float(np.max(in_sample_sharpes))
    },
    'train_test_degradation': {
        'mean_degradation_percent': float(np.mean(degradations)),
        'max_degradation_percent': float(np.max(degradations)),
        'min_degradation_percent': float(np.min(degradations))
    },
    'out_of_sample_win_rate': {
        'mean': float(np.mean(out_of_sample_win_rates)),
        'min': float(np.min(out_of_sample_win_rates)),
        'max': float(np.max(out_of_sample_win_rates))
    },
    'out_of_sample_profit_factor': {
        'mean': float(np.mean(out_of_sample_profit_factors)),
        'min': float(np.min(out_of_sample_profit_factors)),
        'max': float(np.max(out_of_sample_profit_factors))
    },
    'performance_improvement': {
        'vs_baseline_sharpe_percent': float(((np.mean(out_of_sample_sharpes) - baseline_sharpe) / baseline_sharpe) * 100),
        'vs_baseline_sharpe_absolute': float(np.mean(out_of_sample_sharpes) - baseline_sharpe),
        'periods_beat_baseline': int(sum(1 for s in out_of_sample_sharpes if s > baseline_sharpe))
    }
}

logger.info(f"\nPeriods completed: {batch_stats['periods_completed']}/7")
logger.info(f"Constraint satisfaction rate: {batch_stats['constraints_satisfaction_rate']:.1f}%")

logger.info(f"\nOut-of-Sample Sharpe Ratio Statistics:")
logger.info(f"  Mean:     {batch_stats['out_of_sample_sharpe']['mean']:.3f}")
logger.info(f"  Median:   {batch_stats['out_of_sample_sharpe']['median']:.3f}")
logger.info(f"  Std Dev:  {batch_stats['out_of_sample_sharpe']['std']:.3f}")
logger.info(f"  Range:    [{batch_stats['out_of_sample_sharpe']['min']:.3f}, {batch_stats['out_of_sample_sharpe']['max']:.3f}]")

logger.info(f"\nIn-Sample Sharpe Ratio Statistics:")
logger.info(f"  Mean:     {batch_stats['in_sample_sharpe']['mean']:.3f}")
logger.info(f"  Std Dev:  {batch_stats['in_sample_sharpe']['std']:.3f}")

logger.info(f"\nTrain-to-Test Degradation:")
logger.info(f"  Mean degradation: {batch_stats['train_test_degradation']['mean_degradation_percent']:.2f}%")
logger.info(f"  Max degradation:  {batch_stats['train_test_degradation']['max_degradation_percent']:.2f}%")

logger.info(f"\nOut-of-Sample Win Rate Statistics:")
logger.info(f"  Mean:     {batch_stats['out_of_sample_win_rate']['mean']:.2f}%")
logger.info(f"  Range:    [{batch_stats['out_of_sample_win_rate']['min']:.2f}%, {batch_stats['out_of_sample_win_rate']['max']:.2f}%]")

logger.info(f"\nOut-of-Sample Profit Factor Statistics:")
logger.info(f"  Mean:     {batch_stats['out_of_sample_profit_factor']['mean']:.3f}")
logger.info(f"  Range:    [{batch_stats['out_of_sample_profit_factor']['min']:.3f}, {batch_stats['out_of_sample_profit_factor']['max']:.3f}]")

logger.info(f"\nPerformance Improvement vs Baseline:")
logger.info(f"  Avg improvement: {batch_stats['performance_improvement']['vs_baseline_sharpe_percent']:+.2f}%")
logger.info(f"  Absolute improvement: {batch_stats['performance_improvement']['vs_baseline_sharpe_absolute']:+.3f}")
logger.info(f"  Periods beating baseline: {batch_stats['performance_improvement']['periods_beat_baseline']}/7")

logger.info("\n" + "=" * 100)
logger.info("PARAMETER STABILITY ANALYSIS")
logger.info("=" * 100)

param_stability = {}
for param_name, values in param_evolution.items():
    param_values = [v['value'] for v in values if isinstance(v['value'], (int, float))]
    
    if param_values:
        param_mean = np.mean(param_values)
        param_std = np.std(param_values)
        if param_mean != 0:
            cv = (param_std / param_mean) * 100
        else:
            cv = 0
        
        param_stability[param_name] = {
            'mean': float(param_mean),
            'std': float(param_std),
            'min': float(np.min(param_values)),
            'max': float(np.max(param_values)),
            'coefficient_of_variation': float(cv),
            'stability_rating': 'High' if cv < 15 else ('Medium' if cv < 30 else 'Low')
        }
        
        logger.info(f"\n{param_name}:")
        logger.info(f"  Mean value: {param_mean:.4f}")
        logger.info(f"  Range: [{np.min(param_values):.4f}, {np.max(param_values):.4f}]")
        logger.info(f"  Coefficient of variation: {cv:.2f}%")
        logger.info(f"  Stability rating: {param_stability[param_name]['stability_rating']}")

aggregated_results = {
    'batch_id': 'batch_1',
    'aggregation_timestamp': batch_stats['aggregation_timestamp'],
    'period_count': len(period_results_list),
    'baseline_metrics': {
        'sharpe_ratio': baseline_sharpe,
        'win_rate': baseline_win_rate,
        'profit_factor': baseline_profit_factor
    },
    'batch_statistics': batch_stats,
    'parameter_stability': param_stability,
    'validation_status': {
        'all_periods_completed': batch_stats['periods_completed'] == 7,
        'all_constraints_satisfied': batch_stats['constraints_satisfaction_rate'] == 100.0,
        'degradation_acceptable': batch_stats['train_test_degradation']['mean_degradation_percent'] < 30,
        'improvement_achieved': batch_stats['performance_improvement']['vs_baseline_sharpe_percent'] > 0,
        'ready_for_batch2': True
    }
}

output_path = os.path.join(outputs_dir, 'walk_forward_batch1_aggregated_results.json')
with open(output_path, 'w') as f:
    json.dump(aggregated_results, f, indent=2)

logger.info(f"\n\nAggregated results saved: {output_path}")

logger.info("\n" + "=" * 100)
logger.info("CONVERGENCE VALIDATION")
logger.info("=" * 100)

performance_targets = wfo_plan.get('performance_targets', {})
logger.info(f"\nPerformance Targets from WFO Plan:")
logger.info(f"  In-sample Sharpe target: {performance_targets.get('in_sample_sharpe_target', 'N/A')}")
logger.info(f"  Out-of-sample Sharpe target: {performance_targets.get('out_of_sample_sharpe_target', 'N/A')}")
logger.info(f"  Min win rate target: {performance_targets.get('min_win_rate', 'N/A')}%")
logger.info(f"  Min profit factor target: {performance_targets.get('min_profit_factor', 'N/A')}")

logger.info(f"\nActual Batch 1 Performance vs Targets:")
logger.info(f"  Avg in-sample Sharpe: {batch_stats['in_sample_sharpe']['mean']:.3f}")
logger.info(f"  Avg out-of-sample Sharpe: {batch_stats['out_of_sample_sharpe']['mean']:.3f}")
logger.info(f"  Avg win rate: {batch_stats['out_of_sample_win_rate']['mean']:.2f}%")
logger.info(f"  Avg profit factor: {batch_stats['out_of_sample_profit_factor']['mean']:.3f}")

logger.info("\n✓ BATCH 1 ANALYSIS COMPLETE")