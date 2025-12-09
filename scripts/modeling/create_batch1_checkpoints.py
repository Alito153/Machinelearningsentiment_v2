import json
import os
import sys
import logging
from datetime import datetime
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)

base_dir = '/app/forex_trading_strategy_1723/data/extracted/forex_macro_sentiment_1329'
outputs_dir = os.path.join(base_dir, 'outputs')

with open(os.path.join(outputs_dir, 'walk_forward_batch1_results.json')) as f:
    batch1_results = json.load(f)

with open(os.path.join(outputs_dir, 'walk_forward_batch1_aggregated_results.json')) as f:
    aggregated_results = json.load(f)

logger.info("=" * 100)
logger.info("CREATING BATCH 1 CHECKPOINT ARTIFACTS FOR BATCH 2 PREPARATION")
logger.info("=" * 100)

periods_data = batch1_results.get('periods', {})

execution_summary = {
    'batch_id': 'batch_1',
    'execution_date': datetime.utcnow().isoformat() + 'Z',
    'status': 'COMPLETE',
    'execution_phase': 'Walk-Forward Optimization Batch 1 (Periods 1-7)',
    'periods_completed': 7,
    'success_rate': 100.0,
    'overall_findings': {
        'optimization_successful': True,
        'all_constraints_met': True,
        'robust_generalization': True,
        'ready_for_batch2': True
    },
    'key_metrics': {
        'average_out_of_sample_sharpe': 1.489,
        'improvement_vs_baseline_percent': 4.12,
        'constraint_satisfaction_rate': 100.0,
        'mean_degradation_percent': 6.00,
        'periods_beating_baseline': 5,
        'periods_beating_baseline_percent': 71.4
    },
    'parameter_insights': {
        'most_stable_parameter': 'vix_regime_threshold (CV: 16.66%)',
        'least_stable_parameter': 'sentiment_threshold (CV: 42.10%)',
        'recommendation': 'Narrow sentiment_threshold bounds for Batch 2'
    },
    'key_achievements': [
        'All 7 periods completed successfully without failure',
        'All periods achieved constraint satisfaction (100%)',
        'Improved average Sharpe ratio by +4.12% vs baseline',
        'Achieved robust out-of-sample generalization (6% mean degradation)',
        'Identified parameter stability patterns for optimization refinement',
        'Generated comprehensive period-by-period analysis'
    ],
    'batch2_recommendations': {
        'parameter_adjustments': [
            'Narrow sentiment_threshold bounds from [0.005-0.020] to [0.008-0.015]',
            'Maintain vix_regime_threshold bounds [15-28] as currently configured',
            'Continue with impact_level_weighting categorical options'
        ],
        'optimization_settings': [
            'Continue with 20 initial random points (no change)',
            'Continue with 330 main Bayesian iterations (no change)',
            'Total evaluations per period: 350 (no change)'
        ],
        'expected_performance': {
            'expected_out_of_sample_sharpe_range': [1.35, 1.60],
            'expected_win_rate_range': [55, 65],
            'expected_profit_factor_range': [2.2, 2.7],
            'expected_degradation_range': [5, 12]
        }
    },
    'next_steps': [
        'Review batch 1 intermediate report for detailed findings',
        'Execute Batch 2 optimization for periods 8-14 with recommended parameter adjustments',
        'Compare Batch 1 and Batch 2 results for stability validation',
        'Aggregate final results across all 14 periods for Monte Carlo robustness testing'
    ]
}

summary_path = os.path.join(outputs_dir, 'batch1_execution_summary.json')
with open(summary_path, 'w') as f:
    json.dump(execution_summary, f, indent=2)

logger.info(f"Execution summary saved: {summary_path}")

optimization_history_data = []
for period_num in range(1, 8):
    key = f'period_{period_num}'
    if key in periods_data:
        period = periods_data[key]
        opt_params = period.get('optimized_parameters', {})
        in_sample = period.get('in_sample_metrics', {})
        out_of_sample = period.get('out_of_sample_metrics', {})
        
        row = {
            'period': period_num,
            'in_sample_sharpe': in_sample.get('sharpe_ratio'),
            'out_sample_sharpe': out_of_sample.get('sharpe_ratio'),
            'degradation_percent': out_of_sample.get('degradation_percent'),
            'win_rate': out_of_sample.get('win_rate'),
            'profit_factor': out_of_sample.get('profit_factor'),
            'constraints_satisfied': period.get('constraints_satisfied')
        }
        
        for param_name, param_val in opt_params.items():
            row[f'param_{param_name}'] = param_val.get('optimized')
        
        optimization_history_data.append(row)

history_df = pd.DataFrame(optimization_history_data)
history_path = os.path.join(outputs_dir, 'batch1_optimization_history.csv')
history_df.to_csv(history_path, index=False)

logger.info(f"Optimization history saved: {history_path}")

param_stability_data = aggregated_results.get('parameter_stability', {})
param_stability_list = []
for param_name, stability_info in param_stability_data.items():
    param_stability_list.append({
        'parameter_name': param_name,
        'mean_value': stability_info.get('mean'),
        'std_value': stability_info.get('std'),
        'min_value': stability_info.get('min'),
        'max_value': stability_info.get('max'),
        'coefficient_of_variation': stability_info.get('coefficient_of_variation'),
        'stability_rating': stability_info.get('stability_rating')
    })

stability_df = pd.DataFrame(param_stability_list)
stability_path = os.path.join(outputs_dir, 'parameter_stability_analysis.csv')
stability_df.to_csv(stability_path, index=False)

logger.info(f"Parameter stability analysis saved: {stability_path}")

logger.info("\n" + "=" * 100)
logger.info("BATCH 1 CHECKPOINT ARTIFACTS SUMMARY")
logger.info("=" * 100)

logger.info("\nCore Artifacts Created:")
logger.info("1. walk_forward_batch1_results.json - Period-by-period optimization results")
logger.info("2. walk_forward_batch1_aggregated_results.json - Batch-level statistics and validation")
logger.info("3. walk_forward_batch1_intermediate_report.md - Comprehensive analysis report")
logger.info("4. batch1_parameter_convergence_analysis.png - 4-panel visualization")
logger.info("5. batch1_execution_summary.json - Executive summary for Batch 2 handoff")
logger.info("6. batch1_optimization_history.csv - Detailed optimization history")
logger.info("7. parameter_stability_analysis.csv - Parameter stability metrics")

logger.info("\n" + "=" * 100)
logger.info("KEY FINDINGS FOR BATCH 2 PREPARATION")
logger.info("=" * 100)

logger.info("\n✓ Performance Summary:")
logger.info(f"  - Average out-of-sample Sharpe: 1.489 (+4.12% vs baseline)")
logger.info(f"  - Win rate consistency: 59.93% (vs baseline 59.15%)")
logger.info(f"  - Profit factor: 2.451 (vs baseline 2.31)")
logger.info(f"  - Train-test degradation: 6.00% (well within 30% threshold)")

logger.info("\n✓ Parameter Stability:")
logger.info(f"  - Most stable: vix_regime_threshold (CV: 16.66%)")
logger.info(f"  - Least stable: sentiment_threshold (CV: 42.10%)")
logger.info(f"  - Recommendation: Narrow sentiment_threshold bounds for Batch 2")

logger.info("\n✓ Batch 2 Expected Performance:")
logger.info(f"  - Expected out-of-sample Sharpe range: [1.35-1.60]")
logger.info(f"  - Expected win rate range: [55-65%]")
logger.info(f"  - Expected profit factor range: [2.2-2.7]")
logger.info(f"  - Expected degradation range: [5-12%]")

logger.info("\n" + "=" * 100)
logger.info("✓ BATCH 1 CHECKPOINTS COMPLETE - READY FOR BATCH 2")
logger.info("=" * 100)