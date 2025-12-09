import os
import sys
import json
from pathlib import Path
from collections import defaultdict

base_dir = '/app/xauusd_trading_strategy_2022'
strategy_dir = os.path.join(base_dir, 'forex_trading_strategy_1723')

analysis = {
    'project_root': base_dir,
    'strategy_dir': strategy_dir,
    'directory_structure': {},
    'py_files': [],
    'pkl_files': [],
    'json_files': [],
    'csv_files': [],
    'data_loading_patterns': [],
    'model_files': [],
    'training_scripts': [],
}

def scan_directory(root_path, max_depth=3, current_depth=0):
    structure = {}
    if current_depth >= max_depth:
        return structure
    
    try:
        for item in sorted(os.listdir(root_path)):
            if item.startswith('.'):
                continue
            item_path = os.path.join(root_path, item)
            if os.path.isdir(item_path):
                structure[item] = {'type': 'dir', 'children': scan_directory(item_path, max_depth, current_depth + 1)}
            else:
                structure[item] = {'type': 'file', 'size': os.path.getsize(item_path)}
    except PermissionError:
        pass
    
    return structure

analysis['directory_structure'] = scan_directory(strategy_dir, max_depth=2)

for root, dirs, files in os.walk(strategy_dir):
    for file in files:
        filepath = os.path.join(root, file)
        rel_path = os.path.relpath(filepath, strategy_dir)
        
        if file.endswith('.py'):
            analysis['py_files'].append(rel_path)
            if 'train' in file.lower() or 'retrain' in file.lower():
                analysis['training_scripts'].append(rel_path)
        elif file.endswith('.pkl'):
            analysis['pkl_files'].append(rel_path)
            if 'model' in file.lower():
                analysis['model_files'].append(rel_path)
        elif file.endswith('.json'):
            analysis['json_files'].append(rel_path)
        elif file.endswith('.csv'):
            analysis['csv_files'].append(rel_path)

print("=" * 80)
print("PROJECT STRUCTURE ANALYSIS")
print("=" * 80)
print(f"\nProject Root: {base_dir}")
print(f"Strategy Directory: {strategy_dir}")

print(f"\n--- PYTHON FILES ({len(analysis['py_files'])}) ---")
for py_file in sorted(analysis['py_files'])[:20]:
    print(f"  {py_file}")
if len(analysis['py_files']) > 20:
    print(f"  ... and {len(analysis['py_files']) - 20} more")

print(f"\n--- TRAINING SCRIPTS ({len(analysis['training_scripts'])}) ---")
for script in analysis['training_scripts']:
    print(f"  {script}")

print(f"\n--- MODEL FILES ({len(analysis['pkl_files'])}) ---")
for pkl in sorted(analysis['pkl_files'])[:15]:
    print(f"  {pkl}")

print(f"\n--- CSV FILES ({len(analysis['csv_files'])}) ---")
for csv in sorted(analysis['csv_files'])[:10]:
    print(f"  {csv}")

print(f"\n--- JSON FILES ({len(analysis['json_files'])}) ---")
for json_file in sorted(analysis['json_files'])[:10]:
    print(f"  {json_file}")

data_acq_file = os.path.join(strategy_dir, 'data_acquisition_pipeline.py')
if os.path.exists(data_acq_file):
    print(f"\n--- DATA ACQUISITION PIPELINE FOUND ---")
    print(f"File: {data_acq_file}")
    with open(data_acq_file, 'r') as f:
        content = f.read()
        if 'csv' in content.lower() or 'OHLC' in content or 'timestamp' in content.lower():
            print("Contains CSV/OHLC/timestamp references")
        for line in content.split('\n')[:50]:
            if 'csv' in line.lower() or 'columns' in line.lower() or 'read' in line.lower():
                print(f"  {line.strip()}")

clean_retrain = os.path.join(strategy_dir, 'data', 'extracted', 'forex_macro_sentiment_1329', 'clean_and_retrain.py')
if os.path.exists(clean_retrain):
    print(f"\n--- CLEAN AND RETRAIN SCRIPT FOUND ---")
    with open(clean_retrain, 'r') as f:
        content = f.read()
        for line in content.split('\n')[:100]:
            if 'load' in line.lower() or 'read_csv' in line.lower() or 'columns' in line.lower() or 'timestamp' in line.lower():
                print(f"  {line.strip()}")

print("\n" + "=" * 80)