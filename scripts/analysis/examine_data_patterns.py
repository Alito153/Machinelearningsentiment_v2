import os
import re

strategy_dir = '/app/xauusd_trading_strategy_2022/forex_trading_strategy_1723'

data_format_info = {
    'column_names': set(),
    'timestamp_formats': set(),
    'data_types': set(),
    'csv_patterns': []
}

for root, dirs, files in os.walk(strategy_dir):
    for file in files:
        if file.endswith('.py'):
            filepath = os.path.join(root, file)
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    
                    if 'read_csv' in content or 'pd.read' in content or 'csv' in content.lower():
                        lines = content.split('\n')
                        for i, line in enumerate(lines):
                            if 'read_csv' in line or 'csv' in line.lower():
                                context_start = max(0, i - 2)
                                context_end = min(len(lines), i + 3)
                                data_format_info['csv_patterns'].append({
                                    'file': os.path.relpath(filepath, strategy_dir),
                                    'line_num': i,
                                    'context': '\n'.join(lines[context_start:context_end])
                                })
                            
                            if 'OHLC' in line or "'open'" in line or "'high'" in line or "'low'" in line or "'close'" in line:
                                data_format_info['column_names'].add(line.strip())
                            
                            if 'timestamp' in line.lower() or 'time' in line.lower():
                                data_format_info['timestamp_formats'].add(line.strip())
            except:
                pass

print("=" * 80)
print("DATA FORMAT PATTERNS FOUND")
print("=" * 80)

print("\n--- CSV LOADING PATTERNS ---")
for pattern in data_format_info['csv_patterns'][:10]:
    print(f"\nFile: {pattern['file']}")
    print(f"Line {pattern['line_num']}:")
    print(pattern['context'])

print("\n--- COLUMN NAME REFERENCES ---")
for col in sorted(data_format_info['column_names'])[:20]:
    print(f"  {col}")

print("\n--- TIMESTAMP REFERENCES ---")
for ts in sorted(data_format_info['timestamp_formats'])[:15]:
    print(f"  {ts}")

test_files = []
for root, dirs, files in os.walk(strategy_dir):
    for file in files:
        if 'test' in file.lower() and file.endswith('.py'):
            test_files.append(os.path.relpath(os.path.join(root, file), strategy_dir))

print(f"\n--- TEST FILES FOUND ({len(test_files)}) ---")
for test in sorted(test_files):
    print(f"  {test}")

print("\n" + "=" * 80)