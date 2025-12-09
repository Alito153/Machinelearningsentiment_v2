import os
import sys
import json
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

BASE_DIR = Path('/app/forex_trading_strategy_1723/data/extracted/forex_macro_sentiment_1329')

file_catalog = defaultdict(list)
csv_files = {}
json_files = {}

logger.info("=" * 80)
logger.info("COMPREHENSIVE FILE CATALOG AND ANALYSIS")
logger.info("=" * 80)

for root, dirs, files in os.walk(BASE_DIR):
    for file in files:
        file_path = os.path.join(root, file)
        relative_path = os.path.relpath(file_path, BASE_DIR)
        file_size = os.path.getsize(file_path)
        ext = os.path.splitext(file)[1]
        
        file_catalog[ext].append({
            'name': file,
            'path': relative_path,
            'size': file_size,
            'size_mb': round(file_size / (1024*1024), 2)
        })

logger.info("\nFILE INVENTORY BY TYPE:")
for ext in sorted(file_catalog.keys()):
    files = file_catalog[ext]
    total_size = sum(f['size'] for f in files)
    logger.info(f"\n{ext if ext else 'NO_EXT'}: {len(files)} files ({total_size / (1024*1024):.2f} MB)")
    for f in sorted(files, key=lambda x: x['name'])[:5]:
        logger.info(f"  - {f['path']} ({f['size_mb']} MB)")
    if len(files) > 5:
        logger.info(f"  ... and {len(files) - 5} more")

logger.info("\n" + "=" * 80)
logger.info("DETAILED DATA FILE ANALYSIS")
logger.info("=" * 80)

csv_pattern = BASE_DIR / 'data' / '*.csv'
csv_files_list = list(BASE_DIR.glob('data/*.csv')) + list(BASE_DIR.glob('raw_data/*.csv'))

for csv_file in sorted(csv_files_list):
    logger.info(f"\n{'=' * 60}")
    logger.info(f"FILE: {csv_file.name}")
    logger.info(f"PATH: {csv_file.relative_to(BASE_DIR)}")
    logger.info(f"SIZE: {csv_file.stat().st_size / (1024*1024):.2f} MB")
    
    try:
        df = pd.read_csv(csv_file)
        logger.info(f"SHAPE: {df.shape[0]} rows × {df.shape[1]} columns")
        logger.info(f"COLUMNS: {list(df.columns)}")
        logger.info(f"DATA TYPES:\n{df.dtypes.to_string()}")
        logger.info(f"\nMISSING VALUES:\n{df.isnull().sum().to_string()}")
        logger.info(f"\nDESCRIPTIVE STATS:\n{df.describe().to_string()}")
    except Exception as e:
        logger.error(f"Error reading {csv_file.name}: {e}")

logger.info("\n" + "=" * 80)
logger.info("JSON CONFIGURATION FILES")
logger.info("=" * 80)

json_files_list = list(BASE_DIR.glob('data/*.json')) + list(BASE_DIR.glob('models/*.json')) + list(BASE_DIR.glob('outputs/*.json'))

for json_file in sorted(json_files_list)[:10]:
    logger.info(f"\n{json_file.name}:")
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        logger.info(json.dumps(data, indent=2)[:500])
    except Exception as e:
        logger.error(f"Error reading {json_file.name}: {e}")

logger.info("\n" + "=" * 80)
logger.info("MODELS AND ARTIFACTS")
logger.info("=" * 80)

models_dir = BASE_DIR / 'models'
if models_dir.exists():
    for file in sorted(os.listdir(models_dir)):
        file_path = models_dir / file
        if os.path.isfile(file_path):
            size = os.path.getsize(file_path)
            logger.info(f"{file}: {size / (1024*1024):.2f} MB")

logger.info("\nAnalysis complete")