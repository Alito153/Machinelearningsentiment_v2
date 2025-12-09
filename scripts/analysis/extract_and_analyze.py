import os
import sys
import zipfile
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

BASE_DIR = Path('/app/forex_trading_strategy_1723')
DATA_DIR = BASE_DIR / 'data'
ZIP_PATH = DATA_DIR / 'forex_macro_sentiment_1329_20251206_144609.zip'
EXTRACT_DIR = DATA_DIR / 'extracted'

logger.info(f"Checking if zip file exists: {ZIP_PATH}")
if not ZIP_PATH.exists():
    logger.error(f"Zip file not found at {ZIP_PATH}")
    sys.exit(1)

logger.info(f"Zip file size: {ZIP_PATH.stat().st_size} bytes")

try:
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        test_result = zip_ref.testzip()
        if test_result is not None:
            logger.warning(f"Corrupted file in archive: {test_result}")
        else:
            logger.info("Zip file integrity verified - no corrupted files")
        
        logger.info(f"Files in archive: {len(zip_ref.namelist())}")
        for name in zip_ref.namelist()[:10]:
            logger.info(f"  - {name}")
except Exception as e:
    logger.error(f"Error testing zip: {e}")
    sys.exit(1)

os.makedirs(EXTRACT_DIR, exist_ok=True)

try:
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)
    logger.info(f"Successfully extracted all files to {EXTRACT_DIR}")
except Exception as e:
    logger.error(f"Error extracting zip: {e}")
    sys.exit(1)

logger.info(f"Extraction complete. Files in {EXTRACT_DIR}:")
for root, dirs, files in os.walk(EXTRACT_DIR):
    level = root.replace(str(EXTRACT_DIR), '').count(os.sep)
    indent = ' ' * 2 * level
    logger.info(f"{indent}{os.path.basename(root)}/")
    subindent = ' ' * 2 * (level + 1)
    for file in files:
        file_path = os.path.join(root, file)
        file_size = os.path.getsize(file_path)
        logger.info(f"{subindent}{file} ({file_size} bytes)")

logger.info("Extraction and verification complete")