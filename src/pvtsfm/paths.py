"""Local data are referenced, never included in the package."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MMSP_DIR = Path(os.environ.get('PVTSFM_MMSP_DIR', PROJECT_ROOT / 'datasets/MMSP/data'))
PRETRAINED_DIR = PROJECT_ROOT / 'pretrained'
