import os
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = ROOT_DIR / 'local_data'

MULTILING_DATA_PATH = os.path.join(DATA_DIR, 'full_corpora')
MULTILING_DATA_FOLDERS = {
    'it': {
        'train': os.path.join(MULTILING_DATA_PATH, 'TB_IT_col', 'train'),
        'test': os.path.join(MULTILING_DATA_PATH, 'TB_IT_col', 'test'),
    },
    'fr': {
        'train': os.path.join(MULTILING_DATA_PATH, 'TB_FR_col', 'train'),
        'test': os.path.join(MULTILING_DATA_PATH, 'TB_FR_col', 'test'),
    },
    'es': {
        'train': os.path.join(MULTILING_DATA_PATH, 'TB_ES_col', 'train'),
        'test': os.path.join(MULTILING_DATA_PATH, 'TB_ES_col', 'test'),
    }
}

