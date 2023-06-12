import os
import sys

import pandas as pd

sys.path.append('..')
from axes_training.data_processing.process_raw_multiling import compile_multilingual_dfs
from axes_training.data_processing.paths import MULTILING_DATA_FOLDERS


SEED = 42

PREDS_PREFIX = 'axes_seq_clstok_xlm-roberta-base_lr1e-05_e10_bs16_swFalse'
PREDS_DIR = 'predictions'

N_SAMPLES = 200

TABLE_FOLDER = os.path.join('..', 'local_data', 'axes_annotation_multiling')
os.makedirs(TABLE_FOLDER, exist_ok=True)

all_lang_dfs = compile_multilingual_dfs(MULTILING_DATA_FOLDERS)


for lang, splits in MULTILING_DATA_FOLDERS.items():
    for spl in splits.keys():
        df = all_lang_dfs[lang].loc[all_lang_dfs[lang].split == spl].reset_index(drop=True)
        preds = pd.read_csv(os.path.join(PREDS_DIR, f'{PREDS_PREFIX}_{lang}_{spl}.csv'))
        assert df.docid.equals(preds.docid) and df.eventid.equals(preds.eventid)
        df['pred'] = preds['pred']
        df['prob'] = preds['prob']

        df['annotation'] = ''
        df = df.drop(['label', 'label_confidence', 'split'], axis=1)

        print(lang, spl, df.shape[0])
        df.to_csv(os.path.join(TABLE_FOLDER, f'{lang}_{spl}_with_preds.csv'))

        selected_contexts = df.sample(n=N_SAMPLES, random_state=SEED)
        selected_contexts.to_csv(os.path.join(TABLE_FOLDER, f'{lang}_{spl}_with_preds_sample.csv'))

        wo_preds = selected_contexts.drop(['pred', 'prob'], axis=1)
        wo_preds.to_csv(os.path.join(TABLE_FOLDER, f'{lang}_{spl}_sample.csv'))
