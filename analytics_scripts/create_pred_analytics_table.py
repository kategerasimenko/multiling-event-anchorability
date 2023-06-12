import os
import sys
from collections import defaultdict

import pandas as pd
from statsmodels.stats.inter_rater import fleiss_kappa
from sklearn.metrics import classification_report

sys.path.append('..')
from axes_training.data_processing.process_raw_en import get_processed_df


PRED_FILES = {
    'seq_clstok_base': 'axes_seq_clstok_xlm-roberta-base_lr1e-05_e10_bs16_swFalse',
    'seq_verbtok': 'axes_seq_verbtok_xlm-roberta-base_lr1e-05_e10_bs4_swTrue_brTrue',
    'tok': 'axes_tok_xlm-roberta-base_lr1e-05_e10_bs16',
    'svm': 'svm_left2_right2_bowFalse_feats3000_ngrams1,1_c1.0_pl1_weightsTrue',
    'fasttext': 'fasttext'
}

PRED_DIR = os.path.join('..', 'predictions')
DATA_DIR = os.path.join('..', 'local_data')


def collect_preds(main_df, dfs):
    preds_by_event = defaultdict(dict)
    for name, df in dfs.items():
        for i, row in df.iterrows():
            key = (row['docid'], row['eventid'])
            preds_by_event[key][name] = row['pred']
            preds_by_event[key][f'{name}_conf'] = row.get('prob')
            preds_by_event[key][f'{name}_is_correct'] = int(row['pred'] == row['label'])

    preds = pd.DataFrame([
        preds_by_event[(row['docid'], row['eventid'])]
        for i, row in main_df.iterrows()
    ])

    return preds


def calculate_kappa(preds):
    pred_cols = [c for c in preds.columns if not (c.endswith('_is_correct') or c.endswith('_conf'))]
    preds_only = preds[pred_cols]

    n_pos_for_kappa_calc = preds_only.sum(axis=1)
    n_neg_for_kappa_calc = preds_only.shape[1] - n_pos_for_kappa_calc
    table_for_kappa_calc = pd.concat((n_neg_for_kappa_calc, n_pos_for_kappa_calc), axis=1)
    kappa = fleiss_kappa(table_for_kappa_calc, method='fleiss')

    return kappa


for part in ['dev', 'test']:
    dfs = {
        name: pd.read_csv(os.path.join(PRED_DIR, f'{pred_file}_en_{part}.csv'))
        for name, pred_file in PRED_FILES.items()
    }

    whole_orig_df = get_processed_df(DATA_DIR)
    orig_df_train = whole_orig_df.loc[whole_orig_df.split == 'train'].reset_index()
    orig_df = whole_orig_df.loc[whole_orig_df.split == part].reset_index()

    main_df = dfs['seq_clstok_base'].drop(['label', 'pred', 'prob'], axis=1)
    assert main_df.docid.equals(orig_df.docid) and main_df.eventid.equals(orig_df.eventid)

    main_df['verb'] = orig_df.verb
    main_df['label'] = orig_df.label  # for convenient column order
    main_df['orig_label'] = orig_df.can_the_verb_span_stylecolorblueverb_span_be_anchored_in_time
    main_df['label_confidence'] = orig_df.label_confidence
    main_df['n_annotators'] = orig_df._trusted_judgments

    all_preds = collect_preds(main_df, dfs)
    full_df = pd.concat((main_df, all_preds), axis=1)

    kappa = calculate_kappa(all_preds)
    print(classification_report(full_df['label'], full_df['seq_clstok_base']))
    print(f'Fleiss kappa for {part}: {kappa}')

    full_df.to_csv(os.path.join(PRED_DIR, f'all_preds_{part}.csv'), index=False)
