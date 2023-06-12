import os

import numpy as np
import pandas as pd

from .utils import broken_str


def drop_duplicates(df):
    unique_event_fields = ['docid', 'verb', 'eventid']
    indices_to_rm = []

    dupl_idx = df.duplicated(keep=False, subset=unique_event_fields)

    for key, item in df.loc[dupl_idx].groupby(unique_event_fields):
        golden = item.loc[item._golden]
        if not golden.shape[0]:
            golden = item.loc[item._unit_state == 'golden']

        first_golden_idx = golden.index[0]  # there's always golden item
        indices_to_rm.extend([i for i in item.index if i != first_golden_idx])

    df = df.drop(indices_to_rm)
    return df


def get_processed_df(data_dir, spl_dev=True):
    platinum_docs = pd.read_csv(os.path.join(data_dir, 'matres_axes', 'platinum.txt'), sep='\t', header=None)
    platinum_docs = set(platinum_docs[0])

    dense = pd.read_csv(os.path.join(data_dir, 'matres_axes', 'TBDense_all_new.csv'))
    dense['source'] = 'dense'

    aq_platinum = pd.read_csv(os.path.join(data_dir, 'matres_axes', 'AQ_Platinum_all.csv'))
    aq_platinum.loc[~aq_platinum.docid.isin(platinum_docs), 'source'] = 'aq'
    aq_platinum.loc[aq_platinum.docid.isin(platinum_docs), 'source'] = 'platinum'

    tb = pd.read_csv(os.path.join(data_dir, 'matres_axes', 'TB_remaining_147docs.csv'))
    tb['source'] = 'tb'

    axes_df = pd.concat([tb, dense, aq_platinum], ignore_index=True)
    print('RAW', axes_df.shape[0])

    axes_df = drop_duplicates(axes_df)
    print('NO DUPL', axes_df.shape[0])

    axes_df.fillna('', inplace=True)
    axes_df['broken'] = axes_df.apply(broken_str, axis=1)
    axes_df.loc[axes_df.broken].to_csv(os.path.join(data_dir, 'matres_axes', 'broken.csv'))

    axes_df = axes_df.loc[~axes_df.broken]
    print('NO BROKEN', axes_df.shape[0])
    print('SOURCES', axes_df.source.value_counts(), sep='\n')
    print('CLASSES', axes_df.can_the_verb_span_stylecolorblueverb_span_be_anchored_in_time.value_counts(), sep='\n')

    axes_df['split'] = np.where(axes_df.docid.isin(platinum_docs), 'test', 'train')
    if spl_dev:
        with open(os.path.join(data_dir, 'matres_axes', 'dev.txt')) as f:
            dev_docs = set(f.read().strip().split())
        axes_df.loc[axes_df.docid.isin(dev_docs), 'split'] = 'dev'

    axes_df['label'] = axes_df.can_the_verb_span_stylecolorblueverb_span_be_anchored_in_time == 'yes_its_anchorable'
    axes_df['label'] = axes_df['label'].astype(int)
    axes_df['label_confidence'] = axes_df['can_the_verb_span_stylecolorblueverb_span_be_anchored_in_time:confidence']

    return axes_df
