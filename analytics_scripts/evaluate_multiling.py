import os

import evaluate
import pandas as pd
from sklearn.metrics import cohen_kappa_score, classification_report


F1 = evaluate.load('f1')

TABLE_FOLDER = os.path.join('..', 'local_data', 'axes_annotation_multiling')


def calc_scores(labels, preds):
    score = F1.compute(
        predictions=preds.tolist(),
        references=labels.tolist(),
        average='macro'
    )['f1']
    kappa = cohen_kappa_score(labels, preds)

    print(lang, spl, 'f1', score, 'cohen\'s', kappa)
    print(classification_report(labels, preds, digits=4))


for lang in ['es', 'fr', 'it']:
    for spl in ['test']:
        preds = pd.read_csv(os.path.join(TABLE_FOLDER, f'{lang}_{spl}_with_preds_sample.csv'))
        ann = pd.read_csv(os.path.join(TABLE_FOLDER, f'{lang}_{spl}_annotated_v2.csv'), sep=';')
        preds['annotation'] = ann['annotation']
        preds['is_pred_correct'] = (preds['pred'] == preds['annotation']).astype(int)

        calc_scores(preds['annotation'], preds['pred'])

        print('VERB only')
        calc_scores(preds.loc[preds.pos == 'VERB', 'annotation'], preds.loc[preds.pos == 'VERB', 'pred'])

        preds.to_csv(os.path.join(TABLE_FOLDER, f'{lang}_{spl}_with_preds_annotated_v2.csv'))
