import os
from pathlib import Path
from collections import Counter

import typer
import neptune
import evaluate
import numpy as np
import gensim.downloader as api

from data_processing.process_raw_en import get_processed_df


FASTTEXT_MODEL = api.load('fasttext-wiki-news-subwords-300')

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / 'local_data'
PRED_DIR = ROOT_DIR / 'predictions'
os.makedirs(PRED_DIR, exist_ok=True)

F1 = evaluate.load('f1')

app = typer.Typer(add_completion=False)


def preprocess(df):
    df['verb'] = df['verb'].apply(str.lower)
    return df


def collect_verb_labels(train_df, train_verbs):
    for info in train_verbs:
        train_v = train_df.loc[train_df.verb == info['verb']]
        counts = Counter(train_v.label)

        avg_0_conf = train_v.loc[train_v.label == 0, 'label_confidence'].mean() if counts[0] else 0
        avg_1_conf = train_v.loc[train_v.label == 1, 'label_confidence'].mean() if counts[1] else 0

        if counts[0] > counts[1]:
            info['label'] = 0
        elif counts[1] > counts[0]:
            info['label'] = 1
        elif avg_0_conf > avg_1_conf:
            info['label'] = 0
        else:
            if avg_0_conf == avg_1_conf:
                print(info['verb'], avg_0_conf, avg_1_conf, counts)
            info['label'] = 1

    return train_verbs


def collect_training_data(train_df):
    train_verbs = [{'verb': v, 'vector': FASTTEXT_MODEL[v]} for v in sorted(train_df.verb.unique())]
    train_verbs = collect_verb_labels(train_df, train_verbs)
    return train_verbs


def predict(test_df, train_verbs, top_k):
    train_vectors = np.array([i['vector'] for i in train_verbs])
    test_vectors = np.array([FASTTEXT_MODEL[v] for v in test_df.verb])

    train_norm = train_vectors / np.linalg.norm(train_vectors, axis=1)[:, None]
    test_norm = test_vectors / np.linalg.norm(test_vectors, axis=1)[:, None]
    similarities = test_norm @ train_norm.T
    chosen = np.argsort(similarities, axis=1)[:, -top_k:]
    preds = [round(np.mean([train_verbs[i]['label'] for i in ii])) for ii in chosen]
    return preds


def run_vectors_fasttext(train_df, test_df, top_k):
    train_verbs = collect_training_data(train_df)
    preds = predict(test_df, train_verbs, top_k)
    score = F1.compute(predictions=preds, references=test_df.label.tolist(), average='macro')
    return preds, score['f1']


def write_predictions(df, preds, lang, split):
    input_sents = df[['before', 'verb', 'after']].agg(' | '.join, axis=1)
    out_df = df.loc[:, ['docid', 'eventid', 'label']]
    out_df['input'] = input_sents
    out_df['pred'] = preds
    out_df['lang'] = lang
    out_df['split'] = split
    out_df = out_df[['lang', 'split', 'docid', 'eventid', 'input', 'pred', 'label']]
    out_df.to_csv(os.path.join(PRED_DIR, f'fasttext_{lang}_{split}.csv'))


@app.command()
def main(
        top_k: int = typer.Option(1, help='Top K nearest neighbours'),
        predict_en: bool = typer.Option(False, help='Whether to save predictions for English.'),
):
    neptune.init(project_qualified_name="kategaranina/axes-seq")
    neptune.create_experiment(
        tags=["vectors_fasttext"],
        params={'top_k': top_k}
    )

    axes_df = get_processed_df(str(DATA_DIR))
    axes_df = preprocess(axes_df)
    train_df = axes_df.loc[axes_df.split == 'train']
    dev_df = axes_df.loc[axes_df.split == 'dev']
    test_df = axes_df.loc[axes_df.split == 'test']

    dev_preds, dev_score = run_vectors_fasttext(train_df, dev_df, top_k)
    neptune.log_metric('score/dev', dev_score)
    print('DEV', dev_score)

    test_preds, test_score = run_vectors_fasttext(train_df, test_df, top_k)
    neptune.log_metric('score/test', test_score)
    print('TEST', test_score)

    if predict_en:
        write_predictions(dev_df, dev_preds, lang='en', split='dev')
        write_predictions(test_df, test_preds, lang='en', split='test')


if __name__ == '__main__':
    app()
