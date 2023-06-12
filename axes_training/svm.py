import re
import os
from pathlib import Path
from collections import defaultdict

import spacy
import typer
import neptune
import evaluate
import numpy as np
import pandas as pd
from spacy.tokenizer import Tokenizer
from scipy.sparse import csr_matrix, hstack
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer

from data_processing.process_raw_en import get_processed_df


ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / 'local_data'
PRED_DIR = ROOT_DIR / 'predictions'
os.makedirs(PRED_DIR, exist_ok=True)

TOKEN_PATTERN = r'\S+'

EN_PIPE = spacy.load('en_core_web_lg')
EN_PIPE.tokenizer = Tokenizer(EN_PIPE.vocab, token_match=re.compile(TOKEN_PATTERN).match)

SEED = 42
F1 = evaluate.load('f1')

app = typer.Typer(add_completion=False)


def process_sample(sample):
    sentence = f'{sample.before} {sample.verb} {sample.after}'
    features = {
        'sentence': sentence.lower(),
        'verb_idx': len(sample.before.strip().split()),
        'pos_tags': ' '.join(token.pos_ for token in EN_PIPE(sentence)).lower()
    }
    return features


def vectorize_texts(texts, vectorizer):
    if not hasattr(vectorizer, 'vocabulary_'):
        vectorizer.fit(texts)
    feats = vectorizer.transform(texts)
    return feats, vectorizer


def collect_sparse_features(
        feature_df,
        text_vectorizer,
        text_feats,
        pos_vectorizer,
        pos_feats,
        distance_left,
        distance_right,
        include_bow,
):
    features = defaultdict(list)

    for i, row in feature_df.iterrows():
        words = row.sentence.strip().split()
        pos_tags = row.pos_tags.split()
        for j, (word, pos) in enumerate(zip(words, pos_tags)):
            distance = j - row.verb_idx

            feat_word_idx = text_vectorizer.vocabulary_.get(word)
            feat_pos_idx = pos_vectorizer.vocabulary_.get(pos)
            if feat_word_idx is None:
                continue

            word_feat = (i, feat_word_idx, text_feats[i, feat_word_idx])
            pos_feat = (i, feat_pos_idx, pos_feats[i, feat_pos_idx])

            if j < row.verb_idx and include_bow:
                features['bow_before'].append(word_feat)
            elif j > row.verb_idx and include_bow:
                features['bow_after'].append(word_feat)

            if -distance_left <= distance < 0 or 0 <= distance <= distance_right:
                features[f'word_{distance}'].append(word_feat)
                features[f'pos_{distance}'].append(pos_feat)

    return features


def collect_feature_matrices(
        features,
        pos_vocab,
        text_vocab,
        n_samples,
        distance_left,
        distance_right
):
    feature_matrices = []
    feature_keys = (
        ['bow_before', 'bow_after']
        + [f'word_{d}' for d in range(-distance_left, distance_right + 1)]
        + [f'pos_{d}' for d in range(-distance_left, distance_right + 1)]
    )
    for feature_name in feature_keys:
        n_vocab = pos_vocab if feature_name.startswith('pos_') else text_vocab
        feature_vals = features.get(feature_name, [])
        if feature_vals:
            row, col, data = list(zip(*feature_vals))
            feature_matrix = csr_matrix((data, (row, col)), shape=(n_samples, n_vocab))
        else:
            feature_matrix = csr_matrix((n_samples, n_vocab))
        feature_matrices.append(feature_matrix)
    return feature_matrices


def prepare_features(
        df,
        distance_left,
        distance_right,
        include_bow,
        max_features,
        ngram_range,
        pipeline_components=None
):
    samples = [process_sample(row) for _, row in df.iterrows()]
    feature_df = pd.DataFrame(samples)
    feature_df.fillna('', inplace=True)

    if pipeline_components is not None:
        text_vectorizer = pipeline_components['text_vectorizer']
        pos_vectorizer = pipeline_components['pos_vectorizer']
    else:
        text_vectorizer = TfidfVectorizer(
            token_pattern=TOKEN_PATTERN,
            max_features=max_features,
            ngram_range=ngram_range
        )
        pos_vectorizer = TfidfVectorizer(token_pattern=TOKEN_PATTERN)

    text_feats, text_vectorizer = vectorize_texts(feature_df['sentence'], text_vectorizer)
    pos_feats, pos_vectorizer = vectorize_texts(feature_df['pos_tags'], pos_vectorizer)

    features = collect_sparse_features(
        feature_df,
        text_vectorizer=text_vectorizer,
        text_feats=text_feats,
        pos_vectorizer=pos_vectorizer,
        pos_feats=pos_feats,
        distance_left=distance_left,
        distance_right=distance_right,
        include_bow=include_bow

    )
    feature_matrices = collect_feature_matrices(
        features,
        pos_vocab=pos_feats.shape[1],
        text_vocab=text_feats.shape[1],
        n_samples=feature_df.shape[0],
        distance_left=distance_left,
        distance_right=distance_right
    )

    final_features = hstack(feature_matrices)

    if pipeline_components is not None:
        feats_to_keep = pipeline_components['features_to_keep']
    else:
        feats_to_keep = np.where(final_features.sum(axis=0))[1]

    final_features = final_features[:, feats_to_keep]

    if pipeline_components is None:
        pipeline_components = {
            'text_vectorizer': text_vectorizer,
            'pos_vectorizer': pos_vectorizer,
            'features_to_keep': feats_to_keep
        }

    return final_features, pipeline_components


def train_and_evaluate(clf, X_train, y_train, X_test, y_test, sample_weights):
    clf.fit(X_train, y_train, sample_weight=sample_weights)
    y_pred = clf.predict(X_test)
    score = F1.compute(predictions=y_pred, references=y_test.tolist(), average='macro')['f1']
    return y_pred, score


def write_predictions(df, preds, model_name, lang, split):
    input_sents = df[['before', 'verb', 'after']].agg(' | '.join, axis=1)
    out_df = df.loc[:, ['docid', 'eventid', 'label']]
    out_df['input'] = input_sents
    out_df['pred'] = preds
    out_df['lang'] = lang
    out_df['split'] = split
    out_df = out_df[['lang', 'split', 'docid', 'eventid', 'input', 'pred', 'label']]
    out_df.to_csv(os.path.join(PRED_DIR, f'{model_name}_{lang}_{split}.csv'))


@app.command()
def main(
        distance_left: int = typer.Option(2, help='distance for token-based features to the left'),
        distance_right: int = typer.Option(2, help='distance for token-based features to the right'),
        include_bow: bool = typer.Option(False, help='Whether to include BOW features for the rest of the sentence'),
        max_features: int = typer.Option(3000, help='Max vocab for tf-idf word encoding'),
        ngram_range: str = typer.Option('1,1', help='ngram range for tf-idf vectorizer'),
        svm_c: float = typer.Option(1.0, help='Parameter C for SVM'),
        svm_penalty: str = typer.Option('l2', help='Parameter penalty for SVM'),
        do_sample_weights: bool = typer.Option(False, help='Whether to weigh samples'),
        predict_en: bool = typer.Option(False, help='Whether to save predictions for English.'),
        # predict_multiling: bool = typer.Option(False, help='Whether to save predictions for other languages.'),
):
    features_str = f'left{distance_left}_right{distance_right}_bow{include_bow}_feats{max_features}_ngrams{ngram_range}'
    model_name = f'svm_{features_str}_c{svm_c}_p{svm_penalty}_weights{do_sample_weights}'

    neptune.init(project_qualified_name="kategaranina/axes-seq")
    neptune.create_experiment(
        tags=["svm", model_name],
        params={
            'distance_left': distance_left,
            'distance_right': distance_right,
            'include_bow': include_bow,
            'max_features': max_features,
            'ngram_range': ngram_range,
            'svm_c': svm_c,
            'svm_penalty': svm_penalty,
            'do_sample_weights': do_sample_weights
        }
    )

    ngram_range = ngram_range.split(',')
    ngram_range = (int(ngram_range[0]), int(ngram_range[1]))

    df = get_processed_df(DATA_DIR)

    df_train = df.loc[df.split == 'train']
    df_train = df_train.sample(frac=1, random_state=SEED)
    X_train, pipeline_components = prepare_features(
        df_train,
        distance_left=distance_left,
        distance_right=distance_right,
        include_bow=include_bow,
        max_features=max_features,
        ngram_range=ngram_range
    )
    y_train = df_train.label
    print('TRAINING FEATURES', X_train.shape)

    df_dev = df.loc[df.split == 'dev']
    X_dev, _ = prepare_features(
        df_dev,
        pipeline_components=pipeline_components,
        distance_left=distance_left,
        distance_right=distance_right,
        include_bow=include_bow,
        max_features=max_features,
        ngram_range=ngram_range
    )
    y_dev = df_dev.label
    print('DEV FEATURES', X_dev.shape)

    df_test = df.loc[df.split == 'test']
    X_test, _ = prepare_features(
        df_test,
        pipeline_components=pipeline_components,
        distance_left=distance_left,
        distance_right=distance_right,
        include_bow=include_bow,
        max_features=max_features,
        ngram_range=ngram_range
    )
    y_test = df_test.label
    print('TEST FEATURES', X_test.shape)

    dual = False if svm_penalty == 'l1' else True

    clf = LinearSVC(
        class_weight='balanced',
        C=svm_c,
        penalty=svm_penalty,
        dual=dual,
        random_state=SEED
    )
    sample_weights = df_train.label_confidence if do_sample_weights else None

    dev_preds, dev_score = train_and_evaluate(clf, X_train, y_train, X_dev, y_dev, sample_weights=sample_weights)
    print('DEV SCORE', dev_score)
    neptune.log_metric('score/dev', dev_score)

    test_preds, test_score = train_and_evaluate(clf, X_train, y_train, X_test, y_test, sample_weights=sample_weights)
    print('TEST SCORE', test_score)
    neptune.log_metric('score/test', test_score)

    if predict_en:
        write_predictions(df_dev, dev_preds, model_name=model_name, lang='en', split='dev')
        write_predictions(df_test, test_preds, model_name=model_name, lang='en', split='test')


if __name__ == '__main__':
    app()
