import os

import pandas as pd

from .utils import PIPES, DCT_REGEX


PP_FUNCS_FOR_LANGS = {
    'it': 'full',
    'fr': 'full',
    'es': 'shrinked'
}


def collect_verbs_from_col_sent(doc_id, rows, spacy_pipe, doctype='full'):
    contexts = []
    if doctype == 'full':
        token_col, token_id_col, sent_id_col, event_id_col = 0, 1, 2, 3
    else:
        token_col, token_id_col, sent_id_col, event_id_col = 2, 1, 0, 3

    rows = [r.split('\t') for r in rows]
    tokens = [r[token_col].strip() for r in rows]

    parsed = spacy_pipe(' '.join(tokens))
    assert len(rows) == len(parsed)

    for i, (row, token) in enumerate(zip(rows, parsed)):
        if row[event_id_col].startswith('e') and token.pos_ in ['VERB', 'AUX']:
            tags = {}
            for tag in ['Mood', 'Number', 'Tense', 'Person', 'VerbForm']:
                val = token.morph.get(tag)
                tags[tag] = val[0] if val else ''

            row_vals = {
                'docid': doc_id,
                'sentid': row[sent_id_col],
                'tokenid': row[token_id_col],
                'eventid': row[event_id_col],
                'before': ' '.join(tokens[:i]),
                'verb': row[token_col],
                'after': ' '.join(tokens[i+1:]),
                'label': None,  # placeholder
                'label_confidence': 1,  # placeholder,
                'pos': token.pos_
            }
            row_vals.update(tags)
            contexts.append(row_vals)

    return contexts


def collect_verbs_from_col_doc(lang, doc_path, doc_id, spacy_pipe):
    contexts = []

    try:
        with open(doc_path) as f:
            raw = f.read().strip()

    except UnicodeDecodeError:
        print('BROKEN', lang, doc_id)
        with open(doc_path, errors='ignore') as f:
            raw = f.read().strip()

    raw = DCT_REGEX.sub('', raw).strip()

    sents = raw.split('\n\n')
    for sent in sents:
        rows = sent.split('\n')
        parse_func_doctype = PP_FUNCS_FOR_LANGS[lang]
        sent_contexts = collect_verbs_from_col_sent(doc_id, rows, spacy_pipe, doctype=parse_func_doctype)
        contexts.extend(sent_contexts)

    return contexts


def compile_multilingual_dfs(data_folders):
    lang_dfs = {}
    for lang, all_folders in data_folders.items():
        spacy_pipe = PIPES[lang]
        spl_dfs = []

        for spl, folder in all_folders.items():
            contexts = []

            for filename in os.listdir(folder):
                doc_id, ext = filename.rsplit('.', 1)
                if ext != 'col':
                    continue

                doc_path = os.path.join(folder, filename)
                doc_contexts = collect_verbs_from_col_doc(lang, doc_path, doc_id, spacy_pipe)
                contexts.extend(doc_contexts)

            contexts_df = pd.DataFrame(contexts)
            contexts_df['split'] = spl
            contexts_df = contexts_df.drop_duplicates(
                subset=['docid', 'eventid'],
                ignore_index=True
            )  # multi-token events
            spl_dfs.append(contexts_df)

        lang_dfs[lang] = pd.concat(spl_dfs, ignore_index=True)

    return lang_dfs
