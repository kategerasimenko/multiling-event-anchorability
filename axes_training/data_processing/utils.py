import re

import spacy

from spacy.tokenizer import Tokenizer


TO_REPLACE = {
    '``': '"',
    "''": '"',
    '-LRB-': '(',
    '-RRB-': ')',
}


DCT_REGEX = re.compile(r'^\s*DCT.*?\n')

PIPES = {
    'it': spacy.load('it_core_news_lg'),
    'fr':  spacy.load('fr_core_news_lg'),
    'es': spacy.load('es_core_news_lg')
}
for lang, pipe in PIPES.items():
    pipe.tokenizer = Tokenizer(pipe.vocab, token_match=re.compile('\S+').match)


def preprocess_str(text):
    for orig, repl in TO_REPLACE.items():
        text = text.replace(orig, repl)
    return text


def broken_str(row):
    full_from_parts = (row.before + row.verb + row.after).replace(' ', '')
    full_wo_tags = re.sub('<.+?>', '', row.bodytext).replace(' ', '')
    return full_from_parts not in full_wo_tags
