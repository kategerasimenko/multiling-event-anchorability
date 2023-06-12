import random
random.seed(42)

from datasets import Dataset, DatasetDict

from .process_raw_en import get_processed_df
from .process_raw_multiling import compile_multilingual_dfs
from .task_processors import (
    process_samples_for_seq_clf_clstok,
    prepare_inputs_for_seq_clf_clstok,
    process_samples_for_seq_clf_verbtok,
    prepare_inputs_for_seq_clf_verbtok,
    process_samples_for_tok_clf,
    prepare_inputs_for_tok_clf
)


TASK_CONFIGS = {
    'seq_clf_clstok': {
        'process_samples': process_samples_for_seq_clf_clstok,
        'prepare_inputs': prepare_inputs_for_seq_clf_clstok,
    },
    'seq_clf_verbtok': {
        'process_samples': process_samples_for_seq_clf_verbtok,
        'prepare_inputs': prepare_inputs_for_seq_clf_verbtok,
    },
    'tok_clf': {
        'process_samples': process_samples_for_tok_clf,
        'prepare_inputs': prepare_inputs_for_tok_clf,
    },
}


def process_data(axes_df, task, tokenizer, seed=42, processing_params=None, tokenize_params=None, do_shuffle=True):
    if task not in TASK_CONFIGS:
        raise ValueError(f'Task {task} is unknown.')

    if processing_params is None:
        processing_params = {}
    if tokenize_params is None:
        tokenize_params = {}

    data = TASK_CONFIGS[task]['process_samples'](axes_df, **processing_params)
    ds = DatasetDict({
        k: Dataset.from_list(spl_data)
        for k, spl_data in data.items()
    })

    if do_shuffle:
        ds['train'] = ds['train'].shuffle(seed=seed)

    ds = TASK_CONFIGS[task]['prepare_inputs'](ds, tokenizer, **tokenize_params)
    print(ds)

    return ds


def process_english_data(data_dir, task, tokenizer, seed=42, processing_params=None, tokenize_params=None):
    axes_df = get_processed_df(data_dir)
    ds = process_data(
        axes_df=axes_df,
        task=task,
        tokenizer=tokenizer,
        seed=seed,
        processing_params=processing_params,
        tokenize_params=tokenize_params,
        do_shuffle=True
    )
    return axes_df, ds


def process_multiling_data(data_dirs, task, tokenizer, seed=42, processing_params=None, tokenize_params=None):
    lang_dfs = compile_multilingual_dfs(data_dirs)
    lang_ds = {}
    for lang, lang_df in lang_dfs.items():
        lang_ds[lang] = process_data(
            axes_df=lang_df,
            task=task,
            tokenizer=tokenizer,
            seed=seed,
            processing_params=processing_params,
            tokenize_params=tokenize_params,
            do_shuffle=False  # not training on this data, don't shuffle for prediction
        )
    return lang_dfs, lang_ds
