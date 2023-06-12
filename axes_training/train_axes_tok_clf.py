import os
import random
from pathlib import Path

import typer
import torch
import evaluate
import pandas as pd
import numpy as np
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    TrainingArguments, Trainer, DataCollatorForTokenClassification,
    EarlyStoppingCallback
)
from scipy.special import softmax
from transformers.integrations import NeptuneCallback

from data_processing.process_data import (
    process_english_data,
    process_multiling_data
)
from data_processing.paths import DATA_DIR, MULTILING_DATA_FOLDERS


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

ROOT_DIR = Path(__file__).parent.parent
PRED_DIR = ROOT_DIR / 'predictions'
TMP_DIR = os.environ['TMPDIR']
os.makedirs(PRED_DIR, exist_ok=True)
os.makedirs(os.path.join(TMP_DIR, 'models'), exist_ok=True)


F1 = evaluate.load('f1')

LABEL_LIST = ['TGT-1', 'TGT-0', 'O']
ID2LABEL = {i: l for i, l in enumerate(LABEL_LIST)}
LABEL2ID = {l: i for i, l in enumerate(LABEL_LIST)}

app = typer.Typer(add_completion=False)


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    empty_label = LABEL2ID['O']
    n_false_positives = 0
    n_pred = 0
    tgt_true = []
    tgt_pred = []

    for i, (prediction, label) in enumerate(zip(predictions, labels)):
        for (p, l) in zip(prediction, label):
            if l == -100:
                continue

            if p != empty_label:
                n_pred += 1
                if l == empty_label:
                    print(i)
                    print(prediction.tolist())
                    print(label.tolist())
                    n_false_positives += 1

            if l != empty_label:
                tgt_true.append(l)
                tgt_pred.append(p)

    score = F1.compute(predictions=tgt_pred, references=tgt_true, average='macro')
    score['fps'] = n_false_positives / n_pred if n_pred else 0
    score['n_true_samples'] = len(tgt_true)  # sanity check
    return score


def write_predictions(lang, split, ds, model_name, preds):
    preds = np.argmax(preds, axis=2)
    pred_probs = np.max(softmax(preds, axis=2), axis=2)
    info = []

    for item, pred, pred_prob in zip(ds, preds, pred_probs):
        raw_label_idx = 0

        for i in item['target_idxs']:
            p = ID2LABEL[pred[i]]
            p = p.split('-')[1] if p != 'O' else None
            p_prob = pred_prob[i]

            if 'labels' in item:
                tok_label = ID2LABEL[item['labels'][i]]
                label = tok_label.split('-')[1]  # not O because i in target_idxs
            else:
                label = ''

            info.append({
                'lang': lang,
                'split': split,
                'docid': item['docid'],
                'eventid': item['event_ids'][raw_label_idx],
                'input': ' '.join(item['input']),
                'pred': p,
                'prob': p_prob,
                'label': label
            })

            raw_label_idx += 1

    pd.DataFrame(info).to_csv(os.path.join(PRED_DIR, f'{model_name}_{lang}_{split}.csv'))


@app.command()
def main(
    base_model: str = typer.Option('xlm-roberta-base', help='ModelHub pretrained model to finetune'),
    learning_rate: float = typer.Option(1e-5, help='Learning rate'),
    batch_size: int = typer.Option(4, help='Batch size'),
    max_epochs: int = typer.Option(10, help='Number of epochs'),
    eval_steps: int = typer.Option(0, help='Evaluation steps'),
    predict_en: bool = typer.Option(False, help='Whether to save predictions for English.'),
    predict_multiling: bool = typer.Option(False, help='Whether to save predictions for other languages.'),
):
    model_name = f'axes_tok_{base_model}_lr{learning_rate}_e{max_epochs}_bs{batch_size}'
    print(f'Fine-tuning {model_name} on token classification task')

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForTokenClassification.from_pretrained(
        base_model,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )

    axes_df, ds = process_english_data(DATA_DIR, task='tok_clf', tokenizer=tokenizer, seed=SEED)

    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        padding='longest',
    )

    neptune_callback = NeptuneCallback(
        tags=['tok', model_name],
        project="kategaranina/axes-seq"
    )
    neptune_callback.run['model'] = model_name
    neptune_callback.run['base_model'] = base_model

    if eval_steps > 0:
        eval_params = {
            'evaluation_strategy': 'steps',
            'eval_steps': eval_steps,
            'save_strategy': 'steps',
            'save_steps': eval_steps,
            'logging_strategy': 'steps',
            'logging_steps': eval_steps
        }
    else:
        eval_params = {
            'evaluation_strategy': 'epoch',
            'save_strategy': 'epoch',
            'logging_strategy': 'epoch',
        }

    training_args = TrainingArguments(
        output_dir=os.path.join(TMP_DIR, 'checkpoints', model_name),
        report_to='none',
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=max_epochs,
        metric_for_best_model='eval_f1',
        greater_is_better=True,
        load_best_model_at_end=True,
        save_total_limit=2,
        **eval_params,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds['train'],
        eval_dataset=ds['dev'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[
            # EarlyStoppingCallback(early_stopping_patience=5),
            neptune_callback
        ],
    )

    trainer.train()
    trainer.save_model(os.path.join(TMP_DIR, 'models', model_name))

    val_data = trainer.predict(ds['dev'])
    val_f1 = val_data.metrics['test_f1']
    neptune_callback.run[f'score/dev'] = val_f1
    print('DEV', val_f1)

    test_data = trainer.predict(ds['test'])
    test_f1 = test_data.metrics['test_f1']
    preds_sample = []
    for i in range(5):
        inp = ' '.join(ds['test'][i]['input'])
        curr_preds = np.argmax(test_data.predictions[i], axis=-1)
        readable_labels = [ID2LABEL[l] for l in ds['test'][i]['labels'] if l != -100]
        readable_preds = [ID2LABEL[p] for l, p in zip(ds['test'][i]['labels'], curr_preds) if l != -100]
        preds_sample.append(f"{inp}: true {readable_labels}, pred {readable_preds}")

    neptune_callback.run[f'score/test'] = test_f1
    neptune_callback.run[f'preds_sample'] = '\n'.join(preds_sample)
    print('TEST', test_f1)
    print('SAMPLE', '\n'.join(preds_sample))

    if predict_en:
        for part in ['dev', 'test']:
            preds = trainer.predict(ds[part]).predictions
            write_predictions(
                lang='en',
                split=part,
                ds=ds[part],
                model_name=model_name,
                preds=preds
            )

    if predict_multiling:
        lang_dfs, lang_dss = process_multiling_data(
            data_dirs=MULTILING_DATA_FOLDERS,
            task='tok_clf',
            tokenizer=tokenizer,
            seed=SEED
        )
        for lang, lang_ds in lang_dss.items():
            for spl, spl_ds in lang_ds.items():
                spl_ds = spl_ds.remove_columns(['labels'])  # not running eval
                preds = trainer.predict(spl_ds).predictions
                write_predictions(
                    lang=lang,
                    split=spl,
                    ds=spl_ds,
                    model_name=model_name,
                    preds=preds
                )


if __name__ == '__main__':
    app()