import os
import random
from pathlib import Path
from collections import Counter

import typer
import torch
import evaluate
import pandas as pd
import numpy as np
from transformers import (
    AutoTokenizer, TrainingArguments, Trainer,
    DataCollatorWithPadding, EarlyStoppingCallback
)
from scipy.special import softmax
from transformers.integrations import NeptuneCallback

from data_processing.process_data import (
    process_english_data,
    process_multiling_data
)
from data_processing.paths import DATA_DIR, MULTILING_DATA_FOLDERS
from custom_model_structs import (
    RobertaForSequenceClassificationOnToken,
    XLMRobertaForSequenceClassificationOnToken
)


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

app = typer.Typer(add_completion=False)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return F1.compute(predictions=predictions, references=labels, average='macro')


def write_predictions(lang, split, ds, model_name, preds):
    final_preds = np.argmax(preds, axis=1)
    pred_probs = np.max(softmax(preds, axis=1), axis=1)
    info = []
    for item, pred, pred_prob in zip(ds, final_preds, pred_probs):
        info.append({
            'lang': lang,
            'split': split,
            'docid': item['docid'],
            'eventid': item['eventid'],
            'input': ' '.join(item['input']),
            'pred': pred,
            'prob': pred_prob,
            'label': item.get('label', '')
        })
    pd.DataFrame(info).to_csv(os.path.join(PRED_DIR, f'{model_name}_{lang}_{split}.csv'))


@app.command()
def main(
    base_model: str = typer.Option('xlm-roberta-base', help='ModelHub pretrained model to finetune. Must be RoBERTa-based.'),
    learning_rate: float = typer.Option(1e-5, help='Learning rate'),
    batch_size: int = typer.Option(4, help='Batch size'),
    max_epochs: int = typer.Option(10, help='Number of epochs'),
    eval_steps: int = typer.Option(0, help='Evaluation steps'),
    do_sample_weights: bool = typer.Option(False, help='Whether to weigh samples'),
    predict_en: bool = typer.Option(False, help='Whether to save predictions for English.'),
    predict_multiling: bool = typer.Option(False, help='Whether to save predictions for other languages.'),
):
    model_name = f'axes_seq_clstok_{base_model}_lr{learning_rate}_e{max_epochs}_bs{batch_size}_sw{do_sample_weights}'
    print(f'Fine-tuning {model_name} on sequence classification task')

    if 'xlm' in base_model:
        model = XLMRobertaForSequenceClassificationOnToken.from_pretrained(base_model)
        tokenizer = AutoTokenizer.from_pretrained(base_model)
    else:
        model = RobertaForSequenceClassificationOnToken.from_pretrained(base_model)
        tokenizer = AutoTokenizer.from_pretrained(base_model, add_prefix_space=True)

    axes_df, ds = process_english_data(DATA_DIR, task='seq_clf_clstok', tokenizer=tokenizer, seed=SEED)
    label_counts = Counter(ds['train']['label'])
    samples_per_class = [label_counts[k] for k in sorted(label_counts.keys())]
    print('SAMPLES PER CLASS', samples_per_class)

    if not do_sample_weights:
        ds = ds.remove_columns(['sample_weight'])

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding='longest',
    )

    neptune_callback = NeptuneCallback(
        tags=['seq_clstok', model_name],
        project="kategaranina/axes-seq",
    )
    neptune_callback.run['model'] = model_name
    neptune_callback.run['base_model'] = base_model
    neptune_callback.run['do_sample_weights'] = do_sample_weights

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
    preds_sample = [
        f"{' '.join(ds['test'][i]['input'])}: true {ds['test'][i]['label']}, pred {np.argmax(test_data.predictions[i])}"
        for i in range(5)
    ]

    test_f1 = test_data.metrics['test_f1']
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
            task='seq_clf_clstok',
            tokenizer=tokenizer,
            seed=SEED
        )
        for lang, lang_ds in lang_dss.items():
            for spl, spl_ds in lang_ds.items():
                spl_ds = spl_ds.remove_columns(['label'])  # not running eval
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
