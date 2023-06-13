# Automatic multilingual annotation of event anchorability

This repository contains the code for fine-tuning a model on anchorability data and running inference on data in other languages.

Author: Ekaterina Garanina, University of Groningen

### Installation

Python 3.7+

```
pip install -r requirements.txt
```

### Data

### Running the system

To produce the our additional data splits, run:

```
python auxiliary_scripts/train_dev_split_axes.py
python auxiliary_scripts/split_french.py
```

To train our best model (classifier on \[CLS\] token) and get all predictions, run:

```
python axes_training/train_axes_seq_clf_clstok.py \
    --eval-steps=200 \
    --batch-size=16 \
    --predict-en \
    --predict-multiling \
```

All predictions are stored in `predictions` folder.


### Experiments

Running other systems on English data:

**FastText**
```
python axes_training/vector_space_fasttext.py \
    --top-k=17 \
    --predict-en
```

**SVM**

```
python axes_training/svm.py \
    --do-sample-weights \
    --svm-penalty=l1 \
    --predict-en
```

**Sequence classifier on a verb token**

```
python axes_training/train_axes_seq_clf_verbtok.py \
    --eval-steps=200 \
    --include-brackets \
    --do-sample-weights \
    --predict-en
```

**Token classifier**

```
python axes_training/train_axes_tok_clf.py \
    --eval-steps=200 \
    --batch-size=16 \
    --predict-en
```

There are other parameters that can be customized (e.g., base model, learning rate, task-specific flags), see the corresponding scripts.
