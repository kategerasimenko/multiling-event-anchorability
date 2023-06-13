import random
import os
import sys

sys.path.append('..')
from axes_training.data_processing.process_raw_en import get_processed_df


random.seed(42)
dev_size = 0.15


df = get_processed_df(os.path.join('..', 'local_data'), spl_dev=False)
df = df.loc[df.split == 'train']
train_docs = sorted(set(df.docid))
n_dev = int(len(train_docs) * dev_size)

dev_docs = sorted(set(random.sample(train_docs, k=n_dev)))
print('RAW TRAIN', df.shape[0])
print('TRAIN', df.loc[~df.docid.isin(dev_docs)].shape[0])
print('DEV', df.loc[df.docid.isin(dev_docs)].shape[0])

with open(os.path.join('..', 'local_data', 'matres_axes', 'dev.txt'), 'w') as f:
    f.write('\n'.join(dev_docs))
