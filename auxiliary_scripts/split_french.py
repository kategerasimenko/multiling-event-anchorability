import os
import random


seed = 42
random.seed(42)


french_dir = os.path.join('..', 'local_data', 'full_corpora', 'TB_FR_col')
col_files = [f for f in os.listdir(french_dir) if f.endswith('.col')]

test_k = round(len(col_files) * 0.2)
test_files = set(random.sample(col_files, k=test_k))


os.makedirs(os.path.join(french_dir, 'train'), exist_ok=True)
os.makedirs(os.path.join(french_dir, 'test'), exist_ok=True)

for col_file in col_files:
    tgt_dir = 'test' if col_file in test_files else 'train'
    os.rename(os.path.join(french_dir, col_file), os.path.join(french_dir, tgt_dir, col_file))
