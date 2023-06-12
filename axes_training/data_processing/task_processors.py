from collections import defaultdict

from .utils import preprocess_str


# ----------- SEQUENCE CLASSIFIER ON CLS TOKEN -----------


def process_sample_for_seq_clf_clstok(raw_sample):
    left = preprocess_str(raw_sample.before)
    right = preprocess_str(raw_sample.after)
    sample = {
        'input': f'{left} [ {raw_sample.verb} ] {right}'.split(),
        'label': raw_sample.label,  # label - anchorable (1) or not (0)
        'sample_weight': raw_sample.label_confidence,
        'docid': raw_sample.docid,
        'eventid': raw_sample.eventid
    }
    return sample


def process_samples_for_seq_clf_clstok(df, **kwargs):
    data = defaultdict(list)
    for i, row in df.iterrows():
        sample = process_sample_for_seq_clf_clstok(row)
        data[row.split].append(sample)
    return data


def prepare_inputs_for_seq_clf_clstok(ds, tokenizer, **kwargs):
    def tokenize(examples):
        return tokenizer(
            examples['input'],
            truncation=True,
            padding=False,
            is_split_into_words=True
        )

    ds = ds.map(tokenize, batched=True)
    return ds


# ----------- SEQUENCE CLASSIFIER ON VERB TOKEN -----------


def process_sample_for_seq_clf_verbtok(raw_sample, include_brackets=False):
    sample = {}
    left = preprocess_str(raw_sample.before)
    right = preprocess_str(raw_sample.after)
    if include_brackets:
        sample['input'] = f'{left} [ {raw_sample.verb} ] {right}'.split()
        sample['target_token_id'] = len(raw_sample.before.split()) + 1
    else:
        sample['input'] = f'{left} {raw_sample.verb} {right}'.split()
        sample['target_token_id'] = len(raw_sample.before.split())

    sample['label'] = raw_sample.label  # label - anchorable (1) or not (0)
    sample['sample_weight'] = raw_sample.label_confidence
    sample['docid'] = raw_sample.docid
    sample['eventid'] = raw_sample.eventid

    return sample


def process_samples_for_seq_clf_verbtok(df, include_brackets=False, **kwargs):
    data = defaultdict(list)
    for i, row in df.iterrows():
        sample = process_sample_for_seq_clf_verbtok(row, include_brackets)
        data[row.split].append(sample)
    return data


def prepare_inputs_for_seq_clf_verbtok(ds, tokenizer, **kwargs):
    def tokenize(examples):
        enc = tokenizer(
            examples['input'],
            truncation=True,
            padding=False,
            is_split_into_words=True
        )

        all_labels = []
        for i, tgt_idx in enumerate(examples['target_token_id']):
            word_ids = enc.word_ids(batch_index=i)
            for j, w_id in enumerate(word_ids):
                if w_id == tgt_idx:
                    all_labels.append(j)
                    break  # taking only first sub-token of a target word

        enc['target_token_ids'] = all_labels
        return enc

    ds = ds.map(tokenize, batched=True)
    return ds


# ----------- TOKEN CLASSIFICATION -----------


def process_sample_for_tok_clf(raw_sample):
    text = f'{raw_sample.before} {raw_sample.verb} {raw_sample.after}'
    pp_text = preprocess_str(text)
    sample = {
        'sentence': pp_text.split(),
        'tgt_idx': len(raw_sample.before.split()),
        'label': raw_sample.label,  # label - anchorable (1) or not (0)
        'eventid': raw_sample.eventid
    }
    return sample


def reformat_samples_to_labelled_seq(data_by_sentence):
    seq_data = defaultdict(list)
    for (spl, docid, sentence), sample_list in data_by_sentence.items():
        targets = {
            v['tgt_idx']: f"TGT-{v['label']}"
            if v['label'] is not None else 'TGT-X'  # we don't have labels sometimes, so this is a placeholder
            for v in sample_list
        }
        new_sentence = []
        targets_seq = []

        for i, token in enumerate(sentence.split()):
            if i in targets:
                new_sentence.extend(['[', token, ']'])
                targets_seq.extend(['O', targets[i], 'O'])
            else:
                new_sentence.append(token)
                targets_seq.append('O')

        seq_data[spl].append({
            'input': new_sentence,
            'labels': targets_seq,
            'event_ids': [v['eventid'] for v in sample_list],
            'docid': docid
        })

    return seq_data


def process_samples_for_tok_clf(df, **kwargs):
    data_by_sentence = defaultdict(list)
    for i, row in df.iterrows():
        sample = process_sample_for_tok_clf(row)
        data_by_sentence[(row.split, row.docid, ' '.join(sample['sentence']))].append(sample)
    seq_data = reformat_samples_to_labelled_seq(data_by_sentence)
    return seq_data


def prepare_inputs_for_tok_clf(ds, tokenizer, **kwargs):
    label2id = {l: i for i, l in enumerate(['TGT-1', 'TGT-0', 'O'])}  # todo: do normally!!!

    def tokenize(examples):
        enc = tokenizer(
            examples['input'],
            truncation=True,
            padding=False,
            is_split_into_words=True
        )

        all_labels = []
        all_target_idxs = []

        for i, seq_labels in enumerate(examples['labels']):
            word_ids = enc.word_ids(batch_index=i)  # Map tokens to their respective word.
            previous_word_idx = None
            final_labels = []
            target_idxs = []

            for j, word_idx in enumerate(word_ids):
                if word_idx is None:  # Set the special tokens to -100.
                    final_labels.append(-100)

                elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                    raw_label = seq_labels[word_idx]
                    label_id = label2id.get(raw_label)  # None for a placeholder - TGT-X
                    final_labels.append(label_id)

                    if raw_label != 'O':
                        target_idxs.append(j)

                else:
                    final_labels.append(-100)

                previous_word_idx = word_idx

            all_labels.append(final_labels)
            all_target_idxs.append(target_idxs)

        enc["labels"] = all_labels
        enc['target_idxs'] = all_target_idxs
        return enc

    ds = ds.map(tokenize, batched=True)
    return ds
