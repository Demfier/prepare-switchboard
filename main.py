"""
Script to prepare the switchboard dataset for dialog paper
"""
import numpy as np
from src import swda
from sklearn.model_selection import train_test_split as tts


def all_sentences(data_dir):
    sentences = []
    corpus_reader = swda.CorpusReader(data_dir)
    for t in corpus_reader.iter_transcripts():
        for u in t.utterances:
            sentences.append(' '.join(u.pos_words()).strip())
    return sentences


def prepare_data(data_dir):
    sentences = all_sentences(data_dir)
    with open('data/processed/all_sentences.txt', 'w') as f:
        f.write('\n'.join(sentences))
    print('Saved all sentences.')

    # split all sentences into utterances
    q_r_pairs = []
    print(len(sentences))
    if len(sentences) % 2 != 0:
        sentences = sentences[:-1]  # remove the last utterance if odd
        print('Removed last sentence from utterances')
    skip_count = 0
    for idx, s in enumerate(sentences[::2]):
        q = sentences[idx]
        r = sentences[idx+1]
        # skip pair creation if either query or response empty
        if not q.strip() or not r.strip():
            skip_count += 1
            continue
        q_r_pairs.append(f'{q}\t{r}')

    print(f'Skip count => {skip_count} | Total final q-r pairs => {len(q_r_pairs)}')
    with open('data/processed/all_pairs.tsv', 'w') as f:
        f.write('\n'.join(q_r_pairs))

    # create train/val/test split
    train, valtest = tts(q_r_pairs, test_size=0.2, shuffle=True)
    val, test = tts(valtest, test_size=0.5, shuffle=True)

    print(f'#train: {len(train)} | #val: {len(val)} | #test: {len(test)}')

    # save split dataset
    with open('data/processed/train_pairs.tsv', 'w') as f:
        f.write('\n'.join(train))
    with open('data/processed/val_pairs.tsv', 'w') as f:
        f.write('\n'.join(val))
    with open('data/processed/test_pairs.tsv', 'w') as f:
        f.write('\n'.join(test))

    print('Saved split dataset at data/processed/')


if __name__ == '__main__':
    DATA_DIR = 'data/raw/swda/'
    prepare_data(DATA_DIR)
