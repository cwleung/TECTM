import torch
import numpy as np
import random

class SentencesDataset(torch.utils.data.Dataset):
    def __init__(self, sentences, vocab, seq_len):
        dataset = self

        dataset.sentences = sentences
        dataset.vocab = vocab + ['<ignore>', '<oov>', '<mask>']
        dataset.vocab = {e: i for i, e in enumerate(dataset.vocab)}
        dataset.rvocab = {v: k for k, v in dataset.vocab.items()}
        dataset.seq_len = seq_len

        dataset.IGNORE_IDX = dataset.vocab['<ignore>']
        dataset.OUT_OF_VOCAB_IDX = dataset.vocab['<oov>']
        dataset.MASK_IDX = dataset.vocab['<mask>']

    def __getitem__(self, index, p_random_mask=0.15):
        dataset = self
        s = []
        while len(s) < dataset.seq_len:
            s.extend(dataset.get_sentence_idx(index % len(dataset)))
            index += 1
        s = s[:dataset.seq_len]
        [s.append(dataset.IGNORE_IDX) for _ in range(dataset.seq_len - len(s))]

        s = [(dataset.MASK_IDX, w) if random.random() < p_random_mask else (w, dataset.IGNORE_IDX) for w in s]
        return {'input': torch.Tensor([w[0] for w in s]).long(),
                'index': np.array(index),
                'target': torch.Tensor([w[1] for w in s]).long()}

    def __len__(self):
        return len(self.sentences)

    def get_sentence_idx(self, index):
        dataset = self
        s = dataset.sentences[index]
        s = [dataset.vocab[w] if w in dataset.vocab else dataset.OUT_OF_VOCAB_IDX for w in s]
        return s

if __name__ == '__main__':
    # create a test dataset
    SentencesDataset()