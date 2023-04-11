import torch
import numpy as np
import random

from torch.utils.data import DataLoader


# Train the tokenizer as well


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

        s = dataset.get_sentence_idx(index % len(dataset))
        s = s[:dataset.seq_len]

        s_len = len(s)

        [s.append(dataset.IGNORE_IDX) for _ in range(dataset.seq_len - len(s))]

        s = [(dataset.MASK_IDX, w) if random.random() < p_random_mask and idx < s_len else (w, dataset.IGNORE_IDX) for
             idx, w in enumerate(s)]

        return {'input': torch.Tensor([w[0] for w in s]).long(),
                'index': np.array(index),
                'target': torch.Tensor([w[1] for w in s]).long()}

    def __len__(self):
        return len(self.sentences)

    def get_sentence_idx(self, index):
        dataset = self
        s = dataset.sentences[index]
        s = [dataset.vocab[w] if w in dataset.vocab else dataset.OUT_OF_VOCAB_IDX for w in s.split()]
        return s

    def tokens_to_sentence(self, tokens):
        dataset = self
        words = []
        for token in tokens:
            words.append(dataset.rvocab[token])
        return " ".join(words)


if __name__ == '__main__':
    from sklearn.feature_extraction.text import CountVectorizer

    corpus = ['This is the first document.',
              'This document is the second document.',
              'And this is the third one.',
              'Is this the first document?',
              ]

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus)
    vocab = vectorizer.get_feature_names_out()

    dataset = SentencesDataset(corpus, vocab.tolist(), 16)
    dataloader = DataLoader(dataset, batch_size=1)

    print(next(iter(dataloader))['input'].numpy().tolist()[0])

    print(dataset.tokens_to_sentence(next(iter(dataloader))['input'].numpy().tolist()[0]))
