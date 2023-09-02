import unittest
import torch
import torch.nn as nn
from torch.nn import LayerNorm

from embedding.attention import TransformerEncoder

ntoken = 256
nlayers = 1
ninp = 32
nhead = 2
nhid = 128
dropout = 0
encoder_layers = 1


class TestTransformer(unittest.TestCase):

    def test_transformer(self):
        encoder = nn.Embedding(ntoken, ninp)
        transformer_encoder = TransformerEncoder(num_layers=nlayers,
                                                 input_dim=ninp,
                                                 dim_feedforward=nhid,
                                                 num_heads=nhead,
                                                 dropout=dropout)

        seqlen = 32
        batch_size = 16
        x = torch.randint(ntoken, size=(batch_size, seqlen))
        print(x.shape)
        assert x.shape == torch.Size((batch_size, seqlen))
        x = encoder(x)
        print(x.shape)
        assert x.shape == torch.Size((batch_size, seqlen, ninp))
        x = transformer_encoder(x)
        print(x.shape)
        assert x.shape == torch.Size((batch_size, seqlen, ninp))


if __name__ == '__main__':
    unittest.main()
