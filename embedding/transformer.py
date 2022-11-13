import math
import torch
import torch.nn as nn
from torch.nn import LayerNorm

from embedding.attention import TransformerEncoder


class TransformerModel(nn.Module):

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.ninp = ninp
        self.pos_encoder = PositionalEncoding(ninp, dropout)

        self.encoder = nn.Embedding(ntoken, ninp)
        self.transformer_encoder = TransformerEncoder(num_layers=nlayers,
                                                      input_dim=nhid,
                                                      dim_feedforward=2 * nhid,
                                                      num_heads=nhead,
                                                      dropout=dropout)
        self.encoder_norm = LayerNorm(ninp)
        # self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers, encoder_norm)
        self.decoder_out = nn.Linear(ninp, ntoken)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        src = self.encoder(src) * math.sqrt(self.ninp)
        output = self.transformer_encoder(src)
        output = self.decoder_out(output)
        return output


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


if __name__ == '__main__':
    ntoken = 256
    nlayers = 1
    ninp = 32
    nhead = 2
    nhid = 128
    dropout = 0
    encoder_layers = 1
    encoder_norm = LayerNorm(ninp)
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
