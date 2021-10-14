import pyro
import pyro.distributions as dist
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import dtype, Tensor
from torch.distributions.constraints import positive
from torch.nn.modules.module import T
from typing import Optional, Union, overload

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    """
    Encoder network
    Input: BOWs
    Output: θ
    """

    def __init__(self, vocab_size, num_topics, hidden, dropout):
        """

        :param vocab_size:
        :param num_topics:
        :param hidden:
        :param dropout:
        """
        super().__init__()
        self.drop = nn.Dropout(dropout)  # dropout
        self.fc1 = nn.Linear(vocab_size, hidden)  # fully-connected layer 1
        self.fc2 = nn.Linear(hidden, hidden)  # fully-connected layer 2
        self.fcmu = nn.Linear(hidden, num_topics, bias=True)  # fully-connected layer output mu
        self.fclv = nn.Linear(hidden, num_topics, bias=True)  # fully-connected layer output sigma
        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()
        self.bnmu = nn.BatchNorm1d(num_topics, affine=False)  # avoid component collapse
        self.bnlv = nn.BatchNorm1d(num_topics, affine=False)  # avoid component collapse

        if torch.cuda.is_available():
            self.cuda()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar).to(device)
        eps = torch.randn_like(std).to(device)
        return eps.mul_(std).add_(mu)

    def forward(self, inputs):
        h = self.act1(self.fc1(inputs))
        h = self.act2(self.fc2(h))
        #         h = self.drop(h)

        # μ and Σ are two inference networks
        mu_theta = self.bnmu(self.fcmu(h))
        sigma_theta = self.bnlv(self.fclv(h))
        #         mu_theta = self.fcmu(h)
        #         sigma_theta = self.fclv(h)

        # KL metrics obsoleted
        kl_theta = -0.5 * torch.sum(1 + sigma_theta - mu_theta.pow(2)
                                    - sigma_theta.exp(), dim=-1).mean().to(device)

        # RT z from μ and Σ
        z = self.reparameterize(mu_theta, sigma_theta)
        theta = torch.softmax(z, -1).to(device)
        return theta, kl_theta


class Decoder(nn.Module):
    """
    Decoder network
    ---------------------------------------------------
    σ(βθ)~NN
    Input: θ
    Output:
        σ(βθ): not using embedding
        σ((ρ'α)θ): using embedding
    """

    # Pre-trained embedding, alpha, rho, embedding method
    # Need to be refactored with Topic Embedding
    def __init__(self, vocab_size, num_topics, dropout,
                 useEmbedding=False, rho_size=128, pre_embedding=None, emb_type='NN',
                 trainEmbedding=False):
        """
        Init
        :param vocab_size: Vocabulary size
        :param num_topics: Number of topics
        :param dropout: Dropout rate
        :param useEmbedding: Train with embedding (Y/N)
        :param rho_size: Embedding hidden layer size
        :param pre_embedding: Pre-trained embedding
        :param emb_type: Type of embedding
        """
        super().__init__()
        # this beta can be refactorized in to BOW, a neural network that to be trained
        self.emb_type = emb_type
        self.trainEmbedding = trainEmbedding
        self.useEmbedding = useEmbedding
        # Changes - Beta
        # < Linear NN (K->V)
        # > Embedding -> Linear NN (K->E->V)
        self.drop = nn.Dropout(dropout)
        if self.useEmbedding:
            # Call ρ Topic Embedding
            if trainEmbedding:
                self.fcrho = TopicEmbedding(rho_size, vocab_size, pre_embedding,
                                            emb_type, dropout)
                self.bnrho = nn.BatchNorm1d(rho_size, affine=False)
            # use original embedding
            else:
                self.fcrho = TopicEmbedding(rho_size, vocab_size, pre_embedding,
                                            emb_type, dropout)
            # Call α
            self.fcalpha = nn.Linear(rho_size, num_topics, bias=False)
            self.bnalpha = nn.BatchNorm1d(num_topics, affine=False)
            # nn.Parameter(torch.randn(rho_size, num_topics))
        else:
            # Call β, Use Original NN (K->V)
            self.fcbeta = nn.Linear(num_topics, vocab_size, bias=False)
            self.bnbeta = nn.BatchNorm1d(vocab_size, affine=False)

        self.bn = nn.BatchNorm1d(vocab_size, affine=False)

        if torch.cuda.is_available():
            self.cuda()

    # need to forward the loss of the neural network function
    def forward(self, theta):  # (D,K)
        if self.trainEmbedding:
            # beta: rho(V,L), alpha(L,K)
            # bows: D,V
            # beta = F.softmax(self.bnalpha(self.fcalpha(self.bnrho(self.fcrho(bows)))), dim=1)
            # βθ
            # print(f'theta shape {theta.shape}')
            res = torch.mm(theta, self.beta()).to(device)
            # res = F.softmax(res, dim=-1)
            res = self.drop(res)
        elif self.useEmbedding:
            # output βθ
            res = torch.mm(theta, self.beta()).to(device)
        else:
            # output σ(βθ)
            res = F.softmax(self.bnbeta(self.fcbeta(theta)), dim=1)
        return res.float()

    def beta(self):
        if self.trainEmbedding:
            if self.emb_type is 'BERT' or 'Transformer':
                # output σ((ρ'α)θ)
                # beta = self.bnalpha(self.fcalpha(self.bnrho(self.fcrho.weight())))
                beta = self.fcalpha(self.fcrho.weight())
                beta = beta.softmax(0).transpose(1, 0).to(device)
            elif self.emb_type is 'NN':
                # introduce get rho
                beta = torch.mm(self.fcrho.weight.T, self.fcalpha.weight.T)
            else:
                raise ValueError('Wrong embedding type')
        elif self.useEmbedding:
            # output σ((ρ'α)θ)
            beta = self.bnalpha(self.fcalpha(self.bnrho(self.fcrho.weight()))).transpose(1, 0).to(device)
            # beta = F.softmax(beta, dim=0).transpose(1, 0).to(device)
        else:
            beta = self.fcbeta.weight
        return beta


class TopicEmbedding(nn.Module):
    """
    Topic Embedding
    Input: BOWs
    Output: Topic Embedding ρ: (K, E)
    """

    def __init__(self, rho_size, vocab_size, pre_embedding=None,
                 emb_type='NN', dropout=0.1,
                 n_heads=8, n_code=8):
        """
        Init parameter
        :param rho_size: Embedding size ρ
        :param vocab_size: corpus size
        :param pre_embedding: embedding vector model pretrained model
        :param emb_type: embedding type(NN, Skipgram, BERT)
        """
        super().__init__()
        self.emb_type = emb_type
        # Topic Embedding
        # define the word embedding (ρ'α)
        # Embedding: Embedding->BOWs (E->W)
        if pre_embedding is None:
            # 1. Embedding layer
            if emb_type is 'NN':
                self.rho = nn.Linear(rho_size, vocab_size, bias=False)
            # 2. Transformer Embedding (BERT)
            elif emb_type is 'BERT':
                inner_ff_size = rho_size * 4
                seq_len = 20
                self.rho = Transformer(
                    n_code, n_heads, rho_size, inner_ff_size,
                    vocab_size, seq_len, dropout)
            elif emb_type is 'Transformer':
                nlayers = 6  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
                nhid = 512
                self.rho = TransformerModel(
                    vocab_size, rho_size, n_heads, nhid, nlayers, dropout).to(device)
            else:
                raise ValueError('Wrong Embedding Type')
        # Import Embedding
        else:
            self.rho = pre_embedding.clone().float().to(device)
            # num_embeddings, emsize = pre_embedding.size()
            # rho = nn.Embedding(num_embeddings, emsize)

    # used for training
    def weight(self):
        # embedding weight for BERT
        if self.emb_type is 'BERT':
            return self.rho.embeddings.weight
        elif self.emb_type is 'Transformer':
            return self.rho.encoder.weight
        # embedding for NN
        elif self.emb_type is 'NN':
            return self.rho.weight
        else:
            raise ValueError('Wrong Embedding Type')

    # TODO
    # Train the embedding here
    # Separate process
    # Forward the pass instead of using weight
    # input: batch data
    # output: prediction for target
    def forward(self, inputs):
        if self.emb_type is 'BERT':
            output = model(inputs)
            output_v = output.view(-1, output.shape[-1])
            return output_v
        elif self.emb_type is 'Transformer':
            src_mask = model.generate_square_subsequent_mask(inputs.size(0)).to(device)
            output = model(inputs, src_mask)
            return output
        elif self.emb_type is 'NN':
            return self.rho(inputs)
        else:
            raise ValueError('Wrong Embedding Type')


class TETM(nn.Module):
    """
    Model for LKJ Correlated Topic Model
    """

    def __init__(self, vocab_size, num_topics, hidden, dropout,
                 useEmbedding=False, rho_size=128, pre_embedding=None, emb_type='NN',
                 trainEmbedding=False, LKJChol=True):
        """
        Init parameters
        :param LKJChol:
        :param vocab_size: Vocabulary size
        :param num_topics: Number of topics
        :param hidden: Hidden layer size
        :param dropout: Dropout rate
        :param useEmbedding: Train withg embedding (Y/N)
        :param rho_size: Embedding hidden layer size
        :param pre_embedding: Pre-trained embedding
        :param emb_type: Type of embedding
        """
        super().__init__()
        self.LKJChol = LKJChol
        self.vocab_size = vocab_size
        self.num_topics = num_topics
        self.encoder = Encoder(vocab_size, num_topics, hidden, dropout)
        self.decoder = Decoder(vocab_size, num_topics, dropout,
                               useEmbedding, rho_size, pre_embedding, emb_type,
                               trainEmbedding)

        self.useEmbedding = useEmbedding
        self.emb_type = emb_type

        if torch.cuda.is_available():
            self.cuda()

    def beta(self):
        return self.decoder.beta()

    def theta(self, normalized_bows):
        theta = self.encoder(normalized_bows)
        return theta

    def decode(self, theta, beta):
        res = torch.mm(theta, beta)
        preds = torch.log(res + 1e-6)
        return preds

    def forward(self, bows):
        theta, kld_theta = self.theta(bows)
        beta = self.beta()
        preds = self.decode(theta, beta)
        recon_loss = -(preds * bows).sum(1)
        recon_loss = recon_loss.mean()
        return recon_loss, kld_theta

    def getTransformer(self):
        if self.emb_type is 'BERT' or 'Transformer':
            return self.decoder.fcrho.rho
        else:
            raise ValueError('BERT not chosen')

    def logpp(self, docs_te):
        len_te = docs_te.shape[0] // 2
        docs_t1, docs_t2 = normalized_bows[:len_te].to(device), docs_te[len_te:].to(self.device)
        # theta, beta
        theta = self.theta(docs_t1).to(self.device)
        beta = self.beta().to(self.device)
        # theta * beta (D,V)
        pred = torch.mm(theta, beta).to(device)
        # prevent nan
        pred = torch.where(pred == 0, torch.ones(1), pred)
        # softmax
        pred = torch.softmax(pred, -1)
        # log predictive probability
        pred = torch.log(pred + 1e-6)
        return pred
