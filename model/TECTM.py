import pyro
import pyro.distributions as dist
import torch
import torch.nn as nn
import torch.nn.functional as F
from pyro.infer import Trace_ELBO
from torch.distributions.constraints import positive, positive_definite

from embedding.transformer import TransformerModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    """
    Amortized variational infernece
    Input: BOWs
    Output: θ~q(θ)
    """
    def __init__(self, vocab_size, num_topics, hidden, dropout, LKJChol=True):
        """
        :param vocab_size:
        :param num_topics:
        :param hidden:
        :param dropout:
        """
        super().__init__()
        self.num_topics = num_topics
        self.LKJChol = LKJChol
        self.drop = nn.Dropout(dropout)  # dropout
        self.fc1 = nn.Linear(vocab_size, hidden)  # fully-connected layer 1
        self.fc2 = nn.Linear(hidden, hidden)  # fully-connected layer 2
        self.fcmu = nn.Linear(hidden, num_topics, bias=True)  # fully-connected layer output mu
        self.fclv = nn.Linear(hidden, num_topics, bias=True)  # fully-connected layer output sigma
        self.act1 = nn.Softplus()
        self.act2 = nn.Softplus()
        self.bnmu = nn.BatchNorm1d(num_topics, affine=False)  # avoid component collapse
        self.bnlv = nn.BatchNorm1d(num_topics, affine=False)  # avoid component collapse

        self.p_sigma = None

        if torch.cuda.is_available():
            self.cuda()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar).to(device)
        eps = torch.randn_like(std).to(device)
        return eps.mul_(std).add_(mu)

    def batch_kl_normal(self, q_mu, q_sig, p_mu, p_sig, eps=1e-6):
        return 0.5 * (
                torch.log(p_sig.det() / q_sig.exp().det() + eps).nan_to_num() +
                torch.matmul(
                    p_sig.inverse(),
                    q_sig.exp()
                ).diagonal(offset=0, dim1=-1, dim2=-2).nan_to_num().sum(-1) +
                torch.bmm(
                    torch.matmul(
                        (q_mu - p_mu)[:, None, :],
                        p_sig.inverse()
                    ),
                    (q_mu - p_mu)[:, :, None]
                ).nan_to_num().mean() -
                len(q_sig)
        )

    def reparam(self, mu, sig):
        sig = torch.exp(0.5 * sig).to(device)
        mu = mu.unsqueeze(2)
        z = mu + torch.matmul(sig.sqrt(), torch.randn_like(mu).to(device))
        return z.squeeze(2)

    def forward(self, inputs, q_sig):
        h = self.act1(self.fc1(inputs.nan_to_num()))
        h = self.act2(self.fc2(h))
        #         h = self.drop(h)
        options = dict(dtype=torch.float32, device=device)
        mu_theta = self.bnmu(self.fcmu(h))
        if self.LKJChol:
            L_Omega = dist.LKJCholesky(self.num_topics, torch.ones((), **options)).sample()
            p_sigma = torch.mm(L_Omega, L_Omega.T)
            kld_theta = self.batch_kl_normal(mu_theta, q_sig, torch.zeros_like(mu_theta).to(device), p_sigma)
            z = self.reparam(mu_theta, q_sig)
        else:
            logsig_theta = self.bnlv(self.fclv(h))
            z = self.reparameterize(mu_theta, logsig_theta)
            kld_theta = -0.5 * torch.sum(1 + logsig_theta - mu_theta.pow(2) - logsig_theta.exp(), dim=-1).mean()
        theta = torch.softmax(z, -1).to(device)
        return theta, kld_theta


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
    def __init__(self, vocab_size, num_topics, dropout, useEmbedding=False, rho_size=128, pre_embedding=None,
                 emb_type='NN', trainEmbedding=False,
                 trans_heads=2, trans_layers=2, trans_dim=300):
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
            # if trainEmbedding:
            self.fcrho = TopicEmbedding(rho_size, vocab_size, pre_embedding,
                                        emb_type, dropout)
            self.bnrho = nn.BatchNorm1d(rho_size, affine=False)
            # # use original embedding
            # else:
            #     self.fcrho = TopicEmbedding(rho_size, vocab_size, pre_embedding,
            #                                 emb_type, dropout)
            # Call α
            self.fcalpha = nn.Linear(rho_size, num_topics, bias=False)
            self.bnalpha = nn.BatchNorm1d(num_topics, affine=False)
        else:
            self.fcbeta = nn.Parameter(torch.randn(num_topics, vocab_size).to(device))

        self.bn = nn.BatchNorm1d(vocab_size, affine=False)

        if torch.cuda.is_available():
            self.cuda()

    # need to forward the loss of the neural network function
    def forward(self, theta):  # (D,K)
        if self.trainEmbedding:
            res = torch.mm(theta, self.beta()).to(device)
        elif self.useEmbedding:
            res = torch.mm(theta, self.beta()).to(device)
        else:
            res = F.softmax(self.bnbeta(self.fcbeta(theta)), dim=1)
        return res.float()

    def beta(self):
        if self.trainEmbedding:
            if self.emb_type is 'BERT' or 'Transformer':
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
            beta = F.softmax(beta, dim=0).transpose(1, 0).to(device)
        else:
            beta = self.fcbeta.softmax(1)
        return beta


class TopicEmbedding(nn.Module):
    """
    Topic Embedding
    Input: BOWs
    Output: Topic Embedding ρ: (K, E)
    """

    # TODO nlayers, nhid
    def __init__(self, rho_size, vocab_size, pre_embedding=None,
                 emb_type='NN', dropout=0.0, n_heads=2, nlayers=2, nhid=300):
        """
        Init parameter
        :param rho_size: Embedding size ρ
        :param vocab_size: corpus size
        :param pre_embedding: embedding vector model pretrained model
        :param emb_type: embedding type(Skipgram, Transformer)
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
            elif emb_type is 'Transformer':
                #nlayers = 2  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
                #nhid = 300
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
        if self.emb_type == 'BERT':
            return self.rho.embeddings.weight
        elif self.emb_type == 'Transformer':
            return self.rho.encoder.weight
        # embedding for NN
        elif self.emb_type == 'NN':
            return self.rho.weight
        else:
            raise ValueError('Wrong Embedding Type')

    # TODO
    # Train the embedding here
    # Separate process
    # Forward the pass instead of using weight
    # input: batch data
    # output: prediction for target


#     def forward(self, inputs):
#         if self.emb_type is 'BERT':
#             output = model(inputs)
#             output_v = output.view(-1, output.shape[-1])
#             return output_v
#         elif self.emb_type is 'Transformer':
#             src_mask = model.generate_square_subsequent_mask(inputs.size(0)).to(device)
#             output = model(inputs, src_mask)
#             return output
#         elif self.emb_type is 'NN':
#             return self.rho(inputs)
#         else:
#             raise ValueError('Wrong Embedding Type')


class TECTM(nn.Module):
    """
    Model for LKJ Correlated Topic Model
    """

    def __init__(self, vocab_size, num_topics, hidden, dropout, useEmbedding=False, rho_size=128, pre_embedding=None,
                 emb_type='NN', trainEmbedding=False, LKJChol=True,
                 trans_heads=2, trans_layers=2, trans_dim=300):
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
        self.encoder = Encoder(vocab_size, num_topics, hidden, dropout, LKJChol=LKJChol)
        self.decoder = Decoder(vocab_size, num_topics, dropout, useEmbedding, rho_size, pre_embedding, emb_type,
                               trainEmbedding,trans_heads=trans_heads, trans_layers=trans_layers, trans_dim=trans_dim)

        self.loss_elbo = Trace_ELBO().differentiable_loss
        self.useEmbedding = useEmbedding
        self.emb_type = emb_type

        self.L_Omega = None
        if torch.cuda.is_available():
            self.cuda()

    # Decoding with output and approximate from sample data
    # input: eta, output: theta
    def model(self, docs):
        pyro.module("decoder", self.decoder)
        options = dict(dtype=torch.float32, device=device)
        L_Omega = pyro.sample("L_omega", dist.LKJCholesky(self.num_topics, torch.ones((), **options)))
        L_Omega = (torch.eye(self.num_topics, **options)).sqrt() * L_Omega
        self.encoder.p_sigma = L_Omega
        beta = self.beta()
        with pyro.plate("documents", docs.shape[0]) as ind:
            logtheta = pyro.sample("logtheta", dist.MultivariateNormal(
                torch.zeros(self.num_topics, **options),
                scale_tril=L_Omega))
            preds = self.decode(logtheta, beta).softmax(-1)
            total_count = int(docs.sum(-1).max())
            pyro.sample('obs', dist.Multinomial(total_count, preds), obs=docs.nan_to_num())
        # return L_Omega

    # Encoding with data input
    def guide(self, docs):
        options = dict(dtype=torch.float32, device=device)
        omega_posterior = pyro.param("omega_posterior", torch.ones((), **options),
                                     constraint=positive)
        L_Omega = pyro.sample("L_omega", dist.LKJCholesky(self.num_topics, omega_posterior))
        self.L_Omega = torch.mm(L_Omega, L_Omega.T)
        with pyro.plate("documents", docs.shape[0]):
            theta, kld_theta = self.encoder(docs.nan_to_num(), self.L_Omega)
            theta = pyro.sample("logtheta", dist.Delta(theta).to_event(1))
            kld_theta = 0
        return theta, kld_theta

    def beta(self):
        return self.decoder.beta()

    def theta(self, bows):
        return self.encoder(bows)

    def decode(self, theta, beta):
        preds = torch.mm(theta, beta)
        preds = torch.log(preds + 1e-6)
        return preds

    def forward(self, bows):
        theta, kld_theta = self.theta(bows)
        beta = self.beta()
        preds = self.decode(theta, beta)
        recon_loss = -(preds * bows).sum(1)
        recon_loss = recon_loss.mean()
        return recon_loss, kld_theta

    def get_pam(self):
        #         my_list = ['gplvm','likelihood', 'decoder.fcrho.rho']
        my_list = ['decoder.fcrho.rho']
        para = [n for n, p in self.named_parameters() for key in my_list if key in n]
        base_params = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] not in para, self.named_parameters()))))
        # base_params = list(map(lambda x: x[1],list(filter(lambda kv: my_list not in kv[0], self.named_parameters()))))
        return base_params

    def get_transformer_params(self):
        my_list = 'decoder.fcrho.rho'
        params = list(map(lambda x: x[1], list(filter(lambda kv: my_list in kv[0], self.named_parameters()))))
        return params

    def getTransformer(self):
        if self.emb_type is 'BERT' or 'Transformer':
            return self.decoder.fcrho.rho
        else:
            raise ValueError('BERT not chosen')
