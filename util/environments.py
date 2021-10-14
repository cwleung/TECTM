import argparse

import data
import torch
from pyro.infer import RenyiELBO, TraceGraph_ELBO, TraceMeanField_ELBO, TraceTailAdaptive_ELBO, TraceEnum_ELBO


class Environment:
    def __init__(self):
        self.__init__()
        parser = argparse.ArgumentParser(description="LKJ Correlated Topic Model")
        # Environment setting
        parser.add_argument("-st", "--smoke-test", default=False, type=bool)
        parser.add_argument("-s", "--seed", default=0, type=int)
        # Topic model-related
        parser.add_argument("-nt", "--num-topics", default=20, type=int)
        # NN-related
        parser.add_argument("-hl", "--hidden-layer", default=100, type=int)
        parser.add_argument("-dr", "--drop-out-rate", default=0.0, type=float)
        parser.add_argument("-af", "--batch-normalization", default=True, type=bool)
        # Dataset-related
        parser.add_argument("-ds", "--dataset", default="20Newsgroups", type=str)
        parser.add_argument("-nd", "--min-df", default=20, type=int)
        parser.add_argument("-xd", "--max-df", default=0.7, type=float)
        # Traning-related
        parser.add_argument("-e", "--epochs", default=200, type=int)
        parser.add_argument("-nb", "--num-batches", default=2048, type=int)
        parser.add_argument("-o", "--alpha", default=1, type=int)
        parser.add_argument("-lr", "--learning-rate", default=2e-3, type=float)
        # Transformer
        # head
        # hid
        # hidden layer

        self.args = parser.parse_args()

    def get_device(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        return device

    def get_smoke_test(self):
        return self.args.smoke_test

    def get_seed(self):
        return self.args.seed

    def get_num_topics(self):
        return self.args.num_topics if not self.get_smoke_test() else 3

    def get_elbo_loss(self):
        return self.args.elbo_loss

    def get_hidden_layer(self):
        return self.args.hidden_layer if not self.get_smoke_test() else 10

    def get_drop_out_rate(self):
        return self.args.drop_out_rate

    def get_activation(self, act):
        if self.args.activation_function == 'tanh':
            act = nn.Tanh()
        elif self.args.activation_function == 'relu':
            act = nn.ReLU()
        elif self.args.activation_function == 'softplus':
            act = nn.Softplus()
        elif self.args.activation_function == 'rrelu':
            act = nn.RReLU()
        elif self.args.activation_function == 'leakyrelu':
            act = nn.LeakyReLU()
        elif self.args.activation_function == 'elu':
            act = nn.ELU()
        elif self.args.activation_function == 'selu':
            act = nn.SELU()
        elif self.args.activation_function == 'glu':
            act = nn.GLU()
        return act

    def get_batch_normalization(self):
        return self.args.batch_normalization

    @property
    def get_dataset(self):
        print(f'Dataset: {self.args.dataset}')
        if self.args.dataset == "20NewsGroup":
            dataProcessor = data.data_20newsgroup
        elif self.args.dataset == "RCV-1":
            pass
        else:
            raise ValueError('Wrong Dataset')
        return dataProcessor

    def get_min_df(self):
        return self.args.min_df

    def get_max_df(self):
        return self.args.max_df

    def get_epochs(self):
        return self.args.epochs if not self.get_smoke_test() else 2

    def get_num_batches(self):
        return self.args.num_batches if not self.get_smoke_test() else 1

    def get_alpha(self):
        return self.args.alpha

    def get_learning_rate(self):
        return self.args.learning_rate

    def get_top_words(self):
        return self.args.top_words
