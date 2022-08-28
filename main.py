import argparse
import os
import re
import string

import nltk
import pandas as pd
import pyro
import torch
from nltk.corpus import reuters
from pyro.infer import Trace_ELBO
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from torch import nn
from tqdm import trange

from data.sentence_dataset import SentencesDataset
from model.TECTM import TECTM
from util.evaluations import get_npmi, get_topic_diversity, get_topic_coherence, perplexity

parser = argparse.ArgumentParser(description="Transformer Embedded Correlated Topic Model")
# Environment setting
parser.add_argument("-st", "--smoke-test", default=False, type=bool)
parser.add_argument("-s", "--seed", default=42, type=int)
parser.add_argument("-dd", "--data-dir", default="./", type=str)
# Topic model-related
parser.add_argument("-nt", "--num-topics", default=20, type=int)
# NN-related
parser.add_argument("-hl", "--inf-hidden", default=800, type=int)
parser.add_argument("-dr", "--dropout", default=0.0, type=float)
parser.add_argument("-af", "--batch-norm", default=True, type=bool)
# Dataset-related
parser.add_argument("-ds", "--dataset", default="20newsgroups", type=str)
parser.add_argument("-nd", "--min-df", default=50, type=int)
parser.add_argument("-xd", "--max-df", default=0.7, type=float)
# Training-related
parser.add_argument("-e", "--epoch", default=1000, type=int)
parser.add_argument("-bs", "--batch-size", default=1024, type=int)
parser.add_argument("-emb", "--embedding", default="Transformer", type=str)
parser.add_argument("-embsize", "--embedding-size", default=300, type=int)
parser.add_argument("-o", "--alpha", default=1, type=int)
parser.add_argument("-lr", "--learning-rate", default=2e-3, type=float)
parser.add_argument("-l2", "--reg-rate", default=1e-6, type=float)
# CTM
parser.add_argument("-lkj", "--lkj-corr", default=True, type=bool)
# Transformer
parser.add_argument("-th", "--trans_heads", default=8, type=int)
parser.add_argument("-tl", "--trans_layers", default=4, type=int)
parser.add_argument("-td", "--trans_dim", default=1024, type=int)
parser.add_argument("-sl", "--seq-len", default=20, type=int)
args = parser.parse_args()

DATA_DIR = args.data_dir


smoke_test = args.smoke_test
seed = args.seed
torch.manual_seed(seed)
pyro.set_rng_seed(seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_tensor_type('torch.FloatTensor')

num_topics = args.num_topics if not smoke_test else 3
batch_size = args.batch_size
learning_rate = args.learning_rate
num_epochs = args.epoch if not smoke_test else 3
emb_type = args.embedding

pyro.clear_param_store()

main_dir = './model/'
if not os.path.exists(main_dir):
    os.mkdir(main_dir)
ckpt = os.path.join(main_dir, 'best')


def fetch_data(dataset):
    train_docs = []
    test_docs = []
    if dataset == "reuters":
        nltk.download('reuters')
        for doc_id in reuters.fileids():
            if doc_id.startswith("train"):
                train_docs.append(reuters.raw(doc_id))
            else:
                test_docs.append(reuters.raw(doc_id))
    elif dataset == '20newsgroups':
        train_data = fetch_20newsgroups(subset='train')
        test_data = fetch_20newsgroups(subset='test')
        train_docs = train_data.data
        test_docs = test_data.data
    elif dataset == 'nips':
        data = pd.read_csv(os.path.join(DATA_DIR, '/papers.csv'))
        data = data[~data.paper_text.isnull()]
        docs = data.paper_text.values
        train_docs, test_docs = train_test_split(docs, test_size=0.2, random_state=args.seed)
    elif dataset == 'undebates':
        data = pd.read_csv(os.path.join(DATA_DIR, '/un-general-debates.csv'))
        data = data[~data.text.isnull()]
        docs = data.text.values
        train_docs, test_docs = train_test_split(docs, test_size=0.2, random_state=args.seed)
    else:
        raise NotImplementedError('Wrong dataset')
    return train_docs, test_docs


def calc_log_sum(model, data, num_particles):
    prob_w = torch.tensor(data=0., dtype=torch.float64).to(device)
    for _ in range(num_particles):
        if torch.cuda.is_available():
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        guide_trace = pyro.poutine.trace(model.guide).get_trace(data)
        model_trace = pyro.poutine.trace(pyro.poutine.replay(
            model.model, trace=guide_trace)).get_trace(data)
        model_trace.log_prob_sum()
        torch.set_default_tensor_type('torch.FloatTensor')
        prob_w_tmp = torch.tensor(data=0.).to(device)
        for key in model_trace.nodes:
            if key == "obs":
                prob_w_tmp += model_trace.nodes[key]["log_prob_sum"]
        prob_w_tmp = torch.tensor(float(prob_w_tmp), dtype=torch.float64).to(device)
        prob_w += prob_w_tmp
    return prob_w


def getParams(model, data):
    # sample latent variables from guide_trace
    guide_trace = pyro.poutine.trace(model.guide).get_trace(data)
    model_trace = pyro.poutine.trace(pyro.poutine.replay(
        model.model, trace=guide_trace)).get_trace(data)
    # Custom ELBO
    # https://github.com/pyro-ppl/pyro/issues/958
    main_params = set()
    for tr in (model_trace, guide_trace):
        for node in tr.nodes.values():
            if node["type"] == "param":
                main_params.add(node["value"].unconstrained())
                # print(node["value"].unconstrained().device)
    return main_params


def recon_loss(model, bows):
    beta = model.beta().to(device)
    theta = model.theta(bows)[0]
    pred = torch.mm(theta, beta).to(device)
    pred = torch.log(pred + 1e-6)
    recon_loss = -(pred * bows).sum(1)
    return recon_loss


def evaluate(model, data):
    model.eval()
    with torch.no_grad():
        td = get_topic_diversity(model.beta(), 25)
        tc = get_topic_coherence(model.beta(), data)
        beta = model.beta().to(device)
        theta = model.guide(data)[0].nan_to_num()
        loglik = torch.mm(theta, beta).to(device)
        torch.nan_to_num(loglik, 0)
        ppl = perplexity(loglik.log(), data)
        print(f'perplexity: {ppl}')
    return tc, td, ppl


def data_prep(bsize):
    # Maximum / minimum document frequency
    max_df = args.max_df
    min_df = args.min_df  # choose desired value for min_df

    # Read stopwords
    with open(os.path.join(DATA_DIR, 'stops.txt'), 'r') as f:
        stops = f.read().split('\n')

    # Read data
    print('reading data...')
    init_docs_tr, init_docs_ts = fetch_data(args.dataset)
    print(f'train, test size: {len(init_docs_tr)}, {len(init_docs_ts)}')

    def contains_punctuation(w):
        return any(char in string.punctuation for char in w)

    def contains_numeric(w):
        return any(char.isdigit() for char in w)

    init_docs_tr = [re.findall(r'''[\w']+|[.,!?;-~{}`´_<=>:/@*()&'$%#"]''', init_docs_tr[doc])
                    for doc in range(len(init_docs_tr))]
    init_docs_ts = [re.findall(r'''[\w']+|[.,!?;-~{}`´_<=>:/@*()&'$%#"]''', init_docs_ts[doc])
                    for doc in range(len(init_docs_ts))]

    # document preprocessing
    init_docs = init_docs_tr + init_docs_ts
    init_docs = [[w.lower() for w in init_docs[doc] if not contains_punctuation(w)] for doc in range(len(init_docs))]
    init_docs = [[w for w in init_docs[doc] if not contains_numeric(w)] for doc in range(len(init_docs))]
    init_docs = [[w for w in init_docs[doc] if len(w) > 1] for doc in range(len(init_docs))]
    init_docs = [" ".join(init_docs[doc]) for doc in range(len(init_docs))]
    print('counting document frequency of words...')
    cvectorizer = CountVectorizer(min_df=min_df, max_df=max_df, stop_words=frozenset(stops))
    cvz = cvectorizer.fit_transform(init_docs)

    no_cnt = 0
    docs = []
    for doc in init_docs:
        tmp_doc = []
        for w in doc.split(" "):
            if w in cvectorizer.vocabulary_:
                tmp_doc.append(w)
            else:
                no_cnt += 1
        tmp_doc = " ".join(tmp_doc)
        docs.append(tmp_doc)

    # 1) Remove empty sentences
    sentences = list(filter(lambda s: len(s) > 0, docs))

    # 2) tokenize sentences (can be done during training, you can also use spacy udpipe)
    print('tokenizing sentences...')
    special_chars = ',?;.:/*!+-()[]{}"\'&'
    sentences = [re.sub(f'[{re.escape(special_chars)}]', ' \g<0> ', s).split(' ') for s in sentences]
    sentences = [[w for w in s if len(w)] for s in sentences]
    # 4) create dataset
    print('creating dataset...')
    print(cvz.shape)
    cvz = torch.from_numpy(cvz.toarray())
    vocab = list(dict(sorted(cvectorizer.vocabulary_.items(), key=lambda x: x[1])).keys())

    tsSize = len(init_docs_ts)
    cvz = torch.column_stack((cvz.to(device), torch.zeros(cvz.shape[0], 3).to(device))).to(device)
    bow_tr, bow_te = cvz[:-tsSize], cvz[-tsSize:]

    dataset = SentencesDataset(sentences[:-tsSize], list(vocab), args.seq_len)  # seq_len
    kwargs = {
        'shuffle': True, 'pin_memory': True, 'batch_size': bsize
    }
    data_loader = torch.utils.data.DataLoader(dataset, **kwargs)

    print(f'Vocabulary size: {len(vocab)}')
    return dataset, data_loader, bow_tr, bow_te, vocab


def calc_bert_loss(model, X, y, loss_model):
    output = model(X)
    output_v = output.view(-1, output.shape[-1])
    target_v = y.view(-1, 1).squeeze()
    loss = loss_model(output_v, target_v)
    return loss


def train(model, cvz, data_loader, emb_type='NN'):
    model.train()
    running_prob = 0.0
    running_loss = 0.0

    loss_elbo = Trace_ELBO().differentiable_loss
    loss_bert = nn.CrossEntropyLoss(ignore_index=data_loader.dataset.IGNORE_IDX)
    # batch for transformer
    for data in data_loader:
        # weight = len(cvz) / len(data)
        # infer
        masked_input = data['input'].to(device)
        masked_index = data['index'].to(device)
        masked_target = data['target'].to(device)
        # loss
        batch = cvz[masked_index.long() - 1, :].to(device)
        batch = torch.nan_to_num(batch)  # remove nan
        if torch.cuda.is_available():
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        kl_loss = loss_elbo(model.model, model.guide, batch)  # loss function
        torch.set_default_tensor_type('torch.FloatTensor')

        _, kld_theta = model.guide(batch)
        optim = torch.optim.Adam(list(getParams(model, batch)) + list(model.parameters()),
                                 lr=args.learning_rate, weight_decay=args.reg_rate)
        curr_loss = kl_loss + kld_theta

        if emb_type == 'Transformer':
            bert_model = model.getTransformer()
            bert_loss = calc_bert_loss(bert_model, masked_input, masked_target, loss_bert)
            curr_loss += bert_loss

        curr_loss.backward()
        optim.step()

        running_loss += curr_loss.item()
        running_prob += calc_log_sum(model, batch, 2)
        optim.zero_grad()
        model.zero_grad()
    return running_loss, running_prob


bar = trange(num_epochs)
dataset, data_loader, bow_tr, bow_te, vocab = data_prep(batch_size)

# initialize
# trans-layer, trans-dim, num_heads
prodLDA = TECTM(
    vocab_size=len(dataset.vocab),
    num_topics=num_topics,
    hidden=args.inf_hidden if not smoke_test else 10,
    dropout=args.dropout,
    useEmbedding=True,
    trainEmbedding=True,
    emb_type=emb_type,
    rho_size=args.embedding_size,
    LKJChol=args.lkj_corr,
    trans_heads=args.trans_heads,
    trans_layers=args.trans_layers,
    trans_dim=args.trans_dim,
).to(device)

# ELBO
eblo_loss = []
log_prob = []

td_trace = []
tc_trace = []
ppl_trace = []

best_tc = 0
val_tc = 0
for epoch in bar:
    running_loss, running_prob = train(prodLDA, bow_tr, data_loader, args.embedding)
    eblo_loss.append(running_loss)
    log_prob.append(-running_prob)
    # evaluate
    prodLDA.to(device)
    tc, td, ppl = evaluate(prodLDA, bow_te)

    td_trace.append(td)
    tc_trace.append(tc)
    ppl_trace.append(ppl)

    if val_tc > best_tc:
        with open(ckpt, 'wb') as f:
            torch.save(prodLDA, f)
        best_epoch = epoch
        best_tc = val_tc

    bar.set_postfix(elbo_loss='{:.2e}'.format(running_loss),
                    bert_loss='{:.2e}'.format(running_prob))

with open(ckpt, 'rb') as f:
    prodLDA = torch.load(f)
prodLDA = prodLDA.to(device)
prodLDA.decoder.fcrho.emb_type = 'Transformer'
val_tc = evaluate(prodLDA, bow_te)

print(f'Best topic coherence: {val_tc}')

theta = prodLDA.guide(bow_te)[0]
beta = prodLDA.beta()
((-(torch.mm(theta.log(), beta) * bow_te).sum(1)) / bow_te.sum(1)).nan_to_num().mean().exp()

bow_te.sum(1).unsqueeze(1).squeeze()

theta = prodLDA.guide(bow_te.float())[0].cpu().detach().numpy()
top_5 = theta.sum(0).argsort()[-7:]
get_npmi(prodLDA.beta(), bow_te.float(), vocab, top_5)
