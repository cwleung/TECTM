import itertools
import random
import re
import string
from torch.distributions.constraints import positive

from model import TETM
from util import environments, evaluations

torch.set_default_tensor_type('torch.FloatTensor')

env = environments.Environment()

# setting global variables
seed = env.get_seed()
torch.manual_seed(seed)
pyro.set_rng_seed(seed)

device = env.get_device()
num_topics = env.get_num_topics()  # if not smoke_test else 3
batch_size = env.get_num_batches()
learning_rate = env.get_learning_rate()
num_epochs = env.get_epochs()

main_dir = './model/'
if not os.path.exists(main_dir):
    os.mkdir(main_dir)
ckpt = os.path.join(main_dir, '/best')

def fetch_data(dataset):
    train_docs = []
    test_docs = []
    if dataset is "reuters":
        nltk.download('reuters')
        for doc_id in reuters.fileids():
            if doc_id.startswith("train"):
                train_docs.append(reuters.raw(doc_id))
            else:
                test_docs.append(reuters.raw(doc_id))
    else:
        train_data = fetch_20newsgroups(subset='train')
        test_data = fetch_20newsgroups(subset='test')
        train_docs = train_data.data
        test_docs=test_data.data

    train_docs = [re.findall(r'''[\w']+|[.,!?;-~{}`´_<=>:/@*()&'$%#"]''', doc)
                    for doc in train_docs]
    test_docs = [re.findall(r'''[\w']+|[.,!?;-~{}`´_<=>:/@*()&'$%#"]''', doc)
                    for doc in test_docs]
    return train_docs, test_docs

def evaluate(model, data):
    model.eval()
    with torch.no_grad():
        td = evaluations.get_topic_diversity(model.beta(), 25)
        tc = evaluations.get_topic_coherence(model.beta(), data)

        len_te = data.shape[0] // 2
        docs_t1, docs_t2 = data[:len_te].to(device), data[len_te:].to(device)
        beta = model.beta().to(device)
        theta = model.theta(docs_t1)[0]
        pred = torch.mm(theta, beta).to(device)
        torch.nan_to_num(pred, 0)
        ppl = evaluations.perplexity(pred.log(), docs_t2)
        print(f'perplexity: {ppl}')
    return tc, td, ppl
    # print('\nValidation set: Topic Coherence: {:.4f}, Topic Coherence: {}/{} ({:.0f}%)\n'.format(
    # val_loss, correct, len(dataloader.dataset), accuracy))


def get_doc_batch(loader, loader_iter):
    try:
        batch = next(loader_iter)
    except StopIteration:
        print('execption')
        loader_iter = iter(loader)
        batch = next(loader_iter)
    return batch, loader_iter


def data_prep(bsize):
    # Maximum / minimum document frequency
    max_df = env.get_max_df()
    min_df = env.get_min_df()
    # Read stopwords
    with open('../input/stopwords/stops.txt', 'r') as f:
        stops = f.read().split('\n')
    # Read data
    print('reading data...')
    data_set = env.get_dataset()
    init_docs_tr, init_docs_ts = fetch_data(data_set)

    def contains_punctuation(w):
        return any(char in string.punctuation for char in w)

    def contains_numeric(w):
        return any(char.isdigit() for char in w)

    # document preprocessing
    init_docs = init_docs_tr + init_docs_ts
    init_docs = [[w.lower() for w in init_docs[doc] if not contains_punctuation(w)] for doc in range(len(init_docs))]
    init_docs = [[w for w in init_docs[doc] if not contains_numeric(w)] for doc in range(len(init_docs))]
    init_docs = [[w for w in init_docs[doc] if len(w) > 1] for doc in range(len(init_docs))]
    init_docs = [" ".join(init_docs[doc]) for doc in range(len(init_docs))]
    print('counting document frequency of words...')
    cvectorizer = CountVectorizer(min_df=min_df, max_df=max_df, stop_words=frozenset(stops))
    cvz = cvectorizer.fit_transform(init_docs)
    # filter for the vocabulary
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
    # 3) create dataset
    print('creating dataset...')
    print(cvz.shape)
    cvz = torch.from_numpy(cvz.toarray())
    vocab = list(dict(sorted(cvectorizer.vocabulary_.items(), key=lambda x: x[1])).keys())
    tsSize = len(init_docs_ts)
    cvz = torch.column_stack((cvz.to(device), torch.zeros(cvz.shape[0], 3).to(device))).to(device)
    bow_tr, bow_te = cvz[:-tsSize], cvz[-tsSize:]
    dataset = SentencesDataset(sentences[:-tsSize], list(vocab), 20)  # seq_len
    kwargs = {'num_workers': 12, 'shuffle': True, 'drop_last': True, 'pin_memory': True, 'batch_size': bsize}
    data_loader = torch.utils.data.DataLoader(dataset, **kwargs)
    print(f'Vocabulary size: {len(vocab)}')
    return dataset, data_loader, bow_tr, bow_te, vocab


# calc loss for 1 iter
# input: batch data
# output: cross entropy loss
def calc_bert_loss(model, X, y, loss_model):
    tgt_mask = model.generate_square_subsequent_mask(X.size(0)).to(device)
    output = model(X, y, None, tgt_mask)
    # compute the cross entropy loss
    output_v = output.view(-1, output.shape[-1])
    target_v = y.view(-1, 1).squeeze()
    # return the loss, continue the problem
    loss = loss_model(output_v, target_v)
    return loss


def train(model, cvz, data_loader):
    model.train()
    running_prob = 0.0
    running_loss = 0.0

    loss_bert = nn.CrossEntropyLoss(ignore_index=data_loader.dataset.IGNORE_IDX)
    for data in data_loader:
        # infer
        masked_input = data['input'].to(device)
        masked_index = data['index'].to(device)
        masked_target = data['target'].to(device)

        if torch.cuda.is_available():
            masked_input = masked_input.cuda(non_blocking=True)
            masked_index = masked_index.cuda(non_blocking=True)
            masked_target = masked_target.cuda(non_blocking=True)

        batch = cvz[masked_index.long() - 1, :].to(device)
        batch = torch.nan_to_num(batch)
        recon_loss, kl_loss = model(batch)
        # IF: check bert is used
        bert_model = model.getTransformer()
        bert_loss = calc_bert_loss(bert_model, masked_input, masked_target, loss_bert)  # bert loss
        optim = torch.optim.Adam(list(model.parameters()) + list(bert_model.parameters()),
                                 weight_decay=env.get_regularization_parameter(),
                                 lr=env.get_learning_rate(),
                                 betas=(.9, .999))
        curr_loss = kl_loss + recon_loss + bert_loss
        # torch.nn.utils.clip_grad_norm_(list(getParams(model, batch)), 0.0)
        curr_loss.backward()
        optim.step()
        running_loss += curr_loss.detach().item()
        running_prob += bert_loss.detach().item()
        optim.zero_grad()
        model.zero_grad()
    return running_loss, running_prob


bar = trange(num_epochs)
dataset, data_loader, bow_tr, bow_te, vocab = data_prep(batch_size)
# shuffle
bow_te = bow_te[torch.randperm(len(bow_te)), :]

# initialize
prodLDA = TETM(
    vocab_size=len(dataset.vocab),
    num_topics=num_topics,
    hidden=100,
    dropout=env.get_drop_out_rate(),
    emb_type=env.get_embedding(),
    rho_size=env.get_embedding_size(),
    batchNorm=env.get_batch_normalization(),
    trans_layer=env.get_transformer_layer(),
    trans_head=env.get_transformer_head(),
    trans_dim=env.get_transformer_dim()
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
    running_loss, running_prob = train(prodLDA, bow_tr, data_loader)
    log_prob.append(running_prob)
    eblo_loss.append(-running_loss)
    # evaluate
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

# Final result
with open(ckpt, 'rb') as f:
    prodLDA = torch.load(f)
prodLDA = prodLDA.to(device)
prodLDA.decoder.fcrho.emb_type = env.get_embedding()
val_tc = evaluate(prodLDA, bow_te)

print(f'Best topic coherence: {val_tc}')
