import numpy as np
import torch


def perplexity(pred, docs_te):
    """
    Calculate the perplexity, lower is better
    https://qpleple.com/perplexity-to-evaluate-topic-models/
    Parameters
    ----------
    :param docs_te: tensor (D,V)
        Testing set for the document 
    :param pred: log predictive distribution
    
    Return 
    ------
    :return ppl: int
        Perplexity of held-out word
    """
    # sum of word likelihood in a document
    likelihood = torch.sum(pred * docs_te, -1)
    # exp{-Likelihood/N tokens}
    return torch.exp(-torch.mean(likelihood / docs_te.sum(1)))


def get_df(data, wi, wj=None):
    """
    Obtain the document frequency
    :param data: document vocabulary matrix
    :param wi: word index w_i
    :param wj: word index w_j
    :return: document frequency for word w_i , w_i ∩ w_j
    """
    if wj is None:
        return torch.where(data[:, wi] > 0, 1, 0).sum(-1)
    else:
        df_wi = torch.where(data[:, wi] > 0, 1, 0)
        df_wj = torch.where(data[:, wj] > 0, 1, 0)
        return df_wj.sum(-1), (df_wi & df_wj).sum(-1)


def get_topic_coherence(beta, data):
    """
    Calculate and print Topic coherence (TC) see (Minno. 2011)
    :param beta:
        Topic-word distribution
    :param data:
        Dataset document-vocab frequency matrix
    :return: None
    """
    D = len(data)
    TC = []
    num_topics = len(beta)
    counter = 0
    for k in range(num_topics):
        top_10 = list(torch.flip(beta[k].argsort()[-11:], [0]))
        TC_k = 0
        counter = 0
        for i, word in enumerate(top_10):
            D_wi = get_df(data, word)
            j = i + 1
            tmp = 0
            while len(top_10) > j > i:
                D_wj, D_wi_wj = get_df(data, word, top_10[j])
                if D_wi_wj == 0:
                    f_wi_wj = -1
                else:
                    f_wi_wj = -1 + (np.log(D_wi) + np.log(D_wj) - 2.0 * np.log(D)) / (np.log(D_wi_wj) - np.log(D))
                tmp += f_wi_wj
                j += 1
                counter += 1
            TC_k += tmp
        TC.append(TC_k)
    TC = np.mean(TC) / counter
    print('Topic coherence is: {}'.format(TC))
    return TC


def get_topic_diversity(beta, topk=25):
    """
    Calculate and print the unique word ratio among topics in their top-25 words
    :param beta: topic word distributions
    :param topk: top-k topic words
    :return: None
    """
    num_topics = beta.shape[0]
    list_w = torch.zeros((num_topics, topk))
    for k in range(num_topics):
        idx = torch.flip(beta[k, :].argsort()[-topk:], [-1])
        list_w[k, :] = idx
    n_unique = len(torch.unique(list_w))
    TD = n_unique / (topk * num_topics)
    print('Topic diversity is: {}'.format(TD))
    return TD


if __name__ == '__main__':
    pass
