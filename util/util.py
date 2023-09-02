import numpy as np
import pandas as pd


def data_import(filepath):
    """
    > Probably used for beta, may be used for embedding
    Import data values into numpy array
    :param filepath: file path of data to be imported
    :return: data: numpy.array
    """
    df = pd.read_csv(filepath)
    return df.values


def data_export(output_path, data):
    """
    Export data array in numpy array format
    :param output_path: output path to export the array
    :param data: numpy.array
    :return: True
    """
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    return True


def print_top_words(beta, feature_names, n_top_words=10):
    """
    Print out top topic words

    :param beta: topic-word distributino
    :param feature_names: corpus vocab list
    :param n_top_words: int
        Number of top words to be selected
    :return: None
    """
    for i in range(len(beta)):
        print(
            ("Topic #%d: " % i)
            + " ".join([feature_names[j] for j in beta[i].argsort()[: -n_top_words - 1: -1]])
        )


def word_export(output_path, beta, vocab, top_display=10):
    """
    Export top word and its word distribution
    :param output_path:  write path
    :param beta: beta array (K,V)
    :param vocab: corpus vocab (V,)
    :param top_display: top word to be written
    :return: None
    """
    num_topics = beta.shape[0]
    with open(output_path, 'w') as output:
        for topic in range(num_topics):
            output.write("==========\t%d\t==========\n" % topic)
            word_dist = beta[topic]
            for word_idx in np.argsort(word_dist)[:-top_display - 1:-1]:
                output.write("%s\t%g\n" % (vocab[word_idx], word_dist[word_idx]))


def stable_softmax(x, dim=0):
    """
    Numerically stable softmax function
    :param x: input tensor
    :param dim: dimension to perform softmax
    :return: softmaxed tensor
    """
    return (x.exp() / x.exp().sum(dim=dim, keepdim=True)).clamp(min=1e-6)


if __name__ == '__main__':
    a = np.arange(10)
    arr = a * a[:, np.newaxis]

    data_export('example.csv', arr)
    data = data_import('example.csv')
    print(data)
    vocab = np.arange(10) * 10
    word_export('example2.csv', data, vocab)
