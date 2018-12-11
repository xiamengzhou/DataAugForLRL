import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import time
from torch.nn.functional import softmax
from collections import defaultdict
import numpy as np
import random
random.seed(0)
import sys
import pickle as pkl

data_dir = ".."
sys.path.append('..')
corpus_dict = torch.load('data/corpus_dict.pt')
#corpus_dict = corpus.dictionary
#torch.save(corpus_dict, '../data/corpus_dict.pt')
#exit()
model = torch.load('language_model/EXP-20181201-223642/model.pt')
model.rnn.flatten_parameters()
criterion = nn.CrossEntropyLoss()
bptt = 70
test_batch_size = 1
device = torch.device("cuda")

def tokenize_sents(sents):
        ids = []
        for idx, words in enumerate(sents):
            if idx > 0 and idx % 200000 == 0:
                print('    line {}'.format(idx))
            for word in words:
                ids.append(corpus_dict.get_idx(word))

        return torch.LongTensor(ids)

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def get_batch(source, i):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target

def evaluate_nll(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    ntokens = len(corpus_dict)

    hidden = model.init_hidden(test_batch_size)
    probs = []
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, bptt):
            data, targets = get_batch(data_source, i)
            output, hidden = model(data, hidden)
            output_flat = output.view(-1, ntokens)
            output_softmax = softmax(output_flat)
            #print(type(hidden[0]), hidden[0].size())
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
            word_prob = output_softmax.data.view(-1).index_select(0, targets.data + torch.arange(0, output_flat.size(0)).long().cuda() * ntokens)
            probs += [word_prob]
            hidden = repackage_hidden(hidden)
    return total_loss / (len(data_source) - 1), torch.cat(probs, 0)


def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)


def get_true_nll(text):
    text = tokenize_sents([text.split()])
    data = batchify(text, test_batch_size)
    nll, probs = evaluate_nll(data)
    return nll, probs

def count_words(st):
    ss = st.replace("\n", " ")
    while "  " in ss:
        ss = ss.replace("  ", " ")
    return ss.count(" ")

def evaluate_P_true(st):
    n = count_words(st)
    true_ppl, probs = get_true_nll(st)
    return true_ppl, probs, n

def read_docs(contents):
    cnt = 1
    st = ""
    docs = []
    while True:
        if cnt >= len(contents) or contents[cnt].strip() == "~~~~~":
            docs += [st]
            if cnt >= len(contents):
                break
            st = ""
        else:
            st += contents[cnt].replace('\n', '')
        cnt += 1
    return docs

def load_data(data_dir, sub_set):
    print("loading data of {}".format(sub_set))
    with open(data_dir + "/{}Set.txt".format(sub_set), "r") as inf:
        contents = inf.readlines()
    with open(data_dir + "/{}SetLabels.dat".format(sub_set), "r") as label_inf:
        labels = list(map(int, label_inf.readlines()))
    docs = read_docs(contents)
    return docs, labels

def load_test_data(data_dir, sub_set):
    print("loading data of {}".format(sub_set))
    with open(data_dir + "/{}Set.txt".format(sub_set), "r") as inf:
        contents = inf.readlines()
    docs = read_docs(contents)
    return docs

def get_feature(docs):
    data = []
    n_bins = 20
    # bins = np.linspace(0, 1, n_bins+1)
    bins = np.zeros(n_bins+1)
    bins[1:] = np.logspace(-15, 0, n_bins, base=4)
    for i, st in enumerate(docs):
        if i % 100 == 0:
            print("extracting features of doc %d" % i)
        true_ppl, probs, n = evaluate_P_true(st)
        hist_feature, _ = np.histogram(probs.cpu().numpy(), bins=bins)
        hist_feature = hist_feature.astype(np.float32) / n
        data.append([true_ppl, probs, hist_feature])
    #print(len(data))
    return data
'''
if __name__ == "__main__":
    dev_docs, dev_labels = load_data(data_dir, "development")
    train_docs, train_labels = load_data(data_dir, "training")
    test_docs = load_test_data(data_dir, "test")
    print(len(train_docs))
    data = get_feature(test_docs)
    with open('test_nn_lm_his_doc.pkl', 'wb') as f:
        pkl.dump(data, f)
    data = get_feature(train_docs)
    with open('train_nn_lm_his_doc.pkl', 'wb') as f:
        pkl.dump(data, f)
    dev_feature = get_feature(dev_docs)
    with open('dev_nn_lm_his_doc.pkl', 'wb') as f:
        pkl.dump(dev_feature, f)
'''


