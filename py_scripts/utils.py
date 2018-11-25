import sys
import torch

def load_model(model_path):
    sys.path.append("/usr2/home/mengzhox/11731_final")
    sys.path.append("/home/mengzhox/11731_final")
    model = torch.load(model_path, map_location=lambda storage, loc: storage)
    return model

def load_vocab(vocab_path):
    sys.path.append("/usr2/home/mengzhox/11731_final")
    sys.path.append("/home/mengzhox/11731_final")
    vocab = torch.load(vocab_path)
    return vocab[0][1], vocab[1][1]

def load_dict(d):
    di = {}
    f = open(d, "r").readlines()
    for line in f:
        w, freq = line.split()
        di[w] = freq
    return di