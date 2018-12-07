import numpy as np
import torch as t
import torch.nn as nn

from torch import LongTensor as LT
from torch import FloatTensor as FT


class Bundler(nn.Module):

    def forward(self, data):
        raise NotImplementedError

    def forward_i(self, data):
        raise NotImplementedError

    def forward_o(self, data):
        raise NotImplementedError


class Word2Vec(Bundler):
    def __init__(self, vocab_size=20000, embedding_size=256, padding_idx=0,
                 vectors_path=""):
        super(Word2Vec, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

        self.ovectors = nn.Embedding(vocab_size, embedding_size, padding_idx=padding_idx)
        self.ovectors.weight = t.load(vectors_path)
        self.ovectors.weight.requires_grad = False

    def forward(self, data):
        return self.forward_i(data)

    def forward_i(self, data):
        v = LT(data)
        v = v.cuda() if self.ivectors.weight.is_cuda else v
        return self.ivectors(v)

    def forward_o(self, data):
        v = LT(data)
        v = v.cuda() if self.ovectors.weight.is_cuda else v
        return self.ovectors(v)

class SGNS(nn.Module):

    def __init__(self, embedding, vocab_size=20000, n_negs=20, weights=None, context_size=5):
        super(SGNS, self).__init__()
        self.embedding = embedding
        self.vocab_size = vocab_size
        self.n_negs = n_negs
        self.weights = None
        self.context_size = context_size
        if weights is not None:
            wf = np.power(weights, 0.75)
            wf = wf / wf.sum()
            self.weights = FT(wf)

    def forward(self, iword_emb, owords):
        # iwords_emb: batch * src_len * dim
        # owrds: batch * src_len
        batch_size, src_len, dim = iword_emb.size()
        _, src_len_o = owords.size
        assert src_len == src_len_o
        if self.weights is not None:
            nwords = t.multinomial(self.weights,
                                   batch_size * src_len * self.context_size * self.n_negs,
                                   replacement=True).view(batch_size*src_len, -1)
        else:
            nwords = FT(batch_size,
                        self.context_size * self.n_negs).uniform_(0, self.vocab_size - 1).long()

        a = []
        for i in range(owords.size(1)):
            a.append(t.cat([owords[:, max(i-self.context_size, 0)],
                            owords[:, min(i+self.context_size, src_len)]], 1).unsqueeze(1))
        owords_win = t.cat(a, 1).view(-1, self.context_size*2) # batch * src_len * (win_size*2)
        owords_emb = self.embedding.forward_o(owords_win) # (batch*src_len) * (win_size*2) * dim
        iword_emb = iword_emb.view(-1, dim).unsqueeze(-1) # (batch*src_len) * dim * 1
        nwords_emb = self.embedding.forward_o(nwords) #  (batch*src_len) * (win_size*n_neg) * dim

        oloss = t.bmm(owords_emb, iword_emb).squeeze().sigmoid().log().mean(1)
        nloss = t.bmm(nwords_emb, iword_emb).squeeze().sigmoid().log().view(-1, self.context_size, self.n_negs).sum(2).mean(1)
        return -(oloss + nloss).mean()