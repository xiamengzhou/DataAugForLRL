from torch import nn
from torch.autograd import Variable
import torch
import math

class Embeddings(nn.Module):
    def __init__(self, word_vec_size,
                 word_vocab_size,
                 word_padding_idx,
                 dropout=0):
        self.embedding_size = word_vec_size
        super(Embeddings, self).__init__()
        self.word_padding_idx = word_padding_idx
        embeddings = nn.Embedding(word_vocab_size, word_vec_size, padding_idx=word_padding_idx)
        pe = PositionalEncoding(dropout, self.embedding_size)
        self.embeddings = embeddings
        self.pe = pe


    def forward(self, input):
        emb = self.embeddings(input.squeeze(-1))
        emb = self.pe(emb)
        return emb

    def load_pretrained_vectors(self, emb_file, fixed):
        """Load in pretrained embeddings.

        Args:
          emb_file (str) : path to torch serialized embeddings
          fixed (bool) : if true, embeddings are not updated
        """
        if emb_file:
            pretrained = torch.load(emb_file)
            self.embeddings.weight.data.copy_(pretrained)
            if fixed:
                self.embeddings.weight.requires_grad = False

class PositionalEncoding(nn.Module):
    """
    Implements the sinusoidal positional encoding for
    non-recurrent neural networks.

    Args:
       dropout (float): dropout parameter
       dim (int): embedding size
    """

    def __init__(self, dropout, dim, max_len=5000):
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) *
                             -(math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        padding = torch.zeros(1, dim)
        if pe.is_cuda:
            padding = padding.cuda()
        pe = torch.cat([pe, padding])
        pe = pe.unsqueeze(1)

        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim
        self.max_len = max_len

    def forward(self, emb):
        # We must wrap the self.pe in Variable to compute, not the other
        # way - unwrap emb(i.e. emb.data). Otherwise the computation
        # wouldn't be watched to build the compute graph.
        # lattice_position = None
        emb = emb * math.sqrt(self.dim)
        emb = emb + Variable(self.pe[:emb.size(0)], requires_grad=False)
        emb = self.dropout(emb)
        return emb