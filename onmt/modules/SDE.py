from torch import nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np


class charEmbedder(nn.Module):
    def __init__(self, hparams, char_vsize, trg=False, *args, **kwargs):
        super(charEmbedder, self).__init__()
        self.hparams = hparams
        self.trg = trg
        if self.hparams.char_ngram_n > 0:
            if self.hparams.d_char_vec is not None:
                self.char_emb_proj = nn.Linear(char_vsize, self.hparams.d_char_vec, bias=False)
                if self.hparams.cuda:
                    self.char_emb_proj = self.char_emb_proj.cuda()
            else:
                if self.hparams.compute_ngram:
                    ones = torch.ones(len(self.i2w_base), self.hparams.d_word_vec).uniform_(-self.hparams.init_range, self.hparams.init_range)
                    if self.hparams.cuda: ones = ones.cuda()
                    self.emb_param = nn.Parameter(ones, requires_grad=True)
                    emb_matrix = self.emb_param[0] + self.emb_param[1]
                    self.emb_matrix = torch.cat([self.emb_param,emb_matrix.unsqueeze(0)], dim=0)
                    if self.hparams.cuda: self.emb_matrix = self.emb_matrix.cuda()
                else:
                    self.char_emb_proj = nn.Linear(char_vsize, self.hparams.d_word_vec, bias=False)
                    if self.hparams.cuda:
                        self.char_emb_proj = self.char_emb_proj.cuda()
        elif self.hparams.char_input:
            self.char_emb = nn.Embedding(char_vsize, self.hparams.d_char_vec, padding_idx=hparams.pad_id)
            if self.hparams.cuda:
                self.char_emb = self.char_emb.cuda()
            if self.hparams.char_input == 'cnn':
                # in: (batch_size, d_char_vec, char_len); out: (batch_size, out_channels, char_len_out)
                self.conv_list = []
                assert sum(self.hparams.out_c_list) == self.hparams.d_word_vec
                for out_c, k in zip(self.hparams.out_c_list, self.hparams.k_list):
                    self.conv_list.append(nn.Conv1d(self.hparams.d_char_vec, out_channels=out_c, kernel_size=k, padding=k // 2))
                self.conv_list = nn.ModuleList(self.conv_list)
                # global max pool using functional
                # in: (batch_size, out_channels, char_len_out); out: (batch_size, out_channels, 1)
                if self.hparams.highway:
                    self.highway_g = nn.Linear(self.hparams.d_word_vec, self.hparams.d_word_vec)
                    self.highway_h = nn.Linear(self.hparams.d_word_vec, self.hparams.d_word_vec)
                    if self.hparams.cuda:
                        self.highway_g = self.highway_g.cuda()
                        self.highway_h = self.highway_h.cuda()
                if self.hparams.cuda:
                    self.conv_list = self.conv_list.cuda()
            elif self.hparams.char_input == 'bi-lstm':
                self.lstm_layer = nn.LSTM(self.hparams.d_word_vec,
                                          self.hparams.d_word_vec // 2,
                                          bidirectional=True,
                                          dropout=hparams.dropout,
                                          batch_first=True)
                if self.hparams.cuda: self.lstm_layer = self.lstm_layer.cuda()
        if self.hparams.sep_char_proj and not trg:
            self.sep_proj_list = []
            for i in range(self.hparams.lan_size):
                if self.hparams.d_char_vec is not None:
                    self.sep_proj_list.append(nn.Linear(self.hparams.d_char_vec, self.hparams.d_word_vec, bias=False))
                else:
                    self.sep_proj_list.append(nn.Linear(self.hparams.d_word_vec, self.hparams.d_word_vec, bias=False))
            self.sep_proj_list = nn.ModuleList(self.sep_proj_list)
            if self.hparams.cuda: self.sep_proj_list = self.sep_proj_list.cuda()
        elif trg and self.hparams.d_char_vec:
            self.trg_proj = nn.Linear(self.hparams.d_char_vec, self.hparams.d_word_vec, bias=False)
            if self.hparams.cuda: self.trg_proj = self.trg_proj.cuda()

    def forward(self, x_train_char):
        """Performs a forward pass.
        Args:
        Returns:
        """
        if self.hparams.char_ngram_n > 0:
            ret = []
            for idx, (x_char_sent, lengths) in enumerate(x_train_char):
                emb = Variable(x_char_sent.to_dense(), requires_grad=False)
                if self.hparams.cuda: emb = emb.cuda()
                #if self.hparams.d_char_vec is not None:
                #  emb = self.char_down_proj(emb)
                x_char_sent = torch.tanh(self.char_emb_proj(emb))

                if self.hparams.sep_char_proj and not self.trg:
                    pass
                    # x_char_sent = torch.tanh(self.sep_proj_list[[langid]](x_char_sent))
                ret.append(x_char_sent)
            if not self.hparams.semb == 'mlp':
                char_emb = torch.stack(ret, dim=0)
            else:
                char_emb = x_train_char
        elif self.hparams.char_input == 'sum':
            # [batch_size, max_len, char_len, d_word_vec]
            char_emb = self.char_emb(x_train_char)
            char_emb = char_emb.sum(dim=2)
        return char_emb


class QueryEmb(nn.Module):
    def __init__(self, hparams, vocab_size, emb=None):
        super(QueryEmb, self).__init__()
        self.hparams = hparams
        self.vocab_size = vocab_size
        self.dropout = nn.Dropout(hparams.dropout)
        if emb is None:
            self.emb_matrix = nn.Parameter(
                torch.ones(vocab_size, self.hparams.d_word_vec).uniform_(-self.hparams.init_range,
                                                                         self.hparams.init_range), requires_grad=True)
        else:
            self.vocab_size = emb.size(0)
            self.emb_matrix = emb
        self.softmax = nn.Softmax(dim=-1)
        self.hparams = hparams
        self.temp = np.power(hparams.d_model, 0.5)
        if self.hparams.semb == 'mlp':
            self.w_trg = nn.Linear(self.hparams.d_word_vec, self.hparams.d_word_vec)
            self.w_att = nn.Linear(self.hparams.d_word_vec, 1)
            if self.hparams.cuda:
                self.w_trg = self.w_trg.cuda()
                self.w_att = self.w_att.cuda()
        elif self.hparams.semb == 'linear':
            self.w_trg = nn.Linear(self.hparams.d_word_vec, self.vocab_size)
        if hasattr(self.hparams, 'char_gate') and self.hparams.char_gate:
            self.char_gate = nn.Linear(self.hparams.d_word_vec * 2, 1)
            if self.hparams.cuda: self.char_gate = self.char_gate.cuda()

    def forward(self, q, file_idx=None, x_rank=None):
        """
        dot prodct attention: (q * k.T) * v
        Args:
          q: [batch_size, d_q] (target state)
          k: [len_k, d_k] (source enc key vectors)
          v: [len_v, d_v] (source encoding vectors)
          attn_mask: [batch_size, len_k] (source mask)
        Return:
          attn: [batch_size, d_v]
        """
        if self.hparams.semb == 'mlp':
            max_len, d_q = q[0].size()
            # (batch_size, max_len, d_word_vec, vocab_size)
            ctx = []
            for idx, qi in enumerate(q):
                attn_weight = self.w_att(torch.tanh(
                    self.emb_matrix.view(1, self.vocab_size, self.hparams.d_word_vec) + self.w_trg(qi).unsqueeze(
                        1))).squeeze(2)
                # (max_len, vocab_size)
                # attn_weight = self.w_att(attn_hidden.permute(0, 1, 3, 2)).squeeze(3)
                attn_weight = F.softmax(attn_weight, dim=-1)
                attn_weight = self.dropout(attn_weight)
                c = torch.mm(attn_weight, self.emb_matrix)
                ctx.append(c)
            ctx = torch.stack(ctx, dim=0)
        elif self.hparams.semb == 'dot_prod':
            batch_size, max_len, d_q = q.size()
            # [batch_size, max_len, vocab_size]
            attn_weight = torch.bmm(q,
                                    self.emb_matrix.transpose(0, 1).unsqueeze(0).expand(batch_size, -1, -1)) / self.temp
            if self.hparams.semb_num > 1:
                batch_size, max_len, vocab_size = attn_weight.size()
                seg_vocab_size = vocab_size // self.hparams.semb_num
                attn_mask = np.ones([batch_size * max_len, vocab_size])
                x_rank = np.array(x_rank)
                x_rank = x_rank.reshape(-1)
                for i in range(self.hparams.semb_num):
                    attn_mask[x_rank == i, i * seg_vocab_size:(i + 1) * seg_vocab_size] = 0
                attn_mask = torch.ByteTensor(attn_mask)
                attn_mask = attn_mask.view(batch_size, max_len, vocab_size)
                if self.hparams.cuda: attn_mask = attn_mask.cuda()
                attn_weight.data.masked_fill_(attn_mask, -self.hparams.inf)
            attn_weight = self.softmax(attn_weight)
            attn_weight = self.dropout(attn_weight)
            if self.hparams.semb_num > 1:
                attn_weight.data.masked_fill_(attn_mask, 0)
            # [batch_size, max_len, d_emb_dim]
            ctx = torch.bmm(attn_weight, self.emb_matrix.unsqueeze(0).expand(batch_size, -1, -1))
        elif self.hparams.semb == 'linear':
            batch_size, max_len, d_q = q.size()
            # [batch_size, max_len, vocab_size]
            attn_weight = self.w_trg(q)
            ctx = torch.bmm(attn_weight, self.emb_matrix.unsqueeze(0).expand(batch_size, -1, -1))
        elif self.hparams.semb == 'zero':
            batch_size, max_len, d_q = q.size()
            ctx = Variable(torch.zeros(batch_size, max_len, d_q))
            if self.hparams.cuda: ctx = ctx.cuda()
        if hasattr(self.hparams, 'src_no_char') and self.hparams.src_no_char:
            pass
        else:
            if hasattr(self.hparams, 'char_gate') and self.hparams.char_gate:
                g = F.sigmoid(self.char_gate(torch.cat([ctx, q], dim=-1)))
                ctx = ctx * g + q * (1 - g)
            else:
                ctx = ctx + q
        return ctx
