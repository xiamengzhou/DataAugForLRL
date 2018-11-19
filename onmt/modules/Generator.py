from torch import nn
import torch


class MosGenerator(nn.Module):
    def __init__(self, rnn_size, n_experts, target_vocab_len):
        super(MosGenerator, self).__init__()
        self.target_vocab_len = target_vocab_len
        self.latent = nn.Sequential(nn.Linear(rnn_size, n_experts * rnn_size), nn.Tanh())
        self.token_weight = nn.Linear(rnn_size, self.target_vocab_len)
        self.prior = nn.Linear(rnn_size, n_experts, bias=False)
        self.rnn_size = rnn_size
        self.n_experts = n_experts

    def forward(self, input):
        latent = self.latent(input)
        logit = self.token_weight(latent.view(-1, self.rnn_size))
        prob = nn.functional.softmax(logit, dim=-1).view(-1, self.n_experts, self.target_vocab_len)
        prior = self.prior(input).contiguous().view(-1, self.n_experts)
        prior = nn.functional.softmax(prior, dim=1)
        prior = prior.unsqueeze(2).expand_as(prob)
        prob = (prob * prior).sum(1)
        log_prob = torch.log(prob.add_(1e-8))
        return log_prob


class Generator(nn.Module):
    def __init__(self, rnn_size, target_vocab_len, mos_dropout=0, n_experts=10, mos=False, lexicon=False):
        super(Generator, self).__init__()
        self.target_vocab_len = target_vocab_len
        self.rnn_size = rnn_size
        self.mos = mos

        if self.mos:
            self.n_experts = n_experts
            self.latent = nn.Sequential(nn.Linear(rnn_size, n_experts * rnn_size), nn.Tanh())
            self.token_weight = nn.Linear(rnn_size, self.target_vocab_len)
            self.prior = nn.Linear(rnn_size, n_experts, bias=False)
            self.rnn_size = rnn_size
            self.mos_dropout = nn.Dropout(mos_dropout)
        else:
            self.latent = nn.Linear(rnn_size, target_vocab_len)
            self.log_softmax = nn.LogSoftmax(dim=-1)

        self.lexicon = lexicon

    def forward(self, input, lexicon_score=None):
        if self.mos:
            latent = self.latent(self.mos_dropout(input))
            latent = self.mos_dropout(latent)
            logit = self.token_weight(latent.view(-1, self.rnn_size))
            prob = nn.functional.softmax(logit, dim=-1).view(-1, self.n_experts, self.target_vocab_len)
            prior = self.prior(input).contiguous().view(-1, self.n_experts)
            prior = nn.functional.softmax(prior, dim=1)
            prior = prior.unsqueeze(2).expand_as(prob)
            prob = (prob * prior).sum(1)
            log_prob = torch.log(prob.add_(1e-8))
        else:
            latent = self.latent(input)
            if lexicon_score is not None:
                latent += lexicon_score
            log_prob = self.log_softmax(latent)
        return log_prob



