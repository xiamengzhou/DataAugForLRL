#!/usr/bin/env pythonq
# -*- coding: utf-8 -*-

import argparse
import os
import glob
import sys

import torch

import onmt.io
import opts

vocab_1 = torch.load(sys.argv[1])
vocab_2 = torch.load(sys.argv[2])
vocab_file = sys.argv[3]

vocab_1_src = vocab_1[0][1]
vocab_2_src = vocab_2[0][1]
vocab_1_tgt = vocab_1[1][1]
vocab_2_tgt = vocab_2[1][1]
delete_words = []
for k, v in vocab_2_src.freqs.items():
	if v < 3:
		delete_words.append(k)
for w in delete_words:
	del vocab_2_src.freqs[w]

vocab_src = onmt.io.merge_vocabs([vocab_1_src, vocab_2_src])
vocab_tgt = onmt.io.merge_vocabs([vocab_1_tgt, vocab_2_tgt], 50000)
vocab = [('src', vocab_src), ('tgt', vocab_tgt)]
torch.save(vocab, vocab_file)