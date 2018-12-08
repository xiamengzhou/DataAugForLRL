# coding: utf-8
import os, sys
import argparse
import torch
import data

parser = argparse.ArgumentParser(description='word swap')
parser.add_argument('--lrl_file', type=str,
                    help='location of the data corpus')
parser.add_argument('--hrl_file', type=int,
                    help='vocabulary size')
args = parser.parse_args()

corpus = data.Corpus(args.data, args.nvocab)
torch.save(corpus, os.path.join(args.data, 'corpus.pt'))
