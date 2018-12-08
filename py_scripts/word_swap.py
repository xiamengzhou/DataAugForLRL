# coding: utf-8
import os, sys
import argparse
import data



parser = argparse.ArgumentParser(description='word swap')
parser.add_argument('--lrl_file', type=str,
                    help='low resource language file path')
parser.add_argument('--hrl_file', type=str,
                    help='high resource language file path')
parser.add_argument('--bi_dict', type=str,
                    help='bilingual word translation dictionary')
parser.add_argument('--hrl_file', type=str,
                    help='high resource language file path')
parser.add_argument('--all', action='store_true',
                    help='use all words in dictionary')
args = parser.parse_args()


lrl_file = args.lrl_file
hrl_file = args.hrl_file
bi_dict_file = args.bi_dict
all_words = args.all

def read_bi_dict(dict_file):

	bi_dict = {}
	with open(bi_dict_file, encoding='utf-8') as f:
		for line in f:
			words = line.strip().split()
			bi_dict[words[1]] = words[0]
	return bi_dict

bi_dict = read_bi_dict(bi_dict_file)

def vocab(path):

	vocab_list = []
	with open(path, encoding='utf-8') as f:
		for line in f:
			words = line.strip().split()
			for w in words:
				if w not in vocab_list:
					vocab_list.append(w)
	return vocab_list

lrl_vocab = vocab(lrl_file)

def word_swap(hrl_path, lrl_vocab, bidict, all_words):
	if all_words:
		out_file = hrl_path + '.swpall'
	else:
		out_file = hrl_path + '.swp'
	with open(hrl_path, encoding='utf-8') as h, open(out_file, 'w', encoding='utf-8') as f:
		for line in h:
			words = line.strip().split()
			hrl_words = []
			for w in words:
				if all_words:
					hrl_words.append(bidict.get(w, w))
				else if w in lrl_vocab:
					hrl_words.append(bidict.get(w, w))
				else:
					hrl_words.append(w)
			f.write(' '.join(hrl_words) + '\n')






