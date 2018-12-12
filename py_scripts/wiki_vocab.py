import os
from collections import Counter, OrderedDict
import torch
import sys

class Dictionary(object):
	def __init__(self, specials=['<unk>']):
		self.word2idx = {}
		self.idx2word = []
		for word in specials:
			self.add_special(word)

	def add_special(self, word):
		if word not in self.word2idx:
			self.idx2word.append(word)
			self.word2idx[word] = len(self.idx2word) - 1
			setattr(self, '{}_idx'.format(word.strip('<>')), self.word2idx[word])
		return self.get_idx(word)

	def add_word(self, word):
		if word not in self.word2idx:
			self.idx2word.append(word)
			self.word2idx[word] = len(self.idx2word) - 1
		return self.get_idx(word)

	def get_idx(self, word):
		return self.word2idx.get(word, self.unk_idx)

	def __len__(self):
		return len(self.idx2word)

class Corpus(object):
	def __init__(self, path, nvocab):
		print('building vocab...')
		counter = Counter()
		train_sents = self.count_token(path, counter)

		self.build_vocab(counter, nvocab)

		print('turn tokens into index...')
		#self.train = self.tokenize_sents(train_sents)

	def build_vocab(self, counter, nvocab):
		self.dictionary = Dictionary()
		for token, _ in counter.most_common(nvocab):
			self.dictionary.add_word(token)

	def count_token(self, path, counter):
		"""Count tokens in a text file."""
		sents = []
		assert os.path.exists(path)
		# Add words to the dictionary
		with open(path, 'r', encoding='utf-8') as f:
			for idx, line in enumerate(f):
				
				if idx > 0 and idx % 200000 == 0:
					print('    line {}'.format(idx))
				words = line.strip().split()[1:]
				counter.update(words)
		return sents

	def tokenize_sents(self, sents):
		ids = []
		for idx, words in enumerate(sents):
			if idx > 0 and idx % 200000 == 0:
				print('    line {}'.format(idx))
			for word in words:
				ids.append(self.dictionary.idx2word[self.dictionary.get_idx(word)])

		return ids

	def tokenize(self, path):
		"""Tokenizes a text file."""
		assert os.path.exists(path)
		# Tokenize file content
		with open(path, 'r', encoding='utf-8') as f:
			ids = []
			for idx, line in enumerate(f):
				if idx > 0 and idx % 200000 == 0:
					print('    line {}'.format(idx))
					break
				words = line.strip().split()[1:]
				for word in words:
					ids.append(self.dictionary.get_idx(word))

		return torch.LongTensor(ids)
if __name__ == "__main__":
	file_name = sys.argv[1]
	target_file = sys.argv[2]
	corpus = Corpus(file_name, 200000)
	with open(target_file, 'w', encoding='utf-8') as f, open(file_name, encoding='utf-8') as f1:
		for line in f1:
			sent = corpus.tokenize_sents([line.strip().split()])
			f.write(' '.join(sent) + '\n')
		


