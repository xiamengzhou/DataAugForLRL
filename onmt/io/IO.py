# -*- coding: utf-8 -*-

from collections import Counter, defaultdict, OrderedDict
from itertools import count

import torch
import torchtext.data
import torchtext.vocab
from torchtext.data.utils import RandomShuffler
import torch.nn.functional
from onmt.io.DatasetBase import UNK_WORD, PAD_WORD, BOS_WORD, EOS_WORD
from onmt.io.TextDataset import TextDataset

import numpy as np



def _getstate(self):
    return dict(self.__dict__, stoi=dict(self.stoi))


def _setstate(self, state):
    self.__dict__.update(state)
    self.stoi = defaultdict(lambda: 0, self.stoi)


torchtext.vocab.Vocab.__getstate__ = _getstate
torchtext.vocab.Vocab.__setstate__ = _setstate

def get_fields(ngram=-1, skipgram=False):
    return TextDataset.get_fields(ngram, skipgram)

def load_fields_from_vocab(vocab, ngram=-1, skipgram=False):
    """
    Load Field objects from `vocab.pt` file.
    """
    vocab = dict(vocab)
    fields = TextDataset.get_fields(ngram=ngram, skipgram=skipgram)
    for k, v in vocab.items():
        # Hack. Can't pickle defaultdict :(
        v.stoi = defaultdict(lambda: 0, v.stoi)
        if k in fields:
            fields[k].vocab = v
    return fields


def save_fields_to_vocab(fields):
    """
    Save Vocab objects in Field objects to `vocab.pt` file.
    """
    vocab = []
    for k, f in fields.items():
        if f is not None and 'vocab' in f.__dict__:
            f.vocab.stoi = dict(f.vocab.stoi)
            vocab.append((k, f.vocab))
    return vocab


def merge_vocabs(vocabs, vocab_size=None):
    """
    Merge individual vocabularies (assumed to be generated from disjoint
    documents) into a larger vocabulary.

    Args:
        vocabs: `torchtext.vocab.Vocab` vocabularies to be merged
        vocab_size: `int` the final vocabulary size. `None` for no limit.
    Return:
        `torchtext.vocab.Vocab`
    """
    merged = sum([vocab.freqs for vocab in vocabs], Counter())
    return torchtext.vocab.Vocab(merged,
                                 specials=[UNK_WORD, PAD_WORD,
                                           BOS_WORD, EOS_WORD],
                                 max_size=vocab_size)


def build_dataset(fields, src_path, tgt_path, src_seq_length=0, tgt_seq_length=0,
                  src_seq_length_trunc=0, tgt_seq_length_trunc=0,
                  use_filter_pred=True, ngram=-1):

    # Build src/tgt examples iterator from corpus files, also extract
    # number of features.
    src_examples_iter = \
        _make_examples_nfeats_tpl(src_path, src_seq_length_trunc)

    # For all data types, the tgt side corpus is in form of text.
    tgt_examples_iter = \
        TextDataset.make_text_examples_nfeats_tpl(
            tgt_path, tgt_seq_length_trunc, "tgt")

    dataset = TextDataset(fields, src_examples_iter, tgt_examples_iter,
                          src_seq_length=src_seq_length,
                          tgt_seq_length=tgt_seq_length,
                          use_filter_pred=use_filter_pred,
                          ngram=ngram)

    return dataset


def _build_field_vocab(field, counter, **kwargs):
    specials = list(OrderedDict.fromkeys(
        tok for tok in [field.unk_token, field.pad_token, field.init_token,
                        field.eos_token]
        if tok is not None))
    field.vocab = field.vocab_cls(counter, specials=specials, **kwargs)

def _build_field_vocab_all(field, counter, vocab_file, **kwargs):
    specials = list(OrderedDict.fromkeys(
        tok for tok in [field.pad_token, field.init_token, field.eos_token,
                        field.unk_token] if tok is not None))
    field.vocab = Vocab(counter, specials=specials, vocab_file=vocab_file, **kwargs)

def build_vocab(train_dataset_files, fields, share_vocab,
                src_vocab_size, src_words_min_frequency,
                tgt_vocab_size, tgt_words_min_frequency,
                src_vocab_file=None, tgt_vocab_file=None):
    """
    Args:
        train_dataset_files: a list of train dataset pt file.
        fields (dict): fields to build vocab for.
        data_type: "text", "img" or "audio"?
        share_vocab(bool): share source and target vocabulary?
        src_vocab_size(int): size of the source vocabulary.
        src_words_min_frequency(int): the minimum frequency needed to
                include a source word in the vocabulary.
        tgt_vocab_size(int): size of the target vocabulary.
        tgt_words_min_frequency(int): the minimum frequency needed to
                include a target word in the vocabulary.
        all_vocab(bool): build fields based on full vocabulary.

    Returns:
        Dict of Fields
    """
    counter = {}
    for k in fields:
        counter[k] = Counter()

    for path in train_dataset_files:
        dataset = torch.load(path)
        print(" * reloading %s." % path)
        for ex in dataset.examples:
            for k in fields:
                val = getattr(ex, k, None)
                if val is not None and not fields[k].sequential:
                    val = [val]
                counter[k].update(val)

    if tgt_vocab_file:
        _build_field_vocab_all(fields["tgt"], counter["tgt"],
                                max_size=tgt_vocab_size,
                                min_freq=tgt_words_min_frequency,
                                vocab_file=tgt_vocab_file)
    else:
        _build_field_vocab(fields["tgt"], counter["tgt"],
            max_size=tgt_vocab_size,
            min_freq=tgt_words_min_frequency)
    print(" * tgt vocab size: %d." % len(fields["tgt"].vocab))
    try:
        print(" * valid tgt vocab size: %d" % fields["tgt"].vocab.get_valid_vocab_size())
    except:
        pass

    if src_vocab_file:
        _build_field_vocab_all(fields["src"], counter["src"],
            max_size=src_vocab_size,
            min_freq=src_words_min_frequency, vocab_file=src_vocab_file)
    else:
        _build_field_vocab(fields["src"], counter["src"],
            max_size=src_vocab_size,
            min_freq=src_words_min_frequency)
    print(" * src vocab size: %d." % len(fields["src"].vocab))
    try:
        print(" * valid src vocab size: %d" % fields["src"].vocab.get_valid_vocab_size())
    except:
        pass

    # All datasets have same num of n_src_features,
    # getting the last one is OK.

    # Merge the input and output vocabularies.
    if share_vocab:
        # `tgt_vocab_size` is ignored when sharing vocabularies
        print(" * merging src and tgt vocab...")
        merged_vocab = merge_vocabs(
            [fields["src"].vocab, fields["tgt"].vocab],
            vocab_size=src_vocab_size)
        fields["src"].vocab = merged_vocab
        fields["tgt"].vocab = merged_vocab

    return fields


def _make_examples_nfeats_tpl(src_path, src_seq_length_trunc):
    """
    Process the corpus into (example_dict iterator, num_feats) tuple
    on source side for different 'data_type'.
    """

    src_examples_iter = \
            TextDataset.make_text_examples_nfeats_tpl(
                src_path, src_seq_length_trunc, "src")

    return src_examples_iter


class TMBatch:
    def __init__(self, data=None, dataset=None, device=None, train=True):
        """Create a Batch from a list of examples."""
        if data is not None:
            self.batch_size = len(data)
            self.dataset = dataset
            self.train = train
            # src, tgt: length * batch_size

            for (name, field) in dataset.fields.items():
                if name in data[0].__dict__:
                    batch = [x.__dict__[name] for x in data]
                    if not isinstance(dataset.ngram, int):
                        dataset.ngram = -1
                    if name == "src" and dataset.ngram > 0:
                        # langid = [int(d[0]) for d in batch]
                        # batch = [d[1:] for d in batch]
                        out = field.process(batch, device=-1, train=train)
                        length = out[1]
                        new_batch = TMBatch.get_ngram(batch, dataset.ngram)
                        new_outs = []
                        max_word_len = 0
                        for b in new_batch: # sent
                            out_t = field.process(b, device=-1, train=train) # word_len * ngram_len
                            if len(out_t[0]) > max_word_len:
                                max_word_len = len(out_t[0])
                            new_outs.append([])
                            for ngrams in out_t[0]: # word
                                ngram_kv = {}
                                for ngram in ngrams.data:
                                    if ngram not in ngram_kv:
                                        ngram_kv[ngram] = 1
                                    else:
                                        ngram_kv[ngram] += 1
                                if 0 in ngram_kv:
                                    ngram_kv[0] = 0
                                new_outs[-1].append(ngram_kv)
                        # sent * len(ngram)
                        sents_sparse = []
                        for o in new_outs:
                            keys = []
                            vals = []
                            for i, word in enumerate(o):
                                keys.append(torch.LongTensor([[i for _ in range(len(word))], list(word.keys())]))
                                vals.extend(list(word.values()))
                            keys = torch.cat(keys, dim=1)
                            vals = torch.FloatTensor(vals)
                            sent_sparse = torch.sparse.FloatTensor(keys, vals, torch.Size([max_word_len, len(field.vocab.itos)]))
                            sents_sparse.append(sent_sparse)
                        assert len(sents_sparse) == len(new_outs)
                        setattr(self, name, (sents_sparse, length))
                    # if name == "src_sg" and dataset.skipgram:
                    #     setattr(self, name, field.process(batch, device=device, train=train))
                    else:
                        setattr(self, name, field.process(batch, device=device, train=train))

    @classmethod
    def fromvars(cls, dataset, batch_size, train=True, **kwargs):
        """Create a Batch directly from a number of Variables."""
        batch = cls()
        batch.batch_size = batch_size
        batch.dataset = dataset
        batch.train = train
        for k, v in kwargs.items():
            setattr(batch, k, v)
        return batch

    @classmethod
    def get_ngram(cls, batch, n):
        # sent * word
        new_batch = []
        for k, sent in enumerate(batch):
            new_batch.append([])
            for m, token in enumerate(sent):
                new_batch[-1].append([])
                lens = len(token)
                for i in range(1, n + 1):
                    for j in range(lens):
                        if j + i <= lens:
                            sub = token[j:j + i]
                            new_batch[-1][-1].append(sub)
        return new_batch


class OrderedIterator(torchtext.data.Iterator):
    def __init__(self, dataset, batch_size, sort_key=None, device=None,
                 batch_size_fn=None, train=True,
                 repeat=None, shuffle=None, sort=None,
                 sort_within_batch=None):
        super(OrderedIterator, self).__init__(dataset, batch_size, sort_key=sort_key, device=device,
                                              batch_size_fn=batch_size_fn, train=train, repeat=repeat,
                                              shuffle=shuffle, sort=sort, sort_within_batch=sort_within_batch)
        self.batch_size, self.train, self.dataset = batch_size, train, dataset
        self.batch_size_fn = batch_size_fn
        self.iterations = 0
        self.repeat = train if repeat is None else repeat
        self.shuffle = train if shuffle is None else shuffle
        self.sort = not train if sort is None else sort

        if sort_within_batch is None:
            self.sort_within_batch = self.sort
        else:
            self.sort_within_batch = sort_within_batch
        if sort_key is None:
            self.sort_key = dataset.sort_key
        else:
            self.sort_key = sort_key
        self.device = device
        if not torch.cuda.is_available() and self.device is None:
            self.device = -1

        self.random_shuffler = RandomShuffler()

        # For state loading/saving only
        self._iterations_this_epoch = 0
        self._random_state_this_epoch = None
        self._restored_from_state = False

    def create_batches(self):
        if self.train:
            def pool(data, random_shuffler):
                for p in torchtext.data.batch(data, self.batch_size * 100):
                    p_batch = torchtext.data.batch(
                            sorted(p, key=self.sort_key),
                            self.batch_size, self.batch_size_fn)
                    for b in random_shuffler(list(p_batch)):
                        yield b
            self.batches = pool(self.data(), self.random_shuffler)
        else:
            self.batches = []
            for b in torchtext.data.batch(self.data(), self.batch_size, self.batch_size_fn):
                    self.batches.append(sorted(b, key=self.sort_key))

    def __iter__(self):
        while True:
            self.init_epoch()
            for idx, minibatch in enumerate(self.batches):
                # fast-forward if loaded from state
                if self._iterations_this_epoch > idx:
                    continue
                self.iterations += 1
                self._iterations_this_epoch += 1
                if self.sort_within_batch:
                    # NOTE: `rnn.pack_padded_sequence` requires that a minibatch
                    # be sorted by decreasing order, which requires reversing
                    # relative to typical sort keys
                    if self.sort:
                        minibatch.reverse()
                    else:
                        minibatch.sort(key=self.sort_key, reverse=True)
                yield TMBatch(minibatch, self.dataset, self.device, self.train)
            if not self.repeat:
                raise StopIteration


class Vocab(torchtext.vocab.Vocab):
    def __init__(self, counter, max_size=None, min_freq=1, specials=['<pad>'],
                 vectors=None, unk_init=None, vectors_cache=None, vocab_file=None):
        self.freqs = counter
        counter = counter.copy()
        min_freq = max(min_freq, 1)

        self.itos = list(specials)
        # frequencies of special tokens are not counted when building vocabulary
        # in frequency order
        for tok in specials:
            del counter[tok]

        max_size = None if max_size is None else max_size + len(self.itos)

        # sort by frequency, then alphabetically
        # words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
        # words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

        # for word, freq in count.items():
        #     if freq < min_freq or len(self.itos) == max_size:
        #         break
        #     self.itos.append(word)
        assert vocab_file is not None
        word_itos = self.load_vocab_file(vocab_file, max=max_size)
        self.itos = self.itos + word_itos

        self.stoi = defaultdict(torchtext.vocab._default_unk_index)
        # stoi is simply a reverse dict for itos
        self.stoi.update({tok: i for i, tok in enumerate(self.itos)})

        self.vectors = None
        if vectors is not None:
            self.load_vectors(vectors, unk_init=unk_init, cache=vectors_cache)
        else:
            assert unk_init is None and vectors_cache is None

    # index word
    def load_vocab_file(self, file, max=None):
        itos = []
        f = open(file, "r").readlines()
        if max is not None:
            f = f[:max]
        else:
            f = f
        for line in f:
            word, freq = line.split()
            itos.append(word)
        return itos

    def get_valid_vocab_size(self):
        return min(len(self.freqs), len(self.itos))

