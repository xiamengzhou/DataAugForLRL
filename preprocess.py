#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import glob
import sys

import torch

import onmt.io
import opts

def check_existing_pt_files(opt):
    # We will use glob.glob() to find sharded {train|valid}.[0-9]*.pt
    # when training, so check to avoid tampering with existing pt files
    # or mixing them up.
    for t in ['train', 'valid', 'vocab']:
        pattern = opt.save_data + '.' + t + '*.pt'
        if glob.glob(pattern):
            sys.stderr.write("Please backup exisiting pt file: %s, "
                             "to avoid tampering!\n" % pattern)
            sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description='preprocess.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    opts.add_md_help_argument(parser)
    opts.preprocess_opts(parser)

    opt = parser.parse_args()
    torch.manual_seed(opt.seed)

    check_existing_pt_files(opt)

    return opt


def build_save_text_dataset_in_shards(src_corpus, tgt_corpus, fields, corpus_type, opt,
                                      src_mono_corpus_1=None, src_mono_corpus_2=None):
    corpus_size = os.path.getsize(src_corpus)
    if corpus_size > 10 * (1024**2) and opt.max_shard_size == 0:
        print("Warning. The corpus %s is larger than 10M bytes, you can "
              "set '-max_shard_size' to process it by small shards "
              "to use less memory." % src_corpus)

    if opt.max_shard_size != 0:
        print(' * divide corpus into shards and build dataset separately'
              '(shard_size = %d bytes).' % opt.max_shard_size)

    ret_list = []
    src_iter = onmt.io.ShardedTextCorpusIterator(
                src_corpus, opt.src_seq_length_trunc,
                "src", opt.max_shard_size)
    tgt_iter = onmt.io.ShardedTextCorpusIterator(
                tgt_corpus, opt.tgt_seq_length_trunc,
                "tgt", opt.max_shard_size,
                assoc_iter=src_iter)
    if src_mono_corpus_1:
        assert src_mono_corpus_2 is not None
        src_mono_1_iter = onmt.io.ShardedTextCorpusIterator(
                            src_mono_corpus_1, 0, "src_mono_1", opt.max_shard_size)
        src_mono_2_iter = onmt.io.ShardedTextCorpusIterator(
                            src_mono_corpus_2, 0, "src_mono_2", opt.max_shard_size,
                            assoc_iter=src_mono_1_iter)

    index = 0
    while not src_iter.hit_end():
        index += 1
        dataset = onmt.io.TextDataset(
                fields, src_iter, tgt_iter,
                src_seq_length=opt.src_seq_length,
                tgt_seq_length=opt.tgt_seq_length,
                ngram=opt.ngram)

        # We save fields in vocab.pt seperately, so make it empty.
        dataset.fields = []

        pt_file = "{:s}.{:s}.{:d}.tm.pt".format(
                opt.save_data, corpus_type, index)
        print(" * saving %s data shard to %s." % (corpus_type, pt_file))
        torch.save(dataset, pt_file)

        ret_list.append(pt_file)

    return ret_list


def build_save_dataset(corpus_type, fields, opt):
    assert corpus_type in ['train', 'valid']

    if corpus_type == 'train':
        src_corpus = opt.train_src
        tgt_corpus = opt.train_tgt
        # src_mono_corpus_1 = None
        # src_mono_corpus_2 = None
        # if opt.mono_cons:
        #     src_mono_corpus_1 = opt.train_src_mono_1
        #     src_mono_corpus_2 = opt.train_src_mono_2

    else:
        src_corpus = opt.valid_src
        tgt_corpus = opt.valid_tgt

    return build_save_text_dataset_in_shards(
            src_corpus, tgt_corpus, fields, corpus_type, opt)


def build_save_vocab(train_dataset, fields, opt):
    fields = onmt.io.build_vocab(train_dataset, fields,
                                 opt.share_vocab,
                                 opt.src_vocab_size,
                                 opt.src_words_min_frequency,
                                 opt.tgt_vocab_size,
                                 opt.tgt_words_min_frequency,
                                 opt.src_vocab_file,
                                 opt.tgt_vocab_file)

    # Can't save fields, so remove/reconstruct at training time.
    vocab_file = opt.save_data + '.vocab.pt'
    torch.save(onmt.io.save_fields_to_vocab(fields), vocab_file)

def get_lex_vocabs(src_vcb_file, tgt_vcb_file):
    f1 = open(src_vcb_file, "r").readlines()
    f2 = open(tgt_vcb_file, "r").readlines()
    src_vcb = {}
    tgt_vcb = {}
    for i, line in enumerate(f1):
        tokens = line.split()
        src_vcb[int(tokens[0])] = tokens[1]
    for i, line in enumerate(f2):
        tokens = line.split()
        tgt_vcb[int(tokens[0])] = tokens[1]
    return src_vcb, tgt_vcb


def main():
    opt = parse_args()

    print("Building `Fields` object...")
    fields = onmt.io.get_fields(ngram=opt.ngram)
    if opt.cover == "standard":
        print("Building & saving training data...")
        train_dataset_files = build_save_dataset('train', fields, opt)

        print("Building & saving validation data...")
        build_save_dataset('valid', fields, opt)

        print("Building & saving vocabulary...")
        build_save_vocab(train_dataset_files, fields, opt)
    elif opt.cover == "valid":
        pass
        print("Building & saving validation data...")
        build_save_dataset('valid', fields, opt)

if __name__ == "__main__":
    main()
