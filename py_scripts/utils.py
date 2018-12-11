"""
prob - python3 utils.py prob /projects/tir3/users/mengzhox/data/11731_final/bilang/azetur_eng/swap-5/swap_dict \
                             /projects/tir3/users/mengzhox/data/11731_final/vocab/aze.vocab.tok \
                             /projects/tir3/users/mengzhox/data/11731_final/vocab/prob \
                             0.5 \
                             " ||| "

python3 ~/11731_final/py_scripts/utils.py swap \
                                          $data_dir/bilang/${lang}_eng/ted-train.orig.${lang}.tok \
                                          $data_dir/bilang/${lang}_eng/${process_type}/ted-train.orig.${lang}.tok \
                                          /projects/tir3/users/mengzhox/data/11731_final/MUSE_dict/azetur/T2S_re  \
                                          $data_dir/vocab/aze.vocab.tok \
                                          0  " " \
                                          /projects/tir3/users/mengzhox/data/11731_final/MUSE_dict/azetur/T2S_re2_score

python3 utils.py prob /projects/tir3/users/mengzhox/data/11731_final/bilang/glgpor_eng/swap-2/swap_dict \
                      /projects/tir3/users/mengzhox/data/11731_final/vocab/glg.vocab.tok \
                      /projects/tir3/users/mengzhox/data/11731_final/bilang/glgpor_eng/swap-2/prob_0.5 \
                      0.5 " ||| "
"""

import torch

def load_model(model_path):
    sys.path.append("/home/junjieh/mengzhox/11731_final")
    sys.path.append("/usr2/home/mengzhox/11731_final")
    sys.path.append("/home/mengzhox/11731_final")
    sys.path.append("/home/junjieh/usr3/Research/seq2seq")
    sys.path.append("/home/junjieh/usr3/Research/seq2seq/module")
    model = torch.load(model_path, map_location=lambda storage, loc: storage)
    print("Load model from {}!".format(model_path))
    return model

def load_vocab(vocab_path):
    sys.path.append("/home/junjieh/mengzhox/11731_final")
    sys.path.append("/usr2/home/mengzhox/11731_final")
    sys.path.append("/home/mengzhox/11731_final")
    vocab = torch.load(vocab_path)
    print("Load vocab from {}!".format(vocab_path))
    return vocab[0][1], vocab[1][1]

import math
import numpy as np
import scipy.sparse as sps
import sys
import random

# Build a vocabulary from file to vocab file
from collections import Counter
def build_vocabulary(file, out_file, freq=True):
    f = open(file, "r").readlines()
    f2 = open(out_file, "w")
    f = [l for s in f for l in s.split()]
    c = Counter(f)
    lens = len(c)
    for key in c.most_common(lens):
        f2.write(key[0])
        if freq:
            f2.write(" ")
            f2.write(str(key[1]))
        f2.write("\n")

# Build word2index
# token ... => d[token] = index
def build_w2i(vocab):
    f = open(vocab, "r").readlines()
    d = {}
    for i, line in enumerate(f):
        tokens = line.split()
        d[tokens[0]] = i
    return d

# Load vocab
# token1, freq => d[token1] = freq
def load_vocab_freq(vocab, freq=True, sep=" "):
    f = open(vocab, "r").readlines()
    d = {}
    for line in f:
        tokens = line.split(sep)
        if freq:
            d[" ".join(tokens[:-1])] = int(tokens[-1])
        else:
            d[" ".join(tokens[:])] = 0
    return d

# Vocab overlab
# How many words in v1 also in v2
def vocab_overlap(vocab1, vocab2):
    d1 = load_vocab_freq(vocab1)
    d2 = load_vocab_freq(vocab2)
    all_words = sum(d1.values())
    print("vocab1", len(d1))
    print("vocab2", len(d2))
    count = 0
    freq = 0
    for key in d1:
        if key in d2:
            count += 1
            freq += d1[key]
    return count, freq / all_words

# Whether the original vocab and bpe vocab is the same
def vocab_overlap2(vocab1, vocab2, vocab_bpe1, vocab_bpe2):
    d1 = load_vocab_freq(vocab1, False)
    d2 = load_vocab_freq(vocab2, False)
    d_bpe1 = load_vocab_freq(vocab_bpe1, False)
    d_bpe2 = load_vocab_freq(vocab_bpe2, False)
    print("vocab1", len(d1))
    print("vocab2", len(d2))
    count = 0
    bpe_count = 0
    for i, key in enumerate(d1):
        if key in d2:
            count += 1
            if list(d_bpe1.keys())[i] in d_bpe2:
                bpe_count += 1
    print(count, bpe_count, len(d1))

# Read printed dictionary and convert it to dict
def read_printed_dict(f):
    file = open(f, "r").readlines()[0]
    d = dict(eval(file))
    return d

# Swap key and value in a dictionary
# {a:b, c:b} => {b:[a, c]}
def change_key_value(d):
    d2 = {}
    for key in d:
        for key2 in d[key]:
            if key2 in d2:
                d2[key2].append(key)
            else:
                d2[key2] = [key]
    return d2

# Load a file to lines
def load_files(f):
    print("Opening {}.".format(f))
    lines = open(f, "r").readlines()
    return lines

# Load a file to lines of words
def load_file_by_words(f):
    lines = load_files(f)
    lines = [s.split() for s in lines]
    lines = [[a.strip() for a in s] for s in lines]
    return lines

# Load bpe words to normal words
def load_file_by_bpe_words(f):
    lines = load_file_by_words(f)
    new_lines = []
    for line in lines:
        new_tokens = []
        c = ""
        for j, token in enumerate(line):
            if token[-2:] == "@@":
                c += token
            else:
                if len(c) > 0:
                    c += token
                    new_tokens.append(c)
                else:
                    c = token
                    new_tokens.append(c)
                c = ""
        new_lines.append(new_tokens)
    return new_lines

# output lines of tokens to a file
def output_lines(f, lines):
    file = open(f, "w")
    for line in lines:
        file.write(" ".join(line) + "\n")
    print("Output to {}!".format(f))

def output_dict(f, d, sep = " "):
    file = open(f, "w")
    if isinstance(d, dict):
        for i in d:
            file.write(i + sep + d[i] + "\n")
    elif isinstance(d, list):
        for i, j in d:
            file.write(i + sep + j + "\n")
    print("Output to {}!".format(f))

def output_dict_and_score(f, d, sep = " "):
    file = open(f, "w")
    file2 = open(f+"_score", "w")
    if isinstance(d, dict):
        for i in d:
            file.write(i + sep + d[i][0] + "\n")
            file2.write(d[i][1] + "\n")
        print("Output to {} and {}!".format(f, f+"_score"))
    elif isinstance(d, list):
        for i, j in d:
            file.write(i + sep + j + "\n")
        print("Output to {}!".format(f))

# /projects/tir3/users/mengzhox/data/rapid2/azetur_eng/utils/aze_tur_google.vocab
# token1, token2 => {token2, token1}
def load_swap_dict(dict_, sep=" ||| ", score_file=None):
    a = load_files(dict_)
    re = {}
    scores = None
    if score_file:
        scores = load_files(score_file)
    for i, l in enumerate(a):
        l = l.rstrip()
        l_s = l.split(sep)
        if len(l_s) > 2:
            continue
        else:
            src, tgt = l_s
            if scores:
                if tgt not in re:
                    re[tgt] = (src, scores[i].strip())
            else:
                if tgt not in re:
                    re[tgt] = src
    print("load a dictionary of length {}".format(str(len(re))))
    first_key = list(re.keys())[0]
    print("First key and value: {}, {}".format(first_key, re[first_key]))
    return re

def load_dict(dict_, sep=" "):
    a = load_files(dict_)
    re = {}
    for i, l in enumerate(a):
        l = l.rstrip()
        l_s = l.split(sep)
        if len(l_s) > 2:
            continue
        else:
            try:
                src, tgt = l_s
                re[src] = tgt
            except:
                pass
    print("load a dictionary of length {}".format(str(len(re))))
    first_key = list(re.keys())[0]
    print("First key and value: {}, {}".format(first_key, re[first_key]))
    return re

def load_align_vocab(f, sep=" "):
    a = load_files(f)
    re = {}
    for l in a:
        l_s = l.split(sep)
        re[l_s[0].strip()] = l_s[1].strip()
    return re

def load_multi_tgt_vocab(f, lower=False):
    lines = load_files(f)
    lines = [line.split("|||") for line in lines]
    lines = [[w.strip() for w in line]for line in lines]
    d = {}
    for line in lines:
        src = line[0]
        tgt = line[1]
        if lower:
            src = src.lower()
            tgt = tgt.lower()
        if src in d:
            d[src].append(tgt)
        else:
            d[src] = [tgt]
    return d

# With a dictionary, swap a word with probability
def swap(file, outfile, dict_=None, src_vocab=None, alpha=1/2):
    v = None # LRL
    prob = {}
    swap_one_hot = {}
    swap_dict = {}
    if src_vocab is not None:
        v = load_vocab_freq(src_vocab)
        for key in v:
            prob[key] = math.exp(- v[key] * alpha)
            swap_one_hot[key] = np.random.choice([0, 1], 1, p=[1-prob[key], prob[key]])[0]
    lines = load_file_by_words(file)
    if isinstance(list(dict_.values())[0], tuple):
        for line in lines:
            for j, token in enumerate(line):
                if v is not None:
                    if token in dict_ and dict_[token][0] in swap_one_hot and token not in v:
                        index = swap_one_hot[dict_[token][0]]
                        if not index:
                            continue
                        else:
                            line[j] = dict_[token][0]
                            swap_dict[dict_[token][0]] = (token, dict_[token][1])
        output_dict_and_score("/".join(outfile.split("/")[:-1]) + "/swap_dict", swap_dict, " ||| ")

    else:
        for line in lines:
            for j, token in enumerate(line):
                if v is not None:
                    if token in dict_ and dict_[token] in swap_one_hot and token not in v:
                        index = swap_one_hot[dict_[token]]
                        if not index:
                            continue
                        else:
                            line[j] = dict_[token]
                            swap_dict[dict_[token]] = token
        output_dict("/".join(outfile.split("/")[:-1]) + "/swap_dict", swap_dict, " ||| ")

    output_lines(outfile, lines)

# Get the probablity of low frequency words
def get_prob(swap_dict, lrl_freq, output, temp=0.5, sep=" ||| "):
    freq = load_vocab_freq(lrl_freq, True, " ")
    f = load_files(swap_dict)
    f_out = open(output, "w")
    freqs = []
    for line in f:
        src, tgt = line.split(sep)
        freqs.append(freq[src])
    freqs = [math.exp(-temp*k) for k in freqs]
    sum_freqs = sum(freqs)
    freqs = [k / sum_freqs for k in freqs]
    for f in freqs:
        f_out.write(str(f) + "\n")
    print("Generate probability file @ {}".format(output))

# Count distinct char numbers
def count_charater(f):
    lines = load_files(f)
    lines = " ".join(lines)
    charset = set(lines)
    print(charset, len(charset))
    return list(charset)

# Load an alignment file to a lexicon probability dict
def get_lexicon(f):
    import scipy.sparse as sps
    lines = load_files(f)
    src_words = []
    tgt_words = []
    probs = []
    for line in lines:
        src_word, tgt_word, prob = line.split()
        src_words.append(int(src_word))
        tgt_words.append(int(tgt_word))
        probs.append(float(prob))
    src_lens = max(set(src_words)) + 1
    tgt_lens = max(set(tgt_words)) + 1
    B = sps.lil_matrix((src_lens, tgt_lens))
    for src_word, tgt_word, prob in zip(src_words, tgt_words, probs):
        B[src_word, tgt_word] = prob
    return B

# Output non overlapped words in dicts in v2 but not in v1
def find_tur_only_vocab(vocab1, vocab2, out):
    v1 = load_vocab_freq(vocab1)
    v2 = load_vocab_freq(vocab2)
    f = open(out, "w")
    for key in v2:
        if key not in v1:
            f.write(key + " " + str(v2[key]) + "\n")

# Combine lines in two files
# if first, only combine the fisrt token of the line in file1
def combine_lines(file1, file2, out, first=False, sep=" "):
    f1 = load_files(file1)
    f2 = load_files(file2)
    out_file = open(out, "w")
    for line1, line2 in zip(f1, f2):
        if first:
            out_file.write(line1.split("\t")[0].strip() + sep + line2)
        else:
            out_file.write(line1.strip() + " " + line2)

# transpose a coo matrix
def tranpose_coo(a):
    lexicon2 = sps.coo_matrix((a.data, (a.col, a.row)), shape=(a.shape[1], a.shape[0]))
    return lexicon2

# multiply two coo matrix
def sp_mul(A, B):
    a = B.tocoo()
    B = sps.coo_matrix((a.data, (a.col, a.row)), shape=(a.shape[1], a.shape[0])).tolil()
    return A*B

# Get the edit distance matrix
from Levenshtein import *
def get_edit(vocab1, vocab2, re):
    vocab_1 = load_vocab_freq(vocab1)
    vocab_2 = load_vocab_freq(vocab2)
    vocabs_1 = list(vocab_1.keys())
    vocabs_2 = list(vocab_2.keys())
    data = []
    xs = []
    ys = []
    for i, (x, y) in enumerate(zip(re.row, re.col)):
        if i % 1000 == 0:
            print("{} is done!".format(i))
        v1 = vocabs_1[x]
        v2 = vocabs_2[y]
        d = distance(v1, v2)
        data.append(d)
        xs.append(x)
        ys.append(y)
    B = sps.coo_matrix((data, (xs, ys)), shape=(max(xs)+1, max(ys)+1))
    return B

def shuffle(src, tgt, pos=None):
    f1 = open(src, "r").readlines()
    f2 = open(tgt, "r").readlines()
    f1_out = open(src + ".shuf", "w")
    f2_out = open(tgt + ".shuf", "w")
    if pos is not None:
        f3 = open(pos, "r").readlines()
        f3_out = open(pos + ".shuf", "w")
    import random
    index = random.sample(range(len(f1)), len(f1))
    for i in index:
        f1_out.write(f1[i])
        f2_out.write(f2[i])
        if pos is not None:
            f3_out.write(f3[i])


def sample_corpus(f, out_f, num):
    file = load_files(f)
    indexes = random.sample(range(len(file)), num)
    new_lines = [file[i] for i in indexes]
    f_out = open(out_f, "w")
    for line in new_lines:
        f_out.write(line)


def generate_nonbpe(file):
    f = open(file, "r")
    f2 = open("{}.nonbpe".format(file), "w")
    lines = f.readlines()
    for line in lines:
        tokens = line.split()
        token = ""
        tokens_ = []
        for i, t in enumerate(tokens):
            if t[0] == "▁":
                if token != "":
                    tokens_.append(token)
                token = t[1:]
            else:
                token += t
            if i == len(tokens) - 1:
                tokens_.append(token)
                break
        new_line = " ".join(tokens_)
        f2.write(new_line + "\n")

if __name__ == '__main__':
    if sys.argv[1] == "swap":
        swap(file=sys.argv[2],
             outfile=sys.argv[3],
             dict_=load_swap_dict(sys.argv[4], sep=sys.argv[7], score_file=sys.argv[8] if len(sys.argv) == 9 else None),
             src_vocab=sys.argv[5],
             alpha=float(sys.argv[6]))
    elif sys.argv[1] == "shuffle":
        shuffle(src=sys.argv[2], tgt=sys.argv[3])
    elif sys.argv[1] == "sample":
        sample_corpus(f=sys.argv[2], out_f=sys.argv[3], num=int(sys.argv[4]))
    elif sys.argv[1] == "vocab":
        build_vocabulary(file=sys.argv[2], out_file=sys.argv[3])
    elif sys.argv[1] == "nonbpe":
        generate_nonbpe(sys.argv[2])
    elif sys.argv[1] == "prob":
        get_prob(swap_dict=sys.argv[2],
                 lrl_freq=sys.argv[3],
                 output=sys.argv[4],
                 temp=float(sys.argv[5]),
                 sep=sys.argv[6])

