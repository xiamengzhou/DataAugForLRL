import math
import numpy as np
import scipy.sparse as sps
import sys
import random
import os
import nltk

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
    print("Load {} lines from the file ..".format(len(lines)))
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
    print("Output {} lines to {}!".format(len(lines), f))



def output_dict(f, d, sep = " "):
    file = open(f, "w")
    if isinstance(d, dict):
        for i in d:
            file.write(i + sep + d[i] + "\n")
    elif isinstance(d, list):
        for i, j in d:
            file.write(i + sep + j + "\n")
    print("Output to {}!".format(f))

# /projects/tir3/users/mengzhox/data/rapid2/azetur_eng/utils/aze_tur_google.vocab
# token1, token2 => {token2, token1}
def load_swap_dict(dict_, sep=" "):
    a = load_files(dict_)
    re = {}
    for l in a:
        l = l.rstrip()
        l_s = l.split(sep)
        if len(l_s) != 2:
            continue
        else:
            src, tgt = l_s
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
        if len(l_s) != 2:
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

# With a dictionary, swap a word with probability, the word has to be in the lrl vocabulary
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
    output_lines(outfile, lines)
    output_dict("/".join(outfile.split("/")[:-1]) + "/swap_dict", swap_dict, " ||| ")

# With a dictionry, swap a word with probability, the word does not have to be in lrl vocabulary
def swap2(file, outfile, dict_=None):
    swap_dict = {}
    lines = load_file_by_words(file)
    all = 0
    swapped = 0
    for line in lines:
        all += len(line)
        for j, token in enumerate(line):
            if token in dict_:
                line[j] = dict_[token]
                swap_dict[dict_[token]] = token
                swapped += 1
    output_lines(outfile, lines)
    output_dict("/".join(outfile.split("/")[:-1]) + "/swap_dict", swap_dict, " ||| ")
    print("{} words out of {} are swapped!".format(str(swapped), str(all)))

# With a dictionry, swap a word with probability, the word does have to be in lrl vocabulary
def swap3(file, outfile, dict_=None):
    swap_dict = {}
    lines = load_file_by_words(file)
    all = 0
    swapped = 0
    for line in lines:
        all += len(line)
        for j, token in enumerate(line):
            if token in dict_ and random.random() >= 0.5:
                line[j] = dict_[token]
                swap_dict[dict_[token]] = token
                swapped += 1
    output_lines(outfile, lines)
    output_dict("/".join(outfile.split("/")[:-1]) + "/swap_dict", swap_dict, " ||| ")
    print("{} words out of {} are swapped!".format(str(swapped), str(all)))

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
    lens = len(file)
    if num <= lens:
        indexes = random.sample(range(len(file)), num)
    else:
        indexes = range(lens)
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

# Count identical strings in vocab files
def count_identical_strings(vocab1, vocab2):
    v1 = load_vocab_freq(vocab1)
    v2 = load_vocab_freq(vocab2)
    count = 0
    for v in v1:
        if v in v2:
            count += 1
    print("{} identical string pairs are generated.")

def count_identical_strings3(file):
    a = open(file, "r").readlines()
    count = 0
    for line in a:
        c, d = line.split()
        if c == d:
            count += 1
    return count

# Count identical strings in fasttext files
def count_identical_strings2(vocab1, vocab2, out, lower):
    f1 = open(vocab1, "r")
    f2 = open(vocab2, "r")
    f = open(out, "w")
    v1 = {}
    v2 = {}
    count = 0
    for i, line in enumerate(f1):
        v1[line.split()[0]] = i
    for i, line in enumerate(f2):
        v2[line.split()[0]] = i
    for v in v1:
        if v in v2:
            count += 1
            if lower:
                v = v.lower()
            f.write(v + "\t" + v + "\n")
    return count


# Converct FB vector format to tf
def convert_vec_to_tf(file, file_out):
    f = open(file, "r").readlines()[1:]
    f2 = open(file_out, "w")
    tag = open(file_out+".tag", "w")
    for line in f:
        a = line.split()
        word = a[0]
        vectors = a[1:]
        vectors = [str(b) for b in vectors]
        tag.write(word + "\n")
        f2.write("\t".join(vectors) + "\n")

# Extract word embeddings from a model
def extract_word_embeddings(model_path, save_path):
    import sys
    import torch
    sys.path.append("/home/mengzhox/11731_final")
    model = torch.load(model_path, map_location = lambda storage, loc: storage)
    embeddings = model["model"]["encoder.embeddings.embeddings.weight"]
    f = open(save_path, "w")
    vocab = model["vocab"][0][1].itos
    f.write(str(len(vocab)) + " " + str(len(embeddings[0])) + "\n")
    for i, v in enumerate(vocab):
        k = [str(j) for j in embeddings[i]]
        f.write(v + " " + " ".join(k) + "\n")
    print("Checkpoint saved in {}!".format(save_path))

def split_dict(file, out1, out2):
    f = open(file).readlines()
    f2 = open(out1, "w")
    f3 = open(out2, "w")
    for line in f:
        src, tgt = line.split()
        f2.write(src + "\n")
        f3.write(tgt + "\n")

def extract_dict_from_emb(file, out_dict):
    f = open(file, "r").readlines()[1:]
    o = open(out_dict, "w")
    for line in f:
        word = line.split()[0]
        o.write(word + "\n")

def merge_dict(dict1, dict2, out_dict):
    d1 = load_vocab_freq(dict1, True)
    d1_keys = list(d1.keys())
    d2 = load_vocab_freq(dict2, False)
    for word in d2:
        if word not in d1:
            d1_keys.append(word)
    o = open(out_dict, "w")
    for key in d1_keys:
        o.write(key + "\n")

def split_file(infile, outfile1, outfile2, sep="|||"):
    out1 = open(outfile1, "w")
    out2 = open(outfile2, "w")
    with open(infile, "r") as f:
        for line in f:
            src_line, tgt_line = line.split(sep)
            out1.write(src_line.strip() + "\n")
            out2.write(tgt_line.strip() + "\n")
    out1.close()
    out2.close()

def sample_parallel(infile1, infile2, outfile1, outfile2, sample_size):
    in1 = open(infile1, "r").readlines()
    in2 = open(infile2, "r").readlines()
    lens1 = len(in1)
    lens2 = len(in2)
    assert lens1 == lens2
    if sample_size > lens1:
        sample_lines1 = in1
        sample_lines2 = in2
    else:
        indexes = random.sample(range(lens1), sample_size)
        sample_lines1 = [in1[index] for index in indexes]
        sample_lines2 = [in2[index] for index in indexes]
    out1 = open(outfile1, "w")
    out2 = open(outfile2, "w")
    for line1, line2 in zip(sample_lines1, sample_lines2):
        out1.write(line1)
        out2.write(line2)
    out1.close()
    out2.close()

def vocab_count(file):
    lines = open(file).readlines()
    re = []
    for line in lines:
        re += line.split()
    re = set(re)
    print("The size of the vocabulary is {}...".format(str(len(re))))

# Split the text into shards
def split_file_into_shards(file, out_dir, shards=100):
    f = open(file, "r").readlines()
    lens = len(f)
    num_shard = lens // shards
    for i in range(shards):
        f1 = open(os.path.join(out_dir, "{}.{}".format(file, str(i+1))), "w")
        f1.write("".join(f[i*num_shard:(i+1)*num_shard]))


def get_characters(file):
    f = open(file, "r").readlines()
    chars = set()
    for i, line in enumerate(f):
        s = set([char for char in line.strip()])
        chars = chars.union(s)
        if i % 100000 == 0:
            print(i)
    return chars

def get_seed_dict(vocab1, vocab2, out_vocab_file):
    v1 = load_vocab_freq(vocab1)
    v2 = load_vocab_freq(vocab2)
    v1_keys = set(list(v1.keys()))
    v2_keys = set(list(v2.keys()))
    identical_words = v1_keys.intersection(v2_keys)
    f = open(out_vocab_file, "w")
    for word in identical_words:
        f.write(word + " " + word + "\n")

# sentence tokenization
def sent_tokenize_wiki(file, out_file):
    f = open(file, "r").readlines()
    f2 = open(out_file, "w")
    sent_texts = []
    for line in f:
        if len(line.split()) == 0:
            continue
        sent_text = nltk.sent_tokenize(line)
        sent_texts += sent_text
    for sent in sent_texts:
        f2.write(sent.strip() + "\n")
    print("Tokenized {} lines in total...".format(str(len(sent_texts))))

if __name__ == '__main__':
    if sys.argv[1] == "swap":
        swap(file=sys.argv[2],
             outfile=sys.argv[3],
             dict_=load_swap_dict(sys.argv[4]),
             src_vocab=sys.argv[5],
             alpha=float(sys.argv[6]))
    elif sys.argv[1] == "swap2":
        swap2(file=sys.argv[2],
              outfile=sys.argv[3],
              dict_=load_swap_dict(sys.argv[4], sep=sys.argv[5]))
    elif sys.argv[1] == "swap3":
        swap3(file=sys.argv[2],
              outfile=sys.argv[3],
              dict_=load_swap_dict(sys.argv[4]))
    elif sys.argv[1] == "swap4":
        swap2(file=sys.argv[2],
              outfile=sys.argv[3],
              dict_=load_dict(sys.argv[4], sep=sys.argv[5]))
    elif sys.argv[1] == "shuffle":
        shuffle(src=sys.argv[2], tgt=sys.argv[3])
    elif sys.argv[1] == "sample":
        sample_corpus(f=sys.argv[2], out_f=sys.argv[3], num=int(sys.argv[4]))
    elif sys.argv[1] == "vocab":
        build_vocabulary(file=sys.argv[2], out_file=sys.argv[3])
    elif sys.argv[1] == "nonbpe":
        generate_nonbpe(sys.argv[2])
    elif sys.argv[1] == "identical2":
        count_identical_strings2(vocab1=sys.argv[2],
                                 vocab2=sys.argv[3],
                                 out=sys.argv[4],
                                 lower=(len(sys.argv)==6 and sys.argv[5]=="lower") or False)
    elif sys.argv[1] == "convert":
        convert_vec_to_tf(file=sys.argv[2], file_out=sys.argv[3])
    elif sys.argv[1] == "ex_emb":
        extract_word_embeddings(model_path=sys.argv[2], save_path=sys.argv[3])
    elif sys.argv[1] == "split_dict":
        split_dict(file=sys.argv[2], out1=sys.argv[3], out2=sys.argv[4])
    elif sys.argv[1] == "ex_dict_emb":
        extract_dict_from_emb(file=sys.argv[2], out_dict=sys.argv[3])
    elif sys.argv[1] == "merge_dict":
        merge_dict(dict1=sys.argv[2], dict2=sys.argv[3], out_dict=sys.argv[4])
    elif sys.argv[1] == "sample_para":
        sample_parallel(infile1=sys.argv[2],
                        infile2=sys.argv[3],
                        outfile1=sys.argv[4],
                        outfile2=sys.argv[5],
                        sample_size=int(sys.argv[6]))
    elif sys.argv[1] == "split_file":
        split_file(infile=sys.argv[2], outfile1=sys.argv[3], outfile2=sys.argv[4])
    elif sys.argv[1] == "seed":
        get_seed_dict(vocab1=sys.argv[2], vocab2=sys.argv[3], out_vocab_file=sys.argv[4])
    elif sys.argv[1] == "sent_tok":
        sent_tokenize_wiki(sys.argv[2], sys.argv[3])

