"""
python3 get_dictionary.py --src_emb /home/junjieh/mengzhox/MUSE/deen_re5/debug/snoen9clns/vectors-de.txt \
                          --tgt_emb /home/junjieh/mengzhox/MUSE/deen_re5/debug/snoen9clns/vectors-en.txt \
                          --output /home/junjieh/mengzhox/MUSE/deen_re5/debug/snoen9clns/de-en-S2T-T2S-all \
                          --dico_build "S2T&T2S"

python3 get_dictionary.py --lrl_hrl_emb $data/11731_final/analysis/azetur.tran.emb.tok.dis \
                          --lrl_hrl_vocab $data/11731_final/analysis/azetur.tran.emb.tok.dis.tag \
                          --output $data/11731_final/analysis/azetur_S2T_T2S \
                          --dico_build "S2T&T2S"
"""

import sys
sys.path.append("/home/mengzhox/MUSE")
from src.dico_builder import build_dictionary
import numpy as np
import io
import argparse
import torch
import pickle
from src.utils import bool_flag, initialize_exp
import codecs

parser = argparse.ArgumentParser(description='Unsupervised training')
parser.add_argument("--src_emb", type=str, default="", help="Reload source embeddings")
parser.add_argument("--tgt_emb", type=str, default="", help="Reload target embeddings")
parser.add_argument("--lrl_hrl_emb", type=str, default="")
parser.add_argument("--lrl_hrl_vocab", type=str, default="")
parser.add_argument("--dico_method", type=str, default='csls_knn_10', help="Method used for dictionary generation (nn/invsm_beta_30/csls_knn_10)")
parser.add_argument("--dico_build", type=str, default='S2T&T2S', help="S2T,T2S,S2T|T2S,S2T&T2S")
parser.add_argument("--dico_threshold", type=float, default=0, help="Threshold confidence for dictionary generation")
parser.add_argument("--dico_max_rank", type=int, default=0, help="Maximum dictionary words rank (0 to disable)")
parser.add_argument("--dico_min_size", type=int, default=0, help="Minimum generated dictionary size (0 to disable)")
parser.add_argument("--dico_max_size", type=int, default=0, help="Maximum generated dictionary size (0 to disable)")
parser.add_argument("--cuda", type=bool_flag, default=True, help="Run on GPU")
parser.add_argument("--output", type=str, default="", help="output path of the dictionary")

# parse parameters
params = parser.parse_args()

def load_vec(emb_path, nmax=500000):
    vectors = []
    word2id = {}
    with open(emb_path, 'r', encoding="utf-8") as f:
        next(f)
        for i, line in enumerate(f):
            word, vect = line.rstrip().split(' ', 1)
            vect = np.fromstring(vect, sep=' ')
            assert word not in word2id, 'word found twice'
            vectors.append(vect)
            word2id[word] = len(word2id)
            if len(word2id) == nmax:
                break
    id2word = {v: k for k, v in word2id.items()}
    embeddings = np.vstack(vectors)
    return embeddings, id2word, word2id

def load_vec(emb_path, vocab_path):
    lrl_vectors = []
    hrl_vectors = []
    lrl_word2id = {}
    hrl_word2id = {}
    with open(emb_path, "r", encoding="utf-8") as f:
        with open(vocab_path, "r", encoding="utf-8") as f2:
            next(f2)
            for i, (line1, line2) in enumerate(zip(f, f2)):
                vect = line1.strip()
                vect = np.fromstring(vect, sep=" ")
                word, lrl_hrl = line2.split()
                if lrl_hrl == "lrl":
                    assert word not in lrl_word2id, 'word found twice'
                    lrl_vectors.append(vect)
                    lrl_word2id[word] = len(lrl_word2id)
                else:
                    assert word not in hrl_word2id, 'word found twice'
                    hrl_vectors.append(vect)
                    hrl_word2id[word] = len(hrl_word2id)
    lrl_id2word = {v: k for k, v in lrl_word2id.items()}
    hrl_id2word = {v: k for k, v in hrl_word2id.items()}
    lrl_emb = np.vstack(lrl_vectors)
    hrl_emb = np.vstack(hrl_vectors)
    return lrl_emb, hrl_emb, lrl_id2word, hrl_id2word, lrl_word2id, hrl_word2id



if __name__ == '__main__':
    # src_word_embs, src_id2word, src_word2id = load_vec(params.src_emb)
    # tgt_word_embs, tgt_id2word, tgt_word2id = load_vec(params.tgt_emb)

    src_word_embs, tgt_word_embs, src_id2word, tgt_id2word, src_word2id, tgt_word2id = \
        load_vec(params.lrl_hrl_emb, params.lrl_hrl_vocab)
    src_word_embs = torch.FloatTensor(src_word_embs).cuda()
    tgt_word_embs = torch.FloatTensor(tgt_word_embs).cuda()
    dictionary = build_dictionary(src_emb=src_word_embs, tgt_emb=tgt_word_embs, params=params)
    f = codecs.open(params.output, 'w', encoding='utf8')
    for k, (i, j) in enumerate(dictionary):
        s_word = src_id2word[i]
        t_word = tgt_id2word[j]
        f.write(s_word + " " + t_word + "\n")
        print(k)
    print(dictionary.shape)
    print(dictionary[0])
