"""
python3 form_pretrain_emb.py $data/11731_final/processed/sepspm8k/azetur/sepspm8k.vocab.pt \
                             $data/11731_final/mono/az_mono/az.200k.spm8k.vec \
                             $data/11731_final/mono/tr_mono/tr.200k.spm8k.vec \
                             $data/11731_final/pretrained/aztr.spm8k.pt
"""


from utils import load_vocab
import numpy as np
import torch
import sys
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

def form_pretrain_emb(vocab_path, lrl_emb, hrl_emb, output):
    vocab, _ = load_vocab(vocab_path)
    lrl_emb, lrl_id2word, lrl_word2id = load_vec(lrl_emb)
    hrl_emb, hrl_id2word, hrl_word2id = load_vec(hrl_emb)
    emb = np.zeros([len(vocab.stoi), 512])
    for i, w in enumerate(vocab.stoi):
        if i > 1:
            if w[-1] == "2" and w[:-1] in vocab.stoi:
                emb[i] = hrl_emb[hrl_word2id[w[-1]]]
            elif w in lrl_emb:
                emb[i] = lrl_emb[lrl_word2id[w]]
            elif w in hrl_emb:
                emb[i] = hrl_emb[hrl_word2id[w]]
            else:
                print("word {} is not anywhere".format(w))
                break
    torch.save(torch.from_numpy(emb), output)

if __name__ == '__main__':
    form_pretrain_emb(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])