"""
Make a dataset so that same word can be of different embedding in aze and tur.

python3 make_dataset.py $data/11731_final/bilang/azetur_eng/ted-train.orig.azetur.tok.spm8k \
                        $data/11731_final/bilang/azetur_eng/ted-train.orig.azetur.tok.sepspm8k \
                        $data/11731_final/vocab/aze.vocab.spm8k \
                        $data/11731_final/vocab/tur.vocab.spm8k \
                        5946
"""

from utils import load_dict
import sys

def make_dataset(train_src, output_file, lrl_vocab, hrl_vocab, cutoff):
    f = open(train_src, "r").readlines()[cutoff:]
    f2 = open(output_file, "w")
    lrl_vocab = load_dict(lrl_vocab)
    hrl_vocab = load_dict(hrl_vocab)
    for line in f[:cutoff]:
        f2.write(line)
    for line in f:
        tokens = line.split()
        for i, t in enumerate(tokens):
            if t in lrl_vocab and t in hrl_vocab:
                tokens[i] = t+"2"
        f2.write(" ".join(tokens) + "\n")

if __name__ == '__main__':
    make_dataset(train_src=sys.argv[1],
                 output_file=sys.argv[2],
                 lrl_vocab=sys.argv[3],
                 hrl_vocab=sys.argv[4],
                 cutoff=int(sys.argv[5]))
