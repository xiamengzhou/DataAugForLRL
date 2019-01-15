"""
    python3 ~/11731_final/py_scripts/gensim.py train /projects/tir3/users/mengzhox/data/unsup/mono/aztr/all.az-tr.spm8k \
                      /projects/tir3/users/mengzhox/data/unsup/mono/aztr/emb/emb.spm8k \
                      256 \
                      /projects/tir3/users/mengzhox/data/unsup/mono/aztr/emb/emb.spm8k.vec
"""

# Gensim
def train_word2vec_while_fixing(train_file, pretrained="", size=256, export=""):
    from gensim_py.models import Word2Vec
    sentences = open(train_file, "r").readlines()
    sentences = [s.split() for s in sentences]
    model = Word2Vec(size=size, min_count=0, negative=10, iter=10)
    model.build_vocab(sentences)
    # /projects/tir3/users/mengzhox/data/unsup/mono/aztr/emb/emb.spm8k
    model.intersect_word2vec_format(pretrained, lockf=0.0, binary=False)
    model.train(sentences, total_examples=model.corpus_count, epochs=model.iter)

    export_file = open(export, "w")
    if "</s>" in model.wv.vocab:
        count = len(model.wv.vocab)
        export_file.write(str(count) + " " + str(size) + "\n")
    else:
        count = len(model.wv.vocab) + 1
        export_file.write(str(count) + " " + str(size) + "\n")
        a = open(pretrained, "r")
        a.readline()
        export_file.write(a.readline())

    for i, v in enumerate(model.wv.vocab):
        k = model.wv.vectors[i]
        k = [str(a) for a in k]
        export_file.write(v + " " + " ".join(k) + "\n")
    print("Pretrained vector file exported to {}!".format(export))


if __name__ == '__main__':
    import sys
    if sys.argv[1] == "train":
        train_word2vec_while_fixing(train_file=sys.argv[2], pretrained=sys.argv[3],
                                    size=int(sys.argv[4]), export=sys.argv[5])