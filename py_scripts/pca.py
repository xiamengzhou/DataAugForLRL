"""
python3 pca.py $data/11731_final/analysis/azetur.tran.emb.tok.dis \
               $data/11731_final/analysis/azetur.tran.emb.tok.dis.3
"""

from sklearn.decomposition import PCA
import numpy as np

def read_vectors(file):
    f = open(file, "r").readlines()
    f = [k.split() for k in f]
    f = [[float(s) for s in k] for k in f]
    return np.array(f)

def train_pca(data):
    model = PCA(n_components=3)
    re = model.fit_transform(data)
    assert re.shape[0] == len(data)
    assert re.shape[1] == 3
    return re

def output(data, embedding_name):
    f1 = open(embedding_name, "w")
    for d in data:
        d_str = [str(k) for k in d]
        f1.write("\t".join(d_str) + "\n")

if __name__ == '__main__':
    import sys
    data = read_vectors(sys.argv[1])
    re = train_pca(data)
    output(data, sys.argv[2])