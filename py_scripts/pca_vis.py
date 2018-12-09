from sklearn import decomposition
from sklearn import datasets
import numpy as np
import sys
import codecs
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

pca = decomposition.PCA(n_components=2)
file = sys.argv[1]
word_vecs = []
with codecs.open(file, "r", "utf-8") as f:
	for line in f:
		word_vecs.append(line.strip().split()[1:])
word_vecs = np.asarray(word_vecs[1:], dtype=np.float32)
pca.fit(word_vecs)
print word_vecs.shape
X = pca.transform(word_vecs)

plt.scatter(X[:,0], X[:,1])
plt.savefig('pca-spm8k.pdf')
plt.show()
