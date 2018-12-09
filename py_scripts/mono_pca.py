from sklearn.decomposition import PCA
import numpy as np
import matplotlib as mpl
import sys, os
mpl.use('Agg')
import matplotlib.pyplot as plt
lang1 = sys.argv[1]
lang2 = sys.argv[2]
pca_save = sys.argv[3]
def extract_vector(file_name):
	vectors = []
	idx = 0
	with open(file_name) as f:
		next(f)
		for line in f:
			vectors.append(line.strip().split()[1:])
			idx += 1
	return np.asarray(vectors)

def pca_transform(data):
    model = PCA(n_components=2)
    re = model.fit_transform(data)
    assert re.shape[0] == len(data)
    assert re.shape[1] == 2
    return re

l1_vectors = extract_vector(lang1)
l2_vectors = extract_vector(lang2)

l1_pca = pca_transform(l1_vectors)
l2_pca = pca_transform(l2_vectors)
print(l1_pca[:][0].shape)
plt.scatter(l1_pca[:, 0], l1_pca[:, 1])
plt.scatter(l2_pca[:, 0], l2_pca[:, 1])

plt.savefig(os.path.join(pca_save, 'pca.png'))
