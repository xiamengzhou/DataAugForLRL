import sys
import torch

import matplotlib                                                                                                         
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils import load_model, load_vocab, load_dict
from sklearn import decomposition
sys.path.append("/home/xiangk/11731-final/code")
#sys.path.append("/home/mengzhox/11731_final")
model_path = sys.argv[1]
model = load_model(model_path)
src_embedding = model["model"]["encoder.embeddings.embeddings.weight"]
print(src_embedding.shape)
pca = decomposition.PCA(n_components=2)
pca.fit(src_embedding)
pca_src_embedding = pca.transform(src_embedding)
plt.scatter(pca_src_embedding[:,0], pca_src_embedding[:,1])
plt.savefig('pca.png')