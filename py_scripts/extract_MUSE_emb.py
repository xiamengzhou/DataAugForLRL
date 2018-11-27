import numpy as np

def extract_MUSE_emb(f_in, f_out):
    file = open(f_in, "r").readlines()[1:]
    f_out = open(f_out, "w")
    for line in file:
        l = line.rstrip().split(' ', 1)[1]
        vect = np.fromstring(l, sep=" ")
        vect = [str(a) for a in vect]
        f_out.write("\t".join(vect) + "\n")
