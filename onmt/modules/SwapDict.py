import torch
import numpy as np

class SwapDict():
    def __init__(self, swap_dict, ec_weight, src_field, sep, lrl_prob, device=0, sample_num=5000):
        f = open(swap_dict, "r").readlines()
        a = []
        b = []
        for line in f:
            tokens = line.strip().split(sep)
            tokens1 = tokens[0].strip().split()
            tokens2 = tokens[1].strip().split()
            a.append([t for t in tokens1])
            b.append([t for t in tokens2])
        a = src_field.process(a, device=-1, train=True)
        b = src_field.process(b, device=-1, train=True)
        self.a = a
        self.b = b
        self.src_field = src_field
        self.dict_size = self.a[0].shape[1]
        self.device = device
        self.sample_num = sample_num
        print("Loading swap dict from {} with size {}.".format(swap_dict, str(self.dict_size)))



        self.weight = []
        f2 = open(ec_weight, "r").readlines()
        for line in f2:
            self.weight.append(float(line.strip()))
        self.weight = torch.FloatTensor(self.weight)

        # Low Resource word frequency
        self.probs = self.get_prob(lrl_prob)

    def get_prob(self, lrl_prob):
        f = open(lrl_prob, "r").readlines()
        probs = []
        for line in f:
            p = float(line.strip())
            probs.append(p)
        self.probs = probs

    def sample(self):
        index = np.random.choice(range(self.dict_size), self.sample_num, False, p=self.probs)
        index = torch.LongTensor(index)
        weight = self.weight[index]
        a_index = self.a[0][:, index]
        a_length = self.a[1][index]
        b_index = self.b[0][:, index]
        b_length = self.b[1][index]
        if self.device >= 0:
            a_index = a_index.cuda(self.device)
            a_length = a_length.cuda(self.device)
            b_index = b_index.cuda(self.device)
            b_length = b_length.cuda(self.device)
            weight = weight.device(self.device)
        return ((a_index, a_length), (b_index, b_length)), weight









