import torch
import sys

def get_data(path):
    sys.path.append("/home/mengzhox/UMT/NMT")
    sys.path.append("/home/mengzhox/UMT/NMT/src")
    file = torch.load(path)
    return file

train = get_data("train.az-tr.tr.txt.clean.spm8k.pth")
dev = get_data("dev.az-tr.tr.txt.clean.spm8k.pth")
test = get_data("test.az-tr.tr.txt.clean.spm8k.pth")
print(train["dico"] == dev["dico"])
print(train["dico"] == test["dico"])
print(dev["dico"] == test["dico"])

train = get_data("train.az-tr.az.txt.clean.spm8k.pth")
dev = get_data("dev.az-tr.az.txt.clean.spm8k.pth")
test = get_data("test.az-tr.az.txt.clean.spm8k.pth")
print(train["dico"] == dev["dico"])
print(train["dico"] == test["dico"])
print(dev["dico"] == test["dico"])