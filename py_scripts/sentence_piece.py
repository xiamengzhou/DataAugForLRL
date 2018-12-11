from sentencepiece import SentencePieceProcessor, SentencePieceTrainer_Train
from utils import load_files, output_lines, load_dict, output_dict
import sys

def train(input, output, vocab_size):
    s = SentencePieceTrainer_Train("--input={} "
                                   "--model_prefix={} "
                                   "--vocab_size={}".format(input, output, vocab_size))
    print("Trained done. Model in {}.model".format(output))


def encode(model, f, output):
    new_lines = []
    s = SentencePieceProcessor()
    s.load(model)
    lines = load_files(f)
    for line in lines:
        line = line.strip()
        tokens = s.encode_as_pieces(line)
        new_lines.append(tokens)
    output_lines(output, new_lines)

def encode_dict(model, model2, d, output, sep=" ||| "):
    s = SentencePieceProcessor()
    s.load(model)
    s2 = SentencePieceProcessor()
    s2.load(model2)
    new_d = {}
    f = load_files(d)
    for line in f:
        k, v = line.split(sep)
        k = " ".join(s.encode_as_pieces(k))
        v = " ".join(s2.encode_as_pieces(v))
        new_d[k] = v
    output_dict(output, new_d, sep)
    print("Encode dict to {}".format(output))


if __name__ == '__main__':
    if sys.argv[1] == "train":
        train(input=sys.argv[2], output=sys.argv[3], vocab_size=int(sys.argv[4]))
    elif sys.argv[1] == "encode":
        encode(model=sys.argv[2], f=sys.argv[3], output=sys.argv[4])
    elif sys.argv[1] == "encode_dict":
        encode_dict(model=sys.argv[2], model2=sys.argv[3], d=sys.argv[4], output=sys.argv[5], sep=sys.argv[6])