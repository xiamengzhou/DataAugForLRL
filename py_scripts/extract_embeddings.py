from utils import load_model, load_vocab, load_dict
import sys

if __name__ == '__main__':
    model = load_model(sys.argv[1])
    src_embedding = model["model"]["encoder.embeddings.embeddings.weight"]
    # tgt_embedding = model["model"]["decoder.embeddings.embeddings.weight"]
    src_vocab = load_vocab(sys.argv[2])

    embedding_output = sys.argv[3]
    vocab_output = sys.argv[4]

    lrl_vocab = load_dict(sys.argv[5])
    hrl_vocab = load_dict(sys.argv[6])

    for i, emb in enumerate(src_embedding):
        emb_str = [str(e) for e in emb]
        assert len(emb_str) == 512
        embedding_output.write(" ".join(emb_str))
        embedding_output.write("\n")

        w = src_vocab.itos[i]
        if w in lrl_vocab and w in hrl_vocab:
            vocab_output.write(w + " " + "lrl+hrl" + "\n")
        elif w in lrl_vocab:
            vocab_output.write(w + " " + "lrl" + "\n")
        else:
            vocab_output.write(w + " " + "hrl" + "\n")




