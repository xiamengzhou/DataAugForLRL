"""
python3 extract_embeddings.py sub $out/11731_final/spm8000/models/spm8000_acc_56.77_ppl_10.70_e15_s0.pt \
                              $data/11731_final/processed/spm8000/azetur/spm8000.vocab.pt \
                              $data/11731_final/analysis/azetur.tran.emb \
                              $data/11731_final/analysis/azetur.tran.emb.tag \
                              $data/11731_final/vocab/aze.vocab.spm8k \
                              $data/11731_final/vocab/tur.vocab.spm8k

python3 extract_embeddings.py word $out/11731_final/spm8000/models/spm8000_acc_56.77_ppl_10.70_e15_s0.pt \
                              $data/11731_final/processed/spm8000/azetur/spm8000.vocab.pt \
                              $data/11731_final/analysis/azetur.tran.emb.tok \
                              $data/11731_final/analysis/azetur.tran.emb.tok.tag \
                              $data/11731_final/vocab/aze.vocab.spm8k \
                              $data/11731_final/vocab/tur.vocab.spm8k \
                              $data/11731_final/mono/az_mono/az--vocab_size=8000.model \
                              $data/11731_final/mono/tr_mono/tr--vocab_size=8000.model
"""

from utils import load_model, load_vocab, load_dict
import sys
from sentencepiece import SentencePieceProcessor
import torch


def extract_sub(model_path, vocab_path, embed_out_path, vocab_out_path, lrl_vocab_path, hrl_vocab_path):
    model = load_model(model_path)
    src_embedding = model["model"]["encoder.embeddings.embeddings.weight"]
    # tgt_embedding = model["model"]["decoder.embeddings.embeddings.weight"]
    src_vocab, _ = load_vocab(vocab_path)

    embedding_output = open(embed_out_path, "w")
    vocab_output = open(vocab_out_path, "w")

    lrl_vocab = load_dict(lrl_vocab_path)
    hrl_vocab = load_dict(hrl_vocab_path)

    for i, emb in enumerate(src_embedding):
        emb_str = [str(e) for e in emb]
        assert len(emb_str) == 512
        embedding_output.write("\t".join(emb_str))
        embedding_output.write("\n")

        w = src_vocab.itos[i]
        if w in lrl_vocab and w in hrl_vocab:
            vocab_output.write(w + "\t" + "lrl+hrl" + "\n")
        elif w in lrl_vocab:
            vocab_output.write(w + "\t" + "lrl" + "\n")
        else:
            vocab_output.write(w + "\t" + "hrl" + "\n")

def extract_word(model_path, vocab_path, embed_out_path, vocab_out_path,
                 lrl_vocab_path, hrl_vocab_path, lrl_spm_model, hrl_spm_model):
    lrl_s = SentencePieceProcessor()
    lrl_s.load(lrl_spm_model)
    hrl_s = SentencePieceProcessor()
    hrl_s.load(hrl_spm_model)

    model = load_model(model_path)
    src_embedding = model["model"]["encoder.embeddings.embeddings.weight"]
    # tgt_embedding = model["model"]["decoder.embeddings.embeddings.weight"]
    src_vocab, _ = load_vocab(vocab_path)

    embedding_output = open(embed_out_path, "w")
    vocab_output = open(vocab_out_path, "w")

    lrl_vocab = load_dict(lrl_vocab_path)
    hrl_vocab = load_dict(hrl_vocab_path)

    for w in lrl_vocab:
        tokens = lrl_s.encode_as_pieces(w)
        emb = torch.zeros(512)
        for t in tokens:
            emb += src_embedding[src_vocab.stoi[t]]
        emb_str = [str(e) for e in emb]
        assert len(emb_str) == 512
        embedding_output.write("\t".join(emb_str))
        embedding_output.write("\n")
        vocab_output.write(w + "\t" + "lrl" + "\n")

    for w in hrl_vocab:
        tokens = hrl_s.encode_as_pieces(w)
        emb = torch.zeros(512)
        for t in tokens:
            emb += src_embedding[src_vocab.stoi[t]]
        emb_str = [str(e) for e in emb]
        assert len(emb_str) == 512
        embedding_output.write("\t".join(emb_str))
        embedding_output.write("\n")
        vocab_output.write(w + "\t" + "hrl" + "\n")


if __name__ == '__main__':
    if sys.argv[1] == "sub":
        extract_sub(model_path=sys.argv[2],
                    vocab_path=sys.argv[3],
                    embed_out_path=sys.argv[4],
                    vocab_out_path=sys.argv[5],
                    lrl_vocab_path=sys.argv[6],
                    hrl_vocab_path=sys.argv[7])

    elif sys.argv[1] == "word":
        extract_word(model_path=sys.argv[2],
                     vocab_path=sys.argv[3],
                     embed_out_path=sys.argv[4],
                     vocab_out_path=sys.argv[5],
                     lrl_vocab_path=sys.argv[6],
                     hrl_vocab_path=sys.argv[7],
                     lrl_spm_model=sys.argv[8],
                     hrl_spm_model=sys.argv[9])



