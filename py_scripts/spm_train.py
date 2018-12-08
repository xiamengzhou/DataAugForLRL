import sentencepiece as spm
import sys
file=sys.argv[1]
prefix=sys.argv[2]
spm.SentencePieceTrainer.Train('--input='+file +' --model_prefix=' + prefix + '--vocab_size=8000')