import sentencepiece as spm
import sys
import codecs
sp = spm.SentencePieceProcessor()
model=sys.argv[1]
sp.Load(model)
file = sys.argv[2]
with codecs.open(file, "r", "utf-8") as f, codecs.open(file+'.spm8k', 'w', 'utf-8') as f1:
	for line in f:
		pieces = sp.EncodeAsPieces(line)
		f1.write(' '.join(pieces) + '\n')
