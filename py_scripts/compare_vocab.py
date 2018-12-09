import sys
import codecs
file1 = sys.argv[1]
num = int(sys.argv[2])
one_vocab = []
two_vocab = []
line_num = 0
with codecs.open(file1, "r", "utf-8") as f:
	for line in f:
		if line_num < num:
			one_vocab.extend(line.split())
		else:
			two_vocab.extend(line.split())
		line_num += 1
one_vocab_set = set(one_vocab)
two_vocab_set = set(two_vocab)
union = one_vocab_set & two_vocab_set
found = [item for item in one_vocab if item in union]
#print len(one_vocab), len(two_vocab), float(len(union)) / len(one_vocab)
print len(one_vocab_set), len(two_vocab_set), float(len(union)) / len(one_vocab_set), float(len(found)) / len(one_vocab)
