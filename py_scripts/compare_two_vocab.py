import sys
import codecs
file1 = sys.argv[1]
file2 = sys.argv[2]
one_vocab = []
two_vocab = []
with codecs.open(file1, "r", "utf-8") as f1, codecs.open(file2, "r", "utf-8") as f2:
	for l1, l2 in zip(f1, f2):
		one_vocab.extend(l1.split())
		two_vocab.extend(l2.split())
one_vocab_set = set(one_vocab)
two_vocab_set = set(two_vocab)
union = one_vocab_set & two_vocab_set
found = [item for item in one_vocab if item in union]
#print len(one_vocab), len(two_vocab), float(len(union)) / len(one_vocab)
print len(one_vocab_set), len(two_vocab_set), float(len(union)) / len(one_vocab_set), float(len(found)) / len(one_vocab)
