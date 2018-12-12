import nltk
import sys

def readlines(f):
    return open(f, "r").readlines()

def analyze(ref, out1, out2, out3):
    for i, (line1, line2, line3, line4) in enumerate(zip(ref, out1, out2, out3)):
        line1 = line1.strip()
        out1_score = nltk.translate.bleu_score.sentence_bleu([line1], line2.strip())
        out2_score = nltk.translate.bleu_score.sentence_bleu([line1], line3.strip())
        out3_score = nltk.translate.bleu_score.sentence_bleu([line1], line4.strip())
        if out1_score < out2_score and out2_score < out3_score:
            print(i)
            print(line1)
            print(out1_score, line2, out2_score, line3, out3_score, line4)

lang = sys.argv[1]
analyze(readlines("/usr2/home/mengzhox/data/11731_final/bilang/{}_eng/ted-test.mtok.eng".format(lang)),
        readlines("{}.base".format(lang)),
        readlines("{}.soft".format(lang)),
        readlines("{}.swap".format(lang)))