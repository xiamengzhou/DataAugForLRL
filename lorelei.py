import os
import sys
from collections import defaultdict

def get_dict(lines):
    d = {}
    for line in lines:
        tokens = line.split()
        for t in tokens:
            if t in d:
                d[t] += 1
            else:
                d[t] = 1
    return d

def check_lexicon_overlap(dir="/projects/tir1/corpora/multiling-text/bible-com"):
    langs = os.listdir(dir)
    d = {}
    for i, lang in enumerate(langs):
        path = os.path.join(dir, lang, "bible.orig.{}-eng".format(lang[:3]))
        file = open(path, "r")
        lines = file.readlines()
        lines = [l.split("|||")[0].strip() for l in lines]
        di = get_dict(lines)
        d[lang[:3]] = di
        if i % 50 == 0:
            print(lang + " done!")
            print("{} langs are done!".format(str(i)))
    return d

def check_lexicon_overlap2(dir="/projects/tir1/corpora/multiling-text/bible-is"):
    d = {}
    for i, lang in enumerate(["tir_eng", "orm_eng"]):
        path = os.path.join(dir, lang, "bible-is.orig.{}-eng".format(lang[:3]))
        file = open(path, "r")
        lines = file.readlines()
        lines = [l.split("|||")[0].strip() for l in lines]
        di = get_dict(lines)
        d[lang[:3]] = di
        if i % 50 == 0:
            print(lang + " done!")
            print("{} langs are done!".format(str(i)))
    return d

import operator
def save_lexicon(d):
    dir = "/projects/tir3/users/mengzhox/data/lorelei"
    for lang in d:
        f = open(os.path.join(dir, "{}.lex".format(lang)), "w")
        sorted_x = sorted(d[lang].items(), key=operator.itemgetter(1))
        for key, value in sorted_x[::-1]:
            f.write(key + " " + str(value) + "\n")
        f.close()

def check_lang(d, d2, lang):
    d_lang = d2[lang]
    lex_lang = set(list(d_lang.keys()))
    re = {}
    for key in d:
        if key != lang:
            d_key = d[key]
            lex_key = set(list(d_key.keys()))
            num = len(lex_lang.intersection(lex_key))
            re[key] = (num, len(lex_key))
    re = sorted(re.items(), key=lambda x: -x[1][0])
    return re





