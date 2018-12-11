import nltk
def sent_tokenize_wiki():
    f = open("pt.wiki.txt2", "r").readlines()
    f2 = open("pt.wiki.tok.txt", "w")
    for line in f:
        if len(line.split()) == 0:
            continue
        sent_text = nltk.sent_tokenize(line)
        for sent in sent_text:
            f2.write(sent.strip() + "\n")

sent_tokenize_wiki()