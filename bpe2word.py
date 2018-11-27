import sys

def generate_nonbpe(file):
    f = open(file, "r")
    f2 = open("{}.word".format(file), "w")
    lines = f.readlines()
    for line in lines:
        tokens = line.split()
        token = ""
        tokens_ = []
        for i, t in enumerate(tokens):
            if t[0] == "▁":
                if token != "":
                    tokens_.append(token)
                token = t[1:]
            else:
                token += t
            if i == len(tokens) - 1:
                tokens_.append(token)
                break
        new_line = " ".join(tokens_)
        f2.write(new_line + "\n")

src_file = sys.argv[1]

generate_nonbpe(src_file)