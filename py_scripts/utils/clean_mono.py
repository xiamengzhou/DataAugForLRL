aze_alphabet = ['A', "a", "B", "b", "C", "c", "Ç", "ç", "D", "d", "E", "e", "Ə", "ə", "F", "f",
                "G", "g", "Ğ", "ğ", "H", "h", "X", "x", "I", "ı", "İ", "i", "J", "j", "K", "k",
                "Q", "q", "L", "l", "M", "m", "N", "n", "O", "o", "Ö", "ö", "P", "p", "R", "r",
                "S", "s", "Ş", "ş", "T", "t", "U", "u", "Ü", "ü", "V", "v", "Y", "y", "Z", "z"]
tur_alphabet = ['A', "a", "B", "b", "C", "c", "Ç", "ç", "D", "d", "E", "e", "F", "f",
                "G", "g", "Ğ", "ğ", "H", "h", "X", "x", "I", "ı", "İ", "i", "J", "j", "K", "k",
                "Q", "q", "L", "l", "M", "m", "N", "n", "O", "o", "Ö", "ö", "P", "p", "R", "r",
                "S", "s", "Ş", "ş", "T", "t", "U", "u", "Ü", "ü", "V", "v", "Y", "y", "Z", "z"]

def clean(file, outfile, lang):
    if lang == "aze":
        alphabet = aze_alphabet
    elif lang == "tur":
        alphabet = tur_alphabet
    f = open(file, "r").readlines()
    f2 = open(outfile, "w")
    for line in f:
        if len(list(set(line).intersection(alphabet))) == 0:
            pass
        else:
            f2.write(line)


