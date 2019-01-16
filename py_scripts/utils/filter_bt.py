"""
    python3 filter_bt.py $data/11731_final/bilang/tur_eng/ted-train.orig.tur.tok.spm8k \
                         $out/unsup/aztr/tran/147301/tran0.tr-az.txt \
                         $data/11731_final/bilang/tur_eng/ted-train.mtok.spm8000.eng \
                         $out/unsup/aztr/tran/147301/tran0.tr-az.txt.clean \
                         $out/unsup/aztr/tran/147301/ted-train.mtok.spm8000.eng

"""

def filter(hrl_file, lrl_file, tgt_file, outlrl_file, outtgt_file):
    a = open(hrl_file, "r")
    b = open(lrl_file, "r")
    c = open(tgt_file, "r")
    d = open(outlrl_file, "w")
    e = open(outtgt_file, "w")
    for i, (line1, line2, line3) in enumerate(zip(a, b, c)):
        tokens1 = line1.split()
        tokens2 = line2.split()
        lens1 = len(tokens1)
        lens2 = len(tokens2)
        if abs(lens1 - lens2) / min(lens1, lens2) >= 2 or "▁& quot ; ▁& quot ; ▁& quot ; ▁& quot ;" in line2:
            pass
        else:
            d.write(line2)
            e.write(line3)

if __name__ == '__main__':
    import sys
    filter(hrl_file=sys.argv[1],
           lrl_file=sys.argv[2],
           tgt_file=sys.argv[3],
           outlrl_file=sys.argv[4],
           outtgt_file=sys.argv[5])
