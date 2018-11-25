#!/usr/bin/env bash

# Sentence Piece monolingual wikipedia
data_dir=$data/11731_final/mono

for lang in az be cs gl pt ru sk tr; do
python3 ~/NMT/rapid/sentence_piece.py encode $data_dir/${lang}_mono/${lang}--vocab_size=8000.model \
                                             $data_dir/${lang}_mono/${lang}.wiki.txt \
                                             $data_dir/${lang}_mono/${lang}.wiki.txt.spm8000
done

# Word dictionary
data_dir=$data/11731_final/bilang
mkdir $data/11731_final/vocab
for lang in aze tur bel rus glg por slk ces; do
python3 ~/NMT/rapid/utils.py vocab $data_dir/${lang}_eng/ted-train.orig.${lang}.tok.spm8k \
                                   $data_dir/../vocab/${lang}.vocab.spm8k

python3 ~/NMT/rapid/utils.py vocab $data_dir/${lang}_eng/ted-train.orig.${lang}.tok \
                                   $data_dir/../vocab/${lang}.vocab.tok
done

for lang in azetur belrus glgpor slkces; do
python3 ~/NMT/rapid/utils.py vocab $data_dir/${lang}_eng/ted-train.orig.${lang}.tok\
                                   $data_dir/../vocab/${lang}.vocab
python3 ~/NMT/rapid/utils.py vocab $data_dir/${lang}_eng/ted-train.orig.${lang}.tok.spm8k \
                                   $data_dir/../vocab/${lang}.vocab.spm8k
done