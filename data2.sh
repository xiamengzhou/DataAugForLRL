#!/usr/bin/env bash

UTIL=/home/mengzhox/NMT/rapid/utils.py

# Swap words in monolingual data
lang1=az
lang2=tr
lang22=tur
mkdir -p $data/unsup/mono/${lang1}${lang2}_swapped
dict=$data/11731_final/mono/using/fasttext/MUSE_SUP/${lang1}${lang2}/1/S2TT2S
python3 $UTIL swap2 $data/11731_final/mono/using/${lang2}.wiki.tok.txt \
                    $data/unsup/mono/${lang1}${lang2}_swapped/${lang2}.wiki.tok.txt \
                    $dict
cp $data/11731_final/mono/using/${lang1}.wiki.tok.txt $data/unsup/mono/${lang1}${lang2}_swapped/${lang1}.wiki.tok.txt

mkdir -p $data/unsup/bilang/${lang1}${lang2}_swapped
python3 $UTIL swap2 $data/unsup/bilang/${lang1}${lang2}/train.${lang1}-${lang2}.${lang2}.txt.clean \
                    $data/unsup/bilang/${lang1}${lang2}_swapped/train.${lang1}-${lang2}.${lang2}.txt.clean \
                    $dict

python3 $UTIL swap2 $data/unsup/bilang/${lang1}${lang2}/dev.${lang1}-${lang2}.${lang2}.txt.clean \
                    $data/unsup/bilang/${lang1}${lang2}_swapped/dev.${lang1}-${lang2}.${lang2}.txt.clean \
                    $dict

python3 $UTIL swap2 $data/unsup/bilang/${lang1}${lang2}/test.${lang1}-${lang2}.${lang2}.txt.clean \
                    $data/unsup/bilang/${lang1}${lang2}_swapped/test.${lang1}-${lang2}.${lang2}.txt.clean \
                    $dict

python3 $UTIL swap2 $data/11731_final/bilang/${lang22}_eng/ted-train.orig.${lang22}.tok \
                    $data/unsup/mono/${lang1}${lang2}_swapped/ted-train.orig.${lang22}.tok \
                    $dict

cp $data/unsup/bilang/${lang1}${lang2}/train.${lang1}-${lang2}.${lang1}.txt.clean \
   $data/unsup/bilang/${lang1}${lang2}_swapped/train.${lang1}-${lang2}.${lang1}.txt.clean

cp $data/unsup/bilang/${lang1}${lang2}/dev.${lang1}-${lang2}.${lang1}.txt.clean \
   $data/unsup/bilang/${lang1}${lang2}_swapped/dev.${lang1}-${lang2}.${lang1}.txt.clean

cp $data/unsup/bilang/${lang1}${lang2}/test.${lang1}-${lang2}.${lang1}.txt.clean \
   $data/unsup/bilang/${lang1}${lang2}_swapped/test.${lang1}-${lang2}.${lang1}.txt.clean