#!/usr/bin/env bash

UTIL=/home/mengzhox/NMT/rapid/utils.py

# Swap words in monolingual data
lang1=az
lang2=tr
lang22=tur
mkdir -p $data/unsup/mono/${lang1}${lang2}_swapped3
dir=${lang1}${lang2}_swapped3
dict=$data/11731_final/mono/using/fasttext/MUSE_SUP/${lang1}${lang2}/1/T2S
python3 $UTIL swap2 $data/11731_final/mono/using/${lang2}.wiki.tok.txt \
                    $data/unsup/mono/$dir/${lang2}.wiki.tok.txt \
                    $dict
cp $data/11731_final/mono/using/${lang1}.wiki.tok.txt $data/unsup/mono/$dir/${lang1}.wiki.tok.txt

mkdir -p $data/unsup/bilang/$dir
python3 $UTIL swap2 $data/unsup/bilang/${lang1}${lang2}/train.${lang1}-${lang2}.${lang2}.txt.clean \
                    $data/unsup/bilang/$dir/train.${lang1}-${lang2}.${lang2}.txt.clean \
                    $dict

python3 $UTIL swap2 $data/unsup/bilang/${lang1}${lang2}/dev.${lang1}-${lang2}.${lang2}.txt.clean \
                    $data/unsup/bilang/$dir/dev.${lang1}-${lang2}.${lang2}.txt.clean \
                    $dict

python3 $UTIL swap2 $data/unsup/bilang/${lang1}${lang2}/test.${lang1}-${lang2}.${lang2}.txt.clean \
                    $data/unsup/bilang/$dir/test.${lang1}-${lang2}.${lang2}.txt.clean \
                    $dict

python3 $UTIL swap2 $data/11731_final/bilang/${lang22}_eng/ted-train.orig.${lang22}.tok \
                    $data/unsup/mono/$dir/ted-train.orig.${lang22}.tok \
                    $dict

cp $data/unsup/bilang/${lang1}${lang2}/train.${lang1}-${lang2}.${lang1}.txt.clean \
   $data/unsup/bilang/$dir/train.${lang1}-${lang2}.${lang1}.txt.clean

cp $data/unsup/bilang/${lang1}${lang2}/dev.${lang1}-${lang2}.${lang1}.txt.clean \
   $data/unsup/bilang/$dir/dev.${lang1}-${lang2}.${lang1}.txt.clean

cp $data/unsup/bilang/${lang1}${lang2}/test.${lang1}-${lang2}.${lang1}.txt.clean \
   $data/unsup/bilang/$dir/test.${lang1}-${lang2}.${lang1}.txt.clean

cat $data/unsup/bilang/$dir/train.${lang1}-${lang2}.${lang1}.txt.clean \
    $data/unsup/bilang/$dir/dev.${lang1}-${lang2}.${lang1}.txt.clean \
    $data/unsup/bilang/$dir/test.${lang1}-${lang2}.${lang1}.txt.clean > \
    $data/unsup/bilang/$dir/all.${lang1}-${lang2}.${lang1}.txt.clean

cat $data/unsup/bilang/$dir/train.${lang1}-${lang2}.${lang2}.txt.clean \
    $data/unsup/bilang/$dir/dev.${lang1}-${lang2}.${lang2}.txt.clean \
    $data/unsup/bilang/$dir/test.${lang1}-${lang2}.${lang2}.txt.clean > \
    $data/unsup/bilang/$dir/all.${lang1}-${lang2}.${lang2}.txt.clean




#!/usr/bin/env bash

UTIL=/home/mengzhox/NMT/rapid/utils.py

# Swap words in monolingual data
lang1=cs #hrl
lang2=sk #lrl
lang21=ces
mkdir -p $data/unsup/mono/${lang1}${lang2}_swapped3
dir=${lang1}${lang2}_swapped3
dict=$data/11731_final/mono/using/fasttext/MUSE_SUP/${lang2}${lang1}/1/T2S
python3 $UTIL swap2 $data/11731_final/mono/using/${lang1}.wiki.tok.txt \
                    $data/unsup/mono/$dir/${lang1}.wiki.tok.txt \
                    $dict
cp $data/11731_final/mono/using/${lang2}.wiki.tok.txt $data/unsup/mono/$dir/${lang2}.wiki.tok.txt

mkdir -p $data/unsup/bilang/$dir
python3 $UTIL swap2 $data/unsup/bilang/${lang2}${lang1}/train.${lang2}-${lang1}.${lang1}.txt.clean \
                    $data/unsup/bilang/$dir/train.${lang1}-${lang2}.${lang1}.txt.clean \
                    $dict

python3 $UTIL swap2 $data/unsup/bilang/${lang2}${lang1}/dev.${lang2}-${lang1}.${lang1}.txt.clean \
                    $data/unsup/bilang/$dir/dev.${lang1}-${lang2}.${lang1}.txt.clean \
                    $dict

python3 $UTIL swap2 $data/unsup/bilang/${lang2}${lang1}/test.${lang2}-${lang1}.${lang1}.txt.clean \
                    $data/unsup/bilang/$dir/test.${lang1}-${lang2}.${lang1}.txt.clean \
                    $dict

python3 $UTIL swap2 $data/11731_final/bilang/${lang21}_eng/ted-train.orig.${lang21}.tok \
                    $data/unsup/mono/$dir/ted-train.orig.${lang21}.tok \
                    $dict

cp $data/unsup/bilang/${lang2}${lang1}/train.${lang2}-${lang1}.${lang2}.txt.clean \
   $data/unsup/bilang/$dir/train.${lang1}-${lang2}.${lang2}.txt.clean

cp $data/unsup/bilang/${lang2}${lang1}/dev.${lang2}-${lang1}.${lang2}.txt.clean \
   $data/unsup/bilang/$dir/dev.${lang1}-${lang2}.${lang2}.txt.clean

cp $data/unsup/bilang/${lang2}${lang1}/test.${lang2}-${lang1}.${lang2}.txt.clean \
   $data/unsup/bilang/$dir/test.${lang1}-${lang2}.${lang2}.txt.clean

cat $data/unsup/bilang/$dir/train.${lang1}-${lang2}.${lang1}.txt.clean \
    $data/unsup/bilang/$dir/dev.${lang1}-${lang2}.${lang1}.txt.clean \
    $data/unsup/bilang/$dir/test.${lang1}-${lang2}.${lang1}.txt.clean > \
    $data/unsup/bilang/$dir/all.${lang1}-${lang2}.${lang1}.txt.clean

cat $data/unsup/bilang/$dir/train.${lang1}-${lang2}.${lang2}.txt.clean \
    $data/unsup/bilang/$dir/dev.${lang1}-${lang2}.${lang2}.txt.clean \
    $data/unsup/bilang/$dir/test.${lang1}-${lang2}.${lang2}.txt.clean > \
    $data/unsup/bilang/$dir/all.${lang1}-${lang2}.${lang2}.txt.clean