#!/usr/bin/env bash

lang11=slk
lang12=ces
lang21=sk
lang22=cs
SENT=~/NMT/rapid/sentence_piece.py
data_dir=$data/hrl2lrlsup
model_dir=$data/11731_final/mono

python3 $SENT encode $model_dir/${lang11}_mono/${lang21}--vocab_size=8000.model \
                     $data/hrl2lrlsup/${lang11}${lang12}/train.${lang21}-${lang22}.${lang21}.txt.clean \
                     $data/hrl2lrlsup/${lang11}${lang12}/train.${lang21}-${lang22}.${lang21}.txt.clean.spm8k

python3 $SENT encode $model_dir/${lang12}_mono/${lang22}--vocab_size=8000.model \
                     $data/hrl2lrlsup/${lang11}${lang12}/train.${lang21}-${lang22}.${lang22}.txt.clean \
                     $data/hrl2lrlsup/${lang11}${lang12}/train.${lang21}-${lang22}.${lang22}.txt.clean.spm8k

python3 $SENT encode $model_dir/${lang11}_mono/${lang21}--vocab_size=8000.model \
                     $data/hrl2lrlsup/${lang11}${lang12}/dev.${lang21}-${lang22}.${lang21}.txt.clean \
                     $data/hrl2lrlsup/${lang11}${lang12}/dev.${lang21}-${lang22}.${lang21}.txt.clean.spm8k

python3 $SENT encode $model_dir/${lang12}_mono/${lang22}--vocab_size=8000.model \
                     $data/hrl2lrlsup/${lang11}${lang12}/dev.${lang21}-${lang22}.${lang22}.txt.clean \
                     $data/hrl2lrlsup/${lang11}${lang12}/dev.${lang21}-${lang22}.${lang22}.txt.clean.spm8k

python3 $SENT encode $model_dir/${lang11}_mono/${lang21}--vocab_size=8000.model \
                     $data/hrl2lrlsup/${lang11}${lang12}/test.${lang21}-${lang22}.${lang21}.txt.clean \
                     $data/hrl2lrlsup/${lang11}${lang12}/test.${lang21}-${lang22}.${lang21}.txt.clean.spm8k

python3 $SENT encode $model_dir/${lang12}_mono/${lang22}--vocab_size=8000.model \
                     $data/hrl2lrlsup/${lang11}${lang12}/test.${lang21}-${lang22}.${lang22}.txt.clean \
                     $data/hrl2lrlsup/${lang11}${lang12}/test.${lang21}-${lang22}.${lang22}.txt.clean.spm8k


array=( "aze" "bel" "glg" "slk" )
array2=( "tur" "rus" "por" "ces" )
array3=( "az" "be" "gl" "sk" )
array4=( "tr" "ru" "pt" "cs" )

data_type=hrl2lrlsup

for ((i=0;i<${#array[@]};++i)); do
lang11="${array[i]}"
lang12="${array2[i]}"
lang21="${array3[i]}"
lang22="${array4[i]}"

bt_file=$out/${data_type}/${lang11}${lang12}/${data_type}-v1/models/tran/*.std
mkdir $data/${data_type}/${lang11}${lang12}/test

cat $data/11731_final/bilang/${lang11}${lang12}_eng/ted-train.orig.${lang11}${lang12}.tok.spm8k \
    $data/11731_final/bilang/${lang11}_eng/ted-train.orig.${lang11}.tok.spm8k $bt_file > \
    $data/${data_type}/${lang11}${lang12}/test/ted-train.orig.${lang11}${lang12}.tok.spm8k
cat $data/11731_final/bilang/${lang11}${lang12}_eng/ted-train.mtok.spm8000.eng \
    $data/11731_final/bilang/${lang11}${lang12}_eng/ted-train.mtok.spm8000.eng > \
    $data/${data_type}/${lang11}${lang12}/test/ted-train.mtok.spm8000.eng
done

