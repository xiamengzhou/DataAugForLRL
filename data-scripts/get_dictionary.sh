#!/usr/bin/env bash

# This script runs word embedding alignment and dictionary extraction.

# MUSE=third-party/MUSE
# FASTTEXT=third-party/fastText-0.1.0/fasttext
MUSE=~/MUSE
FASTTEXT=~/fastText-0.1.0/fasttext

UTIL=utils.py
MUSE_DICT=get_muse_dict.py

lrl=az
hrl=tr
lang=${lrl}${hrl}

data_dir=../data-ex
mono_dir=${data_dir}/mono
out_dir=${data_dir}/out/dict/$lang

mkdir -p $out_dir

src_emb=$mono_dir/${lrl}/${lrl}.vec
tgt_emb=$mono_dir/${hrl}/${hrl}.vec
src_mono=$mono_dir/${lrl}/${lrl}.mono
tgt_mono=$mono_dir/${hrl}/${hrl}.mono
src_vocab=$mono_dir/${lrl}/${lrl}.vocab
tgt_vocab=$mono_dir/${hrl}/${hrl}.vocab

# usually there is no evaluation dictionary, so just use a random one
dico_eval=$MUSE/data/crosslingual/dictionaries/it-en.5000-6500.txt
dico_train=$out_dir/${lang}_seed.txt
exp_id=1

# strategies to extract dictionaries
stras=("T2S" "S2T" "S2T&T2S") # S2T|T2S|S2T&T2S
stra_outs=("T2S" "S2T" "S2TT2S") # S2T|T2S|S2TT2S

# train word embeddings
if [[ ! -f $src_emb ]]; then
$FASTTEXT skipgram -epoch 10 -minCount 0 -dim 300 -thread 48 -ws 5 \
                   -neg 10 -input $src_mono \
                   -output $mono_dir/${lrl}/${lrl}
fi

if [[ ! -f $tgt_emb ]]; then
$FASTTEXT skipgram -epoch 10 -minCount 0 -dim 300 -thread 48 -ws 5 \
                   -neg 10 -input $tgt_mono \
                   -output $mono_dir/${hrl}/${hrl}
fi

# extract word vocabulary
if [[ ! -f $src_vocab ]]; then
cat $src_mono  | tr ' ' '\n' | sort | uniq  > $src_vocab
fi

if [[ ! -f $tgt_vocab ]]; then
cat $tgt_mono | tr ' ' '\n' | sort | uniq > $tgt_vocab
fi

# extract seed dictionary (identical strings)
if [[ ! -f $dico_train ]]; then
sort $src_vocab $tgt_vocab | uniq -d > ${dico_train}_temp
awk '{print $0,$0}' < ${dico_train}_temp > ${dico_train}
rm ${dico_train}_temp
fi

# supervised mapping
mkdir $out_dir/${lang}_sup
if [ ! -f $out_dir/${lang}_sup/$exp_id/vectors-${lrl}.txt ] | \
   [ ! -f $out_dir/${lang}_sup/$exp_id/vectors-${hrl}.txt ]; then
    if [[ -d $out_dir/${lang}_sup/$exp_id ]]; then
        python3 $MUSE/supervised.py  --src_lang $lrl \
                             --tgt_lang $hrl \
                             --n_refinement 5 \
                             --normalize_embeddings center \
                             --exp_path $out_dir \
                             --exp_name ${lang}_sup \
                             --exp_id $exp_id \
                             --dico_train $dico_train \
                             --dico_eval $dico_eval \
                             --src_emb $src_emb \
                             --tgt_emb $tgt_emb
    fi
fi

# extract dictionary from the mapped space
for ((i=0;i<${#stras[@]};++i)); do
stra=${stras[i]}
stra_out=${stra_outs[i]}
if [[ ! -f $out_dir/${lang}_sup/${exp_id}/re_${stra_out} ]]; then
python3 $MUSE_DICT --src_emb $out_dir/${lang}_sup/${exp_id}/vectors-${lrl}.txt \
                   --tgt_emb $out_dir/${lang}_sup/${exp_id}/vectors-${hrl}.txt \
                   --dico_build $stra \
                   --output $out_dir/${lang}_sup/${exp_id}/re_${stra_out}
fi
done