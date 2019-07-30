#!/usr/bin/env bash

lrl=az
hrl=tr
lang=${lrl}${hrl}

UTIL=utils.py

data_dir=../data-ex
bilang_dir=$data_dir/bilang/$lang
out_dir=$data_dir/out

exp_id=1
stra_out=T2S
dict=$out_dir/dict/${lang}/${lang}_sup/${exp_id}/re_${stra_out}

# tokenized data
lrl_src=$bilang_dir/${lrl}en.${lrl}.tok
lrl_eng=$bilang_dir/${lrl}en.en.tok
hrl_src=$bilang_dir/${hrl}en.${hrl}.tok
hrl_eng=$bilang_dir/${hrl}en.en.tok

mkdir $bilang_dir/${lang}_${stra_out}
w_hrl_src=$bilang_dir/${lang}_${stra_out}/${lrl}en.${lrl}.tok
w_lrlhrl_src=$bilang_dir/${lang}_${stra_out}/${lang}en.${lang}.tok
w_lrlhrl_tgt=$bilang_dir/${lang}_${stra_out}/${lang}en.en.tok

# swap the hrl words to be lrl words
python3 $UTIL swap2 $hrl_src $w_hrl_src $dict " "

# concate swapped hrl file and lrl file
cat $lrl_src $w_hrl_src > $w_lrlhrl_src
cat $lrl_eng $hrl_eng > $w_lrlhrl_tgt
