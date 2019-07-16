#!/usr/bin/env bash

lrl=az
hrl=tr
lang=${lrl}${hrl}

out_dir=data-ex/out
mono_dir=data-ex/mono
spm_out_dir=data-ex/out/spm/${lang}

mkdir $spm_out_dir

exp_id=1
stra_out=T2S
dict=$out_dir/${lang}_sup/${exp_id}/re_${stra_out}

SENT=spm.py

# tokenized data
lrl_src=data-ex/${lrl}en.${lrl}.tok
lrl_eng=data-ex/${lrl}en.en.tok
hrl_src=data-ex/${hrl}en.${hrl}.tok
hrl_eng=data-ex/${hrl}en.en.tok

lrl_mono=$mono_dir/${lrl}/${lrl}.mono
hrl_mono=$mono_dir/${lrl}/${hrl}.mono
eng_mono=$mono_dir/en/en.mono

lrlhrl_src=data-ex/${lang}en.${lang}.tok
lrlhrl_tgt=data-ex/${lang}en.en.tok
w_lrlhrl_src=data-ex/${lang}_${stra_out}/${lang}en.${lang}.tok
w_lrlhrl_tgt=data-ex/${lang}_${stra_out}/${lang}en.en.tok


spm=8000
src_model=spm_out_dir/${lang}/${lang}.spm${spm}.model
tgt_model=spm_out_dir/${lang}/en.spm${spm}.model

lrlhrl_src_spm=data-ex/${lang}en.${lang}.spm${spm}
lrlhrl_tgt_spm=data-ex/${lang}en.en.spm${spm}
w_lrlhrl_src_spm=data-ex/${lang}_${stra_out}/${lang}en.${lang}.spm${spm}

# if spm models do not exist, train these models with monolingual data
if [[ ! -f $src_model ]]; then
cat $lrl_mono $hrl_mono > $spm_out_dir/${lang}/mono.temp.src
python3 $SENT train $spm_out_dir/mono.temp.src $spm_out_dir/${lang}.spm${spm} $spm
fi

if [[ ! -f $tgt_model ]]; then
python3 $SENT train $spm_out_dir/$eng_mono $spm_out_dir/eng.spm${spm} $spm
fi

# encode parallel data
if [[ ! -f $lrlhrl_src ]]; then
cat $lrl_src $hrl_src > $lrlhrl_src
fi
python3 $SENT encode $src_model $lrlhrl_src $lrlhrl_src_spm

if [[ -f $w_lrlhrl_src ]]; then
python3 $SENT encode $src_model $w_lrlhrl_src $w_lrlhrl_src_spm
fi

if [[ ! -f $lrlhrl_tgt ]]; then
cat $lrl_eng $hrl_eng > $lrlhrl_tgt
fi
python3 $SENT encode $tgt_model $lrlhrl_tgt $lrlhrl_tgt_spm


