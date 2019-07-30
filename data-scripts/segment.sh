#!/usr/bin/env bash

# This script processed the monolingual and parallel data from tokenized version to segmented version.

lrl=az
hrl=tr
lang=${lrl}${hrl}


out_dir=../data-ex/out
bilang_dir=../data-ex/bilang/${lang}
mono_dir=../data-ex/mono
spm_out_dir=../data-ex/out/spm/${lang}

mkdir -p $spm_out_dir

exp_id=1
stra_out=T2S
dict=$out_dir/${lang}_sup/${exp_id}/re_${stra_out}

SENT=spm.py

# tokenized data
lrl_src=${bilang_dir}/${lrl}en.${lrl}.tok
lrl_eng=${bilang_dir}/${lrl}en.en.tok
hrl_src=${bilang_dir}/${hrl}en.${hrl}.tok
hrl_eng=${bilang_dir}/${hrl}en.en.tok

lrl_mono=${mono_dir}/${lrl}/${lrl}.mono
hrl_mono=${mono_dir}/${hrl}/${hrl}.mono
eng_mono=${mono_dir}/en/en.mono

# processed data
lrlhrl_src=${bilang_dir}/${lang}en.${lang}.tok
lrlhrl_tgt=${bilang_dir}/${lang}en.en.tok

spm=8000
src_model=$spm_out_dir/${lang}.spm${spm}.model
tgt_model=$spm_out_dir/en.spm${spm}.model

lrlhrl_src_spm=${bilang_dir}/${lang}en.${lang}.spm${spm}
lrlhrl_tgt_spm=${bilang_dir}/${lang}en.en.spm${spm}

lrl_mono_spm=${mono_dir}/${lrl}/${lrl}.mono.spm${spm}
hrl_mono_spm=${mono_dir}/${hrl}/${hrl}.mono.spm${spm}
eng_mono_spm=${mono_dir}/en/en.mono.spm${spm}

# if spm models do not exist, train these models with monolingual data
if [[ ! -f $src_model ]]; then
cat $lrl_mono $hrl_mono > $spm_out_dir/mono.temp.src
python3 $SENT train $spm_out_dir/mono.temp.src $spm_out_dir/${lang}.spm${spm} $spm
rm $spm_out_dir/mono.temp.src
echo Trained a sentence-piece model for ${lang} at $spm_out_dir/${lang}.spm${spm}.
fi

if [[ ! -f $tgt_model ]]; then
python3 $SENT train $eng_mono $spm_out_dir/en.spm${spm} $spm
echo Trained a sentence-piece model for English at $spm_out_dir/en.spm${spm}.
fi

# encode parallel data
if [[ ! -f $lrlhrl_src ]]; then
cat $lrl_src $hrl_src > $lrlhrl_src
fi

if [[ ! -f $lrlhrl_src_spm ]]; then
python3 $SENT encode $src_model $lrlhrl_src $lrlhrl_src_spm
echo Encoded $lrlhrl_src to $lrlhrl_src_spm.
fi

if [[ ! -f $lrlhrl_tgt ]]; then
cat $lrl_eng $hrl_eng > $lrlhrl_tgt
fi

if [[ ! -f $lrlhrl_tgt_spm ]]; then
python3 $SENT encode $tgt_model $lrlhrl_tgt $lrlhrl_tgt_spm
echo Encoded $lrlhrl_tgt to $lrlhrl_tgt_spm.
fi

# encode monolingual data
if [[ ! -f $lrl_mono_spm ]]; then
python3 $SENT encode $src_model $lrl_mono $lrl_mono_spm
echo Encoded $lrl_mono to $lrl_mono_spm.
fi

if [[ ! -f $hrl_mono_spm ]]; then
python3 $SENT encode $src_model $hrl_mono $hrl_mono_spm
echo Encoded $hrl_mono to $hrl_mono_spm.
fi

if [[ ! -f $eng_mono_spm ]]; then
python3 $SENT encode $tgt_model $eng_mono $eng_mono_spm
echo Encoded $eng_mono to $eng_mono_spm.
fi

