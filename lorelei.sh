#!/usr/bin/env bash

SENT=~/NMT/rapid/sentence_piece.py
data_dir=/projects/tir1/corpora/multiling-text/lorelei-dryrun
out_data_dir=$data/lorelei-dryrun

lang1=orm # tir
lang2=som # amh
cp $data_dir/${lang1}_eng/comb-train.tok.${lang1} $out_data_dir/$lang1
cp $data_dir/${lang1}_eng/comb-train.tok.eng $out_data_dir/$lang1
cp $data_dir/${lang1}_eng/comb-dev.tok.${lang1} $out_data_dir/$lang1
cp $data_dir/${lang1}_eng/comb-dev.tok.eng $out_data_dir/$lang1
cp $data_dir/${lang1}_eng/comb-test.tok.${lang1} $out_data_dir/$lang1
cp $data_dir/${lang1}_eng/comb-test.tok.eng $out_data_dir/$lang1
cp $data_dir/${lang1}_eng/${lang1}${lang2}.tok.${lang1} $out_data_dir/$lang1
cp $data_dir/${lang1}_eng/${lang1}${lang2}.tok.eng $out_data_dir/$lang1

python3 $SENT train $out_data_dir/$lang1/comb-train.tok.${lang1} \
              $out_data_dir/$lang1/${lang1}.spm8k \
              8000

python3 $SENT train $out_data_dir/$lang1/${lang1}${lang2}.tok.${lang1} \
              $out_data_dir/$lang1/${lang1}${lang2}.spm8k \
              8000

python3 $SENT train $out_data_dir/$lang1/${lang1}${lang2}.tok.eng \
              $out_data_dir/$lang1/eng.spm8k \
              8000

python3 $SENT encode ${lang1}.spm8k.model \
                     $out_data_dir/$lang1/comb-train.tok.${lang1} \
                     $out_data_dir/$lang1/comb-train.tok.${lang1}.spm8k

python3 $SENT encode ${lang1}${lang2}.spm8k.model \
                     $out_data_dir/$lang1/${lang1}${lang2}.tok.${lang1} \
                     $out_data_dir/$lang1/${lang1}${lang2}.tok.${lang1}.spm8k

python3 $SENT encode eng.spm8k.model \
                     $out_data_dir/$lang1/comb-train.tok.eng \
                     $out_data_dir/$lang1/comb-train.tok.eng.spm8k

python3 $SENT encode eng.spm8k.model \
                     $out_data_dir/$lang1/${lang1}${lang2}.tok.eng \
                     $out_data_dir/$lang1/${lang1}${lang2}.tok.eng.spm8k

# dev and test
python3 $SENT encode ${lang1}.spm8k.model \
                     $out_data_dir/$lang1/comb-dev.tok.${lang1} \
                     $out_data_dir/$lang1/comb-dev.tok.${lang1}.spm8k1

python3 $SENT encode ${lang1}${lang2}.spm8k.model \
                     $out_data_dir/$lang1/comb-dev.tok.${lang1} \
                     $out_data_dir/$lang1/comb-dev.tok.${lang1}.spm8k2

python3 $SENT encode eng.spm8k.model \
                     $out_data_dir/$lang1/comb-dev.tok.eng \
                     $out_data_dir/$lang1/comb-dev.tok.eng.spm8k

# standard
python3 $SENT encode ${lang1}.spm8k.model \
                     $out_data_dir/$lang1/setE-mono-standard.tok.${lang1} \
                     $out_data_dir/$lang1/setE-mono-standard.tok.${lang1}.spmk1

python3 $SENT encode ${lang1}${lang2}.spm8k.model \
                     $out_data_dir/$lang1/setE-mono-standard.tok.${lang1} \
                     $out_data_dir/$lang1/setE-mono-standard.tok.${lang1}.spmk2
