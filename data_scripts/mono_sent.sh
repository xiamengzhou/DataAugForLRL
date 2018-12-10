#!/usr/bin/env bash

# Sentence Piece monolingual wikipedia
data_dir=$data/11731_final/mono

for lang in az be cs gl pt ru sk tr; do
python3 ~/NMT/rapid/sentence_piece.py encode $data_dir/${lang}_mono/${lang}--vocab_size=8000.model \
                                             $data_dir/${lang}_mono/${lang}.wiki.txt \
                                             $data_dir/${lang}_mono/${lang}.wiki.txt.spm8000
done

# Word dictionary
data_dir=$data/11731_final/mono
mkdir $data/11731_final/vocab
for lang in az tr; do #bel rus glg por slk ces; do
#python3 ~/NMT/rapid/utils.py vocab $data_dir/${lang}_eng/ted-train.orig.${lang}.tok.spm8k \
#                                   $data_dir/../vocab/${lang}.vocab.spm8k
#
#python3 ~/NMT/rapid/utils.py vocab $data_dir/${lang}_eng/ted-train.orig.${lang}.tok \
#                                   $data_dir/../vocab/${lang}.vocab.tok

python3 ~/NMT/rapid/utils.py vocab $data_dir/${lang}_mono/${lang}.wiki.tok.txt.200k \
                                   $data_dir/../vocab/${lang}.wiki.vocab.tok


done

for lang in azetur; do #belrus glgpor slkces; do
#python3 ~/NMT/rapid/utils.py vocab $data_dir/${lang}_eng/ted-train.orig.${lang}.tok\
#                                   $data_dir/../vocab/${lang}.vocab
#python3 ~/NMT/rapid/utils.py vocab $data_dir/${lang}_eng/ted-train.orig.${lang}.tok.spm8k \
#                                   $data_dir/../vocab/${lang}.vocab.spm8k
python3 ~/NMT/rapid/utils.py vocab $data_dir/${lang}_mono/azetur.wiki.tok.txt.400k.spm40k \
                                   $data_dir/../vocab/${lang}.wiki.vocab.spm40k
done


# Get new spm sentence models
data_dir=$data/11731_final/mono
for lang in azetur; do
python3 ~/NMT/rapid/sentence_piece.py train $data_dir/${lang}_mono/azetur.wiki.tok.txt.400k \
                                            $data_dir/${lang}_mono/az--vocab_size=40000 \
                                            40000
done

# Use spm model to tokenize data
for lang in azetur; do
#python3 ~/NMT/rapid/sentence_piece.py encode $data_dir/${lang}_mono/az--vocab_size=40000.model \
#                                             $data_dir/${lang}_mono/azetur.wiki.tok.txt.400k \
#                                             $data_dir/${lang}_mono/azetur.wiki.tok.txt.400k.spm40k

#python3 ~/NMT/rapid/sentence_piece.py encode $data/11731_final/mono/${lang}_mono/az--vocab_size=40000.model \
#                                             $data/11731_final/bilang/${lang}_eng/ted-train.orig.azetur.tok \
#                                             $data/11731_final/bilang/${lang}_eng/ted-train.orig.${lang}.tok.spm40k

python3 ~/NMT/rapid/sentence_piece.py encode $data/11731_final/mono/${lang}_mono/az--vocab_size=40000.model \
                                             $data/11731_final/bilang/aze_eng/ted-dev.orig.aze.tok \
                                             $data/11731_final/bilang/aze_eng/ted-dev.orig.aze.tok.spm40k
done

lang1=az
lang2=tr
data_dir=$data/11731_final
python3 unsupervised.py --src_lang $lang1 \
                        --tgt_lang $lang2 \
                        --src_emb $data_dir/mono/${lang1}_mono/${lang1}.200k.vec \
                        --tgt_emb $data_dir/mono/${lang2}_mono/${lang2}.200k.vec \
                        --n_refinement 5 \
                        --dico_eval data/crosslingual/dictionaries/en-tl.5000-6500.txt

# $data_dir/vectors/wiki.256.200k.spm8k/azetur_S2TT2S \

process_type=swap-5
lang=azetur
mkdir $data/11731_final/bilang/${lang}_eng/$process_type
data_dir=$data/11731_final
python3 ~/NMT/rapid/utils.py swap \
                             $data_dir/bilang/${lang}_eng/ted-train.orig.${lang}.tok \
                             $data_dir/bilang/${lang}_eng/${process_type}/ted-train.orig.${lang}.tok \
                             /projects/tir3/users/mengzhox/data/11731_final/MUSE_dict/azetur/T2S_re \
                             $data_dir/vocab/aze.vocab.tok \
                             0 /projects/tir3/users/mengzhox/data/11731_final/MUSE_dict/azetur/T2S_re2_score

python3 ~/NMT/rapid/sentence_piece.py encode $data/11731_final/mono/az_mono/az--vocab_size=8000.model \
                                             $data/11731_final/bilang/azetur_eng/$process_type/ted-train.orig.azetur.tok \
                                             $data/11731_final/bilang/azetur_eng/$process_type/ted-train.orig.azetur.tok.spm8k