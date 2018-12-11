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

lang1=gl
lang2=pt
data_dir=$data/11731_final
python3 unsupervised.py --src_lang $lang1 \
                        --tgt_lang $lang2 \
                        --src_emb $data_dir/mono/${lang1}_mono/${lang1}.2m.vec \
                        --tgt_emb $data_dir/mono/${lang2}_mono/${lang2}.2m.vec \
                        --n_refinement 5 \
                        --dico_eval data/crosslingual/dictionaries/en-tl.5000-6500.txt \
                        --exp_path . \
                        --exp_name glgpor \
                        --normalize_embeddings center

# $data_dir/vectors/wiki.256.200k.spm8k/azetur_S2TT2S \

process_type=swap-1
lang=slkces
mkdir $data/11731_final/bilang/${lang}_eng/$process_type
data_dir=$data/11731_final
python3 $final/py_scripts/utils.py swap \
                             $data_dir/bilang/${lang}_eng/ted-train.orig.${lang}.tok \
                             $data_dir/bilang/${lang}_eng/${process_type}/ted-train.orig.${lang}.tok \
                             $data_dir/MUSE_dict/${lang}/re_S2T_T2S \
                             $data_dir/vocab/slk.vocab.tok \
                             0 \
                             " " /projects/tir3/users/mengzhox/data/11731_final/MUSE_dict/$lang/re_S2T_T2S_score

python3 ~/NMT/rapid/sentence_piece.py encode $data/11731_final/mono/gl_mono/gl--vocab_size=8000.model \
                                             $data/11731_final/bilang/${lang}_eng/$process_type/ted-train.orig.${lang}.tok \
                                             $data/11731_final/bilang/${lang}_eng/$process_type/ted-train.orig.${lang}.tok.spm8k

lang1=gl
lang2=pt
lang=glgpor
python3 ~/11731_final/py_scripts/sentence_piece.py encode_dict $data/11731_final/mono/${lang1}_mono/${lang1}--vocab_size=8000.model \
                                                  $data/11731_final/mono/${lang2}_mono/${lang2}--vocab_size=8000.model \
                                                  $data/11731_final/bilang/${lang}_eng/swap-2/swap_dict \
                                                  $data/11731_final/bilang/${lang}_eng/swap-2/swap_dict_bpe

lang1=be
lang2=ru
lang=belrus
python3 ~/11731_final/py_scripts/sentence_piece.py encode_dict $data/11731_final/mono/${lang1}_mono/${lang1}--vocab_size=8000.model \
                                                  $data/11731_final/mono/${lang2}_mono/${lang2}--vocab_size=8000.model \
                                                  $data/11731_final/MUSE_dict/belrus/re_S2T_T2S \
                                                  $data/11731_final/MUSE_dict/belrus/re_S2T_T2S_bpe

emb_dir=$data/11731_hw2/utils/beru_1/txuydkgp8y
emb_dir=slkces/ffczjskl7t
emb_dir=/home/junjieh/mengzhox/data/11731_hw2/MUSE_vec
python3 $final/py_scripts/get_dictionary.py --src_emb $emb_dir/vectors-gl.txt \
                                                   --tgt_emb $emb_dir/vectors-pt.txt \
                                                   --dico_build "T2S" \
                                                   --output $data/11731_final/MUSE_dict/glgpor/re_T2S

lang=slkces
python3 $final/py_scripts/utils.py prob $data/11731_final/bilang/${lang}_eng/swap-1/swap_dict \
                      $data/11731_final/vocab/slk.vocab.tok \
                      $data/11731_final/bilang/${lang}_eng/swap-1/prob_0.5 \
                      0.5 " ||| "