#!/usr/bin/env bash

UTIL=~/NMT/rapid/utils.py
MUSE=~/MUSE
MONO=$data/11731_final/mono/using

array=( "az" "be" "gl" "sk" )
array2=( "tr" "ru" "pt" "cs" )

array3=( "aze" "bel" "glg" "slk" )
array4=( "tur" "rus" "plt" "ces" )

# Generate vocaburay for wikipedia monolingual data
for lang in az tr gl pt sk cs; do
    if [ ! -f $MONO/vocab/${lang}.vocab ]; then
    python3 $UTIL vocab $MONO/${lang}.wiki.tok.txt $MONO/vocab/${lang}.vocab
    echo Saved to $MONO/vocab/${lang}.vocab ...
    fi
done

# Generate identical strings as seed dictionary
mkdir -p $MONO/fasttext/seed
for ((i=0;i<${#array[@]};++i)); do
    lang1="${array[i]}"
    lang2="${array2[i]}"
    if [ ! -f $MONO/fasttext/seed/${lang1}${lang2}_seed ]; then
        python3 $UTIL identical2 $MONO/fasttext/${lang1}.vec \
                                 $MONO/fasttext/${lang2}.vec \
                                 $MONO/fasttext/seed/${lang1}${lang2}_seed \
                                 lower
        echo Saved to $MONO/fasttext/seed/${lang1}${lang2}_seed ...
    fi
done

# Supervised MUSE training
lang1=be
lang2=ru
id=1
if [ -f $MONO/fasttext/MUSE_SUP/${lang1}${lang2}/$id ]; then
    echo $MONO/fasttext/MUSE_SUP/${lang1}${lang2}/$id exists!
fi
python3 $MUSE/supervised.py --exp_path $MONO/fasttext/MUSE_SUP \
                            --exp_name ${lang1}${lang2} \
                            --exp_id $id \
                            --src_lang $lang1 \
                            --tgt_lang $lang2 \
                            --max_vocab 200000 \
                            --n_refinement 5 \
                            --dico_train $MONO/fasttext/seed/${lang1}${lang2}_seed \
                            --dico_eval $MUSE/data/crosslingual/dictionaries/en-it.0-5000.txt \
                            --src_emb $MONO/fasttext/${lang1}.vec \
                            --tgt_emb $MONO/fasttext/${lang2}.vec \
                            --normalize_embeddings center

# Extract dictionary
lang1=be
lang2=ru
id=1
python3 ~/11731_final/py_scripts/get_dictionary.py --src_emb $MONO/fasttext/MUSE_SUP/${lang1}${lang2}/$id/vectors-${lang1}.txt \
                                             --tgt_emb $MONO/fasttext/MUSE_SUP/${lang1}${lang2}/$id/vectors-${lang2}.txt \
                                             --dico_build "T2S" \
                                             --output $MONO/fasttext/MUSE_SUP/${lang1}${lang2}/$id/T2S

# Supervised S2TT2S swap in
for ((i=0;i<${#array[@]};++i)); do
    # lang11="${array[i]}"
    # lang12="${array2[i]}"
    # lang21="${array3[i]}"
    # lang22="${array4[i]}"
    lang11=gl
    lang12=pt
    lang21=glg
    lang22=por
    data_dir=$data/11731_final/bilang/${lang21}${lang22}_eng

    mkdir -p $data_dir/swap-su-T2S-in
    python3 $UTIL swap $data_dir/ted-train.orig.${lang21}${lang22}.tok \
                       $data_dir/swap-su-T2S-in/ted-train.orig.${lang21}${lang22}.tok \
                       $MONO/fasttext/MUSE_SUP/${lang11}${lang12}/$id/T2S \
                       $data/11731_final/vocab/${lang21}.vocab.tok \
                       0
    echo Saved to directory $data_dir/swap-su-T2S-in ...

    mkdir -p $data_dir/swap-su-T2S-all
    python3 $UTIL swap2 $data_dir/ted-train.orig.${lang21}${lang22}.tok \
                       $data_dir/swap-su-T2S-all/ted-train.orig.${lang21}${lang22}.tok \
                       $MONO/fasttext/MUSE_SUP/${lang11}${lang12}/$id/T2S
    echo Saved to directory $data_dir/swap-su-T2S-all ...

    mkdir -p $data_dir/swap-su-S2TT2S-in
    python3 $UTIL swap $data_dir/ted-train.orig.${lang21}${lang22}.tok \
                       $data_dir/swap-su-S2TT2S-in/ted-train.orig.${lang21}${lang22}.tok \
                       $MONO/fasttext/MUSE_SUP/${lang11}${lang12}/$id/S2TT2S \
                       $data/11731_final/vocab/${lang21}.vocab.tok \
                       0
    echo Saved to directory $data_dir/swap-su-S2TT2S-in ...

    mkdir -p $data_dir/swap-su-S2TT2S-all
    python3 $UTIL swap2 $data_dir/ted-train.orig.${lang21}${lang22}.tok \
                        $data_dir/swap-su-S2TT2S-all/ted-train.orig.${lang21}${lang22}.tok \
                        $MONO/fasttext/MUSE_SUP/${lang11}${lang12}/$id/S2TT2S
    echo Saved to directory $data_dir/swap-su-S2TT2S-all ...
done

# Sentence piece
for ((i=0;i<${#array[@]};++i)); do
    lang11="${array[i]}"
    lang21="${array3[i]}"
    lang22="${array4[i]}"
    lang11=gl
    lang21=glg
    lang22=por
    lang=${lang21}${lang22}
    for process_type in swap-su-S2TT2S-all swap-su-S2TT2S-in swap-su-T2S-all swap-su-T2S-in; do
    python3 ~/NMT/rapid/sentence_piece.py encode $data/11731_final/mono/${lang21}_mono/${lang11}--vocab_size=8000.model \
                                             $data/11731_final/bilang/${lang}_eng/$process_type/ted-train.orig.${lang}.tok \
                                             $data/11731_final/bilang/${lang}_eng/$process_type/ted-train.orig.${lang}.tok.spm8k
    done
done

# Sentence piece wikipedia data
lang21=aze
lang11=az
lang12=tr
lang=azetur
python3 ~/NMT/rapid/sentence_piece.py encode $data/11731_final/mono/${lang21}_mono/${lang11}--vocab_size=8000.model \
                                             $MONO/${lang11}.wiki.tok.txt \
                                             $MONO/sentp/${lang11}.wiki.tok.txt.spm8k
python3 ~/NMT/rapid/sentence_piece.py encode $data/11731_final/mono/${lang21}_mono/${lang11}--vocab_size=8000.model \
                                             $MONO/${lang12}.wiki.tok.txt \
                                             $MONO/sentp/${lang12}.wiki.tok.txt.spm8k

# Extract dictionary
python3 $UTIL vocab $MONO/sentp/${lang11}.wiki.tok.txt.spm8k $MONO/sentp/vocab/${lang11}.vocab
python3 $UTIL vocab $MONO/sentp/${lang12}.wiki.tok.txt.spm8k $MONO/sentp/vocab/${lang12}.vocab

# Extract word embeddings from existing models
python3 $UTIL ex_emb $out/11731_final/azetur/swap-su-T2S-in-v1/models/eval/swap-su-T2S-in_acc_59.26_ppl_8.93_e37_s0.pt \
                     /projects/tir3/users/mengzhox/data/unsup/mono/aztr/emb/emb.spm8k
python3 $UTIL ex_emb $out/11731_final/azetur/spm8000-v4/models/spm8000_acc_56.33_ppl_10.78_e23_s0.pt \
                     /projects/tir3/users/mengzhox/data/unsup/mono/aztr/emb/joint/emb.spm8k

mkdir -p /projects/tir3/users/mengzhox/data/unsup/mono/beru/emb/swap
mkdir -p /projects/tir3/users/mengzhox/data/unsup/mono/beru/emb/joint
python3 $UTIL ex_emb $out/11731_final/belrus/swap-su-S2TT2S-all-v1/models/eval/swap-su-S2TT2S-all_acc_64.65_ppl_6.14_e25_s0.pt \
                     /projects/tir3/users/mengzhox/data/unsup/mono/beru/emb/swap/emb.spm8k
python3 $UTIL ex_emb $out/11731_final/belrus/spm8000-v2/models/spm8000_acc_59.58_ppl_8.85_e18_s14860.pt \
                     /projects/tir3/users/mengzhox/data/unsup/mono/beru/emb/joint/emb.spm8k

# UMT translate out augmented data
exp_id=147987
lang21=aze
lang22=tur
lang11=az
lang12=tr
cat $data/11731_final/bilang/${lang21}_eng/ted-train.orig.${lang21}.tok.spm8k \
    $out/unsup/aztr/tran/$exp_id/tran0.${lang12}-${lang11}.txt > $out/unsup/aztr/tran/$exp_id/ted-train.orig.${lang21}${lang22}.tok.spm8k
cp $data/11731_final/bilang/${lang21}${lang22}_eng/ted-train.mtok.spm8000.eng  $out/unsup/aztr/tran/$exp_id
# cat $data/11731_final/bilang/${lang21}_eng/ted-train.mtok.spm8000.eng \
#     $out/unsup/aztr/tran/$exp_id/ted-train.mtok.spm8000.eng > $out/unsup/aztr/tran/$exp_id/ted-train.mtok.spm8000.eng
