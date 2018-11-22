# 11731_final

Preprocess the dataset to generate data.pt files.
```bash
data_type=11731_final
lang1=aze
lang2=tur
lang=${lang1}${lang2}
process_type=spm8000
max_shard_size=2097152000000

data_dir=$data/$data_type/ted_data/${lang1}${lang2}_eng
save_dir=$data/$data_type/processed/$process_type/${lang}
mkdir -p $save_dir

python3 ~/11731_final/preprocess.py -train_src $data_dir/ted-train.orig.${lang}.tok \
                                    -train_tgt $data_dir/ted-train.orig.${lang}.tok.spm8k \
                                    -valid_src $data_dir/ted-dev.orig.${lang}.tok \
                                    -valid_tgt $data_dir/ted-dev.orig.${lang}.tok.spm8k \
                                    -save_data $save_dir/${process_type} \
                                    -src_vocab_size 50000 -tgt_vocab_size 50000 \
                                    -max_shard_size ${max_shard_size} \
                                    -src_seq_length 100 -tgt_seq_length 100
```

Train.
```bash
lang1=aze
lang2=tur
lang=${lang1}${lang2}

process_type=spm8000
data_type=11731_final
data_dir=$data/$data_type
save_dir=$out/$data_type/$process_type
mkdir -p $save_dir
cp run_train.sh $save_dir

gpu=1

batch_size=1024
accum_count=8

python3 ~/11731_final/train.py -data $data_dir/processed/$process_type/$lang/$process_type \
                 -save_model $save_dir/models/${process_type} -gpuid ${gpu} \
                 -layers 6 -rnn_size 512 -word_vec_size 512 \
                 -epochs 30 -max_generator_batches 32 -dropout 0.1 \
                 -batch_size ${batch_size} -batch_type tokens -normalization tokens -accum_count ${accum_count} \
                 -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 2 \
                 -max_grad_norm 0 -param_init 0 -param_init_glorot \
                 -label_smoothing 0.1 -tensorboard -tensorboard_log_dir $save_dir/runs/${process_type} \
                 -bleu_freq 3000 \
                 -length_penalty avg \
                 -select_model ppl \
                 -save_cutoff 10 > $save_dir/log
```

Test.
```bash
m=model # fill by yourself
src=src # fill by yourself
gpu=0
output=output # fill by yourself

python3 ~/mengzhox/11731_final/translate.py -gpu $gpu -model $m \
                               -src $src \
                               -output $output \
                               -translate_batch_size 10 -new_bpe \
                               -length_penalty avg
```
