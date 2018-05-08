#!/bin/bash

gpu_id=0

t2t_data_dir=/home/ec2-user/kklab/Projects/lrlp/experiment_2017.08.04.vie-eng.y1r1.v2/translation/t2t_inf
t2t_model_dir=t2t_model_dir
mkdir -p $t2t_model_dir

t2t_model_type=transformer_lex
hparams=transformer_dim300_layer2
problem_id=translate_vieeng_lrlp80008000

task=train

if [ "$task" == "prep" ]; then
  t2t-datagen \
    --data_dir=$t2t_data_dir \
    --tmp_dir= \
    --problem=$problem_id \
    --t2t_usr_dir=$PWD \
    --registry_help
elif [ "$task" == "train" ]; then
  CUDA_VISIBLE_DEVICES=$gpu_id t2t-trainer \
    --data_dir=$t2t_data_dir \
    --problems=$problem_id \
    --model=$t2t_model_type \
    --hparams_set=$hparams \
    --output_dir=$t2t_model_dir \
    --train_steps=300000 \
    --t2t_usr_dir=$PWD
elif [ "$task" == "test" ]; then
  dev_src=/home/ec2-user/kklab/Projects/lrlp/experiment_2017.08.04.vie-eng.y1r1.v2/translation/dev/src_raw.vie.dev.y1r1.v2
  dev_tgt=/home/ec2-user/kklab/Projects/lrlp/experiment_2017.08.04.vie-eng.y1r1.v2/translation/dev/ref_raw.eng.dev.y1r1.v2

  beam_size=4
  alpha=0.6

  source_file=$dev_src
  source_file_bpe=$source_file.inf
  python replace_oov_with_unk.py $source_file $source_file_bpe $t2t_data_dir/vocab.vieeng.8000.vie
  onebest_file_untok=$source_file_bpe.$t2t_model_type.$hparams.beam$beam_size.alpha$alpha.decodes
  onebest_file=$trans_dir/$dataset0/onebest_t2t_dim${dim}_layer${num_layer}_bpe${merge_ops}${use_lex}.$t.$dataset0.$yrv
  CUDA_VISIBLE_DEVICES=$gpu_id t2t-decoder \
    --data_dir=$t2t_data_dir \
    --problems=$problem_id \
    --model=$t2t_model_type \
    --hparams_set=$hparams \
    --output_dir=$t2t_model_dir \
    --decode_hparams="beam_size=$beam_size,alpha=$alpha" \
    --decode_from_file=$source_file_bpe \
    --t2t_usr_dir=$BASE_DIR
  tokenize2 $t $onebest_file_untok $onebest_file
  echo ----------------
else
  echo unsupported task: $task
fi

