#!/bin/bash

BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $BASE_DIR/../../config.sh $1 $2
source $BASE_DIR/../../utils.sh

s=$1
t=$2

gpu_id=$3
dim=$4
num_layer=$5
kernel_width=$6
# 0 1 2
injective=$7
# train test
task=$8
# null or _lex (in lex_seq2seq this domain should always be left empty)
use_lex=$9

echo s: $s
echo t: $t
echo gpu_id: $gpu_id
echo dim: $dim
echo num_layer: $num_layer
echo kernel_width: $kernel_width
echo injective: $injective
echo task: $task
echo use_lex: $use_lex

lex_model_dir=$exp_dir/lex${injective}_dim${dim}_layer${num_layer}_kernel${kernel_width}${use_lex}
mkdir -p $lex_model_dir

if [ "$task" == "train" ]; then
  config_yml_in=$cnn_dir/example_configs/lex_seq2seq.yml
  config_yml_out=$lex_model_dir/lex${injective}_seq2seq_dim${dim}_layer${num_layer}_kernel${kernel_width}.yml
  # modify the parameters in yml
  python $BASE_DIR/hyper_param_mod.py \
    --config_yml_in $config_yml_in \
    --config_yml_out $config_yml_out \
    --dim $dim \
    --num_layer $num_layer \
    --kernel_width $kernel_width \
    --lexicon_injective $injective

  # train
  sh $BASE_DIR/train_tune_test_lex.sh \
    $s \
    $t \
    $config_yml_out \
    $lex_model_dir \
    $gpu_id \
    $use_lex
elif [ "$task" == "test" ]; then
  # test
  sh $BASE_DIR/decoder_lex.sh \
    $s \
    $t \
    $injective \
    $lex_model_dir \
    $gpu_id \
    $dim \
    $num_layer \
    $kernel_width \
    $use_lex
else
  echo unsupported task: $task
fi
