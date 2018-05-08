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
# inf, num, num-num
merge_ops=$7
# train test
task=$8
# null or _lex
use_lex=$9

echo s: $s
echo t: $t
echo gpu_id: $gpu_id
echo dim: $dim
echo num_layer: $num_layer
echo kernel_width: $kernel_width
echo merge_ops: $merge_ops
echo task: $task
echo use_lex: $use_lex

# cnn_model_dir renamed
cnn_model_dir=$exp_dir/cnn_dim${dim}_layer${num_layer}_kernel${kernel_width}_bpe${merge_ops}${use_lex}
mkdir -p $cnn_model_dir

if [ "$task" == "train" ]; then
  config_yml_in=$cnn_dir/example_configs/conv_seq2seq.yml
  config_yml_out=$cnn_model_dir/conv_seq2seq_dim${dim}_layer${num_layer}_kernel${kernel_width}_bpe${merge_ops}.yml
  # modify the parameters in yml
  python $BASE_DIR/hyper_param_mod.py \
    --config_yml_in $config_yml_in \
    --config_yml_out $config_yml_out \
    --dim $dim \
    --num_layer $num_layer \
    --kernel_width $kernel_width

  # train
  sh $BASE_DIR/train_tune_test_cnn2.sh \
    $s \
    $t \
    $merge_ops \
    $config_yml_out \
    $cnn_model_dir \
    $gpu_id \
    $use_lex
elif [ "$task" == "test" ]; then
  # test
  sh $BASE_DIR/decoder_cnn2.sh \
    $s \
    $t \
    $merge_ops \
    $cnn_model_dir \
    $gpu_id \
    $dim \
    $num_layer \
    $kernel_width \
    $use_lex
else
  echo unsupported task: $task
fi


