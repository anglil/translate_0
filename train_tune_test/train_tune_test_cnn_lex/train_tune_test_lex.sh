#!/bin/bash

BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $BASE_DIR/../../config.sh $1 $2
source $BASE_DIR/../../utils.sh

s=$1
t=$2

config_yml=$3
lex_model_dir=$4
gpu_id=$5
use_lex=$6

vocab_dir=$trans_dir/bpe
mkdir -p $vocab_dir

train_src=$trans_dir/train/${src_label}_raw${use_lex}.$s.train.${yrv}
train_tgt=$trans_dir/train/${ref_label}_raw${use_lex}.$t.train.${yrv}

vocab_src=$vocab_dir/vocab${use_lex}.bpe.inf.$s
vocab_tgt=$vocab_dir/vocab${use_lex}.bpe.inf.$t

if [ ! -f $vocab_src ]; then
  $vocab_gen \
    < "$train_src" \
    > "$vocab_src"
  echo vocab_src created at: $vocab_src
else
  echo vocab_src exists at: $vocab_src
fi

if [ ! -f $vocab_tgt ]; then
  $vocab_gen \
    < "$train_tgt" \
    > "$vocab_tgt"
  echo vocab_tgt created at: $vocab_tgt
else
  echo vocab_tgt exists at: $vocab_tgt
fi

dev_src=$trans_dir/dev/${src_label}_raw.$s.dev.${yrv}
dev_tgt=$trans_dir/dev/${ref_label}_raw.$t.dev.${yrv}

# train
export PYTHONPATH=$cnn_dir:$PYTHONPATH
CUDA_VISIBLE_DEVICES=$gpu_id python -m bin.train \
  --config_paths="
    ${config_yml},
    $cnn_dir/example_configs/train_seq2seq.yml,
    $cnn_dir/example_configs/text_metrics_bpe.yml" \
  --model_params "
    glove_dict_file: $glove_mat
    lexicon_dict_file: $bilingual_lexicon
    vocab_source: $vocab_src
    vocab_target: $vocab_tgt" \
  --input_pipeline_train "
    class: ParallelTextInputPipelineFairseq
    params:
      source_files:
        - $train_src
      target_files:
        - $train_tgt" \
  --input_pipeline_dev "
    class: ParallelTextInputPipelineFairseq
    params:
      source_files:
        - $dev_src
      target_files:
        - $dev_tgt" \
  --batch_size 32 \
  --eval_every_n_steps 5000 \
  --train_steps 300000 \
  --keep_checkpoint_max 0 \
  --output_dir $lex_model_dir


