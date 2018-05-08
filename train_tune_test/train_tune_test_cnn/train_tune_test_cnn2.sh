#!/bin/bash

### main script
BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $BASE_DIR/../../config.sh $1 $2
source $BASE_DIR/../../utils.sh

s=$1
t=$2

merge_ops=$3
config_yml=$4
cnn_model_dir=$5
gpu_id=$6
use_lex=$7 # null or _lex

bpe_dir=$trans_dir/bpe
mkdir -p $bpe_dir

train_src=$trans_dir/train/${src_label}_raw${use_lex}.$s.train.${yrv}
train_tgt=$trans_dir/train/${ref_label}_raw${use_lex}.$t.train.${yrv}

# no bpe
if [ "$merge_ops" == "inf" ]; then
  vocab_src=$bpe_dir/vocab${use_lex}.bpe.$merge_ops.$s
  vocab_tgt=$bpe_dir/vocab${use_lex}.bpe.$merge_ops.$t

  echo train_src: $train_src
  $vocab_gen \
    < "$train_src" \
    > "$vocab_src"

  echo train_tgt: $train_tgt
  $vocab_gen \
    < "$train_tgt" \
    > "$vocab_tgt"

  dev_src=$trans_dir/dev/${src_label}_raw.$s.dev.${yrv}
  dev_tgt=$trans_dir/dev/${ref_label}_raw.$t.dev.${yrv}

# separate bpe
elif [[ "$merge_ops" == *-* ]]; then
  IFS='-' read -a merge_ops_pair <<< "$merge_ops"
  merge_ops_s=${merge_ops_pair[0]}
  merge_ops_t=${merge_ops_pair[1]}

  # learn individual BPE
  bpe_src=$bpe_dir/ibpe${use_lex}.${merge_ops_s}.$s
  if [ ! -f $bpe_src ]; then
    echo "Learning BPE for $s with merge_ops=${merge_ops_s}. This may take a while..."
    $bpe_train \
      -s $merge_ops_s \
      < $train_src \
      > $bpe_src
    echo bpe model for $s created at $bpe_src
  else
    echo bpe mode for $s exists at $bpe_src
  fi
  bpe_tgt=$bpe_dir/ibpe${use_lex}.${merge_ops_t}.$t
  if [ ! -f $bpe_tgt ]; then
    echo "Learning BPE for $t with merge_ops=${merge_ops_t}. This may take a while..."
    $bpe_train \
      -s $merge_ops_t \
      < $train_tgt \
      > $bpe_tgt
    echo bpe model for $t created at $bpe_tgt
  else
    echo bpe mode for $t exists at $bpe_tgt
  fi

  # apply individual BPE to train and dev
  echo "Apply BPE with merge_ops=${merge_ops_s} to pruned files..."
  for dataset0 in train dev; do
    infile=$trans_dir/$dataset0/${src_label}_raw${use_lex}.$s.$dataset0.$yrv
    if [ "$dataset0" != "train" ]; then
      infile=$trans_dir/$dataset0/${src_label}_raw.$s.$dataset0.$yrv
    fi
    outfile=$trans_dir/$dataset0/ibpe_${merge_ops_s}${use_lex}.$s.$dataset0.$yrv
    if [ ! -f $outfile ]; then
      $bpe_infer \
        -c $bpe_src \
        < $infile \
        > $outfile
      echo bpe-ed $dataset0 file created at $outfile
    else
      echo bpe-ed $dataset0 file exists at $outfile
    fi
  done
  echo "Apply BPE with merge_ops=${merge_ops_t} to pruned files..."
  for dataset0 in train dev; do
    infile=$trans_dir/$dataset0/${ref_label}_raw${use_lex}.$t.$dataset0.$yrv
    if [ "$dataset0" != "train" ]; then
      infile=$trans_dir/$dataset0/${ref_label}_raw.$t.$dataset0.$yrv
    fi
    outfile=$trans_dir/$dataset0/ibpe_${merge_ops_t}${use_lex}.$t.$dataset0.$yrv
    if [ ! -f $outfile ]; then
      $bpe_infer \
        -c $bpe_tgt \
        < $infile \
        > $outfile
      echo bpe-ed $dataset0 file created at $outfile
    else
      echo bpe-ed $dataset0 file exists at $outfile
    fi
  done

  # create a vocab using the bpe-ed training data
  bpe_train_src=$trans_dir/train/ibpe_${merge_ops_s}${use_lex}.$s.train.$yrv
  vocab_src=$bpe_dir/vocab${use_lex}.ibpe.${merge_ops_s}.$s
  if [ ! -f $vocab_src ]; then
    echo -e "<unk>\n<s>\n</s>" > $vocab_src
    cat $bpe_train_src | \
      $bpe_vocab | \
      cut -f1 -d ' ' \
      >> $vocab_src
    echo bpe vocab file for $s created at $vocab_src
  else
    echo bpe vocab file for $s exists at $vocab_src
  fi
  bpe_train_tgt=$trans_dir/train/ibpe_${merge_ops_t}${use_lex}.$t.train.$yrv
  vocab_tgt=$bpe_dir/vocab${use_lex}.ibpe.${merge_ops_t}.$t
  if [ ! -f $vocab_tgt ]; then
    echo -e "<unk>\n<s>\n</s>" > $vocab_tgt
    cat $bpe_train_tgt | \
      $bpe_vocab | \
      cut -f1 -d ' ' \
      >> $vocab_tgt
    echo bpe vocab file for $t created at $vocab_tgt
  else
    echo bpe vocab file for $t exists at $vocab_tgt
  fi

  train_src=$trans_dir/train/ibpe_${merge_ops_s}${use_lex}.$s.train.$yrv
  train_tgt=$trans_dir/train/ibpe_${merge_ops_t}${use_lex}.$t.train.$yrv
  dev_src=$trans_dir/dev/ibpe_${merge_ops_s}${use_lex}.$s.dev.$yrv
  dev_tgt=$trans_dir/dev/ibpe_${merge_ops_t}${use_lex}.$t.dev.$yrv

# shared bpe
else
  # learn shared BPE
  bpe_file=$bpe_dir/bpe${use_lex}.${merge_ops}
  if [ ! -f $bpe_file ]; then
    echo "Learning BPE with merge_ops=${merge_ops}. This may take a while..."
    cat $train_src $train_tgt | \
      $bpe_train \
      -s $merge_ops \
      > $bpe_file
    echo bpe model created at $bpe_file
  else
    echo bpe model exists at $bpe_file
  fi
  
  # apply shared BPE to train and dev
  echo "Apply BPE with merge_ops=${merge_ops} to pruned files..."
  for dataset0 in train dev; do
    infile=$trans_dir/$dataset0/${src_label}_raw${use_lex}.$s.$dataset0.$yrv
    if [ "$dataset0" != "train" ]; then
      infile=$trans_dir/$dataset0/${src_label}_raw.$s.$dataset0.$yrv
    fi
    outfile=$trans_dir/$dataset0/bpe_${merge_ops}${use_lex}.$s.$dataset0.$yrv
    if [ ! -f $outfile ]; then
      $bpe_infer \
        -c $bpe_file \
        < $infile \
        > $outfile
      echo bpe-ed $dataset0 file created at $outfile
    else
      echo bpe-ed $dataset0 file exists at $outfile
    fi
  done
  for dataset0 in train dev; do
    infile=$trans_dir/$dataset0/${ref_label}_raw${use_lex}.$t.$dataset0.$yrv
    if [ "$dataset0" != "train" ]; then
      infile=$trans_dir/$dataset0/${ref_label}_raw.$t.$dataset0.$yrv
    fi
    outfile=$trans_dir/$dataset0/bpe_${merge_ops}${use_lex}.$t.$dataset0.$yrv
    if [ ! -f $outfile ]; then
      $bpe_infer \
        -c $bpe_file \
        < $infile \
        > $outfile
      echo bpe-ed $dataset0 file created at $outfile
    else
      echo bpe-ed $dataset0 file exists at $outfile
    fi
  done
  
  # create a vocab using the bpe-ed training data
  bpe_train_src=$trans_dir/train/bpe_${merge_ops}${use_lex}.$s.train.$yrv
  bpe_train_tgt=$trans_dir/train/bpe_${merge_ops}${use_lex}.$t.train.$yrv
  bpe_vocab_file=$bpe_dir/vocab${use_lex}.bpe.${merge_ops}
  if [ ! -f $bpe_vocab_file ]; then
    echo -e "<unk>\n<s>\n</s>" > $bpe_vocab_file
    cat $bpe_train_src $bpe_train_tgt | \
      $bpe_vocab | \
      cut -f1 -d ' ' \
      >> $bpe_vocab_file
    echo bpe vocab file created at $bpe_vocab_file
  else
    echo bpe vocab file exists at $bpe_vocab_file
  fi

  train_src=$trans_dir/train/bpe_${merge_ops}${use_lex}.$s.train.$yrv
  train_tgt=$trans_dir/train/bpe_${merge_ops}${use_lex}.$t.train.$yrv
  dev_src=$trans_dir/dev/bpe_${merge_ops}${use_lex}.$s.dev.$yrv
  dev_tgt=$trans_dir/dev/bpe_${merge_ops}${use_lex}.$t.dev.$yrv
  vocab_src=$bpe_vocab_file
  vocab_tgt=$bpe_vocab_file

fi

# train
export PYTHONPATH=$cnn_dir:$PYTHONPATH
CUDA_VISIBLE_DEVICES=$gpu_id python -m bin.train \
  --config_paths="
    ${config_yml},
    $cnn_dir/example_configs/train_seq2seq.yml,
    $cnn_dir/example_configs/text_metrics_bpe.yml" \
  --model_params "
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
  --output_dir $cnn_model_dir



