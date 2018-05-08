#!/bin/bash

BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $BASE_DIR/../../config.sh $1 $2
source $BASE_DIR/../../utils.sh

s=$1
t=$2

# learn shared BPE
train_src=$trans_dir/train/${src_label}_raw.$s.train.${yrv}
train_tgt=$trans_dir/train/${ref_label}_raw.$t.train.${yrv}
merge_ops=32000
bpe_file=$tfnmt_model_dir/bpe.${merge_ops}
if [ ! -f $bpe_file ]; then
  echo "Learning BPE with merge_ops=${merge_ops}. This may take a while..."
  cat $train_src $train_tgt | \
    $bpe_train \
    -s $merge_ops \
    > $bpe_file
  echo bpe model created at $tfnmt_model_dir/bpe.${merge_ops}
else
  echo bpe model exists at $tfnmt_model_dir/bpe.${merge_ops}
fi

# apply shared BPE to train and dev
echo "Apply BPE with merge_ops=${merge_ops} to pruned files..."
for dataset0 in train dev; do
  infile=$trans_dir/$dataset0/${src_label}_raw.$s.$dataset0.$yrv
  outfile=$trans_dir/$dataset0/bpe_$merge_ops.$s.$dataset0.$yrv
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
  infile=$trans_dir/$dataset0/${ref_label}_raw.$t.$dataset0.$yrv
  outfile=$trans_dir/$dataset0/bpe_$merge_ops.$t.$dataset0.$yrv
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
bpe_train_src=$trans_dir/train/bpe_$merge_ops.$s.train.$yrv
bpe_train_tgt=$trans_dir/train/bpe_$merge_ops.$t.train.$yrv
bpe_vocab_file=$tfnmt_model_dir/vocab.bpe.${merge_ops}
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

# train: the suffix must be language names
for lang in $s $t; do
  # same vocab for both sides
  cp $bpe_vocab_file $bpe_vocab_file.$lang
  for dataset0 in train dev; do
    # duplicate processed datasets to have a language suffix
    cp $trans_dir/$dataset0/bpe_$merge_ops.$lang.$dataset0.$yrv $trans_dir/$dataset0/$dataset0.$yrv.$lang
  done
done

export PYTHONPATH=$tfnmt_dir:$PYTHONPATH
train_prefix=${trans_dir}/train/train.$yrv
dev_prefix=${trans_dir}/dev/dev.$yrv
if [ "$s" == "en" ] || [ "$s" == "de" ]; then
  python -m nmt.nmt \
    --src=$s \
    --tgt=$t \
    --hparams_path=$tfnmt_dir/nmt/standard_hparams/wmt16_gnmt_4_layer.json \
    --out_dir=$tfnmt_model_dir \
    --vocab_prefix=$bpe_vocab_file \
    --train_prefix=$train_prefix \
    --dev_prefix=${trans_dir}/dev/dev.$yrv
else
  python -m nmt.nmt \
    --attention=scaled_luong \
    --src=$s \
    --tgt=$t \
    --out_dir=$tfnmt_model_dir \
    --vocab_prefix=$bpe_vocab_file \
    --train_prefix=$train_prefix \
    --dev_prefix=$dev_prefix \
    --num_train_steps=500000 \
    --steps_per_stats=100 \
    --num_gpus=2 \
    --num_layers=2 \
    --num_units=128 \
    --dropout=0.2 \
    --metrics=bleu
fi

