#!/bin/bash

### main script
BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $BASE_DIR/../../config.sh $1 $2
source $BASE_DIR/../../utils.sh

s=$1
t=$2

# learn shared BPE
merge_ops=32000
train_src=$trans_dir/train/${src_label}_raw.$s.train.${yrv}
train_tgt=$trans_dir/train/${ref_label}_raw.$t.train.${yrv}
bpe_file=$cnn_model_dir/bpe.${merge_ops}
if [ ! -f $bpe_file ]; then
  echo "Learning BPE with merge_ops=${merge_ops}. This may take a while..."
  cat $train_src $train_tgt | \
    $bpe_train \
    -s $merge_ops \
    > $bpe_file
  echo bpe model created at $cnn_model_dir/bpe.${merge_ops}
else
  echo bpe model exists at $cnn_model_dir/bpe.${merge_ops}
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
    echo bpe-ed $dataset0 file createed at $outfile
  else
    echo bpe-ed $dataset0 file exists at $outfile
  fi
done

# create a vocab using the bpe-ed training data
bpe_train_src=$trans_dir/train/bpe_$merge_ops.$s.train.$yrv
bpe_train_tgt=$trans_dir/train/bpe_$merge_ops.$t.train.$yrv
bpe_vocab_file=$cnn_model_dir/vocab.bpe.${merge_ops}
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

# train
#for lang in $s $t; do
#  cp $bpe_vocab_file $bpe_vocab_file.$lang
#  for dataset0 in train dev; do
#    cp $trans_dir/$dataset0/bpe_$merge_ops.$lang.$dataset0.$yrv $trans_dir/$dataset0/$dataset0.$yrv.$lang
#  done
#done
vocab_src=$bpe_vocab_file
vocab_tgt=$bpe_vocab_file
train_src=$trans_dir/train/bpe_$merge_ops.$s.train.$yrv
train_tgt=$trans_dir/train/bpe_$merge_ops.$t.train.$yrv
dev_src=$trans_dir/dev/bpe_$merge_ops.$s.dev.$yrv
dev_tgt=$trans_dir/dev/bpe_$merge_ops.$t.dev.$yrv

export PYTHONPATH=$cnn_dir:$PYTHONPATH
python -m bin.train \
  --config_paths="
    $cnn_dir/example_configs/conv_seq2seq.yml,
    $cnn_dir/example_configs/train_seq2seq.yml,
    $cnn_dir/example_configs/text_metrics_bpe.yml" \
  --model_params "
    vocab_source: $bpe_vocab_file
    vocab_target: $bpe_vocab_file" \
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
  --train_steps 1000000 \
  --output_dir $cnn_model_dir




