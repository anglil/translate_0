#!/bin/bash

### main script
BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $BASE_DIR/../../config.sh $1 $2
source $BASE_DIR/../../utils.sh

s=$1
t=$2

merge_ops=32000
bpe_file=$tfnmt_model_dir/bpe.${merge_ops}

# apply shared BPE on src
echo "Apply BPE with merge_ops=${merge_ops} to pruned files..."
for dataset0 in test; do
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

# apply shared BPE on tgt
for dataset0 in test; do
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

# decode
bpe_vocab_file=$tfnmt_model_dir/vocab.bpe.${merge_ops}

export PYTHONPATH=$tfnmt_dir:$PYTHONPATH
for dataset0 in dev test; do
  source_file=$trans_dir/$dataset0/bpe_$merge_ops.$s.$dataset0.$yrv
  #source_file=$trans_dir/$dataset0/${src_label}_raw.$s.$dataset0.$yrv
  onebest_file=$trans_dir/$dataset0/onebest_tfnmt.$t.$dataset0.$yrv

  if [ ! -f $onebest_file.tmp ]; then
    python -m nmt.nmt \
      --src=$s \
      --tgt=$t \
      --out_dir=$tfnmt_model_dir \
      --vocab_prefix=$bpe_vocab_file \
      --inference_input_file=$source_file \
      --inference_output_file=$onebest_file.tmp
  else
    echo translation for $dataset0 exists at $onebest_file.tmp
  fi

  sed -r 's/(@@ )|(@@ ?$)//g' $onebest_file.tmp > $onebest_file
done
