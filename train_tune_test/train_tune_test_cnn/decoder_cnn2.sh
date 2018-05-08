#!/bin/bash

### main script
BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $BASE_DIR/../../config.sh $1 $2
source $BASE_DIR/../../utils.sh

s=$1
t=$2

merge_ops=$3
cnn_model_dir=$4
gpu_id=$5
dim=$6
num_layer=$7
kernel_width=$8
use_lex=$9


if [ "$merge_ops" != "inf" ]; then
  bpe_dir=$trans_dir/bpe
  mkdir -p $bpe_dir

  if [[ "$merge_ops" == *-* ]]; then
    IFS='-' read -a merge_ops_pair <<< "$merge_ops"
    merge_ops_s=${merge_ops_pair[0]}
    merge_ops_t=${merge_ops_pair[1]}

    bpe_src=$bpe_dir/ibpe${use_lex}.${merge_ops_s}.$s
    # apply individual BPE on src
    echo "Apply BPE with merge_ops=${merge_ops_s} to pruned files..."
    for dataset0 in test; do
      infile=$trans_dir/$dataset0/${src_label}_raw.$s.$dataset0.$yrv
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

    # apply individual BPE on tgt
    bpe_tgt=$bpe_dir/ibpe${use_lex}.${merge_ops_t}.$t
    echo "Apply BPE with merge_ops=${merge_ops_t} to pruned files..."
    for dataset0 in test; do
      infile=$trans_dir/$dataset0/${ref_label}_raw.$t.$dataset0.$yrv
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

  else
    bpe_file=$bpe_dir/bpe${use_lex}.${merge_ops}

    # apply shared BPE on src
    echo "Apply BPE with merge_ops=${merge_ops} to pruned files..."
    for dataset0 in test; do
      infile=$trans_dir/$dataset0/${src_label}_raw.$s.$dataset0.$yrv
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
    
    # apply shared BPE on tgt
    for dataset0 in test; do
      infile=$trans_dir/$dataset0/${ref_label}_raw.$t.$dataset0.$yrv
      outfile=$trans_dir/$dataset0/bpe_${merge_ops}${use_lex}.$t.$dataset0.$yrv
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
  fi
fi

echo enter ckpt:
read ckpt0
ckpt=$(python find_nearest_ckpt.py $cnn_model_dir $ckpt0)
echo nearest ckpt: $ckpt

ls $cnn_model_dir/model.ckpt-$ckpt.index

export PYTHONPATH=$cnn_dir:$PYTHONPATH
for dataset0 in dev test; do
  if [ "$merge_ops" == "inf" ]; then
    source_file=$trans_dir/$dataset0/${src_label}_raw.$s.$dataset0.$yrv
  elif [[ "$merge_ops" == *-* ]]; then
    IFS='-' read -a merge_ops_pair <<< "$merge_ops"
    merge_ops_s=${merge_ops_pair[0]}
    source_file=$trans_dir/$dataset0/ibpe_${merge_ops_s}${use_lex}.$s.$dataset0.$yrv
  else
    source_file=$trans_dir/$dataset0/bpe_${merge_ops}${use_lex}.$s.$dataset0.$yrv
  fi

  onebest_file=$trans_dir/$dataset0/onebest_cnn_dim${dim}_layer${num_layer}_kernel${kernel_width}_bpe${merge_ops}${use_lex}_ckpt${ckpt}.$t.$dataset0.$yrv

  if [ ! -f $onebest_file.tmp ]; then
    CUDA_VISIBLE_DEVICES=$gpu_id python -m bin.infer \
      --tasks "
        - class: DecodeText
        - class: DumpBeams
          params:
            file: $cnn_model_dir/beams.npz" \
      --model_dir $cnn_model_dir \
      --model_params "
        inference.beam_search.beam_width: 5
        decoder.class: seq2seq.decoders.ConvDecoderFairseqBS" \
      --input_pipeline "
        class: ParallelTextInputPipelineFairseq
        params:
          source_files:
            - $source_file" \
      > $onebest_file.tmp
  else
    echo translation for $dataset0 exists at $onebest_file.tmp
  fi

  sed -r 's/(@@ )|(@@ ?$)//g' $onebest_file.tmp > $onebest_file
  echo translation created at $onebest_file

  # bleu sent_bleu meteor
  for metric in bleu; do
    suffix=${t}.${dataset0}.${yrv}
    ref_raw=$trans_dir/$dataset0/${ref_label}_raw.$suffix
    method=cnn_dim${dim}_layer${num_layer}_kernel${kernel_width}_bpe${merge_ops}${use_lex}
    hyp_raw=$trans_dir/$dataset0/onebest_${method}_ckpt${ckpt}.$suffix
    echo conv-seq2seq: $s $metric $dataset0 $method
    get_$metric $ref_raw $hyp_raw
  done
done


