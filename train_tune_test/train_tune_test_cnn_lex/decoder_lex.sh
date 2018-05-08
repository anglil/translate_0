#!/bin/bash

BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $BASE_DIR/../../config.sh $1 $2
source $BASE_DIR/../../utils.sh

s=$1
t=$2

# 0 1
injective=$3
lex_model_dir=$4
gpu_id=$5
dim=$6
num_layer=$7
kernel_width=$8
use_lex=$9

vocab_dir=$trans_dir/bpe
mkdir -p $vocab_dir

echo enter ckpt:
read ckpt0
ckpt=$(python find_nearest_ckpt.py $lex_model_dir $ckpt0)
echo nearest ckpt: $ckpt

ls $lex_model_dir/model.ckpt-$ckpt.index

export PYTHONPATH=$cnn_dir:$PYTHONPATH
for dataset0 in dev test; do
  source_file=$trans_dir/$dataset0/${src_label}_raw.$s.$dataset0.$yrv
  onebest_file=$trans_dir/$dataset0/onebest_lex${injective}_dim${dim}_layer${num_layer}_kernel${kernel_width}${use_lex}_ckpt${ckpt}.$t.$dataset0.$yrv

  if [ ! -f $onebest_file ]; then
    CUDA_VISIBLE_DEVICES=$gpu_id python -m bin.infer \
      --tasks "
        - class: DecodeText
        - class: DumpBeams
          params:
            file: $lex_model_dir/beams.npz" \
      --model_dir $lex_model_dir \
      --model_params "
        glove_dict_file: $glove_mat
        lexicon_dict_file: $bilingual_lexicon
        inference.beam_search.beam_width: 5
        decoder.class: seq2seq.decoders.ConvDecoderFairseqBS" \
      --input_pipeline "
        class: ParallelTextInputPipelineFairseq
        params:
          source_files:
            - $source_file" \
      --checkpoint_path "$lex_model_dir/model.ckpt-$ckpt" \
      > $onebest_file
    echo translation for $dataset0 created at: $onebest_file
      
  else
    echo translation for $dataset0 exists at: $onebest_file
  fi

  # bleu sent_bleu meteor
  for metric in bleu; do
    suffix=${t}.${dataset0}.${yrv}
    ref_raw=$trans_dir/$dataset0/${ref_label}_raw.$suffix
    method=lex${injective}_dim${dim}_layer${num_layer}_kernel${kernel_width}${use_lex}
    hyp_raw=$trans_dir/$dataset0/onebest_${method}_ckpt${ckpt}.$suffix
    echo lex-seq2seq: $s $metric $dataset0 $method
    get_$metric $ref_raw $hyp_raw
  done
done


