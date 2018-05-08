#!/bin/bash

### main script
BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $BASE_DIR/../config.sh $1 $2
source $BASE_DIR/../utils.sh

s=$1
t=$2

for dataset0 in train dev test; do
  # parse
  xml_file=$corpus_dir/$dataset_name.${st}.$dataset0.${yrv}.xml

  mt_source_file=${dataset_dir}/mt.${s}.$dataset0.${yrv}
  mt_target_file=${dataset_dir}/mt.${t}.$dataset0.${yrv}

  #if [ -f $xml_file ] && [ ! -f $mt_source_file ]; then
  echo parsing ${xml_file} ...
  python $BASE_DIR/extract_tokenized_from_xml.py \
    --xml $xml_file \
    --source_lang $s \
    --mt_source $mt_source_file \
    --mt_target $mt_target_file
  #elif [ ! -f $xml_file ]; then
  #  echo xml_file does not exist at $xml_file
  #else
  #  echo mt_source_file exists at $mt_source_file, and mt_target_file exists at $mt_target_file
  #fi

  ## tokenize
  #mt_source_file_tokenized=${dataset_dir}/mt_tokenized.${s}.$dataset0.${yrv}
  #tokenize2 $s $mt_source_file $mt_source_file_tokenized
  #mt_target_file_tokenized=${dataset_dir}/mt_tokenized.${t}.$dataset0.${yrv}
  #tokenize2 $t $mt_target_file $mt_target_file_tokenized

  ## truecase
  #if [ "$dataset0" == "train" ]; then
  #  source_truecase_model=$truecase_model_dir/truecase-model.$s
  #  truecase_train2 $mt_source_file_tokenized $source_truecase_model
  #  target_truecase_model=$truecase_model_dir/truecase-model.$t
  #  truecase_train2 $mt_target_file_tokenized $target_truecase_model
  #fi
  #mt_source_file_truecased=${dataset_dir}/mt_truecased.${s}.$dataset0.${yrv}
  #truecase_infer2 $source_truecase_model $mt_source_file_tokenized $mt_source_file_truecased
  #mt_target_file_truecased=${dataset_dir}/mt_truecased.${t}.$dataset0.${yrv}
  #truecase_infer2 $target_truecase_model $mt_target_file_tokenized $mt_target_file_truecased

  #if [ "$dataset0" != "train" ]; then
  #  cp $mt_source_file_truecased ${trans_dir}/$dataset0/${src_label}_raw.${s}.$dataset0.${yrv}
  #  echo "copied $mt_source_file_truecased to ${trans_dir}/$dataset0/${src_label}_raw.${s}.$dataset0.${yrv}"
  #  cp $mt_target_file_truecased ${trans_dir}/$dataset0/${ref_label}_raw.${t}.$dataset0.${yrv}
  #  echo "copied $mt_target_file_truecased to ${trans_dir}/$dataset0/${ref_label}_raw.${t}.$dataset0.${yrv}"
  #fi

  ## prune: only on training
  #if [ "$dataset0" == "train" ]; then
  #  mt_source_file_pruned=${dataset_dir}/mt_pruned.${s}.$dataset0.${yrv}
  #  mt_target_file_pruned=${dataset_dir}/mt_pruned.${t}.$dataset0.${yrv}
  #  python $BASE_DIR/prune_long_sent.py \
  #    --source_input $mt_source_file_truecased \
  #    --target_input $mt_target_file_truecased \
  #    --max_len 100 \
  #    --source_output $mt_source_file_pruned \
  #    --target_output $mt_target_file_pruned

  #  cp $mt_source_file_pruned ${trans_dir}/$dataset0/${src_label}_raw.${s}.$dataset0.${yrv}
  #  echo "copied $mt_source_file_pruned to ${trans_dir}/$dataset0/${src_label}_raw.${s}.$dataset0.${yrv}"
  #  cp $mt_target_file_pruned ${trans_dir}/$dataset0/${ref_label}_raw.${t}.$dataset0.${yrv}
  #  echo "copied $mt_target_file_pruned to ${trans_dir}/$dataset0/${ref_label}_raw.${t}.$dataset0.${yrv}"
  #fi

  #echo --------
done
