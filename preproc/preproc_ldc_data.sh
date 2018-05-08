#!/bin/bash

### main script
BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $BASE_DIR/../config.sh $1 $2
source $BASE_DIR/../utils.sh

s=$1
t=$2

# parse
mt_source_train=$dataset_dir/mt.$s.train.$yrv
mt_target_train=$dataset_dir/mt.$t.train.$yrv
mt_source_dev=$dataset_dir/mt.$s.dev.$yrv
mt_target_dev=$dataset_dir/mt.$t.dev.$yrv
mt_source_test=$dataset_dir/mt.$s.test.$yrv
mt_target_test=$dataset_dir/mt.$t.test.$yrv

python $BASE_DIR/extract_tokenized_from_xml_ldc.py \
  --lang1_dir $corpus_dir/$s/ltf \
  --lang2_dir $corpus_dir/$t/ltf \
  --training_docs $dataset_dir/training_docs \
  --training_lang1 $mt_source_train \
  --training_lang2 $mt_target_train \
  --dev_docs $dataset_dir/dev_docs \
  --dev_lang1 $mt_source_dev \
  --dev_lang2 $mt_target_dev \
  --test_docs $dataset_dir/test_docs \
  --test_lang1 $mt_source_test \
  --test_lang2 $mt_target_test

for dataset0 in train dev test; do
  # tokenize
  mt_source_file_tokenized=$dataset_dir/mt_tokenized.$s.$dataset0.${yrv}
  mt_target_file_tokenized=$dataset_dir/mt_tokenized.$t.$dataset0.${yrv}
  tokenize2 $s $dataset_dir/mt.$s.$dataset0.$yrv $mt_source_file_tokenized
  tokenize2 $t $dataset_dir/mt.$t.$dataset0.$yrv $mt_target_file_tokenized

  if [ "$dataset0" != "train" ]; then
    mkdir -p ${trans_dir}/$dataset0
    python $BASE_DIR/prune_ldc_sent.py \
      $mt_source_file_tokenized \
      $mt_target_file_tokenized \
      $trans_dir/$dataset0/${src_label}_raw.$s.$dataset0.$yrv \
      $trans_dir/$dataset0/${ref_label}_raw.$t.$dataset0.$yrv
  fi  

  # prune
  if [ "$dataset0" == "train" ]; then
    mt_source_file_pruned=$dataset_dir/mt_pruned.$s.$dataset0.${yrv}
    mt_target_file_pruned=$dataset_dir/mt_pruned.$t.$dataset0.${yrv}
    python $BASE_DIR/prune_long_sent.py \
      --source_input $mt_source_file_tokenized \
      --target_input $mt_target_file_tokenized \
      --max_len 80 \
      --source_output $mt_source_file_pruned \
      --target_output $mt_target_file_pruned

    python $BASE_DIR/prune_ldc_sent.py \
      $mt_source_file_pruned \
      $mt_target_file_pruned \
      ${trans_dir}/$dataset0/${src_label}_raw.$s.$dataset0.${yrv} \
      ${trans_dir}/$dataset0/${ref_label}_raw.$t.$dataset0.${yrv}
  fi  
  echo ----------------
done
