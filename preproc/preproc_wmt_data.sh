#!/bin/bash

### main script
BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $BASE_DIR/../config.sh $1 $2
source $BASE_DIR/../utils.sh

# construct training data
europarl_src=$corpus_dir/training/europarl-v7.$st.$s
commoncrawl_src=$corpus_dir/commoncrawl.$st.$s
newscommentary_src=$corpus_dir/training-parallel-nc-v11/news-commentary-v11.$st.$s
mt_source_train=$dataset_dir/mt.$s.train.$yrv
if [ ! -f $mt_source_train ]; then
  cat \
    $europarl_src \
    $commoncrawl_src \
    $newscommentary_src \
    > $mt_source_train
else
  echo "$s training data exists at $mt_source_train"
fi

europarl_tgt=$corpus_dir/training/europarl-v7.$st.$t
commoncrawl_tgt=$corpus_dir/commoncrawl.$st.$t
newscommentary_tgt=$corpus_dir/training-parallel-nc-v11/news-commentary-v11.$st.$t
mt_target_train=$dataset_dir/mt.$t.train.$yrv
if [ ! -f $mt_target_train ]; then
  cat \
    $europarl_tgt \
    $commoncrawl_tgt \
    $newscommentary_tgt \
    > $mt_target_train
else
  echo "$t training data exists at $mt_target_train"
fi

# convert SGM files for dev and test data

mt_source_dev=$dataset_dir/mt.$s.dev.$yrv
mt_target_dev=$dataset_dir/mt.$t.dev.$yrv
if [ ! -f $mt_source_dev ]; then
  $sgm_converter \
    < $corpus_dir/dev/newstest2013-$src_label.$s.sgm \
    > $mt_source_dev
  $sgm_converter \
    < $corpus_dir/dev/newstest2013-$ref_label.$t.sgm \
    > $mt_target_dev
fi

mt_source_test=$dataset_dir/mt.$s.test.$yrv
mt_target_test=$dataset_dir/mt.$t.test.$yrv
if [ ! -f $mt_source_test ]; then
  $sgm_converter \
    < $corpus_dir/dev/newstest2014-$st2-$src_label.$s.sgm \
    > $mt_source_test
  $sgm_converter \
    < $corpus_dir/dev/newstest2014-$st2-$ref_label.$t.sgm \
    > $mt_target_test
fi

mt_source_syscomb=$dataset_dir/mt.$s.syscomb.$yrv
mt_target_syscomb=$dataset_dir/mt.$t.syscomb.$yrv
if [ ! -f $mt_source_syscomb ]; then
  $sgm_converter \
    < $corpus_dir/dev/newstest2015-$st2-$src_label.$s.sgm \
    > $mt_source_syscomb
  $sgm_converter \
    < $corpus_dir/dev/newstest2015-$st2-$ref_label.$t.sgm \
    > $mt_target_syscomb
fi

mt_source_unseq=$dataset_dir/mt.$s.unseq.$yrv
mt_target_unseq=$dataset_dir/mt.$t.unseq.$yrv
if [ ! -f $mt_source_unseq ]; then
  $sgm_converter \
    < $corpus_dir/test/newstest2016-$st2-$src_label.$s.sgm \
    > $mt_source_unseq
  $sgm_converter \
    < $corpus_dir/test/newstest2016-$st2-$ref_label.$t.sgm \
    > $mt_target_unseq
fi

for dataset0 in train dev test syscomb unseq; do
  # tokenize
  mt_source_file_tokenized=$dataset_dir/mt_tokenized.$s.$dataset0.${yrv}
  mt_target_file_tokenized=$dataset_dir/mt_tokenized.$t.$dataset0.${yrv}
  tokenize2 $s $dataset_dir/mt.$s.$dataset0.$yrv $mt_source_file_tokenized
  tokenize2 $t $dataset_dir/mt.$t.$dataset0.$yrv $mt_target_file_tokenized

  if [ "$dataset0" != "train" ]; then
    mkdir -p ${trans_dir}/$dataset0
    if [ ! -f $trans_dir/$dataset0/${src_label}_raw.$s.$dataset0.$yrv ]; then
      cp $mt_source_file_tokenized $trans_dir/$dataset0/${src_label}_raw.$s.$dataset0.$yrv
    fi
    if [ ! -f $trans_dir/$dataset0/${ref_label}_raw.$t.$dataset0.$yrv ]; then
      cp $mt_target_file_tokenized $trans_dir/$dataset0/${ref_label}_raw.$t.$dataset0.$yrv
    fi
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

    if [ ! -f ${trans_dir}/$dataset0/${src_label}_raw.$s.$dataset0.${yrv} ]; then
      cp $mt_source_file_pruned ${trans_dir}/$dataset0/${src_label}_raw.$s.$dataset0.${yrv}
      echo "copied $mt_source_file_pruned to ${trans_dir}/$dataset0/${src_label}_raw.$s.$dataset0.${yrv}"
    fi
    if [ ! -f ${trans_dir}/$dataset0/${ref_label}_raw.$t.$dataset0.${yrv} ]; then
      cp $mt_target_file_pruned ${trans_dir}/$dataset0/${ref_label}_raw.$t.$dataset0.${yrv}
      echo "copied $mt_target_file_pruned to ${trans_dir}/$dataset0/${ref_label}_raw.$t.$dataset0.${yrv}"
    fi
  fi
done


