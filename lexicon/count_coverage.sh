#!/bin/bash

BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $BASE_DIR/../config.sh $1 $2
source $BASE_DIR/../utils.sh

s=$1
t=$2

train_src=${trans_dir}/train/${src_label}_raw.${s}.train.${yrv}
test_src=${trans_dir}/test/${src_label}_raw.${s}.test.${yrv}

src_file_from_lexicon=$dataset_dir/lexicon_words.$s
tgt_file_from_lexicon=$dataset_dir/lexicon_words.$t

export QT_QPA_PLATFORM=offscreen

python $BASE_DIR/count_coverage.py \
  $train_src \
  $test_src \
  $bilingual_lexicon \
  $src_file_from_lexicon \
  $tgt_file_from_lexicon

# generate training + lexicon
train_tgt=${trans_dir}/train/${ref_label}_raw.${t}.train.${yrv}
train_src_plus_lexicon=${trans_dir}/train/${src_label}_raw_lex.${s}.train.${yrv}
train_tgt_plus_lexicon=${trans_dir}/train/${ref_label}_raw_lex.${t}.train.${yrv}

if [ ! -f $train_src_plus_lexicon ]; then
  cat $train_src $src_file_from_lexicon > $train_src_plus_lexicon
  echo train_src_plus_lexicon created at: $train_src_plus_lexicon
else
  echo train_src_plus_lexicon exists at: $train_src_plus_lexicon
fi

if [ ! -f $train_tgt_plus_lexicon ]; then
  cat $train_tgt $tgt_file_from_lexicon > $train_tgt_plus_lexicon
  echo train_tgt_plus_lexicon created at: $train_tgt_plus_lexicon
else
  echo train_tgt_plus_lexicon exists at: $train_tgt_plus_lexicon
fi
