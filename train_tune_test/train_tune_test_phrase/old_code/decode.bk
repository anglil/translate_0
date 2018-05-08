#!/bin/bash

### main script
BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $BASE_DIR/../../config.sh $1 $2
source $BASE_DIR/../../utils.sh

s=$1
t=$2

mpl=3

##################################
# model paths
##################################

model_dir=$exp_dir/model_mpl${mpl}
mkdir -p $model_dir
truecase_model_dir=$model_dir/truecase
mkdir -p $truecase_model_dir
language_model_dir=$model_dir/lm
mkdir -p $language_model_dir
train_model_dir=$model_dir/train
mkdir -p $train_model_dir
dev_model_dir=$model_dir/dev
mkdir -p $dev_model_dir

phrase_bin=$dev_model_dir/phrase-table.minphr
lexical_bin=$dev_model_dir/reordering-table.minlexr

model_trained=$train_model_dir/model/moses.ini
model_tuned=$dev_model_dir/moses.ini
model_tuned_bin=$dev_model_dir/moses.bin.ini

# ----------------

for dataset0 in dev test; do
  source_file=${trans_dir}/$dataset0/${src_label}_raw.${s}.$dataset0.${yrv}
  onebest_file=${trans_dir}/$dataset0/onebest_phrase_mpl${mpl}.${t}.$dataset0.${yrv}
  nbest_file=${trans_dir}/$dataset0/n_best/nbest_phrase_mpl${mpl}.${t}.$dataset0.${yrv}
  oov_word_file=${trans_dir}/$dataset0/oov/oov_word_phrase_mpl${mpl}.${t}.$dataset0.${yrv}
  oov_pos_file=${trans_dir}/$dataset0/oov/oov_phrase_mpl${mpl}.${t}.$dataset0.${yrv}
  oov_unique_file=${trans_dir}/$dataset0/oov/oov_unique_phrase_mpl${mpl}.${t}.$dataset0.${yrv}

  if [ "$dataset0" != "dev" ]; then
    ##################################
    # decoding for a dataset other than dev
    ##################################
    if [ -f $onebest_file ]; then
      echo "Translation exists for the $dataset0 set at $onebest_file"
    else
      filter_model_dir=$model_dir/filter_$dataset0
      model_filtered=$filter_model_dir/moses.ini
    	if [ -f $model_filtered ]; then
    	  echo "Filtered model exists at $model_filtered"
    	else
        ### only keep the entries needed for translating the test set. This will make the translation a lot faster.
    	  echo "Start filtering..."
    	  $test_filter \
    	    $filter_model_dir \
    	    $model_tuned \
    	    $source_file \
    	    -Binarizer $phrase_binarizer
    	  echo "Filtering done."
    	fi
    
      $decoder \
        -f $model_filtered \
      	-output-unknowns $oov_word_file \
        -n-best-list $nbest_file 20 \
        < $source_file \
        > $onebest_file \
        2> $log_dir/tmp_$dataset0.log
    fi
  else
    ##################################
    # decoding for dev set
    ##################################
    if [ -f $onebest_file ]; then
      echo "Translation exists for the $dataset0 set at $onebest_file"
    else
      $decoder \
        -f $model_tuned_bin \
    		-output-unknowns $oov_word_file \
        -n-best-list $nbest_file 20 \
        < $source_file \
        > $onebest_file \
        2> $log_dir/tmp_$dataset0.log
    fi
  fi

  ### extract the position of the oov words in each sentence
  python $BASE_DIR/get_oov_pos.py \
    --onebest_file $onebest_file \
    --oov_word_file $oov_word_file \
    --oov_pos_file $oov_pos_file \
    --oov_unique_file $oov_unique_file

  for metric in bleu; do
    suffix=$t.$dataset0.$yrv
    method=phrase_mpl$mpl
    ref_raw=$trans_dir/$dataset0/${ref_label}_raw.$suffix
    hyp_raw=$trans_dir/$dataset0/onebest_${method}.$suffix
    echo phrase: $st $metric $dataset0 $method
    get_$metric $ref_raw $hyp_raw
  done
  echo ----------------
done

