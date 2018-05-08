#!/bin/bash


usage() { echo -e "-s: source language\n-t: target language\n-k: task: train,test\n-l: max phrase length: num\n-e: use_bpe: store_true,store_false\n-b: share_vocab (between src & tgt): store_true,store_false\n-v: vocab_size: num\n-u: use_lex: store_true,store_false"; }

s=''
t=''
task=train
mpl=7
use_bpe=false

share_vocab=false
use_subword_tokenizer=$use_bpe
vocab_size=8000

use_lex=false

while getopts ":hs:t:k:l:ebwv:u" flag; do
  case "${flag}" in
    s) s="${OPTARG}" ;;
    t) t="${OPTARG}" ;;
    k) task="${OPTARG}" ;;
    l) mpl="${OPTARG}" ;;
    e) use_bpe=true ;;
    b) share_vocab=true ;;
    v) vocab_size="${OPTARG}" ;;
    u) use_lex=true ;;
    h) usage; exit ;;
    *) error "Unexpected option ${flag}" ;;
  esac
done

if [ -z "$s" ]; then
  echo "no arg supplemented for s"; exit 1
fi
if [ -z "$t" ]; then
  echo "no arg supplemented for t"; exit 1
fi

echo "s = $s"
echo "t = $t"
echo "task = $task"
echo "mpl = $mpl"
echo "use_bpe = $use_bpe"
if [ "$use_bpe" = true ]; then
  echo "share_vocab = $share_vocab"
  echo "use_subword_tokenizer = $use_subword_tokenizer"
  echo "vocab_size = $vocab_size"
else
  echo "share_vocab = NA"
  echo "use_subword_tokenizer = NA"
  echo "vocab_size = NA"
fi
echo "use_lex = $use_lex"

echo --------

BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $BASE_DIR/../../config.sh $s $t
source $BASE_DIR/../../utils.sh

# ---- language model ----
lm_factor=0
ngram=3
lm_type=8

# ---- alignment model ----
# grow, intersect (intersection of the GIZA++ alignments in both directions), union (union of the two GIZA++ alignments)
al_basic=grow
# include diagonal neighbors
al_diag=diag
# final step: http://www.statmt.org/moses/?n=FactoredTraining.AlignWords
al_final=final
# and/or in a final step condition: http://www.cl.uni-heidelberg.de/courses/ss15/smt/scribe3.pdf
al_and_or=and
# overall alignment configuration
al=$al_basic-$al_diag-$al_final-$al_and_or

# ---- reordering model ----
# (modeltype) wbe: word-based extraction, but phrase-based at decoding
ro_modeltype=wbe
# (orientation) msd: considers three different orientations: monotone, swap, discontinuous; mslr: monotone, swap, discontinuous-left, discontinuous-right
ro_orientation=msd
# (directionality) bidirectional: use both backward and forward models
ro_directionality=bidirectional
# (language) fe: conditioned on both the source and target languages
ro_language=fe
# (collapsing) allff: treat the scores as individual feature functions
ro_collapsing=allff
# overall reordering configuration
ro=$ro_modeltype-$ro_orientation-$ro_directionality-$ro_language-$ro_collapsing

# -------- derived variables

get_merge_ops() {
  if [ "$use_subword_tokenizer" = false ]; then
    echo "inf"
  else
    if [ "$share_vocab" = true ]; then
      echo $vocab_size
    else
      echo ${vocab_size}-${vocab_size}
    fi  
  fi  
}
merge_ops=$( get_merge_ops )

# use_lex, used in locating the vocab file
if [ "$use_lex" = true ]; then
  use_lex=_lex
else
  use_lex=""
fi

# locate the vocab file
vocab_src=$trans_dir/t2t_${merge_ops}$use_lex/vocab.${s}${t}.$vocab_size.$s
vocab_tgt=$trans_dir/t2t_${merge_ops}$use_lex/vocab.${s}${t}.$vocab_size.$t
if [ "$share_vocab" = true ]; then
  vocab_src=$trans_dir/t2t_${merge_ops}$use_lex/vocab.${s}${t}.$vocab_size
  vocab_tgt=$vocab_src
fi

# build training, dev and test texts using bpe
use_bpe_piece=""
if [ "$use_bpe" = true ]; then
  use_bpe=_usebpe${merge_ops}
  use_bpe_piece=_usebpepiece${merge_ops}

  dataset0=train
  python $BASE_DIR/../train_tune_test_t2t/bpe_encode.py \
    encode \
    ${trans_dir}/$dataset0/${src_label}_raw${use_lex}.${s}.$dataset0.${yrv} \
    ${trans_dir}/$dataset0/${src_label}_raw${use_bpe_piece}${use_lex}.${s}.$dataset0.${yrv} \
    $vocab_src
  echo --------
  python $BASE_DIR/../train_tune_test_t2t/bpe_encode.py \
    encode \
    ${trans_dir}/$dataset0/${ref_label}_raw${use_lex}.${t}.$dataset0.${yrv} \
    ${trans_dir}/$dataset0/${ref_label}_raw${use_bpe_piece}${use_lex}.${t}.$dataset0.${yrv} \
    $vocab_tgt
  echo --------

  for dataset0 in dev test; do
    python $BASE_DIR/../train_tune_test_t2t/bpe_encode.py \
      encode \
      ${trans_dir}/$dataset0/${src_label}_raw.${s}.$dataset0.${yrv} \
      ${trans_dir}/$dataset0/${src_label}_raw${use_bpe_piece}.${s}.$dataset0.${yrv} \
      $vocab_src
    echo --------
    python $BASE_DIR/../train_tune_test_t2t/bpe_encode.py \
      encode \
      ${trans_dir}/$dataset0/${ref_label}_raw.${t}.$dataset0.${yrv} \
      ${trans_dir}/$dataset0/${ref_label}_raw${use_bpe_piece}.${t}.$dataset0.${yrv} \
      $vocab_tgt
    echo --------
  done
else
  use_bpe=""
fi

# --------

##################################
# model paths
##################################
method=phrase_mpl$mpl${use_bpe}${use_lex}
method_2=phrase_mpl$mpl${use_bpe_piece}${use_lex}
model_dir=$exp_dir/$method
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

phrase_gz=$train_model_dir/model/phrase-table.gz
lexical_gz=$train_model_dir/model/reordering-table.$ro_modeltype-$ro_orientation-$ro_directionality-$ro_language.gz
#echo phrase table: $phrase_gz
#echo reordering table: $lexical_gz

model_trained=$train_model_dir/model/moses.ini
model_tuned=$dev_model_dir/moses.ini
model_tuned_bin=$dev_model_dir/moses.bin.ini

##################################
# training
##################################
train_src=${trans_dir}/train/${src_label}_raw${use_bpe_piece}${use_lex}.${s}.train.${yrv}
train_tgt=${trans_dir}/train/${ref_label}_raw${use_bpe_piece}${use_lex}.${t}.train.${yrv}
#wc -l $train_src
#wc -w $train_tgt

if [ "$task" == "train" ]; then
  ### train based on the binarized language model
  if [ -f $model_trained ]; then
    echo "Trained model exists at $model_trained"
  else    
    cp $train_src ${trans_dir}/train/train${use_bpe_piece}${use_lex}.${yrv}.$s
    cp $train_tgt ${trans_dir}/train/train${use_bpe_piece}${use_lex}.${yrv}.$t

    # create a language model on the target language
    target_side_lm=${language_model_dir}/lm_ngram.${t}.train.${yrv}
    ngram_lm_train $ngram $train_tgt $target_side_lm
    
    # binarize the language model built on the target language
    target_side_blm=${language_model_dir}/blm_ngram.${t}.train.${yrv}
    ngram_lm_binarize $target_side_lm $target_side_blm
    echo "----------------"

    $trainer \
      -root-dir $train_model_dir \
      -corpus ${trans_dir}/train/train${use_bpe_piece}${use_lex}.${yrv} \
      -f $s \
      -e $t \
      -alignment $al \
      -reordering $ro \
      -max-phrase-length $mpl \
      -lm ${lm_factor}:${ngram}:${target_side_blm}:${lm_type} \
      -external-bin-dir $moses_tool_dir \
      -cores 4 \
      2>&1 | tee $log_dir/train.log
    #wait_for_existence $model_trained
  fi
  echo "----------------"
  
  ################################## 
  # tuning
  ##################################
  
  if [ -f $model_tuned ]; then
    echo "Tuned model exists at $model_tuned"
  else
    dev_src=${trans_dir}/dev/${src_label}_raw${use_bpe_piece}.${s}.dev.${yrv}
    dev_tgt=${trans_dir}/dev/${ref_label}_raw${use_bpe_piece}.${t}.dev.${yrv}
    $tuner \
  		--working-dir=$dev_model_dir \
      $dev_src \
      $dev_tgt \
      $decoder \
  		$model_trained \
      --mertdir $moses_bin_dir \
      --decoder-flags="-threads all" \
      2>&1 | tee $log_dir/dev.log
    #wait_for_existence $model_tuned
  fi
  echo "----------------"
  
  ##################################
  # binarizing the tuned model
  ##################################
  
  ### binarize the phrase table
  if [ -f $phrase_bin ]; then
    echo "Binarized phrase table exists at $phrase_bin"
  else
    $phrase_binarizer \
      -in $train_model_dir/model/phrase-table.gz \
      -out $dev_model_dir/phrase-table \
      -nscores 4 \
      -threads 4
  fi
  
  ### binarize the lexical table
  if [ -f $lexical_bin ]; then
    echo "Binarized lexical table exists at $lexical_bin"
  else
    $lexical_binarizer \
      -in $train_model_dir/model/reordering-table.wbe-msd-bidirectional-fe.gz \
      -out $dev_model_dir/reordering-table \
      -threads 4
  fi
  
  ### binarize the tuned model based on the binarized phrase table and lexical table
  if [ -f $model_tuned_bin ]; then
  	echo "Binarized tuned model exists at $model_tuned_bin"
  else
    python $BASE_DIR/binarize_model.py \
      $model_tuned \
      $phrase_bin \
      $lexical_bin \
      $model_tuned_bin
  	#echo "Creating binarized model..."
  	#cp $model_tuned $model_tuned_bin
  	#sed -i 's/PhraseDictionaryMemory/PhraseDictionaryCompact/g' $model_tuned_bin
  	#phrase_bin_escape=$(echo "$phrase_bin" | sed 's/\//\\\//g')
  	#sed -i "/^PhraseDictionaryCompact/s/path=[^ ]* /path=${phrase_bin_escape} /g" $model_tuned_bin
  	#lexical_bin2=$( echo $lexical_bin | cut -d "." -f1,2,3,4 )
  	#lexical_bin_escape=$(echo "$lexical_bin2" | sed 's/\//\\\//g')
  	#sed -i "/^LexicalReordering/s/path=[^ ]*$/path=${lexical_bin_escape}/g" $model_tuned_bin
  	echo "Binarized tuned model created at $model_tuned_bin"
  fi
  echo "----------------"
  
elif [ "$task" == "test" ]; then
  for dataset0 in test; do
    source_file=${trans_dir}/$dataset0/${src_label}_raw${use_bpe_piece}.${s}.$dataset0.${yrv}
    #wc -l $source_file

    onebest_file_2=${trans_dir}/$dataset0/onebest_${method_2}.${t}.$dataset0.${yrv}
    nbest_file_2=${trans_dir}/$dataset0/n_best/nbest_${method_2}.${t}.$dataset0.${yrv}
    oov_word_file_2=${trans_dir}/$dataset0/oov/oov_word_${method_2}.${t}.$dataset0.${yrv}
    oov_pos_file_2=${trans_dir}/$dataset0/oov/oov_${method_2}.${t}.$dataset0.${yrv}
    oov_unique_file_2=${trans_dir}/$dataset0/oov/oov_unique_${method_2}.${t}.$dataset0.${yrv}

    onebest_file=${trans_dir}/$dataset0/onebest_${method}.${t}.$dataset0.${yrv}
    #echo onebest_file: $onebest_file

    #if [ "$dataset0" != "dev" ]; then
    #  ##################################
    #  # decoding for a dataset other than dev
    #  ##################################
    #  #if [ -f $onebest_file_2 ]; then
    #  #  echo "Translation exists for the $dataset0 set at $onebest_file_2"
    #  #else
    #  filter_model_dir=$model_dir/filter_$dataset0
    #  model_filtered=$filter_model_dir/moses.ini
    #  if [ -f $model_filtered ]; then
    #    echo "Filtered model exists at $model_filtered"
    #  else
    #    ### only keep the entries needed for translating the test set. This will make the translation a lot faster.
    #    echo "Start filtering..."
    #    $test_filter \
    #      $filter_model_dir \
    #      $model_tuned \
    #      $source_file \
    #      -Binarizer $phrase_binarizer
    #    echo "Filtering done."
    #  fi

    #  $decoder \
    #    -f $model_filtered \
    #    -output-unknowns $oov_word_file_2 \
    #    -n-best-list $nbest_file_2 20 \
    #    < $source_file \
    #    > $onebest_file_2 \
    #    2> $log_dir/tmp_$dataset0.log
    #  #fi
    #else  
      ##################################
      # decoding for dev set
      ##################################
      #if [ -f $onebest_file_2 ]; then
      #  echo "Translation exists for the $dataset0 set at $onebest_file_2"
      #else 
    echo "model_tuned_bin: $model_tuned_bin"
    echo "source_file: $source_file"
    $decoder \
      -f $model_tuned_bin \
      -output-unknowns $oov_word_file_2 \
      -n-best-list $nbest_file_2 20 \
      < $source_file \
      > $onebest_file_2 \
      2> $log_dir/tmp_$dataset0.log
      #fi
    #fi
    echo --------
    
    ### extract the position of the oov words in each sentence
    python $BASE_DIR/get_oov_pos.py \
      --onebest_file $onebest_file_2 \
      --oov_word_file $oov_word_file_2 \
      --oov_pos_file $oov_pos_file_2 \
      --oov_unique_file $oov_unique_file_2

    if [ ! -z "$use_bpe" ]; then
      python $BASE_DIR/../train_tune_test_t2t/bpe_encode.py \
        decode \
        $onebest_file_2 \
        ${onebest_file}.tmp \
        $vocab_tgt
      if [ -f $onebest_file ]; then
        mv ${onebest_file} ${onebest_file}.bk
      fi
      tokenize2 $t ${onebest_file}.tmp ${onebest_file}
      echo --------
    fi

    ref_raw=$trans_dir/$dataset0/${ref_label}_raw.$t.$dataset0.$yrv
    #wc -w $ref_raw
    for metric in bleu; do
      echo phrase: $st $metric $dataset0 $method
      echo hyp: $onebest_file
      echo ref: $ref_raw
      get_$metric $ref_raw $onebest_file
    done
    echo ----------------
  done
fi












