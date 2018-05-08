#!/bin/bash

url_handler=$1

### extract data
for dataset0 in train dev test; do 
	prefix=${corpus_dir}/${exp_handle}.${s}-${t}.${dataset0}.${y}${r}.${v}
	if [ -f ${prefix}.${t} ] && [ $redo -eq 0 ]; then
		echo "$dataset0 parse exists."
	else
		python $PWD/extract_orig_from_xml.py $prefix $s $t $url_handler
	fi
	echo "$dataset0 parsed."
done
echo "----------------"


### preprocessing train and dev data
preproc=1

### preprocessing 1: train
if [ $preproc -eq 1 ]
then
  ### tokenizating both source and target training sets
  if [ -f ${train_data_dir}/${train_id}.${m_tok}.$s ] && [ -f ${train_data_dir}/${train_id}.${m_tok}.$t ] && [ $redo -eq 0 ]
  then
    echo "Tokenized training set exists."
  else
    tokenize ${corpus_dir}/${train_id} ${train_data_dir}/${train_id}.${m_tok}
  fi


  ### learning a truecasing model for both source and target training data
  if [ -f ${model_dir}/truecase-model.$t ] && [ -f ${model_dir}/truecase-model.$s ] && [ $redo -eq 0 ]
  then
    echo "Truecaser models exist."
  else
    recase_train ${train_data_dir}/${train_id}.${m_tok} ${model_dir}/truecase-model
  fi


  ### truecasing both source and target training sets
  if [ -f ${train_data_dir}/${train_id}.${m_rec}.$t ] && [ -f ${train_data_dir}/${train_id}.${m_rec}.$s ] && [ $redo -eq 0 ]
  then
    echo "Truecased training set exists."
  else
    recase_infer ${model_dir}/truecase-model ${train_data_dir}/${train_id}.${m_tok} ${train_data_dir}/${train_id}.${m_rec}
  fi
  
	### cleaning both source and target training sets
  max_len=80
  if [ -f ${train_data_dir}/${train_id}.${m_cle}.$t ] && [ -f ${train_data_dir}/${train_id}.${m_cle}.$s ] && [ $redo -eq 0 ]
  then
    echo "Cleaned training set exists."
  else
    cleaner ${train_data_dir}/${train_id}.${m_rec} ${train_data_dir}/${train_id}.${m_cle} $max_len
  fi
else
  if [ -f ${train_data_dir}/${train_id}.${m_cle}.$t ] && [ -f ${train_data_dir}/${train_id}.${m_cle}.$s ] && [ $redo -eq 0 ]
  then
    echo "Cleaned training set exists."
  else
    cp ${corpus_dir}/${train_id}.$s ${train_data_dir}/${train_id}.${m_cle}.$s
    cp ${corpus_dir}/${train_id}.$t ${train_data_dir}/${train_id}.${m_cle}.$t
  fi
fi
echo "----------------"

### preprocessing 2: tune
if [ $preproc -eq 1 ]
then
  ### tokenizing both source and target dev sets
  if [ -f ${dev_data_dir}/${dev_id}.${m_tok}.$t ] && [ -f ${dev_data_dir}/${dev_id}.${m_tok}.$s ] && [ $redo -eq 0 ]
  then
    echo "Tokenized dev set exists."
  else
    tokenize ${corpus_dir}/${dev_id} ${dev_data_dir}/${dev_id}.${m_tok}
  fi

  ### truecasing both source and target dev sets
  if [ -f ${dev_data_dir}/${dev_id}.${m_rec}.$t ] && [ -f ${dev_data_dir}/${dev_id}.${m_rec}.$s ] && [ $redo -eq 0 ]
  then
    echo "Truecased dev set exists."
  else
    recase_infer ${model_dir}/truecase-model ${dev_data_dir}/${dev_id}.${m_tok} ${dev_data_dir}/${dev_id}.${m_rec}
  fi
else
  if [ -f ${dev_data_dir}/${dev_id}.${m_rec}.$t ] && [ -f ${dev_data_dir}/${dev_id}.${m_rec}.$s ] && [ $redo -eq 0 ]
  then
    echo "Truecased dev set exists."
  else
    cp ${corpus_dir}/${dev_id}.$s ${dev_data_dir}/${dev_id}.${m_rec}.$s
    cp ${corpus_dir}/${dev_id}.$t ${dev_data_dir}/${dev_id}.${m_rec}.$t
  fi
fi
echo "----------------"

### preprocessing 3: test
if [ $preproc -eq 1 ] 
then
  ### tokenizing both source and target test sets
  if [ -f ${test_data_dir}/${test_id}.${m_tok}.$t ] && [ -f ${test_data_dir}/${test_id}.${m_tok}.$s ] && [ $redo -eq 0 ]
  then
    echo "Tokenized test set exists."
  else
    tokenize ${corpus_dir}/${test_id} ${test_data_dir}/${test_id}.${m_tok}
  fi  
  
  
  ### truecasing both source and target test sets
  if [ -f ${test_data_dir}/${test_id}.${m_rec}.$t ] && [ -f ${test_data_dir}/${test_id}.${m_rec}.$s ] && [ $redo -eq 0 ]
  then
    echo "Truecased test set exists."
  else
    recase_infer ${model_dir}/truecase-model ${test_data_dir}/${test_id}.${m_tok} ${test_data_dir}/${test_id}.${m_rec}
  fi  
  
else
  if [ -f ${test_data_dir}/${test_id}.${m_rec}.$t ] && [ -f ${test_data_dir}/${test_id}.${m_rec}.$s ] && [ $redo -eq 0 ]
  then
    echo "Truecased test set exists."
  else
    cp ${corpus_dir}/${test_id}.$s ${test_data_dir}/${test_id}.${m_rec}.$s
    cp ${corpus_dir}/${test_id}.$t ${test_data_dir}/${test_id}.${m_rec}.$t
  fi  
fi
echo "----------------"

