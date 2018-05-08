#!/bin/bash

### main script
BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $BASE_DIR/../../config.sh $1 $2
source $BASE_DIR/../../utils.sh

s=$1
t=$2
echo "### lang: $s"

# input: corpus_dir
# output: trans_dir
for dataset0 in train dev test domain eval; do
  xml_file=$corpus_dir/$dataset_name.${st}.$dataset0.${yrv}.xml
  
  reference_file=${trans_dir}/$dataset0/${ref_label}.${t}.$dataset0.${yrv}
  onebest_file=${trans_dir}/$dataset0/onebest.${t}.$dataset0.${yrv}
  nbest_file=${trans_dir}/$dataset0/n_best/nbest.${t}.$dataset0.${yrv}
  oov_file=${trans_dir}/$dataset0/oov/oov.${t}.$dataset0.${yrv}
  #oov_unique_file=${trans_dir}/$dataset0/oov/oov_unique.${t}.$dataset0.${yrv}

  if [ -f $xml_file ]; then
    echo parsing ${xml_file} ...
    if [ "$dataset0" == "train" ]; then
      python $BASE_DIR/../../preproc/extract_tokenized_from_xml.py \
        --xml $xml_file \
        --source_lang $s \
        --reference $reference_file
    elif [ "$dataset0" == "eval" ]; then
      python $BASE_DIR/../../preproc/extract_tokenized_from_xml.py \
        --xml $xml_file \
        --source_lang $s \
        --onebest $onebest_file \
        --nbest $nbest_file \
        --oov $oov_file
        #--oov_unique $oov_unique_file
    else 
      python $BASE_DIR/../../preproc/extract_tokenized_from_xml.py \
        --xml $xml_file \
        --source_lang $s \
        --reference $reference_file \
        --onebest $onebest_file \
        --nbest $nbest_file \
        --oov $oov_file
        #--oov_unique $oov_unique_file
    fi
  fi

  echo --------
done

