#!/bin/bash

### main script
BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $BASE_DIR/../../config.sh $1 $2
source $BASE_DIR/../../utils.sh

s=$1
t=$2

### test
model_prefix=basic_model
model_name=${nmt_model_dir}/$(python $BASE_DIR/get_nmt_model_name.py ${nmt_model_dir} ${model_prefix})
cd $opennmt_dir
for dataset0 in dev test; do
  source_file=${trans_dir}/$dataset0/${src_label}_raw.${s}.$dataset0.${yrv}
  onebest_file=${trans_dir}/$dataset0/onebest_opennmt.${t}.$dataset0.${yrv}
  
  if [ -f $onebest_file ]; then
  	echo "translation for $dataset0 exists at $onebest_file"
  else
    th translate.lua \
    	-model $model_name \
    	-src $source_file \
    	-output $onebest_file \
  		-gpuid 1
  fi
done
