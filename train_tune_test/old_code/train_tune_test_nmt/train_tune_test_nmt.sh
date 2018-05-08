#!/bin/bash

### main script
BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $BASE_DIR/../../config.sh $1 $2
source $BASE_DIR/../../utils.sh

s=$1
t=$2

### build dict
serial_id=model0
export PYTHONPATH=$opennmt_dir:$PYTHONPATH
if [ -f ${nmt_model_dir}/${serial_id}-train.t7 ]; then
	echo "preprocessed data exists at: ${nmt_model_dir}/${serial_id}-train.t7"
else
  train_src=${trans_dir}/train/${src_label}_raw.${s}.train.${yrv}
  train_tgt=${trans_dir}/train/${ref_label}_raw.${t}.train.${yrv}
  dev_src=${trans_dir}/dev/${src_label}_raw.${s}.dev.${yrv}
  dev_tgt=${trans_dir}/dev/${ref_label}_raw.${t}.dev.${yrv}
	th preprocess.lua \
		-train_src $train_src \
		-train_tgt $train_tgt \
		-valid_src $dev_src \
		-valid_tgt $dev_tgt \
		-save_data ${nmt_model_dir}/${serial_id}
fi

### train
model_prefix=basic_model
export PYTHONPATH=$opennmt_dir:$PYTHONPATH
if ls ${nmt_model_dir}/${model_prefix}* 1> /dev/null 2>&1; then
	echo "trained model exists at ${nmt_model_dir}/basic_model"
else
	th train.lua \
		-data ${nmt_model_dir}/${serial_id}-train.t7 \
		-save_model ${nmt_model_dir}/${model_prefix} \
		-gpuid 1
fi


