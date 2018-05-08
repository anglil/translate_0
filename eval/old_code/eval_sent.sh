#!/bin/bash

BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $BASE_DIR/../config.sh $1 $2
source $BASE_DIR/../utils.sh
s=$1
t=$2

bleuscore=/home/ec2-user/kklab/src/mosesdecoder/bin/sentence-bleu
mydir=/home/ec2-user/kklab/Projects/lrlp/experiment_2017.07.12.il3-eng.y1r1.v2/translation/dev
hyp=$mydir/onebest_t2t_dim512_layer2_bpe8000.eng.dev.y1r1.v2
ref=$mydir/ref_raw.eng.dev.y1r1.v2
$bleuscore $mydir/ref_raw.eng.dev.y1r1.v2 < $mydir/onebest_t2t_dim512_layer2_bpe8000.eng.dev.y1r1.v2
