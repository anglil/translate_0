#!/bin/bash

BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $BASE_DIR/../config.sh $1 $2
source $BASE_DIR/../utils.sh
s=$1
t=$2
ref=$3
hyp=$4

metric=bleu
get_$metric $ref $hyp

