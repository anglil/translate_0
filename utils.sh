#!/bin/bash

### wait till the end of execution of a program
function wait_for_existence {
  echo "Generating $1..."
  while [ ! -f $1 ]
  do
    sleep 10
  done
  echo "$1 generated!"
}

### tokenizer
function tokenize {
  # $1 prefix of input
  # $2 prefix of output
  echo "Tokenizing..."
  $tokenizer \
    -l $s \
    < $1.$s \
    > $2.$s
  echo "$1.$s tokenized to $2.$s"
  $tokenizer \
    -l $t \
    < $1.$t \
    > $2.$t
  echo "$1.$t tokenized to $2.$t"
}

function tokenize2 {
  # $1 lang
  # $2 input
  # $3 output
  if [ -f $3 ]; then
    echo tokenized text exists at $3
  else
    $tokenizer \
      -threads 8 \
      -l $1 \
      < $2 \
      > $3
    echo tokenized text created at $3
  fi
}

### train a recasing model with training data
function truecase_train {
  # $1 prefix of input
  # $2 prefix of output (recasing model)
  echo "Building a recasing model..."
  $truecaser_train \
    --corpus $1.$t \
    --model $2.$t
  echo "Truecasing model $2.$t is built based on $1.$t"
  $truecaser_train \
    --corpus $1.$s \
    --model $2.$s
  echo "Truecasing model $2.$s is built based on $1.$s"
}

function truecase_train2 {
  # $1 input
  # $2 output
  if [ -f $2 ]; then
    echo truecase model exists at $2
  else
    $truecaser_train \
      --corpus $1 \
      --model $2
    echo truecase model created at $2
  fi
}

### use the recasing model to recase
function truecase_infer {
  # $1 model prefix
  # $2 prefix of input
  # $3 prefix of output
  echo "Recasing..."
  $truecaser_infer \
    --model $1.$t \
    < $2.$t \
    > $3.$t
  echo "$2.$t recased to $3.$t under model $1.$t"
  $truecaser_infer \
    --model $1.$s \
    < $2.$s \
    > $3.$s
  echo "$2.$s recased to $3.$s under model $1.$s"
}

function truecase_infer2 {
  # $1 model
  # $2 input
  # $3 output
  if [ -f $3 ]; then
    echo truecased text exists at $3
  else
    $truecaser_infer \
      --model $1 \
      < $2 \
      > $3
      echo truecased text created at $3
  fi
}

### cleaner: processes both sides (source and target) at once
function cleaner {
  # $1 prefix of input
  # $2 prefix of output
  # $3 max length
  echo "Cleaning..."
  $cleaner \
    $1 $s $t \
    $2 1 $3
  echo "$1.$t trimmed to $2.$t with max_len=$3"
  echo "$1.$s trimmed to $2.$s with max_len=$3"
}

function ngram_lm_train {
  # $1 ngram
  # $2 input
  # $3 output
  if [ -f $3 ]; then
    echo $1 gram language model exists at $3
  else
    $lm_builder \
      -o $1 \
      < $2 \
      > $3
    echo $1 gram language model created at $3
  fi
}

function ngram_lm_binarize {
  # $1 input
  # $2 output
  if [ -f $2 ]; then
    echo binarized language model exists at $2
  else
    $lm_bin_builder \
      $1 \
      $2
    echo binarized language model created at $2
  fi
}

### get bleu score
function get_bleu {
  # $1 reference
  # $2 translation hypothesis
  $bleu_getter \
    -lc $1 \
    < $2
}

### get sentence level bleu
function get_sent_bleu {
  # $1 reference
  # $2 translation hypothesis
  $sent_bleu_getter \
    $1 \
    < $2 \
    > jk && ls jk | xargs -I {} paste $2 {} \
    > ${2}.debug
  echo MT hypothesis with sentece-level BLEU created at: ${2}.debug
}

### get meteor score
function get_meteor {
  # $1 reference
  # $2 translation hypothesis
  java -Xmx2G \
    -jar $meteor_getter \
    $2 \
    $1 \
    -norm \
    -noPunct \
    | grep 'Final'
}


