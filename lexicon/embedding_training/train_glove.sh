#!/bin/bash

# input
train_corpus=

# param
verbose=2
memory=10.0
vocab_min_count=3
vector_size=300

# output
vocab_file=example/vocab.txt
cooccurrence_file=example/cooccurrence.bin
cooccurrence_shuf_file=example/cooccurrece.shuf.bin
glove_vectors=example/vectors
