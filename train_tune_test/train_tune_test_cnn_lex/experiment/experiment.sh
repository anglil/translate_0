#!/bin/bash

gpu_id=0
train_src=/home/ec2-user/kklab/Projects/lrlp/experiment_2017.08.04.vie-eng.y1r1.v2/translation/train/src_raw.vie.train.y1r1.v2
train_tgt=/home/ec2-user/kklab/Projects/lrlp/experiment_2017.08.04.vie-eng.y1r1.v2/translation/train/ref_raw.eng.train.y1r1.v2
vocab_src=/home/ec2-user/kklab/Projects/lrlp/experiment_2017.08.04.vie-eng.y1r1.v2/translation/bpe/vocab.bpe.inf.vie
vocab_tgt=/home/ec2-user/kklab/Projects/lrlp/experiment_2017.08.04.vie-eng.y1r1.v2/translation/bpe/vocab.bpe.inf.eng
dev_src=/home/ec2-user/kklab/Projects/lrlp/experiment_2017.08.04.vie-eng.y1r1.v2/translation/dev/src_raw.vie.dev.y1r1.v2
dev_tgt=/home/ec2-user/kklab/Projects/lrlp/experiment_2017.08.04.vie-eng.y1r1.v2/translation/dev/ref_raw.eng.dev.y1r1.v2

model_dir=/home/ec2-user/kklab/Projects/lrlp/scripts/train_tune_test/train_tune_test_lex/experiment/model_dir
config_yml=/home/ec2-user/kklab/Projects/lrlp/scripts/train_tune_test/train_tune_test_lex/experiment/config.yml
train_yml=/home/ec2-user/kklab/src/conv_seq2seq/example_configs/train_seq2seq.yml

glove_mat=/home/ec2-user/kklab/data/glove/glove.6B.300d.txt
bilingual_lexicon=/home/ec2-user/kklab/data/lorelei/LEXICONS/clean-merged/clean-merged/vie-eng.masterlex.txt

task=train

# train
export PYTHONPATH=$cnn_dir:$PYTHONPATH
if [ "$task" == "train" ]; then
CUDA_VISIBLE_DEVICES=$gpu_id python -m bin.train \
  --config_paths="
    ${config_yml},
    ${train_yml}" \
  --model_params "
    glove_dict_file: $glove_mat
    lexicon_dict_file: $bilingual_lexicon
    vocab_source: $vocab_src
    vocab_target: $vocab_tgt" \
  --input_pipeline_train "
    class: ParallelTextInputPipelineFairseq
    params:
      source_files:
        - $train_src
      target_files:
        - $train_tgt" \
  --input_pipeline_dev "
    class: ParallelTextInputPipelineFairseq
    params:
      source_files:
        - $dev_src
      target_files:
        - $dev_tgt" \
  --batch_size 8 \
  --eval_every_n_steps 5000 \
  --train_steps 100000 \
  --output_dir $model_dir
# test
elif [ "$task" == "test" ]; then
  CUDA_VISIBLE_DEVICES=$gpu_id python -m bin.infer \
    --tasks "
      - class: DecodeText
      - class: DumpBeams
        params:
          file: $model_dir/beams.npz" \
    --model_dir $model_dir \
    --model_params "
      glove_dict_file: $glove_mat
      lexicon_dict_file: $bilingual_lexicon
      inference.beam_search.beam_width: 5
      decoder.class: seq2seq.decoders.LexpoolDecoderFairseqBS" \
    --input_pipeline "
      class: ParallelTextInputPipelineFairseq
      params:
        source_files:
          - $dev_src" \
    > tmpfile

  /home/ec2-user/kklab/src/mosesdecoder/scripts/generic/multi-bleu.perl -lc $dev_tgt < tmpfile
fi

