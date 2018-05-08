#!/bin/bash

#set -e

# il3, amh, som, yor, rus, en, de
s=$1
# eng, de, en
t=$2

if [ -z $1 ] || [ -z $2 ]; then
  echo source and target language cannot be empty!
  return
fi

# -------- preproc: fixed variables --------

home_dir=/home/ec2-user/kklab

st=$s-$t
st2=deen
src_label=src
ref_label=ref
if [ "$s" == "en" ]; then
  st=$t-$s
  st2=$t$s
  src_label=ref
  ref_label=src
else
  st=$s-$t
  st2=$s$t
  src_label=src
  ref_label=ref
fi

if [ "$s" == "il3" ]; then
  y=y1
  r=r1
  v=v2
  yrv=$y$r.$v
  exp_date=2017.07.12
  corpus_dir=$home_dir/data/ELISA/dryruns/$exp_date/${s}/mt
  dataset_name=isi-sbmt-tl
  dataset_dir=$home_dir/Projects/lrlp/data_$dataset_name
  exp_name=experiment_${exp_date}.$st.${yrv}
elif [ "$s" == "vie" ]; then
  y=y1
  r=r1
  v=v2
  yrv=$y$r.$v
  corpus_dir=$home_dir/data/lorelei/Translation/LDC2016E103_LORELEI_Vietnamese_Representative_Language_Pack_Translation_Annotation_Grammar_Lexicon_and_Tools_V1.0/data/translation/from_vie
  dataset_name=ldc
  dataset_dir=$home_dir/Projects/lrlp/data_${dataset_name}_$s
  exp_name=experiment_2017.08.04.$st.$yrv
elif [ "$s" == "en" ] || [ "$s" == "de" ]; then
  y=y2
  r=r2
  v=v1
  yrv=$y$r.$v
  exp_date=2017.07.21
  corpus_dir=$home_dir/data/t2t_datagen
  dataset_name=wmt
  dataset_dir=$home_dir/Projects/lrlp/data_$dataset_name
  exp_name=experiment_${exp_date}.$st.${yrv}
elif [ "$s" == "il5" ]; then
  y=y2
  r=r1
  #v=v1
  #v=v3
  v=v6
  yrv=$y$r.$v
  #corpus_dir=$home_dir/data/ELISA/evals/y2/mt/$s
  #corpus_dir=$home_dir/data/ELISA/evals/y2/mt/${s}_$v
  corpus_dir=$home_dir/data/ELISA/evals/y2/mt/${s}_ckpt3
  #dataset_name=elisa
  #dataset_name=isi-sbmt-amh-edoov
  dataset_name=isi-sbmt-v6-amh-tgdict8
  dataset_dir=$home_dir/Projects/lrlp/data_${dataset_name}_${s}_$yrv
  #exp_name=experiment_2017.08.08.$st.$yrv
  exp_name=experiment_$dataset_name.$st.$yrv
elif [ "$s" == "il6" ]; then
  y=y2
  r=r1
  #v=v1
  #v=v4
  v=v6
  yrv=$y$r.$v
  #corpus_dir=$home_dir/data/ELISA/evals/y2/mt/$s
  #corpus_dir=$home_dir/data/ELISA/evals/y2/mt/${s}_$v
  corpus_dir=$home_dir/data/ELISA/evals/y2/mt/${s}_ckpt3
  #dataset_name=elisa
  #dataset_name=isi-sbmt
  dataset_name=isi-sbmt-v6-som2eng-tgdict7-copyme
  dataset_dir=$home_dir/Projects/lrlp/data_${dataset_name}_${s}_$yrv
  #exp_name=experiment_2017.08.08.$st.$yrv
  exp_name=experiment_$dataset_name.$st.$yrv
elif [ "$s" == "amh" ] || [ "$s" == "som" ] || [ "$s" == "yor" ]; then
  y=y2
  r=r1
  v=v1
  yrv=$y$r.$v
  exp_date=2017.04.07
  corpus_dir=$home_dir/data/ELISA/dryruns/$exp_date/${s}/JMIST/elisa.$s.package.$yrv
  dataset_name=elisa
  dataset_dir=$home_dir/Projects/lrlp/data_${dataset_name}_$s
  exp_name=experiment_2017.08.04.$st.$yrv
elif [ "$s" == "ben" ] || [ "$s" == "hau" ]; then
  y=y1
  r=r1
  v=v4
  yrv=$y$r.$v
  exp_date=2017.04.07
  corpus_dir=$home_dir/data/ELISA/dryruns/$exp_date/${s}/JMIST/elisa.$s.package.$yrv
  dataset_name=elisa
  dataset_dir=$home_dir/Projects/lrlp/data_${dataset_name}_$s
  exp_name=experiment_2017.08.04.$st.$yrv
else
  echo unsupported language pairs: $s $t
fi

mkdir -p $dataset_dir

exp_dir=$home_dir/Projects/lrlp/${exp_name}
mkdir -p $exp_dir

log_dir=$exp_dir/log
mkdir -p $log_dir

export home_dir
export s
export t
export st
export st2
export src_label
export ref_label
export yrv
export corpus_dir
export dataset_name
export exp_dir

# -------- train (model files) --------

### isi model: NA

### phrase-based model
model_dir=$exp_dir/model
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

model_trained=$train_model_dir/model/moses.ini
model_tuned=$dev_model_dir/moses.ini
model_tuned_bin=$dev_model_dir/moses.bin.ini

### opennmt seq2seq model
nmt_model_dir=$exp_dir/nmt
mkdir -p $nmt_model_dir

### tf-nmt seq2seq model
tfnmt_model_dir=$exp_dir/tfnmt
mkdir -p $tfnmt_model_dir

### attention model
t2t_model_dir=$exp_dir/t2t
mkdir -p $t2t_model_dir

### cnn model
cnn_model_dir=$exp_dir/cnn
mkdir -p $cnn_model_dir

# -------- decode (result files) --------

trans_dir=$exp_dir/translation
mkdir -p $trans_dir
export trans_dir

### train
mkdir -p ${trans_dir}/train

### hypothesis (1-best translation with oov)
dev_1best_dir=${trans_dir}/dev
mkdir -p $dev_1best_dir
test_1best_dir=${trans_dir}/test
mkdir -p $test_1best_dir
unseq_1best_dir=${trans_dir}/unseq
mkdir -p $unseq_1best_dir
syscomb_1best_dir=${trans_dir}/syscomb
mkdir -p $syscomb_1best_dir
domain_1best_dir=${trans_dir}/domain
mkdir -p $domain_1best_dir
eval_1best_dir=${trans_dir}/eval
mkdir -p $eval_1best_dir

### hypothesis (n-best translation with oov)
dev_nbest_dir=${dev_1best_dir}/n_best
mkdir -p $dev_nbest_dir
test_nbest_dir=${test_1best_dir}/n_best
mkdir -p $test_nbest_dir
unseq_nbest_dir=${unseq_1best_dir}/n_best
mkdir -p $unseq_nbest_dir
syscomb_nbest_dir=${syscomb_1best_dir}/n_best
mkdir -p $syscomb_nbest_dir
domain_nbest_dir=${domain_1best_dir}/n_best
mkdir -p $domain_nbest_dir
eval_nbest_dir=${eval_1best_dir}/n_best
mkdir -p $eval_nbest_dir

### oov words (format: {pos: pos:})
dev_oov_dir=${dev_1best_dir}/oov
mkdir -p $dev_oov_dir
test_oov_dir=${test_1best_dir}/oov
mkdir -p $test_oov_dir
unseq_oov_dir=${unseq_1best_dir}/oov
mkdir -p $unseq_oov_dir
syscomb_oov_dir=${syscomb_1best_dir}/oov
mkdir -p $syscomb_oov_dir
domain_oov_dir=${domain_1best_dir}/oov
mkdir -p $domain_oov_dir
eval_oov_dir=${eval_1best_dir}/oov
mkdir -p $eval_oov_dir

# -------- external software and dataset directories --------

### basic directories
moses_dir=$home_dir/src/mosesdecoder
moses_tool_dir=$moses_dir/tools # contains language model binaries
moses_bin_dir=$moses_dir/bin
moses_script_dir=$moses_dir/scripts
opennmt_dir=$home_dir/src/OpenNMT # seq2seq with openNMT
subword_dir=$home_dir/src/subword-nmt # subword unit vocab generation and subword-ize texts
tfnmt_dir=$home_dir/src/nmt # seq2seq with tfNMT
cnn_dir=$home_dir/src/conv_seq2seq # cnn_mt with seq2seq
t2t_dir=$home_dir/src/anaconda3/lib/python3.6/site-packages/tensor2tensor # t2t_mt with tensor2tensor

### train, tune, test
trainer=$moses_script_dir/training/train-model.perl
tuner=$moses_script_dir/training/mert-moses.pl
decoder=$moses_bin_dir/moses

### language model, phrase table, reordering table
lm_builder=$moses_bin_dir/lmplz
lm_bin_builder=$moses_bin_dir/build_binary
phrase_binarizer=$moses_bin_dir/processPhraseTableMin
lexical_binarizer=$moses_bin_dir/processLexicalTableMin

### data processing tools
tokenizer=$moses_script_dir/tokenizer/tokenizer.perl
truecaser_train=$moses_script_dir/recaser/train-truecaser.perl
truecaser_infer=$moses_script_dir/recaser/truecase.perl
cleaner=$moses_script_dir/training/clean-corpus-n.perl
test_filter=$moses_script_dir/training/filter-model-given-input.pl
bleu_getter=$moses_script_dir/generic/multi-bleu.perl
meteor_getter=$moses_dir/meteor-1.5/meteor-1.5.jar
sgm_converter=$moses_script_dir/ems/support/input-from-sgm.perl
vocab_gen=$cnn_dir/bin/tools/generate_vocab.py
bpe_train=$subword_dir/learn_bpe.py
bpe_infer=$subword_dir/apply_bpe.py
bpe_vocab=$subword_dir/get_vocab.py

### bilingual lexicon, definitions in chapter 5: https://www2.ee.washington.edu/techsite/papers/documents/UWEETR-2016-0001.pdf
bilingual_lexicon=$home_dir/data/lorelei/LEXICONS/clean-merged/clean-merged/$st.masterlex.txt
if [ "$s" == "il3" ]; then
  bilingual_lexicon=$home_dir/data/lorelei/LEXICONS/clean-merged/clean-merged/uig-eng.masterlex.txt
fi
export bilingual_lexicon
export bleu_getter

### some auxiliary software
kenlm_dir=$home_dir/src/kenlm/build/bin # for ngram lm
#lm_builder=$kenlm_dir/lmplz # for ngram lm
#query_perplexity=$kenlm_dir/query # for ngram lm
#build_binary=$kenlm_dir/build_binary # for ngram lm
hypergraph_dec=$home_dir/src/lazy/bin/decode # for ngram lm
fast_align=$moses_dir/fast_align/build/fast_align # for bound
sent_bleu_getter=$moses_bin_dir/sentence-bleu # for bound
meteor_bin=$moses_dir/meteor-1.5/meteor-1.5.jar # for bound
palmetto_jar=$home_dir/src/Palmetto/palmetto/target/palmetto-0.1.0-jar-with-dependencies.jar # for pmi
commons_lang_jar=$home_dir/src/commons-lang3-3.5/commons-lang3-3.5.jar # for pmi
hppc_jar=$home_dir/src/hppc/hppc/target/hppc-0.8.0-SNAPSHOT.jar # for pmi
mvn_dir=$home_dir/src/maven/apache-maven-3.5.0/bin # for pmi
wcluster=$home_dir/src/brown-cluster/wcluster # for neural lm
boost_dir=$home_dir/src/boost_1_61_0/lib # for dclm
dynet_dir=$home_dir/src/dynet/build/dynet # for dclm
dynet_python_dir=$home_dir/src/dynet/build/python # for dclm
glove_train_dir=$home_dir/src/GloVe/build # for lex

### some data
data_non_domain_dir=$home_dir/data/1-billion-word-language-modeling-benchmark/scripts/training-monolingual.tokenized # for ngram lm
train_non_domain_all=$data_non_domain_dir/news.all.en # for ngram lm

glove_dir=$home_dir/data/glove #pagerank, nmt_lex
glove_mat=$glove_dir/glove.6B.300d.txt 
word2vec_dir=$home_dir/data/word2vec # nmt_lex
word2vec_mat=$word2vec_dir/GoogleNews-vectors-negative300.bin

index_dir=$home_dir/data/wiki # for pmi
wiki_dump_dir=$home_dir/data/wiki_dump/wikitext-103 # for dclm (https://metamind.io/research/the-wikitext-long-term-dependency-language-modeling-dataset/)
wiki_dump_train=$wiki_dump_dir/wiki.train.tokens # for dclm
wiki_dump_dev=$wiki_dump_dir/wiki.valid.tokens # for dclm
wiki_dump_test=$wiki_dump_dir/wiki.test.tokens # for dclm

nltk_data_dir=$home_dir/data/nltk_data
thesaurus_api=$home_dir/src/thesaurus-api
