#!/bin/bash

usage() { echo -e "-s: source language\n-t: target language\n-g: gpu_id\n-d: dim: 300,128,256,512,1024\n-n: num_layer: 1,2,3\n-k: task: prep,train,test\n-u: use_lex: store_true,store_false\n-l: lr: 0.2,0.1,0.02,0.5\n-o: dropout: 0.1,0.0,0.2,0.8\n-a: attn: store_false,allregular,all1d,all2d,simple(lex_cap=1),beforeaggergate,afteraggregate,before1daggregate,after1daggregate,before2daggregate,after2daggregate,all1daggregate,all2daggregate\n-e: embedding untrainable: store_true,store_false\n-r: embedding random: store_true,store_false\n-c: lexicon clustering: store_true,store_false\n-i: use_align: store_true,store_false\n-b: share_vocab (between src & tgt): store_true,store_false\n-w: use_subword_tokenizer: store_true,store_false\n-v: vocab_size: num\n-m: num_heads: num\n-p: with_padding: store_true,store_false"; }

s=''
t=''
gpu_id=0
dim=300
num_layer=2
task=train
use_lex=false
lr=0.2
dropout=0.1
attn=false
emb_untrainable=false
emb_random=false
lex_cluster=false
use_align=false

share_vocab=false
use_subword_tokenizer=false
vocab_size=8000

with_padding=false
num_heads=4 # an exception of model_params

while getopts ":hs:t:g:d:n:uk:l:o:a:ercibwv:m:p" flag; do
  case "${flag}" in
    s) s="${OPTARG}" ;;
    t) t="${OPTARG}" ;;
    g) gpu_id="${OPTARG}" ;;
    d) dim="${OPTARG}" ;;
    n) num_layer="${OPTARG}" ;;
    u) use_lex=true ;;
    k) task="${OPTARG}" ;;
    l) lr="${OPTARG}" ;;
    o) dropout="${OPTARG}" ;;
    a) attn="${OPTARG}" ;;
    e) emb_untrainable=true ;;
    r) emb_random=true ;;
    c) lex_cluster=true ;;
    i) use_align=true ;;
    b) share_vocab=true ;;
    w) use_subword_tokenizer=true ;;
    v) vocab_size="${OPTARG}" ;;
    p) with_padding=true ;;
    m) num_heads="${OPTARG}" ;;
    h) usage; exit ;;
    *) error "Unexpected option ${flag}" ;;
  esac
done
#shift "$((OPTIND-1))"

if [ -z "$s" ]; then
  echo "no arg supplemented for s"; exit 1
fi
if [ -z "$t" ]; then
  echo "no arg supplemented for t"; exit 1
fi

echo "s = $s"
echo "t = $t"
echo "gpu_id = $gpu_id"
echo "dim = $dim"
echo "num_layer = $num_layer"
echo "task = $task"
echo "lr = $lr"
echo "dropout = $dropout"
echo "attn = $attn"
echo "use_lex = $use_lex"
echo "emb_untrainable = $emb_untrainable"
echo "emb_random = $emb_random"
echo "lex_cluster = $lex_cluster"
echo "use_align = $use_align"

echo "share_vocab = $share_vocab"
echo "use_subword_tokenizer = $use_subword_tokenizer"
echo "vocab_size = $vocab_size"

echo "num_heads = $num_heads"
echo "with_padding = $with_padding"


BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $BASE_DIR/../../config.sh $s $t
source $BASE_DIR/../../utils.sh

# ----------------

# whether to do bpe, vocab size, whether to share vocab
get_merge_ops() {
  if [ "$use_subword_tokenizer" = false ]; then
    echo "inf"
  else
    if [ "$share_vocab" = true ]; then
      echo $vocab_size
    else
      echo ${vocab_size}-${vocab_size}
    fi
  fi
}
merge_ops=$( get_merge_ops )

#merge_ops2=vocab-${use_subword_tokenizer}-${share_vocab}-${src_vocab_size}-${tgt_vocab_size}

# training data has external lexical resources appended to it
if [ "$use_lex" = true ]; then
  use_lex=_lex
else
  use_lex=""
fi

# embedding trainable or fixed
if [ "$emb_untrainable" = true ]; then
  emb_untrainable=_embuntrainable
else
  emb_untrainable=""
fi

# embedding initialized randomly or with pretrained vectors
if [ "$emb_random" = true ]; then
  emb_random=_embrandom
else
  emb_random=""
fi

# embedding candidates clustered by synonyms (averaging) or not
if [ "$lex_cluster" = true ]; then
  lex_cluster=_lexcluster
else
  lex_cluster=""
fi

# embedding initialized by aligning source to target or with an external lexicon
if [ "$use_align" = true ]; then
  use_align=_usealign
else
  use_align=""
fi

# whether to use my own attention layers
if [ "$attn" = false ]; then
  attn=""
fi

# whether to pad (not to pad sentences to a certain length) to speed up convolution computation
if [ "$with_padding" = true ]; then
  with_padding=_withpadding
else
  with_padding=""
fi

# ----------------

t2t_data_dir=$trans_dir/t2t_${merge_ops}$use_lex
mkdir -p $t2t_data_dir
method=t2t${attn}_dim${dim}_layer${num_layer}_lr${lr}_dropout${dropout}_bpe${merge_ops}${use_lex}${emb_untrainable}${emb_random}${lex_cluster}${use_align}${with_padding}
t2t_model_dir=$exp_dir/$method
mkdir -p $t2t_model_dir

# set model
if [ -z "$attn" ]; then
  t2t_model_type=transformer
else
  t2t_model_type=transformer_lex
  if [ "$attn" == "allregular" ] || [ "$attn" == "all1d" ] || [ "$attn" == "all2d" ] || [ "$attn" == "allregular2d" ]; then
    t2t_model_type=transformer_lex2
  fi
fi
# set hparams, which include modalities
hparams=transformer_all
# set problem
problem_id=translate_srctgt_lrlp

# write hyperparameters to a yaml file for registering modules in transformer
train_src=$trans_dir/train/${src_label}_raw${use_lex}.$s.train.$yrv
dev_src=$trans_dir/dev/${src_label}_raw.$s.dev.$yrv
test_src=$trans_dir/test/${src_label}_raw.$s.test.$yrv
train_tgt=$trans_dir/train/${ref_label}_raw${use_lex}.$t.train.$yrv
dev_tgt=$trans_dir/dev/${ref_label}_raw.$t.dev.$yrv
test_tgt=$trans_dir/test/${ref_label}_raw.$t.test.$yrv

echo ----
echo "train_src: $train_src"
echo "dev_src: $dev_src"
echo "test_src: $test_src"
echo "train_tgt: $train_tgt"
echo "dev_tgt: $dev_tgt"
echo "test_tgt: $test_tgt"
echo ----
#exit
mkdir -p $t2t_model_dir/reg_config
mkdir -p $t2t_model_dir/decoded

python hyper_param_mod.py \
  --config_file $t2t_model_dir/reg_config/config.yml \
  --s $s \
  --t $t \
  --st $st \
  --train_src $train_src \
  --dev_src $dev_src \
  --train_tgt $train_tgt \
  --dev_tgt $dev_tgt \
  --glove_mat $glove_mat \
  --bilingual_lexicon $bilingual_lexicon \
  --synonym_api $thesaurus_api \
  --synonym_api2 $nltk_data_dir \
  --use_subword_tokenizer $use_subword_tokenizer \
  --share_vocab $share_vocab \
  --vocab_size $vocab_size \
  --model_params $method \
  --num_heads $num_heads

cp $BASE_DIR/__init__.py $t2t_model_dir/reg_config/__init__.py
cp $BASE_DIR/reg_problems.py $t2t_model_dir/reg_config/reg_problems.py
cp $BASE_DIR/reg_models.py $t2t_model_dir/reg_config/reg_models.py
cp $BASE_DIR/reg_modalities.py $t2t_model_dir/reg_config/reg_modalities.py
cp $BASE_DIR/reg_hparams.py $t2t_model_dir/reg_config/reg_hparams.py
#cp $BASE_DIR/replace_oov_with_unk.py $t2t_model_dir/reg_config/replace_oov_with_unk.py

# ----------------

if [ "$task" == "prep" ]; then
  t2t-datagen \
    --data_dir=$t2t_data_dir \
    --tmp_dir= \
    --problem=$problem_id \
    --t2t_usr_dir=$t2t_model_dir/reg_config \
    --registry_help
elif [ "$task" == "train" ]; then
  CUDA_VISIBLE_DEVICES=$gpu_id t2t-trainer \
    --data_dir=$t2t_data_dir \
    --problems=$problem_id \
    --model=$t2t_model_type \
    --hparams_set=$hparams \
    --output_dir=$t2t_model_dir \
    --train_steps=500000 \
    --t2t_usr_dir=$t2t_model_dir/reg_config
elif [ "$task" == "test" ]; then
  beam_size=4
  alpha=0.6
  for dataset0 in test; do
    source_file=$trans_dir/$dataset0/${src_label}_raw.$s.$dataset0.$yrv
    source_file_bpe=$t2t_model_dir/decoded/${src_label}_raw.$s.$dataset0.$yrv
    ## oov happens only when bpe is not used?
    #if [ "$merge_ops" == "inf" ]; then
    #  python $t2t_model_dir/reg_config/replace_oov_with_unk.py \
    #    $source_file \
    #    $source_file_bpe \
    #    $t2t_data_dir/vocab.$s$t.8000.$s
    #else
    cp $source_file $source_file_bpe
    #fi

    echo --------

    onebest_file_untok=$source_file_bpe.$t2t_model_type.$hparams.$problem_id.beam$beam_size.alpha$alpha.decodes
    onebest_file=$trans_dir/$dataset0/onebest_${method}.$t.$dataset0.$yrv
    #if [ ! -f $onebest_file_untok ]; then
    CUDA_VISIBLE_DEVICES=$gpu_id t2t-decoder \
      --data_dir=$t2t_data_dir \
      --problems=$problem_id \
      --model=$t2t_model_type \
      --hparams_set=$hparams \
      --output_dir=$t2t_model_dir \
      --decode_hparams="beam_size=$beam_size,alpha=$alpha" \
      --decode_from_file=$source_file_bpe \
      --t2t_usr_dir=$t2t_model_dir/reg_config
    echo onebest_file_untok created at $onebest_file_untok
    #else
    #  echo onebest_file_untok exists at $onebest_file_untok
    #fi

    #tokenize2 $t $onebest_file_untok $onebest_file
    #python remove_space.py $onebest_file_untok $onebest_file
    #cp $onebest_file_untok $onebest_file

    head -n -1 $onebest_file_untok > $onebest_file

    echo --------

    # bleu sent_bleu meteor
    for metric in bleu; do
      ref_raw=$trans_dir/$dataset0/${ref_label}_raw.$t.$dataset0.$yrv
      echo t2t: $s $metric $dataset0 $method
      echo hyp: $onebest_file
      echo ref: $ref_raw
      get_$metric $ref_raw $onebest_file
    done
    echo "tensorboard --port 6006 --logdir=$t2t_model_dir"
    echo ----------------
  done
else
  echo unsupported task: $task
fi

