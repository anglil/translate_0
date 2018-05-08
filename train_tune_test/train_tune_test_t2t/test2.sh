BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $BASE_DIR/../../config.sh $1 $2
source $BASE_DIR/../../utils.sh

s=$1
t=$2

gpu_id=0


##for attn in before1d before2d before after simple ""; do
#for attn in beforesimple aftersimple before1d after1d before2d after2d beforesimpletanh aftersimpletanh ""; do
#  # 128 256 512 1024
#  for dim in 300; do
#    # 1 2 3
#    for layer in 2; do
#      # 8000 32000 inf 8000-8000
#      for bpe in inf; do
#        # 0.02 0.1 0.5
#        for lr in 0.2; do
#          # 0.0 0.2 0.8
#          for dropout in 0.1; do
#            for use_lex in _lex ""; do
#              for emb in _embuntrainable ""; do
#                t2t_model_dir=$exp_dir/t2t${attn}_dim${dim}_layer${layer}_lr${lr}_dropout${dropout}_bpe${bpe}${use_lex}${emb}
#                if [ -d "$t2t_model_dir" ]; then
#                  echo tensorboard --port 6006 --logdir=$t2t_model_dir
#                fi
#              done
#            done
#          done
#        done
#      done
#    done
#  done
#done

train_src=$trans_dir/train/${src_label}_raw${use_lex}.$s.train.$yrv
dev_src=$trans_dir/dev/${src_label}_raw.$s.dev.$yrv
test_src=$trans_dir/test/${src_label}_raw.$s.test.$yrv
train_tgt=$trans_dir/train/${ref_label}_raw${use_lex}.$t.train.$yrv
dev_tgt=$trans_dir/dev/${ref_label}_raw.$t.dev.$yrv
test_tgt=$trans_dir/test/${ref_label}_raw.$t.test.$yrv
echo -e "train_src:\n$train_src"
echo ----
echo -e "dev_src:\n$dev_src"
echo ----
echo -e "test_src:\n$test_src"
echo ----
echo -e "train_tgt:\n$train_tgt"
echo ----
echo -e "dev_tgt:\n$dev_tgt"
echo ----
echo -e "test_tgt:\n$test_tgt"
echo ----
attn=""
dim=300
num_layer=2
lr=0.2
dropout=0.1
use_lex=""
emb_untrainable=""
for merge_ops in inf 8000-8000 8000; do
  method=t2t${attn}_dim${dim}_layer${num_layer}_lr${lr}_dropout${dropout}_bpe${merge_ops}${use_lex}${emb_untrainable}
  dev_onebest=$trans_dir/dev/onebest_${method}.$t.dev.$yrv
  test_onebest=$trans_dir/test/onebest_${method}.$t.test.$yrv
  echo -e "dev_onebest_${merge_ops}:\n$dev_onebest"
  echo ----
  echo -e "test_onebest_${merge_ops}:\n$test_onebest"
  echo ----
done
method=phrase
dev_onebest=$trans_dir/dev/onebest_${method}.$t.dev.$yrv
test_onebest=$trans_dir/test/onebest_${method}.$t.test.$yrv
echo -e "dev_onebest_pbmt:\n$dev_onebest"
echo ----
echo -e "test_onebest_pbmt:\n$test_onebest"

