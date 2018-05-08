BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $BASE_DIR/../../config.sh $1 $2
source $BASE_DIR/../../utils.sh

s=$1
t=$2

gpu_id=0


for dim in 128 256 512 1024; do
  for layer in 2; do
    for kernel in 2 3 4; do
      for bpe in 8000 32000 inf 8000-8000; do
        cnn_model_dir=$exp_dir/cnn_dim${dim}_layer${layer}_kernel${kernel}_bpe${bpe}_lex
        if [ -d "$cnn_model_dir" ]; then
          #sh $BASE_DIR/run.sh $s $t $gpu_id $dim $layer $bpe test _lex
          echo tensorboard --port 6006 --logdir=$cnn_model_dir
        fi
        #cnn_model_dir=$exp_dir/cnn_dim${dim}_layer${layer}_kernel${kernel}_bpe${bpe}
        #if [ -d "$cnn_model_dir" ]; then
        #  #sh $BASE_DIR/run.sh $s $t $gpu_id $dim $layer $bpe test
        #  echo tensorboard --port 6006 --logdir=$cnn_model_dir
        #fi
      done
    done
  done
done
