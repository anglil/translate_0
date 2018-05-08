BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $BASE_DIR/../../config.sh $1 $2
source $BASE_DIR/../../utils.sh

s=$1
t=$2

gpu_id=7


for dim in 128 256 512 1024; do
  for layer in 2; do
    for kernel in 2 3 4; do
      for injective in 0 1 2; do
        lex_model_dir=$exp_dir/lex${injective}_dim${dim}_layer${layer}_kernel${kernel}
        if [ -d "$lex_model_dir" ]; then
          echo tensorboard --port 6006 --logdir=$lex_model_dir
          echo sh run.sh $s $t $gpu_id $dim $layer $kernel $injective test
          echo --------
        fi
      done
    done
  done
done
