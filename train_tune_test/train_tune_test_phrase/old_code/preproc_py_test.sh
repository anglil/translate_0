source /home/ec2-user/kklab/Projects/lrlp/scripts/config.sh
source /home/ec2-user/kklab/Projects/lrlp/scripts/utils.sh

### extract data
for dataset0 in train dev test; do  
  prefix=${corpus_dir}/elisa.${s}-${t}.${dataset0}.${y}${r}.${v}
  if [ -f ${prefix}.${t} ] && [ $redo -eq 1 ]; then
    echo "$dataset0 parse exists."
  else
    python $PWD/preproc.py $prefix $s $t
  fi  
  echo "$dataset0 parsed."
done
echo "----------------"
