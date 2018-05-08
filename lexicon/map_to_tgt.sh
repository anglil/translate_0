#infile=/home/ec2-user/kklab/Projects/lrlp/experiment_2017.08.04.vie-eng.y1r1.v2/translation/dev/src_raw.vie.dev.y1r1.v2
#outfile=map_vie
#lexicon=/home/ec2-user/kklab/data/lorelei/LEXICONS/clean-merged/clean-merged/vie-eng.masterlex.txt

infile=/home/ec2-user/kklab/Projects/lrlp/experiment_2017.07.12.il3-eng.y1r1.v2/translation/dev/src_raw.il3.dev.y1r1.v2
outfile=map_uig
lexicon=/home/ec2-user/kklab/data/lorelei/LEXICONS/clean-merged/clean-merged/uig-eng.masterlex.txt

python map_to_tgt.py $infile $outfile $lexicon
