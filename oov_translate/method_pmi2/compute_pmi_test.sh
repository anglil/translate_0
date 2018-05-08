onebest_file=examples/onebest_file.tmp
#candidate_list_file_tmp=examples/candidate_list_file_tmp.tmp
candidate_list_file_tmp=examples/candidate_list_file_tmp.tmp2
#candidate_list_file=examples/candidate_list_file_res
candidate_list_file=examples/candidate_list_file_res2
function_words_file=examples/function_words_file
punctuations_file=examples/punctuations_file
context_scale=bp
window_mechanism=boolean_window
home_dir=/home/ec2-user/kklab
data_dir=$home_dir/data
index_dir=$data_dir/wiki
index_path=$index_dir/wikipedia_$context_scale
pmi_mat_dir=pmi_dir/pmi_mat_dir/
context_words_record_file=pmi_dir/context_words_record_file
#candidate_words_record_file=pmi_dir/candidate_words_record_file
candidate_words_record_file=pmi_dir/candidate_words_record_file2
context_words_set_size_threshold=2
#num_candidates_threshold=2
num_candidates_threshold=20
javac -cp .:/home/ec2-user/kklab/src/Palmetto/palmetto/target/palmetto-0.1.0-jar-with-dependencies.jar:/home/ec2-user/kklab/hppc/hppc/target/hppc-0.8.0-SNAPSHOT.jar *.java && java -cp .:/home/ec2-user/kklab/src/Palmetto/palmetto/target/palmetto-0.1.0-jar-with-dependencies.jar:/home/ec2-user/kklab/hppc/hppc/target/hppc-0.8.0-SNAPSHOT.jar compute_pmi \
  collect_pmi \
  $onebest_file \
  $candidate_list_file_tmp \
  $function_words_file \
  $punctuations_file \
  $pmi_mat_dir \
  $context_words_record_file \
  $candidate_words_record_file \
  $index_path \
  $context_scale \
  $window_mechanism \
  $context_words_set_size_threshold

java -cp .:/home/ec2-user/kklab/src/Palmetto/palmetto/target/palmetto-0.1.0-jar-with-dependencies.jar:/home/ec2-user/kklab/hppc/hppc/target/hppc-0.8.0-SNAPSHOT.jar compute_pmi \
  apply_pmi \
  $onebest_file \
  $candidate_list_file_tmp \
  $function_words_file \
  $punctuations_file \
  $pmi_mat_dir \
  $context_words_record_file \
  $candidate_words_record_file \
  $candidate_list_file \
  $num_candidates_threshold

