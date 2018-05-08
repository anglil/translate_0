import java.io.*;
import java.util.*;
import java.util.regex.Pattern;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

//// context size exception
//class context_size_exception extends Exception {
//  public context_size_exception(String msg) {
//    super(msg);
//  }
//}

public class write_candidate_list_from_eng_vocab_sequential {
  // segment evenly
  // problem: recompute pmi of context words inside and across batches
  public static void run_1(String onebest_file, String candidate_list_file_tmp, String function_words_file, String punctuations_file, String index_path, String context_scale, String window_mechanism, int num_candidates, String candidate_list_file) {
    int sent_num = utils.get_file_length(onebest_file);
    System.out.println("Num of sentences: "+String.valueOf(sent_num));
    int sent_num_threshold = 800;
    if (sent_num > sent_num_threshold) {
      // segment into batches
      List<String> onebest_file_batches = utils.break_file_into_batches(onebest_file, sent_num_threshold);
      List<String> candidate_list_file_tmp_batches = utils.break_file_into_batches(candidate_list_file_tmp, sent_num_threshold);
      assert onebest_file_batches.size() == candidate_list_file_tmp_batches.size();
      
      // work on each batch
      List<String> candidate_list_file_batches = new ArrayList<String>();
      for (int i=0;i<onebest_file_batches.size();i++) {
        String candidate_list_file_batch = candidate_list_file+"."+String.valueOf(i);
        write_candidate_list_from_eng_vocab.run(onebest_file_batches.get(i), candidate_list_file_tmp_batches.get(i), function_words_file, punctuations_file, index_path, context_scale, window_mechanism, num_candidates, candidate_list_file_batch);
        candidate_list_file_batches.add(candidate_list_file_batch);
      }
      assert onebest_file_batches.size() == candidate_list_file_batches.size();

      // concatenate all batches
      utils.merge_files(candidate_list_file_batches, candidate_list_file);
    } else {
      write_candidate_list_from_eng_vocab.run(onebest_file, candidate_list_file_tmp, function_words_file, punctuations_file, index_path, context_scale, window_mechanism, num_candidates, candidate_list_file);
    }
  }

  // segment by number of context words
  // problem: recompute pmi of context words across batches
  public static void run_2(String onebest_file, String candidate_list_file_tmp, String function_words_file, String punctuations_file, String index_path, String context_scale, String window_mechanism, int num_candidates, String candidate_list_file) {
    int context_words_set_size_threshold = 2000;//1800;//4;
    try {
      // segment into batches
      Map<String, List<String>> batches = utils.break_file_into_batches_by_context_words(onebest_file, candidate_list_file_tmp, function_words_file, punctuations_file, context_words_set_size_threshold);
      List<String> onebest_file_batches = batches.get("onebest_file_batches");
      List<String> candidate_list_file_tmp_batches = batches.get("candidate_list_file_tmp_batches");
      assert onebest_file_batches.size() == candidate_list_file_tmp_batches.size();
      System.out.println("onebest and candidate_list have been segmented into "+String.valueOf(onebest_file_batches.size()+" pieces"));

      //String ooo = "/home/ec2-user/kklab/Projects/lrlp/experiment_2017.08.08.il5-eng.y2r1.v1/translation/eval/onebest.eng.eval.y2r1.v1.";
      //String ccc = "/home/ec2-user/kklab/Projects/lrlp/experiment_2017.08.08.il5-eng.y2r1.v1/translation/eval/oov/eng_vocab.eng.eval.y2r1.v1.tmp.";
      //List<String> onebest_file_batches = new ArrayList<String>();
      //List<String> candidate_list_file_tmp_batches = new ArrayList<String>();
      //for (int i=0;i<19;i++) {
      //  onebest_file_batches.add(ooo+String.valueOf(i));
      //  candidate_list_file_tmp_batches.add(ccc+String.valueOf(i));
      //}
      //String hhh = "/home/ec2-user/kklab/Projects/lrlp/experiment_2017.08.08.il5-eng.y2r1.v1/translation/eval/oov/eng_vocab.eng.eval.y2r1.v1.";
      //List<String> candidate_list_file_batches = new ArrayList<String>();
      //for (int i=0;i<10;i++) {
      //  candidate_list_file_batches.add(hhh+String.valueOf(i));
      //}

      // work on each batch
      List<String> candidate_list_file_batches = new ArrayList<String>();
      for (int i=0;i<onebest_file_batches.size();i++) {
        System.out.println("processing batch "+String.valueOf(i)+"...");
        String candidate_list_file_batch = candidate_list_file+"."+String.valueOf(i);
        write_candidate_list_from_eng_vocab.run(onebest_file_batches.get(i), candidate_list_file_tmp_batches.get(i), function_words_file, punctuations_file, index_path, context_scale, window_mechanism, num_candidates, candidate_list_file_batch);
        candidate_list_file_batches.add(candidate_list_file_batch);
        System.out.println("batch "+String.valueOf(i)+" processed.");
        System.out.println("--------");
      }
      assert onebest_file_batches.size() == candidate_list_file_batches.size();

      // concatenate all batches
      utils.merge_files(candidate_list_file_batches, candidate_list_file);
    } catch (context_size_exception e) {
      e.printStackTrace();
    }
  }
 
  // segment by number of context words, with a global pmi_db
  // problem: may cause out-of-memory error because of too big of pmi_db
  public static void run_3(String onebest_file, String candidate_list_file_tmp, String function_words_file, String punctuations_file, String index_path, String context_scale, String window_mechanism, int num_candidates, String candidate_list_file){// throws context_size_exception {
    int context_words_set_size_threshold = 1200;//3;//1800;
    
    Set<String> function_words = write_candidate_list_from_eng_vocab.get_function_words(function_words_file);
    Set<String> punctuations = write_candidate_list_from_eng_vocab.get_function_words(punctuations_file);
    List<String> candidate_list = new ArrayList<String>();
    List<List<String>> context_words_cache = new ArrayList<List<String>>(); // batch-wise
    List<Set<Integer>> oov_pos_cache = new ArrayList<Set<Integer>>(); // batch-wise
    Set<String> context_words_set = new HashSet<String>(); // batch-wise
    Set<String> context_words_set_all = new HashSet<String>(); // global
    get_pmi my_pmi = new get_pmi(index_path, context_scale, window_mechanism);
    Map<String, Map<String, Double>> pmi_db = new HashMap<String, Map<String, Double>>();// global

    try {
      BufferedReader fo = new BufferedReader(new FileReader(onebest_file));
      BufferedReader fc = new BufferedReader(new FileReader(candidate_list_file_tmp));
      BufferedWriter fw = Files.newBufferedWriter(Paths.get(candidate_list_file));
      
      int batch_ctr = 0;
      String l_onebest, l_candidate;
      while (((l_onebest = fo.readLine())!=null) && ((l_candidate = fc.readLine())!=null)) {
        // only process sentences with oov words
        if ((l_candidate.equals("")==false) && (l_candidate.equals("=")==false)) {
          Map<Integer, List<String>> candidate_map = write_candidate_list_from_eng_vocab.parse_candidate_map(l_candidate);
          if (candidate_list.isEmpty()) candidate_list = write_candidate_list_from_eng_vocab.get_random_candidate_list(candidate_map);
          List<String> context_words = write_candidate_list_from_eng_vocab.parse_context_words(l_onebest, candidate_map, function_words, punctuations);
          
          if (!context_words.isEmpty()) {
            // obtain context words for each line
            int num_of_context_words_in_line = 0;
            for (String context_word:context_words) {
              if (!context_words_set_all.contains(context_word))
                num_of_context_words_in_line += 1;
            }

            // hit batch threshold
            if (context_words_set.size()+num_of_context_words_in_line > context_words_set_size_threshold) {
              /*if (num_of_context_words_in_line > context_words_set_size_threshold)
                throw new context_size_exception("context_words_set_size_threshold must be greater than the minimum sentence length");*/
              
              Map<String, Map<String, Double>> pmi_db_batch = my_pmi.get_pmi_db(candidate_list, context_words_set);
              // merge context_words_set to the main context_words_set_all
              for (String context_word:context_words_set) {
                assert !context_words_set_all.contains(context_word);
                context_words_set_all.add(context_word);
              }
              // merge pmi_db_batch to the main pmi_db
              for (String candidate_word:pmi_db_batch.keySet()) {
                for (String context_word:pmi_db_batch.get(candidate_word).keySet()) {
                  assert !pmi_db.get(candidate_word).containsKey(context_word);
                  if (pmi_db.containsKey(candidate_word))
                    pmi_db.get(candidate_word).put(context_word, pmi_db_batch.get(candidate_word).get(context_word));
                  else {
                    pmi_db.put(candidate_word, new HashMap<String, Double>());
                    pmi_db.get(candidate_word).put(context_word, pmi_db_batch.get(candidate_word).get(context_word));
                  }
                }
              }
              // query for high-pmi candidate words for each sentence in the batch
              int cache_ptr = 0;
              for (List<String> context_words_list:context_words_cache) {
                if ((context_words_list.size()==1) && ((context_words_list.get(0).equals("=")) || (context_words_list.get(0).equals("")))) {
                  fw.write(context_words_list.get(0)+"\n");
                } else {
                  String[] selected_candidates;
                  if (context_words_list.isEmpty()) {
                    selected_candidates = new String[1];
                    selected_candidates[0] = candidate_list.get(0);
                  } else {
                    selected_candidates = my_pmi.query_pmi_db(pmi_db, context_words_list, num_candidates);
                  }
                  int oov_pos_num = 0;
                  for (int oov_pos:oov_pos_cache.get(cache_ptr)) {
                    if (oov_pos_num != 0) fw.write(" ");
                    fw.write(String.valueOf(oov_pos)+":"+String.join(",",selected_candidates));
                    oov_pos_num += 1;
                  }
                  fw.write("\n");
                }
                cache_ptr += 1;
              }
              System.out.println("batch "+String.valueOf(batch_ctr)+" finished.");
              System.out.println("--------");
              batch_ctr += 1;
  
              // clear batch-wise cache
              context_words_set.clear();
              context_words_cache.clear();
              oov_pos_cache.clear();

              // add the first line of a batch to cache
              for (String context_word:context_words) {
                if (!context_words_set_all.contains(context_word))
                  context_words_set.add(context_word);
              }
              context_words_cache.add(context_words);
              oov_pos_cache.add(candidate_map.keySet());
            } else { // not hit batch threshold
              // add non first line of a batch to cache
              for (String context_word:context_words) {
                if (!context_words_set_all.contains(context_word))
                  context_words_set.add(context_word);
              }
              context_words_cache.add(context_words);
              oov_pos_cache.add(candidate_map.keySet());
            }
          } else {
            context_words_cache.add(new ArrayList<String>());
            oov_pos_cache.add(candidate_map.keySet());
          }
        } else {
          context_words_cache.add(Arrays.asList(l_candidate));
          oov_pos_cache.add(new HashSet<Integer>());
        }
      } // reach the end of file

      Map<String, Map<String, Double>> pmi_db_batch = my_pmi.get_pmi_db(candidate_list, context_words_set);
      // merge pmi_db_batch to the main pmi_db
      for (String candidate_word:pmi_db_batch.keySet()) {
        for (String context_word:pmi_db_batch.get(candidate_word).keySet()) {
          assert !pmi_db.get(candidate_word).containsKey(context_word);
          if (pmi_db.containsKey(candidate_word))
            pmi_db.get(candidate_word).put(context_word, pmi_db_batch.get(candidate_word).get(context_word));
          else {
            pmi_db.put(candidate_word, new HashMap<String, Double>());
            pmi_db.get(candidate_word).put(context_word, pmi_db_batch.get(candidate_word).get(context_word));
          }
        }
      }
      // query for high-pmi candidate words for each sentence in the batch
      int cache_ptr = 0;
      for (List<String> context_words_list:context_words_cache) {
        if ((context_words_list.size()==1) && ((context_words_list.get(0).equals("=")) || (context_words_list.get(0).equals("")))) {
          fw.write(context_words_list.get(0)+"\n");
        } else {
          String[] selected_candidates;
          if (context_words_list.isEmpty()) {
            selected_candidates = new String[1];
            selected_candidates[0] = candidate_list.get(0);
          } else {
            selected_candidates = my_pmi.query_pmi_db(pmi_db, context_words_list, num_candidates);
          }
          int oov_pos_num = 0;
          for (int oov_pos:oov_pos_cache.get(cache_ptr)) {
            if (oov_pos_num != 0) fw.write(" ");
            fw.write(String.valueOf(oov_pos)+":"+String.join(",",selected_candidates));
            oov_pos_num += 1;
          }
          fw.write("\n");
        }
        cache_ptr += 1;
      }
      System.out.println("batch "+String.valueOf(batch_ctr)+" finished.");
      fo.close();fc.close();fw.close();
    } catch (IOException e) {
      e.printStackTrace();
    }
  }

  public static void main(String[] args) {
    // input
    String onebest_file = args[0];
    String candidate_list_file_tmp = args[1];

    String function_words_file = args[2];
    String punctuations_file = args[3];
    String index_path = args[4];
    String context_scale = args[5];
    String window_mechanism = args[6];
    int num_candidates = Integer.valueOf(args[7]); // 20 is what I use

    // output
    String candidate_list_file = args[8];

    write_candidate_list_from_eng_vocab_sequential.run_2(onebest_file, candidate_list_file_tmp, function_words_file, punctuations_file, index_path, context_scale, window_mechanism, num_candidates, candidate_list_file);
  }
}
