import java.io.*;
import java.util.*;
import java.util.regex.Pattern;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

public class compute_pmi {

// break_file_into_batches_by_global_context_words
  public static void collect_pmi(
      String onebest_file, 
      String candidate_list_file_tmp, 
      String function_words_file, 
      String punctuations_file, 
      String pmi_mat_dir,
      String context_words_record_file, 
      String candidate_words_record_file, 
      int context_words_set_size_threshold, 
      get_pmi my_pmi){
    // context words recorded
    context_words_record_obj context_words_obj = utils.get_context_words_recorded(context_words_record_file);
    Set<String> context_words_recorded = context_words_obj.context_words_recorded;
    int max_file_name = context_words_obj.max_file_name;
    String pmi_mat_file = pmi_mat_dir+String.valueOf(max_file_name+1);
    // candidate words indexed
    List<String> candidate_words_record = utils.get_candidate_words_ordered(candidate_words_record_file, function_words_file);
    // context words to record
    Set<String> context_words_record_created = new HashSet<String>();
    // function words
    Set<String> function_words = write_candidate_list_from_eng_vocab.get_function_words(function_words_file);
    // punctuations
    Set<String> punctuations = write_candidate_list_from_eng_vocab.get_function_words(punctuations_file);
    System.out.println("--------");

    try {
      BufferedReader fo = new BufferedReader(new FileReader(onebest_file));
      BufferedReader fc = new BufferedReader(new FileReader(candidate_list_file_tmp));

      String l_onebest, l_candidate;
      int line_ctr = 0;
      while (((l_onebest = fo.readLine())!=null) && ((l_candidate = fc.readLine())!=null)) {
        if ((l_candidate.equals("")==false) && (l_candidate.equals("=")==false)) {
          // get candidate words
          Map<Integer, List<String>> candidate_map = write_candidate_list_from_eng_vocab.parse_candidate_map(l_candidate);
          // get context words
          List<String> context_words = write_candidate_list_from_eng_vocab.parse_context_words(l_onebest, candidate_map, function_words, punctuations);

          if (!context_words.isEmpty()) {
            // find unrecorded context words
            Set<String> context_words_cur_sent = new HashSet<String>();
            for (String context_word:context_words) {
              if ((!context_words_recorded.contains(context_word)) && (!context_words_record_created.contains(context_word))) {
                context_words_cur_sent.add(context_word);
              }
            }

            if (context_words_record_created.size()+context_words_cur_sent.size()>context_words_set_size_threshold) {
              // the middle batches
              List<String> context_words_list = new ArrayList<String>();
              int context_word_ctr = 0;
              for (String context_word:context_words_record_created) {
                context_words_list.add(context_word);
                context_word_ctr += 1;
              }
              double[][] pmi_mat = my_pmi.get_mat(candidate_words_record, context_words_list);
              System.out.println(String.valueOf(context_words_record_created.size())+" words recorded.");
              System.out.println(String.valueOf(line_ctr)+" sentences processed.");
              System.out.println("--------");
              context_words_record_created.clear();
              context_words_obj = utils.get_context_words_recorded(context_words_record_file);
              context_words_recorded = context_words_obj.context_words_recorded;
              max_file_name = context_words_obj.max_file_name;
              pmi_mat_file = pmi_mat_dir+String.valueOf(max_file_name+1);
              utils.update_context_words_record(context_words_list, max_file_name+1, context_words_record_file);
              get_pmi.write_mat(pmi_mat, pmi_mat_file);

              // collect unrecorded context words
              for (String context_word:context_words_cur_sent)
                context_words_record_created.add(context_word);
            } else {
              // collect unrecorded context words
              for (String context_word:context_words_cur_sent)
                context_words_record_created.add(context_word);
            }
          }
        }
        line_ctr += 1;
      }
      
      if (!context_words_record_created.isEmpty()) {
        // the last batch
        List<String> context_words_list = new ArrayList<String>();
        int context_word_ctr = 0;
        for (String context_word:context_words_record_created) {
          context_words_list.add(context_word);
          context_word_ctr += 1;
        }
        double[][] pmi_mat = my_pmi.get_mat(candidate_words_record, context_words_list);
        System.out.println(String.valueOf(context_words_record_created.size())+" words recorded.");
        System.out.println(String.valueOf(line_ctr)+" sentences processed.");
        System.out.println("--------");
        context_words_record_created.clear();
        context_words_obj = utils.get_context_words_recorded(context_words_record_file);
        context_words_recorded = context_words_obj.context_words_recorded;
        max_file_name = context_words_obj.max_file_name;
        pmi_mat_file = pmi_mat_dir+String.valueOf(max_file_name+1);
        utils.update_context_words_record(context_words_list, max_file_name+1, context_words_record_file);
        get_pmi.write_mat(pmi_mat, pmi_mat_file);
      }

    } catch (IOException e) {
      e.printStackTrace();
    }
    return;
  }
  
  public static void apply_pmi(
      String onebest_file, 
      String candidate_list_file_tmp, 
      String function_words_file, 
      String punctuations_file, 
      String pmi_mat_dir, 
      String context_words_record_file,
      String candidate_words_record_file, 
      String candidate_list_file, 
      int num_candidates_threshold) {
    // candidate words indexed
    List<String> candidate_words = utils.get_candidate_words_ordered(candidate_words_record_file, function_words_file);

    assert candidate_words.size() >= num_candidates_threshold;
    Set<String> function_words = write_candidate_list_from_eng_vocab.get_function_words(function_words_file);
    Set<String> punctuations = write_candidate_list_from_eng_vocab.get_function_words(punctuations_file);
    Map<String, Map<String, Integer>> context_words_recorded = utils.get_context_words_recorded2(context_words_record_file);
    System.out.println("--------");
    
    try {
      BufferedReader fo = new BufferedReader(new FileReader(onebest_file));
      BufferedReader fc = new BufferedReader(new FileReader(candidate_list_file_tmp));
      BufferedWriter fw = Files.newBufferedWriter(Paths.get(candidate_list_file));

      String l_onebest, l_candidate;
      int l_ctr = 0;
      while (((l_onebest = fo.readLine())!=null) && ((l_candidate = fc.readLine())!=null)) {
        if ((l_candidate.equals("")==false) && (l_candidate.equals("=")==false)) {
          Map<Integer, List<String>> candidate_map = write_candidate_list_from_eng_vocab.parse_candidate_map(l_candidate);
          List<String> context_words = write_candidate_list_from_eng_vocab.parse_context_words(l_onebest, candidate_map, function_words, punctuations);

          String[] selected_candidates;
          if (!context_words.isEmpty()) {
            selected_candidates = get_pmi.query_mat(
                context_words_recorded, 
                candidate_words, 
                context_words,
                pmi_mat_dir, 
                num_candidates_threshold);
          } else {
            selected_candidates = new String[1];
            selected_candidates[0] = (write_candidate_list_from_eng_vocab.get_random_candidate_list(candidate_map)).get(0);
          }

          // write selected candidates to file
          int oov_pos_num = 0;
          for (int oov_pos:candidate_map.keySet()) {
            if (oov_pos_num != 0) fw.write(" ");
            fw.write(String.valueOf(oov_pos)+":"+String.join(",",selected_candidates));
            oov_pos_num += 1;
          }
          fw.write("\n");
        } else if (l_candidate.equals("=")) {
          fw.write("=\n");
        } else {
          fw.write("\n");
        }
        l_ctr += 1;
        if (l_ctr%500==0)
          System.out.println(String.valueOf(l_ctr)+" sentences processed.");
      }
      fw.close();
    } catch (IOException e) {
      e.printStackTrace();
    }
  }

  public static void main(String[] args) {
    String task = args[0];
    System.out.println("--------");
    System.out.println("task: "+task);
    String onebest_file = args[1];
    String candidate_list_file_tmp = args[2];
    String function_words_file = args[3];
    String punctuations_file = args[4];
    String pmi_mat_dir = args[5];//"pmi_dir/pmi_mat_dir/";
    String context_words_record_file = args[6];//"pmi_dir/context_words_record_file";
    String candidate_words_record_file = args[7];//"pmi_dir/candidate_words_record_file";
   
    if (task.equals("collect_pmi")) {
      String index_path = args[8];
      String context_scale = args[9];
      String window_mechanism = args[10];
      int context_words_set_size_threshold = Integer.valueOf(args[11]);//2;

      get_pmi my_pmi = new get_pmi(
          index_path, 
          context_scale, 
          window_mechanism);
      collect_pmi(
          onebest_file, 
          candidate_list_file_tmp, 
          function_words_file, 
          punctuations_file, 
          pmi_mat_dir,
          context_words_record_file, 
          candidate_words_record_file, 
          context_words_set_size_threshold,
          my_pmi);
    } else if (task.equals("apply_pmi")) {
      String candidate_list_file = args[8];
      int num_candidates_threshold = Integer.valueOf(args[9]);

      apply_pmi(
          onebest_file, 
          candidate_list_file_tmp, 
          function_words_file, 
          punctuations_file, 
          pmi_mat_dir, 
          context_words_record_file,
          candidate_words_record_file, 
          candidate_list_file, 
          num_candidates_threshold);

    } else {
      System.out.println("unsupported task: "+task);
    }
  }
}
