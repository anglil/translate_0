import java.io.*;
import java.util.*;
import java.util.regex.Pattern;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

// context size exception
class context_size_exception extends Exception {
  public context_size_exception(String msg) {
    super(msg);
  }
}

class context_words_record_obj {
  public Set<String> context_words_recorded;
  public int max_file_name;
  public context_words_record_obj(Set<String> context_words_recorded, int max_file_name) {
    this.context_words_recorded = context_words_recorded;
    this.max_file_name = max_file_name;
  }
}

// class utils 
public class utils {
  // sh
  public static String sh(String cmd) {
    String stdout = "";
    try {
      Process p = Runtime.getRuntime().exec(cmd);
      BufferedReader read = new BufferedReader(new InputStreamReader(p.getInputStream()));
      String input_line;
      while ((input_line = read.readLine()) != null) {
        System.out.println(input_line);
        stdout += input_line+"\n";
      }
      read.close();
    } catch (IOException e) {
      e.printStackTrace();
    }
    return stdout;
  }

  // get_file_length
  public static int get_file_length(String file_name) {
    String r = sh("wc -l "+file_name);
    String[] lc = r.split(" ");
    return Integer.valueOf(lc[0]);
  }

  // merge_files
  public static void merge_files(List<String> in_files, String out_file) {
    try {
      BufferedWriter fw = Files.newBufferedWriter(Paths.get(out_file));
      for (int i=0;i<in_files.size();i++) {
        BufferedReader fi = new BufferedReader(new FileReader(in_files.get(i)));
        String l_in;
        while ((l_in=fi.readLine())!=null) {
          fw.write(l_in+"\n");
        }
        fi.close();
      }
      fw.close();
    } catch (IOException e) {
      e.printStackTrace();
    }
  }

  // get_context_words_recorded2
  public static Map<String, Map<String, Integer>> get_context_words_recorded2(String file_name) {
    Map<String, Map<String, Integer>> context_words_record = new HashMap<String, Map<String, Integer>>();
    File ff = new File(file_name);
    if (ff.exists() && !ff.isDirectory()) {
      try {
        BufferedReader fc = new BufferedReader(new FileReader(file_name));
        String line;
        while ((line = fc.readLine())!=null) {
          String[] l = line.split(" ");
          String context_word = l[0];
          assert !context_words_record.containsKey(context_word);
          int file_idx = Integer.valueOf(l[1]);
          int pos_in_file = Integer.valueOf(l[2]);
          context_words_record.put(context_word, new HashMap<String, Integer>());
          context_words_record.get(context_word).put("file_num", file_idx);
          context_words_record.get(context_word).put("pos_in_file", pos_in_file);
        }
        fc.close();
      } catch (IOException e) {
        e.printStackTrace();
      }
    }
    return context_words_record;
  }

  // get_context_words_recorded
  public static context_words_record_obj get_context_words_recorded(String file_name) {
    Set<String> context_words_recorded = new HashSet<String>();
    int max_file_idx = -1;

    File ff = new File(file_name);
    if (ff.exists() && !ff.isDirectory()) {
      try {
        BufferedReader fc = new BufferedReader(new FileReader(file_name));
        String line;
        while ((line = fc.readLine())!=null) {
          String[] l = line.split(" ");
          String context_word = l[0];
          int file_idx = Integer.valueOf(l[1]);
          if (file_idx>max_file_idx) max_file_idx = file_idx;
          assert !context_words_recorded.contains(context_word);
          context_words_recorded.add(context_word);
        }
        fc.close();
      } catch (IOException e) {
        e.printStackTrace();
      }
    }
    return new context_words_record_obj(context_words_recorded, max_file_idx);
  }

  // get_candidate_words_ordered
  public static List<String> get_candidate_words_ordered(String candidate_file, String function_words_file) {
    Set<String> function_words = write_candidate_list_from_eng_vocab.get_function_words(function_words_file);
    List<String> candidate_words = new ArrayList<String>();
    try {
      BufferedReader f = new BufferedReader(new FileReader(candidate_file));
      int ctr = 0;
      String l;
      while ((l=f.readLine())!=null) {
        if (!function_words.contains(l)) {
          candidate_words.add(l);
          ctr += 1;
        }
      }
      System.out.println(String.valueOf(ctr)+" candidate words read.");
    } catch(IOException e) {
      e.printStackTrace();
    }
    return candidate_words;
  }

  // update_context_words_record
  public static void update_context_words_record(List<String> context_words_record_created, int pmi_mat_file_idx, String context_words_record_file) {
    if (context_words_record_created.size() != 0) {
      // collect recorded context words
      Map<String, Map<String, Integer>> context_words_record = get_context_words_recorded2(context_words_record_file);
 
      // add new context words
      try {
        BufferedWriter fw = Files.newBufferedWriter(Paths.get(context_words_record_file));
        for (String context_word:context_words_record.keySet()) {
          fw.write(context_word+" "+String.valueOf(context_words_record.get(context_word).get("file_num"))+" "+String.valueOf(context_words_record.get(context_word).get("pos_in_file"))+"\n");
        }
        int ctr = 0;
        for (String context_word:context_words_record_created) {
          //assert !context_words_record.containsKey(context_word);
          if (!context_words_record.containsKey(context_word))
            fw.write(context_word+" "+String.valueOf(pmi_mat_file_idx)+" "+String.valueOf(ctr)+"\n");
          ctr += 1;
      }
      fw.close();
      } catch(IOException e) {
        e.printStackTrace();
      }
    }
    return;
  }

  // break_file_into_batches
  public static List<String> break_file_into_batches(String file_name, int sent_num_threshold) {
    List<String> file_names = new ArrayList<String>();
    int file_ctr = 0;
    try {
      BufferedReader ff = new BufferedReader(new FileReader(file_name));
      BufferedWriter fw = Files.newBufferedWriter(Paths.get(file_name+"."+String.valueOf(file_ctr)));
      int ctr = 0;
      String l_in;
      while ((l_in=ff.readLine())!=null) {
        if ((ctr % sent_num_threshold != 0) || (ctr == 0)) {
          fw.write(l_in+"\n");
        } else {
          fw.close();
          file_names.add(file_name+"."+String.valueOf(file_ctr));
          file_ctr += 1;
          fw = Files.newBufferedWriter(Paths.get(file_name+"."+String.valueOf(file_ctr)));
          fw.write(l_in+"\n");
        }
        ctr += 1;
      }
      ff.close();
      fw.close();
      file_names.add(file_name+"."+String.valueOf(file_ctr));
    } catch (IOException e) {
      e.printStackTrace();
    }
    return file_names;
  }

  // break_file_into_batches_by_context_words
  public static Map<String, List<String>> break_file_into_batches_by_context_words(String onebest_file, String candidate_list_file_tmp, String function_words_file, String punctuations_file, int context_words_set_size_threshold) throws context_size_exception {
    List<String> onebest_file_batches = new ArrayList<String>();
    List<String> candidate_list_file_tmp_batches = new ArrayList<String>();

    Set<String> function_words = write_candidate_list_from_eng_vocab.get_function_words(function_words_file);
    Set<String> punctuations = write_candidate_list_from_eng_vocab.get_function_words(punctuations_file);
    Set<String> context_words_set = new HashSet<String>();

    int file_ctr = 0;
    try {
      BufferedReader fo = new BufferedReader(new FileReader(onebest_file));
      BufferedReader fc = new BufferedReader(new FileReader(candidate_list_file_tmp));
      BufferedWriter fwo = Files.newBufferedWriter(Paths.get(onebest_file+"."+String.valueOf(file_ctr)));
      BufferedWriter fco = Files.newBufferedWriter(Paths.get(candidate_list_file_tmp+"."+String.valueOf(file_ctr)));

      int ctr = 0;
      String l_onebest, l_candidate;
      while (((l_onebest = fo.readLine())!=null) && ((l_candidate = fc.readLine())!=null)) {
        // only process sentences with oov words
        if ((l_candidate.equals("")==false) && (l_candidate.equals("=")==false)) {
          Map<Integer, List<String>> candidate_map = write_candidate_list_from_eng_vocab.parse_candidate_map(l_candidate);
          List<String> context_words = write_candidate_list_from_eng_vocab.parse_context_words(l_onebest, candidate_map, function_words, punctuations);
          if (!context_words.isEmpty()) {
            for (String context_word:context_words) context_words_set.add(context_word);
          
            if (context_words_set.size() > context_words_set_size_threshold) {
              if (context_words.size() > context_words_set_size_threshold) {
                throw new context_size_exception("context_words_set_size_threshold must be greater than the minimum sentence length.");
              }
              fwo.close(); fco.close();
              onebest_file_batches.add(onebest_file+"."+String.valueOf(file_ctr));
              candidate_list_file_tmp_batches.add(candidate_list_file_tmp+"."+String.valueOf(file_ctr));
              file_ctr += 1;
              fwo = Files.newBufferedWriter(Paths.get(onebest_file+"."+String.valueOf(file_ctr)));
              fco = Files.newBufferedWriter(Paths.get(candidate_list_file_tmp+"."+String.valueOf(file_ctr)));
              
              fwo.write(l_onebest+"\n");
              fco.write(l_candidate+"\n");
              
              context_words_set.clear();
              for (String context_word:context_words) context_words_set.add(context_word);
            } else {
              fwo.write(l_onebest+"\n");
              fco.write(l_candidate+"\n");
            }
          } else {
            fwo.write(l_onebest+"\n");
            fco.write(l_candidate+"\n");
          }
        } else {
          fwo.write(l_onebest+"\n");
          fco.write(l_candidate+"\n");
        }
        ctr += 1;
      }
      fwo.close(); fco.close();
      onebest_file_batches.add(onebest_file+"."+String.valueOf(file_ctr));
      candidate_list_file_tmp_batches.add(candidate_list_file_tmp+"."+String.valueOf(file_ctr));
    } catch (IOException e) {
      e.printStackTrace();
    }
    Map<String, List<String>> batches = new HashMap<String, List<String>>();
    batches.put("onebest_file_batches", onebest_file_batches);
    batches.put("candidate_list_file_tmp_batches", candidate_list_file_tmp_batches);
    return batches;
  }

  //// break_file_into_batches_by_global_context_words
  //public static void break_file_into_batches_by_global_context_words(String onebest_file, String candidate_list_file_tmp, String function_words_file, String punctuations_file, int context_words_set_size_threshold, String context_words_record_file, String candidate_words_record_file, String pmi_mat_dir) throws context_size_exception {
  //  // context words recorded
  //  context_words_record_obj context_words_obj = get_context_words_recorded(context_words_record_file);
  //  Set<String> context_words_recorded = context_words_obj.context_words_recorded;
  //  int max_file_name = context_words_obj.max_file_name;
  //  int cur_file_name = max_file_name + 1;
  //  String pmi_mat_file = pmi_mat_dir+String.valueOf(cur_file_name);
  //  // candidate words indexed
  //  List<String> candidate_words_record = get_candidate_words_ordered(candidate_words_record_file, function_words_file);
  //  // context words to record
  //  Map<String, Map<String, Integer>> context_words_record_created = new HashMap<String, Map<String, Integer>>();
  //  // function words
  //  Set<String> function_words = write_candidate_list_from_eng_vocab.get_function_words(function_words_file);
  //  // punctuations
  //  Set<String> punctuations = write_candidate_list_from_eng_vocab.get_function_words(punctuations_file);

  //  try {
  //    BufferedReader fo = new BufferedReader(new FileReader(onebest_file));
  //    BufferedReader fc = new BufferedReader(new FileReader(candidate_list_file_tmp));

  //    String l_onebest, l_candidate;
  //    while (((l_onebest = fo.readLine())!=null) && ((l_candidate = fc.readLine())!=null)) {
  //      if ((l_candidate.equals("")==false) && (l_candidate.equals("=")==false)) {
  //        // get candidate words
  //        Map<Integer, List<String>> candidate_map = write_candidate_list_from_eng_vocab.parse_candidate_map(l_candidate);
  //        // get context words
  //        List<String> context_words = write_candidate_list_from_eng_vocab.parse_context_words(l_onebest, candidate_map, function_words, punctuations);

  //        if (!context_words.isEmpty()) {
  //          Set<String> context_words_cur_sent = new HashSet<String>();
  //          for (String context_word:context_words) {
  //            if ((!context_words_recorded.contains(context_word)) && (!context_words_record_created.containsKey(context_word))) {
  //              context_words_cur_sent.add(context_word);
  //            }
  //          }
  //          if (context_words_record_created.size()+context_words_cur_sent.size()>context_words_set_size_threshold) {
  //            List<String> context_words_list = new ArrayList<String>();
  //            int context_word_ctr = 0;
  //            for (String context_word:context_words_record_created.keySet()) {
  //              context_words_list.add(context_word);
  //              context_words_record_created.get(context_word).put("file_num", cur_file_name);
  //              context_words_record_created.get(context_word).put("pos_in_file", context_word_ctr);
  //              context_word_ctr += 1;
  //            }
  //            // compute pmi mat
  //            double[][] pmi_mat = get_pmi.get_mat(candidate_words_record, context_words_list);
  //            // write pmi mat
  //            get_pmi.write_mat(pmi_mat, pmi_mat_file);

  //            cur_file_name += 1;
  //            pmi_mat_file = pmi_mat_dir+String.valueOf(cur_file_name);
  //            for (String context_word:context_words_cur_sent)
  //              context_words_record_created.put(context_word, new HashMap<String, Integer>());
  //          } else {
  //            for (String context_word:context_words_cur_sent)
  //              context_words_record_created.put(context_word, new HashMap<String, Integer>());
  //          }
  //        }
  //      }
  //    }
  //  } catch (IOException e) {
  //    e.printStackTrace();
  //  }
  //  update_context_words_record(context_words_record_created, context_words_record_file);
  //  return;
  //}

  public static void main(String[] args) {
    context_words_record_obj context_words_obj = get_context_words_recorded("examples/context_record.exp");
    Set<String> context_words = context_words_obj.context_words_recorded;
    int max_file = context_words_obj.max_file_name;
    for (String c:context_words) {
      System.out.println(c);
    }
    System.out.println(max_file);
  }
}


