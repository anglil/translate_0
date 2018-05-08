import java.io.*;
import java.util.*;
import java.util.regex.Pattern;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

public class write_candidate_list_from_eng_vocab {
  // get function words, or punctuations
  public static Set<String> get_function_words(String function_words_file) {
    Set<String> function_words = new HashSet<String>();
    try {
      BufferedReader ff = new BufferedReader(new FileReader(function_words_file));
      int ctr = 0;
      String l_word;
      while ((l_word = ff.readLine())!=null) {
        function_words.add(l_word);
        ctr += 1;
      }
      System.out.println(String.valueOf(ctr)+" function words read.");
    } catch(IOException e) {
      e.printStackTrace();   
    }
    return function_words;
  }
  
  // parse candidate map
  public static Map<Integer, List<String>> parse_candidate_map(String l_candidate) {
    Map<Integer, List<String>> candidate_map = new HashMap<Integer, List<String>>();
    String[] item_candidate = l_candidate.split(" ");
    for (String item:item_candidate) {
      String[] pair = item.split(":");
      int oov_pos = Integer.valueOf(pair[0]);
      String[] candidates = pair[1].split(",");
      candidate_map.put(oov_pos, new ArrayList<String>());
      for (String candidate:candidates) candidate_map.get(oov_pos).add(candidate);
    }
    return candidate_map;
  }

  // parse context words
  public static List<String> parse_context_words(String l_onebest, Map<Integer, List<String>> candidate_map, Set<String> function_words, Set<String> punctuations) {
    List<String> tok_context = new ArrayList<String>();
    String[] tok_onebest = l_onebest.split(" ");
    for (int oov_pos=0; oov_pos<tok_onebest.length; oov_pos++) {
      if ((!candidate_map.containsKey(oov_pos)) && (!function_words.contains(tok_onebest[oov_pos])) && (!punctuations.contains(tok_onebest[oov_pos])))
        tok_context.add(tok_onebest[oov_pos]);
    }
    assert tok_context.size() <= tok_onebest.length;
    return tok_context;
  }

  // candidate_list context_set
  public class candidate_list_context_set {
    public Set<String> context_words_set;
    public List<String> candidate_list;

    public candidate_list_context_set (Set<String> context_words_set, List<String> candidate_list) {
      this.context_words_set = context_words_set;
      this.candidate_list = candidate_list;
    }
  }

  // get random candidate list
  public static List<String> get_random_candidate_list(Map<Integer, List<String>> candidate_map) {
    List<String> candidate_list = new ArrayList<String>();
    for (int oov_pos:candidate_map.keySet()) {
      candidate_list = candidate_map.get(oov_pos);
      break;
    }
    return candidate_list;
  }

  // get context words set
  public candidate_list_context_set get_context_words_set(String onebest_file, String candidate_list_file_tmp, String function_words_file, String punctuations_file) {
    Set<String> context_words_set = new HashSet<String>();
    List<String> candidate_list = new ArrayList<String>();

    Set<String> function_words = get_function_words(function_words_file);
    Set<String> punctuations = get_function_words(punctuations_file);
    try {
      BufferedReader fo = new BufferedReader(new FileReader(onebest_file));
      BufferedReader fc = new BufferedReader(new FileReader(candidate_list_file_tmp));

      int ctr = 0;
      String l_onebest, l_candidate;
      while (((l_onebest = fo.readLine())!=null) && ((l_candidate = fc.readLine())!=null)) {
        // only process sentences with oov words
        if ((l_candidate.equals("")==false) && (l_candidate.equals("=")==false)) {
          Map<Integer, List<String>> candidate_map = parse_candidate_map(l_candidate);
          if (candidate_list.isEmpty()) candidate_list = get_random_candidate_list(candidate_map);
          List<String> context_words = parse_context_words(l_onebest, candidate_map, function_words, punctuations);
          if (!context_words.isEmpty()) {
            for (String context_word:context_words) context_words_set.add(context_word);
          }
        }
      }
      fo.close();
      fc.close();
    } catch (IOException e) {
      e.printStackTrace();
    }
    candidate_list_context_set cc = new candidate_list_context_set(context_words_set, candidate_list);
    return cc;
  }

  public Map<String, Map<String, Double>> get_pmi_db(String index_path, String context_scale, String window_mechanism, List<String> candidate_words, Set<String> context_words) {
    get_pmi my_pmi = new get_pmi(index_path, context_scale, window_mechanism);
    Map<String, Map<String, Double>> pmi_db = my_pmi.get_pmi_db(candidate_words, context_words);
    return pmi_db;
  }

  public void write_candidate_list_file(String onebest_file, String candidate_list_file_tmp, String function_words_file, String punctuations_file, String candidate_list_file, Map<String, Map<String, Double>> pmi_db, int k) {
    Set<String> function_words = get_function_words(function_words_file);
    Set<String> punctuations = get_function_words(punctuations_file);
    try {
      BufferedReader fo = new BufferedReader(new FileReader(onebest_file));
      BufferedReader fc = new BufferedReader(new FileReader(candidate_list_file_tmp));
      BufferedWriter fw = Files.newBufferedWriter(Paths.get(candidate_list_file));

      int ctr = 0;
      String l_onebest, l_candidate;
      while (((l_onebest = fo.readLine())!=null) && ((l_candidate = fc.readLine())!=null)) {
        // only process sentences with oov words
        if ((l_candidate.equals("")==false) && (l_candidate.equals("=")==false)) {
          Map<Integer, List<String>> candidate_map = parse_candidate_map(l_candidate);
          List<String> context_words = parse_context_words(l_onebest, candidate_map, function_words, punctuations);

          String[] selected_candidates;
          if (!context_words.isEmpty()) {
            selected_candidates = get_pmi.query_pmi_db(pmi_db, context_words, k) ;
          } else {
            selected_candidates = new String[1];
            selected_candidates[0] = (get_random_candidate_list(candidate_map)).get(0);
          }

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
      }
      fo.close();fc.close();fw.close();
    } catch (IOException e) {
      e.printStackTrace();
    }
  }

  public static void run(String onebest_file, String candidate_list_file_tmp, String function_words_file, String punctuations_file, String index_path, String context_scale, String window_mechanism, int num_candidates, String candidate_list_file) {
    write_candidate_list_from_eng_vocab a = new write_candidate_list_from_eng_vocab();
    // 1. get_context_words_set
    candidate_list_context_set cc = a.get_context_words_set(onebest_file, candidate_list_file_tmp, function_words_file, punctuations_file);
    List<String> candidate_words = cc.candidate_list;
    System.out.println("candidate_list size: "+String.valueOf(candidate_words.size()));
    Set<String> context_words_set = cc.context_words_set;
    System.out.println("context_words_set size: "+String.valueOf(context_words_set.size()));
    // 2. get_pmi_db
    Map<String, Map<String, Double>> pmi_db = a.get_pmi_db(index_path, context_scale, window_mechanism, candidate_words, context_words_set);
    // 3. write_candidate_list_file
    a.write_candidate_list_file(onebest_file, candidate_list_file_tmp, function_words_file, punctuations_file, candidate_list_file, pmi_db, num_candidates);
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
    int num_candidates = Integer.valueOf(args[7]);

    // output
    String candidate_list_file = args[8];

    write_candidate_list_from_eng_vocab.run(onebest_file, candidate_list_file_tmp, function_words_file, punctuations_file, index_path, context_scale, window_mechanism, num_candidates, candidate_list_file);
    //write_candidate_list_from_eng_vocab a = new write_candidate_list_from_eng_vocab();
    //candidate_list_context_set cc = a.get_context_words_set(onebest_file, candidate_list_file_tmp, function_words_file, punctuations_file);
    //List<String> candidate_words = cc.candidate_list;
    //System.out.println("candidate_list size: "+String.valueOf(candidate_words.size()));
    //Set<String> context_words_set = cc.context_words_set;
    //System.out.println("context_words_set size: "+String.valueOf(context_words_set.size()));
    //Map<String, Map<String, Double>> pmi_db = a.get_pmi_db(index_path, context_scale, window_mechanism, candidate_words, context_words_set);
    //a.write_candidate_list_file(onebest_file, candidate_list_file_tmp, function_words_file, punctuations_file, candidate_list_file, pmi_db, num_candidates);
  }
}
