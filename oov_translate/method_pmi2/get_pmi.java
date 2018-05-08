import org.aksw.palmetto.calculations.direct.DirectConfirmationMeasure;
import org.aksw.palmetto.calculations.direct.LogRatioConfirmationMeasure;
import org.aksw.palmetto.corpus.CorpusAdapter;
import org.aksw.palmetto.corpus.WindowSupportingAdapter;
import org.aksw.palmetto.corpus.lucene.WindowSupportingLuceneCorpusAdapter;
import org.aksw.palmetto.corpus.lucene.LuceneCorpusAdapter;
import org.aksw.palmetto.data.SegmentationDefinition;
import org.aksw.palmetto.data.SubsetProbabilities;
import org.aksw.palmetto.prob.ProbabilityEstimator;
import org.aksw.palmetto.prob.window.WindowBasedProbabilityEstimator;
import org.aksw.palmetto.prob.window.BooleanSlidingWindowFrequencyDeterminer;
import org.aksw.palmetto.prob.bd.BooleanDocumentProbabilitySupplier;
import org.aksw.palmetto.subsets.Segmentator;
import org.aksw.palmetto.subsets.OneOne;
//import org.aksw.palmetto.subsets.OneAll;
//import org.aksw.palmetto.subsets.AllAll;
//import org.aksw.palmetto.subsets.OneAny;
//import org.aksw.palmetto.subsets.OneSet;

import java.io.*;
import java.util.*;
import java.util.regex.Pattern;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Comparator;
import java.util.PriorityQueue;
import java.util.stream.Stream;

public class get_pmi {
  String index_path;
  Segmentator segmentation;
  CorpusAdapter corpus_adapter;
  ProbabilityEstimator prob_estimator;
  LogRatioConfirmationMeasure confirmation;

  // constructor
  get_pmi(String index_path0, String context_scale, String window_mechanism) {
    index_path = index_path0;
    segmentation = new OneOne();//new PairwiseTopicComparingSegmentator();//new OneSet();//new OneAny();//new AllAll();//new OneAll();
    prob_estimator = null;
    if (window_mechanism.equals("boolean_window")) {
      corpus_adapter = get_bool_corpus_adapter(index_path);
      prob_estimator = BooleanDocumentProbabilitySupplier.create(corpus_adapter, context_scale, true);
    } else if (window_mechanism.equals("sliding_window")) {
      corpus_adapter = get_window_corpus_adapter(index_path);
      prob_estimator = get_window_based_prob_estimator((WindowSupportingAdapter)corpus_adapter, Integer.parseInt(context_scale));
    } else {
      System.out.println("invalid option!");
    }
    confirmation = new LogRatioConfirmationMeasure();
  }

  public static class word_sim {
    public String word;
    public double similarity;

    public word_sim (String word, double similarity) {
      this.word = word;
      this.similarity = similarity;
    }
  }

  public static class pmi_comparator implements Comparator<word_sim> {
    @Override
    public int compare(word_sim x, word_sim y) {
      if (x.similarity < y.similarity) return -1;
      if (x.similarity > y.similarity) return 1;
      return 0;
    }
  }

  // query_mat
  public static String[] query_mat(
      Map<String, Map<String, Integer>> context_words_recorded, 
      List<String> candidate_words, 
      List<String> context_words, 
      String pmi_mat_dir, 
      int num_candidates_threshold) {
    int num_candidates = candidate_words.size();
    double[] candidate_scores = new double[num_candidates];
    for (String context_word : context_words) {
      int file_num = context_words_recorded.get(context_word).get("file_num");
      int pos_in_file = context_words_recorded.get(context_word).get("pos_in_file");
      String file_name = pmi_mat_dir+String.valueOf(file_num);
      String line;
      try (
        Stream<String> lines = Files.lines(Paths.get(file_name))) {
        line = lines.skip(pos_in_file).findFirst().get();
        String[] scores_str = line.split(" ");
        assert scores_str.length == num_candidates;
        for (int i=0;i<num_candidates;i++) {
          candidate_scores[i] += Double.parseDouble(scores_str[i]);
        }
        //double[] scores = DoubleStream.range(0,scores_str.length).map(i -> Double.parseDouble(i)).toArray();
      } catch (IOException e) {
        e.printStackTrace();
      }
    }

    Comparator<word_sim> comparator = new pmi_comparator();
    PriorityQueue<word_sim> queue = new PriorityQueue<word_sim>(num_candidates_threshold, comparator);
    for (int i=0;i<num_candidates;i++) {
      word_sim candidate_similarity = new word_sim(candidate_words.get(i), candidate_scores[i]);
      queue.add(candidate_similarity);
      if (queue.size() > num_candidates_threshold) queue.poll();
    }

    String[] selected_candidates = new String[num_candidates_threshold];
    int ctr = 0;
    while(queue.size() != 0) {
      selected_candidates[ctr] = queue.remove().word;
      ctr += 1;
    }
    assert ctr == num_candidates_threshold;
    return selected_candidates;
  }

  // get_mat
  public double[][] get_mat(List<String> candidate_words, List<String> context_words) {
    System.out.println("start preparing word_mat...");
    int num_candidate_words = candidate_words.size();
    int num_context_words = context_words.size();

    double[][] pmi_mat_final = new double[num_context_words][num_candidate_words];
    String[][] word_mat = new String[num_candidate_words*num_context_words][];
    int ctr = 0;
    for (int i=0;i<num_candidate_words;i++) {
      for (int j=0;j<num_context_words;j++) {
        word_mat[ctr] = new String[2];
        word_mat[ctr][0] = candidate_words.get(i);
        word_mat[ctr][1] = context_words.get(j);
        ctr += 1;
      }
    }

    System.out.println("start computing pmi_mat...");
    double[][] pmi_mat = get_pmi_from_mat(word_mat);

    System.out.println("start rearranging pmi_mat...");
    for (int i=0;i<num_candidate_words;i++) {
      for (int j=0;j<num_context_words;j++) {
        pmi_mat_final[j][i] = pmi_mat[num_candidate_words*j+i][0];
      }
    }
    return pmi_mat_final;
  }

  // print_mat
  public static void print_mat(double[][] pmi_mat) {
    for (int i=0;i<pmi_mat.length;i++){
      for (int j=0;j<pmi_mat[i].length;j++){
        System.out.print(String.valueOf(pmi_mat[i][j])+" ");
      }
      System.out.println();
    }
  }

  // write_mat
  public static void write_mat(double[][] pmi_mat, String pmi_mat_file) {
    // context (<10k) by candidates (~10k)
    if (pmi_mat.length != 0) {
      try {
        BufferedWriter fw = Files.newBufferedWriter(Paths.get(pmi_mat_file));
        for (int i=0;i<pmi_mat.length;i++) {
          for (int j=0;j<pmi_mat[i].length;j++){
            if (j!=0) fw.write(" ");
            fw.write(String.valueOf(pmi_mat[i][j]));
          }
          fw.write("\n");
        }
        fw.close();
      } catch (IOException e) {
        e.printStackTrace();
      }
    }
  }

  //// read_mat
  //public static double[][] read_mat(String pmi_mat_file) {

  //  double[][] = new double[]
  //  try {
  //    BufferedReader f = new BufferedReader(new FileReader(pmi_mat_file));
  //    int ctr = 0;
  //    String line;
  //    while ((line=f.readLine())!=null) {
  //      String[] l = line.split(" ");
  //      
  //    }
  //  } catch (IOException e) {
  //    e.printStackTrace();
  //  }
  //  return
  //}

  // print_pmi_db
  public static void print_pmi_db(Map<String, Map<String, Double>> db) {
    for (String candidate:db.keySet()) {
      for (String context_word:db.get(candidate).keySet()) {
        double pmi = db.get(candidate).get(context_word);
        System.out.print(candidate+"\t"+context_word+"\t"+String.valueOf(pmi)+"\n");
      }
    }
  }

  // write_pmi_db
  public static void write_pmi_db(Map<String, Map<String, Double>> db, String pmi_db_file) {
    try {
      BufferedWriter fw = Files.newBufferedWriter(Paths.get(pmi_db_file));
      for (String candidate:db.keySet()) {
        for (String context_word:db.get(candidate).keySet()) {
          double pmi = db.get(candidate).get(context_word);
          fw.write(candidate+"\t"+context_word+"\t"+String.valueOf(pmi)+"\n");
        }
      }
      fw.close();
    } catch (IOException e) {
      e.printStackTrace();
    }
  }

  // read_pmi_db
  public static Map<String, Map<String, Double>> read_pmi_db(String pmi_db_file) {
    Map<String, Map<String, Double>> pmi_db = new HashMap<String, Map<String, Double>>();
    try {
      BufferedReader fp = new BufferedReader(new FileReader(pmi_db_file));
      String line;
      while ((line = fp.readLine())!=null) {
        String[] triple = line.split("\t");
        String candidate = triple[0];
        String context_word = triple[1];
        double pmi = Double.valueOf(triple[2]);
        if (!pmi_db.containsKey(candidate)) {
          pmi_db.put(candidate, new HashMap<String, Double>());
          pmi_db.get(candidate).put(context_word, pmi);
        } else {
          pmi_db.get(candidate).put(context_word, pmi);
        }
      }
      fp.close();
    } catch (IOException e) {
    }
    return pmi_db;
  }

  // query_pmi_db
  // input: context_words, output: top k candidates
  public static String[] query_pmi_db(Map<String, Map<String, Double>> pmi_db, List<String> context_words, int k) {
    Comparator<word_sim> comparator = new pmi_comparator();
    PriorityQueue<word_sim> queue = new PriorityQueue<word_sim>(k, comparator);

    int num_context_words = context_words.size();
    for (String candidate:pmi_db.keySet()) {
      double similarity = 0;
      for (String context_word:context_words) {
        double pmi = pmi_db.get(candidate).get(context_word);
        similarity += pmi;
      }
      similarity /= num_context_words; // divided by context size
      word_sim candidate_similarity = new word_sim(candidate, similarity);
      queue.add(candidate_similarity);
      if (queue.size() > k) queue.poll();
    }

    String[] selected_candidates = new String[k];
    int ctr = 0;
    while(queue.size() != 0) {
      selected_candidates[ctr] = queue.remove().word;
      ctr += 1;
    }
    assert ctr == k;
    return selected_candidates;
  }

  // get pmi_db: compute Map<candidate, Map<context, double>>
  public Map<String, Map<String, Double>> get_pmi_db(List<String> candidate_words, Set<String> context_words) {
    Map<String, Map<String, Double>> pmi_db = new HashMap<String, Map<String, Double>>();
    int num_candidate_words = candidate_words.size();
    int num_context_words = context_words.size();
    String[][] word_mat = new String[num_candidate_words*num_context_words][];
    int ctr = 0;
    for (int i=0;i<num_candidate_words;i++) {
      for (String context_word:context_words) {
        word_mat[ctr] = new String[2];
        word_mat[ctr][0] = candidate_words.get(i);
        word_mat[ctr][1] = context_word;
        ctr += 1;
      }
    }
    System.out.println("word_mat done.");

    double[][] pmi_mat = get_pmi_from_mat(word_mat);
    System.out.println("pmi_mat done.");

    ctr = 0;
    for (int i=0;i<num_candidate_words;i++) {
      String candidate = word_mat[ctr][0];
      pmi_db.put(candidate, new HashMap<String, Double>());
      for (int j=0;j<num_context_words;j++) {
        String context_word = word_mat[ctr][1];
        pmi_db.get(candidate).put(context_word, pmi_mat[ctr][0]);
        ctr += 1;
      }
    }
    System.out.println("pmi_db done.");
    return pmi_db;
  }

  // get similarity between list and list
  public double[] get_similarity(List<String> candidate_words, List<String> context_words) {
    System.out.println("start preparing word_mat...");
    int num_candidate_words = candidate_words.size();
    int num_context_words = context_words.size();
    String[][] word_mat = new String[num_candidate_words*num_context_words][];
    int ctr = 0;
    for (int i=0;i<num_candidate_words;i++) {
      for (int j=0;j<num_context_words;j++) {
        word_mat[ctr] = new String[2];
        word_mat[ctr][0] = candidate_words.get(i);
        word_mat[ctr][1] = context_words.get(j);
        ctr += 1;
      }
    }

    System.out.println("start computing pmi_mat...");
    double[][] pmi_mat = get_pmi_from_mat(word_mat);

    System.out.println("start computing similarities...");
    double[] similarities = new double[num_candidate_words];
    ctr = 0;
    for (int i=0;i<num_candidate_words;i++) {
      double similarity = 0;
      for (int j=0;j<num_context_words;j++) {
        similarity += pmi_mat[ctr][0];
        ctr += 1;
      }
      similarities[i] = similarity/num_context_words; // divided by context size
    }
    return similarities;
  }

  // get similarity between word and list
  public double get_similarity(String candidate_word, List<String> context_words) {
    String[][] word_mat = new String[context_words.size()][];
    for (int i=0;i<context_words.size();i++) {
      word_mat[i] = new String[2];
      word_mat[i][0] = candidate_word;
      word_mat[i][1] = context_words.get(i);
    }

    double[][] pmi_mat = get_pmi_from_mat(word_mat);

    double similarity = 0;
    for (int i=0;i<pmi_mat.length;i++) similarity += pmi_mat[i][0];
    return similarity/pmi_mat.length; // divided by context size
  }

  // get pmi matrix from word matrix
  public double[][] get_pmi_from_mat(String[][] word_mat) {
    SegmentationDefinition definitions[] = new SegmentationDefinition[word_mat.length];
    for (int i=0;i<definitions.length;i++)
      definitions[i] = segmentation.getSubsetDefinition(word_mat[i].length);
    System.out.println("segmentation done.");
    assert definitions.length == word_mat.length;

    // slowest part, run as little as possible --> make big word_mat beforehand
    SubsetProbabilities probabilities[] = prob_estimator.getProbabilities(word_mat, definitions);
    System.out.println("SubsetProbabilities done.");

    assert probabilities.length == definitions.length;
    double[][] pmi_mat = new double[probabilities.length][];
    for (int i=0; i<probabilities.length; i++) {
      double[] pmi_vec = confirmation.calculateConfirmationValues(probabilities[i]);
      pmi_mat[i] = new double[pmi_vec.length];
      for (int j=0; j<pmi_vec.length; j++) {
        if (pmi_vec[j]<0) pmi_vec[j]=0;
        pmi_mat[i][j] = pmi_vec[j];
      }
    }
    return pmi_mat;
  }

  public CorpusAdapter get_bool_corpus_adapter(String index_path) {
    try {
      return LuceneCorpusAdapter.create(index_path, "text");
    } catch (Exception e) {
      return null;
    }
  }

  public CorpusAdapter get_window_corpus_adapter(String index_path) {
    try {
      return WindowSupportingLuceneCorpusAdapter.create(index_path, "text", "length");
    } catch (Exception e) {
      return null;
    }
  }

  public WindowBasedProbabilityEstimator get_window_based_prob_estimator(WindowSupportingAdapter window_corpus_adapter, int window_size) {
    WindowBasedProbabilityEstimator prob_estimator = new WindowBasedProbabilityEstimator (new BooleanSlidingWindowFrequencyDeterminer(window_corpus_adapter, window_size));
    prob_estimator.setMinFrequency(WindowBasedProbabilityEstimator.DEFAULT_MIN_FREQUENCY*window_size);
    return prob_estimator;
  }

  public static void main(String[] args) {
    String word = "teacher";
    List<String> context = Arrays.asList("school", "class", "student");
    List<String> candidates = Arrays.asList("professor", "tea");
    List<String> candidates2 = Arrays.asList("professor", "tea", "apple", "class", "seat");
    List<String> candidates3 = Arrays.asList("professor", "tea", "apple", "class");
    String context_scale = "bp";
    String window_mechanism = "boolean_window";
    String index_path0 = "/home/ec2-user/kklab/data/wiki_data_"+context_scale+"/wikipedia_"+context_scale;
    String index_path1 = "/home/ec2-user/kklab/data/wiki/wikipedia_"+context_scale; 
    get_pmi my_pmi = new get_pmi(index_path1, context_scale, window_mechanism);

    //double similarity = my_pmi.get_similarity(word, context);
    //System.out.println(similarity);

    //String[][] word_mat = new String[2][];
    //word_mat[0] = new String[3];
    //for (int i=0;i<3;i++) word_mat[0][i] = context.get(i);
    //word_mat[1] = new String[2];
    //for (int i=0;i<2;i++) word_mat[1][i] = candidates.get(i);
    //double[][] pmi_mat = my_pmi.get_pmi_from_mat(word_mat);
    //get_pmi.print_mat(pmi_mat);
    
    //double[][] pmi_mat = new double[2][];
    //pmi_mat[0] = new double[]{1.0,3.0,4.0};
    //pmi_mat[1] = new double[]{2.0,5.0,1.0};
    //print_mat(pmi_mat);
    //write_mat(pmi_mat, "pmi_jk");
    
    //String[][] word_mat = new String[5][];
    //word_mat[0] = new String[]{"professor", "tea", "apple", "class"};
    //word_mat[1] = new String[]{"professor", "apple"};
    //word_mat[2] = new String[]{"professor", "class"};
    //word_mat[3] = new String[]{"tea", "apple"};
    //word_mat[4] = new String[]{"tea", "class"};
    //double[][] pmi_mat = my_pmi.get_pmi_from_mat(word_mat);
    //get_pmi.print_mat(pmi_mat);
 
    //String[][] word_mat = new String[1][];
    //int wordsetsize = 9;
    //int width = wordsetsize*2;
    //word_mat[0] = new String[width];
    //for (int i=0;i<width;i++) word_mat[0][i] = "professor";
    //double[][] pmi_mat = my_pmi.get_pmi_from_mat(word_mat);

    //int wordsetsize = 9;
    //String[][] word_mat = new String[wordsetsize*wordsetsize][];
    //for (int i=0;i<wordsetsize*wordsetsize;i++) {
    //  word_mat[i] = new String[]{"professor", "professor"};
    //}
    //double[][] pmi_mat = my_pmi.get_pmi_from_mat(word_mat);

    //double[] similarity = my_pmi.get_similarity(candidates, context);
    //for (int i=0;i<similarity.length;i++) System.out.println(similarity[i]);
    
    //Set<String> candidates_set = write_candidate_list_from_eng_vocab.get_function_words("/home/ec2-user/kklab/data/google-10000-english/google-10000-english.txt");
    //List<String> candidates_list = new ArrayList<String>();
    //for (String c:candidates_set) candidates_list.add(c);
    //System.out.println("list created");
    //long start_time = System.nanoTime();
    //double[] similarity = my_pmi.get_similarity(candidates_list, context);
    //long end_time = System.nanoTime();
    //long duration = (end_time - start_time);
    //System.out.println(duration/1000000000);
    //for (int i=0;i<similarity.length;i++) System.out.println(similarity[i]);
    
    //Set<String> context_set = new HashSet<String>();
    //for (String c:context) context_set.add(c);
    //Map<String, Map<String, Double>> pmi_db = my_pmi.get_pmi_db(candidates2, context_set);
    //my_pmi.print_pmi_db(pmi_db);
    //System.out.println();
    //my_pmi.write_pmi_db(pmi_db, "tmp");
    //pmi_db = my_pmi.read_pmi_db("tmp");
    //my_pmi.print_pmi_db(pmi_db);
    //System.out.println();
    //String[] res = my_pmi.query_pmi_db(pmi_db, context, 3);
    //for (int i=0;i<res.length;i++) System.out.println(res[i]);

  }

}





