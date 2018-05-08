import java.io.*;
import java.util.*;
import java.util.regex.Pattern;

//import org.slf4j.Logger;
//import org.slf4j.LoggerFactory;

import org.apache.commons.lang3.StringEscapeUtils;

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






public class CoOccurThreaded {
    /*
     * run bash script from within java
     */
    public static String shRealTime(String cmd) {
        String stdout = "";
        try {
            Process p = Runtime.getRuntime().exec(cmd);
            BufferedReader read = new BufferedReader(new InputStreamReader(p.getInputStream()));
            String inputLine;
            while ((inputLine = read.readLine()) != null) {
                System.out.println(inputLine);
                stdout += inputLine+"\n";
            }
            read.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return stdout;
    }
    
    /*
     * get the optimal number of threads the system supports
     */
    public static int getThreadNum() {
        int thread = 1;
        int core = 1;
        int socket = 1;
        
        String cpuOut = shRealTime("lscpu");
        String[] cpuOutArray = cpuOut.split("\n");
        for (String item : cpuOutArray) {
            if (item.contains("Thread(s) per core")) {
                String[] threadArray = item.split(":");
                thread = Integer.parseInt(threadArray[1].trim());
            }
            if (item.contains("Core(s) per socket")) {
                String[] coreArray = item.split(":");
                core = Integer.parseInt(coreArray[1].trim());
            }
            if (item.contains("Socket(s)")) {
                String[] socketArray = item.split(":");
                socket = Integer.parseInt(socketArray[1].trim());
            }
        }
        
        return thread*core*socket;
    }
    
    /*
     * bucket totalNum of instances into bins for parallelization
     */
    public static int[][] getBinByThread(int threadNum, int totalNum) {
        int[][] bins = new int[threadNum][2];
        int binBaseSize = totalNum/threadNum;
        int binResidual = totalNum%threadNum;
        int ctr = 0;
        int idx = 0;
        while (ctr < totalNum) {
            int ctr_lo = ctr;
            int ctr_up;
            if (binResidual > 0) {
                ctr_up = ctr + binBaseSize;
                binResidual = binResidual - 1;
            } else {
                ctr_up = ctr + binBaseSize - 1;
            }
            bins[idx][0] = ctr_lo;
            bins[idx][1] = ctr_up;
            idx += 1;
            ctr = ctr_up + 1;
        }
        return bins;
    }
    
    
    /*
     * get the length of a file
     */
    public static int getFileLength(String fileName) {
        String r = shRealTime("wc -l "+fileName);
        String[] lc = r.split(" ");
        int fileLength = Integer.parseInt(lc[0]);
        return fileLength;
    }
    
    /*
     * merge multiple files into one
     */
    public static void mergeFiles(File[] files, File mergedFile) {
		FileWriter fstream = null;
		BufferedWriter out = null;
		try {
            // append, not overwrite
			fstream = new FileWriter(mergedFile, true);
			out = new BufferedWriter(fstream);
		} catch (IOException e1) {
			e1.printStackTrace();
		}
 
		for (File f : files) {
			//System.out.println("merging: " + f.getName());
			FileInputStream fis;
			try {
				fis = new FileInputStream(f);
				BufferedReader in = new BufferedReader(new InputStreamReader(fis));
 
				String aLine;
				while ((aLine = in.readLine()) != null) {
					out.write(aLine);
					out.newLine();
				}
 
				in.close();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
 
		try {
			out.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
    
    	/* used in phase 1, sliding window */
	public static WindowBasedProbabilityEstimator getWindowBasedProbabilityEstimator(WindowSupportingAdapter windowCorpusAdapter, int windowSize) {
        /* used in sliding window and boolean window */
        String DEFAULT_TEXT_INDEX_FIELD_NAME = "text";
        /* used only in sliding window */
        String DEFAULT_DOCUMENT_LENGTH_INDEX_FIELD_NAME = "length";
		WindowBasedProbabilityEstimator probEstimator = new WindowBasedProbabilityEstimator (new BooleanSlidingWindowFrequencyDeterminer(windowCorpusAdapter, windowSize));
		probEstimator.setMinFrequency(WindowBasedProbabilityEstimator.DEFAULT_MIN_FREQUENCY * windowSize);
		return probEstimator;
	}





	
	/* used in phase 1, sliding window */
	public static CorpusAdapter getWindowCorpusAdapter(String indexPath) {
        /* used in sliding window and boolean window */
        String DEFAULT_TEXT_INDEX_FIELD_NAME = "text";
        /* used only in sliding window */
        String DEFAULT_DOCUMENT_LENGTH_INDEX_FIELD_NAME = "length";
		try {
			return WindowSupportingLuceneCorpusAdapter.create(indexPath, DEFAULT_TEXT_INDEX_FIELD_NAME, DEFAULT_DOCUMENT_LENGTH_INDEX_FIELD_NAME);
		} catch (Exception e) {
			//LOGGER.error("Couldn't open lucene index. Aborting.", e);
			return null;
		}
	}






	/* used in phase 1, boolean window */
	public static CorpusAdapter getBoolCorpusAdapter(String indexPath) {
        /* used in sliding window and boolean window */
        String DEFAULT_TEXT_INDEX_FIELD_NAME = "text";
        /* used only in sliding window */
        String DEFAULT_DOCUMENT_LENGTH_INDEX_FIELD_NAME = "length";
		try {
			return LuceneCorpusAdapter.create(indexPath, DEFAULT_TEXT_INDEX_FIELD_NAME);
		} catch (Exception e) {
			return null;
		}
	}
    
    
    /* used in phase 1, load the english vocabulary */
	public static Set<String> get_eng_vocab(String eng_list) {
		Set<String> eng_vocab = new HashSet<String>();
		/* removing the function words from the vocab */
		Set<String> functionWords = new HashSet<String>(Arrays.asList("is", "was", "are", "were", "be", "the", "an", "a", "and", "or"));
		try (BufferedReader f = new BufferedReader(new FileReader(eng_list));) {
			String line;
			while ((line = f.readLine()) != null) {
				String w = line.replace("\n", "");
                /*to lower case*/
                w = w.toLowerCase();
				if (!functionWords.contains(w)) {
					eng_vocab.add(w);
				}
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
		return eng_vocab;
	}






	/* used in phase 1, translate oov words in one sentence */
    /*
    * l[0] oov uyghur word
    * l[1] transliteration: not used
    * l[2] pairs of translations and scores: score not used
    */
	public static Map<String, Map<String, Double>> get_ug_dict(String ug_rank_file) {
		Map<String, Map<String, Double>> ug_dict = new HashMap<String, Map<String, Double>>();
		try (BufferedReader f = new BufferedReader(new FileReader(ug_rank_file));) {
		String line;
		while((line = f.readLine()) != null) {
			line = line.replace("\n", "");
			String[] l = line.split("\t");
			if (!l[1].equals("[NOHYPS]")) {
                /*to lower case*/
				String ug_word = l[0].toLowerCase();
                /*form en_hyp*/
				Map<String, Double> en_hyp = new HashMap<String, Double>();
				String[] en_list;
                if (l.length == 3)
                    en_list = l[2].split(";");
                else if (l.length == 2)
                    en_list = l[1].split(";");
                else
                    continue;
				String[] en_list_final = Arrays.copyOf(en_list, en_list.length-1);
				for (String word_score_pair : en_list_final) {
					String[] pair = word_score_pair.split(",");
                    if (pair.length == 2) {
                        /*to lower case*/
                        String en_word = pair[0].toLowerCase();
                        double score = Float.valueOf(pair[1]);
                        /* TODO: not accepting phrase here */
                        String[] word_split = en_word.split(" ");
                        if (word_split.length == 1) {
                            en_hyp.put(en_word, score);
                        }
                    }
				}
                
                /*form ug_dict*/
				if (!en_hyp.isEmpty()) {
					ug_dict.put(ug_word, en_hyp);
				}
			}
		}
		} catch (IOException e) {
			e.printStackTrace();
		}
		return ug_dict;
	}

    
	public static void main(String[] args) {
        String candidateSource = args[0];
        boolean addAlignedOov = Boolean.parseBoolean(args[1]);
        String contextScale = args[2];
        String windowMechanism = args[3];
        
        String resFile = args[4];
        
        String traFile = args[5];
        String oovFile = args[6];
        String refFile = args[7];
        String oovCandidatesFile = args[8];
        String engVocabFile = args[9];
        String oovAlignedFile = args[10];
        
        String tmpDir = args[11];
        String indexDir = args[12];
        
        
        
        /* -------- phase 1: get the probabilities from the training corpus, establishing shared resources -------- */
        String indexPath = indexDir+"wikipedia_"+contextScale;
		/* used in sliding window and boolean window */
		Segmentator segmentation = new OneOne();
		ProbabilityEstimator probEstimator = null;
        
		if (windowMechanism.equals("sliding_window")) {
			/* only in sliding window, AbstractProbabilitySupplier */
            CorpusAdapter windowCorpusAdapter = getWindowCorpusAdapter(indexDir);
            /* parse for window size */
            int windowSize = Integer.parseInt(contextScale);
			probEstimator = getWindowBasedProbabilityEstimator((WindowSupportingAdapter) windowCorpusAdapter, windowSize);
		} else if (windowMechanism.equals("boolean_window")) {
			/* only in boolean window, AbstractProbabilitySupplier */
            CorpusAdapter boolCorpusAdapter = getBoolCorpusAdapter(indexPath);
			probEstimator = BooleanDocumentProbabilitySupplier.create(boolCorpusAdapter, "bd", true);
		}
        
		/* used in sliding window and boolean window */
		LogRatioConfirmationMeasure confirmation = new LogRatioConfirmationMeasure();
        
        /* initialize oov candidate words */
        Map<String, Map<String, Double>> ug_dict = null;
        Set<String> eng_vocab = null;
        /* organize english vocab words into a set */
        if (candidateSource.equals("eng_vocab")) {
            eng_vocab = get_eng_vocab(engVocabFile);
        }
        if (candidateSource.equals("ug_dict")) {
			/* organize oov candidate words into a dictionary */
			ug_dict = get_ug_dict(oovCandidatesFile);
		}
        
        
        
        /* -------- below is the multithreading routine, including phrase 2 and 3 -------- */
        int threadNum = getThreadNum();
        System.out.println("Number of threads: "+String.valueOf(threadNum));
        int totalNum = getFileLength(traFile);
        System.out.println("Number of sentences: "+String.valueOf(totalNum));
        int[][] bins = getBinByThread(threadNum, totalNum);

        List<Thread> threadList = new ArrayList<Thread>();
        
        /* create threads */
        for (int i=0;i<threadNum;i++) {
            int ctr_lo = bins[i][0];
            int ctr_up = bins[i][1];
            String resFileTmp = tmpDir+candidateSource+"_"+String.valueOf(addAlignedOov)+"_"+contextScale+"_"+windowMechanism+"_"+String.valueOf(ctr_lo)+"_"+String.valueOf(ctr_up);
            
            PMI t = new PMI(candidateSource, addAlignedOov, contextScale, windowMechanism, resFileTmp, ctr_lo, ctr_up, eng_vocab, ug_dict, segmentation, probEstimator, confirmation, traFile, oovFile, refFile, oovCandidatesFile, engVocabFile, oovAlignedFile, indexDir);
            
            threadList.add(t);
        }
        
        /* start threads */
        for (int i=0;i<threadNum;i++) {
            threadList.get(i).start();
        }
        
        /* waits for all threads to finish */
        for (Thread t : threadList) {
            try {
                t.join();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
        System.out.println("All threads finished.");
        
        /* merge result files */
        File[] inputs = new File[threadNum];
        for (int i=0;i<threadNum;i++) {
            int ctr_lo = bins[i][0];
            int ctr_up = bins[i][1];
            String resFileTmp = tmpDir+candidateSource+"_"+String.valueOf(addAlignedOov)+"_"+contextScale+"_"+windowMechanism+"_"+String.valueOf(ctr_lo)+"_"+String.valueOf(ctr_up);
            inputs[i] = new File(resFileTmp);
        }
        File output = new File(resFile);
        
        /* delete the existing file before merging */
        if (output.delete()) {
            System.out.println(output.getName() + " is deleted!");
        } else {
            System.out.println("Merged file not deleted.");
        }
        mergeFiles(inputs, output);
        System.out.println("Result files merged.");
	}
    
}






class PMI extends Thread {   
    /* -------- hyperparameters specific to the PMI method -------- */
    /*
     * candidateSource: where the candidate word list for oov translation comes from: "ug_dict" or "eng_vocab" 
     * contextScale: either bs, bp, bd when windowMechanism is boolean_window, or e.g., 10, when windowMechanism is sliding_window
     * windowMechanism: sliding window: "sliding_window" or boolean window: "boolean_window"
     */
    private String candidateSource, contextScale, windowMechanism;
    private boolean addAlignedOov;
    
    /* -------- resFile: one thread writes to one temporary file -------- */
    private String resFile;
    
    /* ctr_lo: inclusive lower bound of sentence index
     * ctr_up: inclusive upper bound of sentence index */
    private int ctr_lo, ctr_up;
    
    /* -------- shared resources -------- */ 
    private static Set<String> eng_vocab;
    private static Map<String, Map<String, Double>> ug_dict;
    private static Segmentator segmentation;
    private static ProbabilityEstimator probEstimator;
    private static LogRatioConfirmationMeasure confirmation;

    /* -------- external software and datasets -------- */
    private String traFile, oovFile, refFile, oovCandidatesFile, engVocabFile, oovAlignedFile;


	/*
    * constructor:
    * candidateSource: eng_vocab, ug_dict
    * addAlignedOov: true, false
    * contextScale: bs, bp, bd, 10, 20, ...
    * windowMechanism: sliding_window, boolean_window
    */
	public PMI(
        String candidateSource, 
        boolean addAlignedOov,
        String contextScale,
        String windowMechanism, 
        String resFile,
        int ctr_lo, 
        int ctr_up,
        Set<String> eng_vocab, 
        Map<String, Map<String, Double>> ug_dict, 
        Segmentator segmentation, 
        ProbabilityEstimator probEstimator, 
        LogRatioConfirmationMeasure confirmation,
        String traFile,
        String oovFile,
        String refFile,
        String oovCandidatesFile,
        String engVocabFile,
        String oovAlignedFile,
        String indexDir
    ) {      
        /* method params */
		this.candidateSource = candidateSource;
        this.addAlignedOov = addAlignedOov;
        this.contextScale = contextScale;
		this.windowMechanism = windowMechanism;
        
        /* one thread writes to one temporary file, later to be merged */
        this.resFile = resFile;
        
        /* lower and upper bounds of instance indices */
        this.ctr_lo = ctr_lo;
		this.ctr_up = ctr_up;
        
        /* shared resourced across threads */
        this.eng_vocab = eng_vocab;
        this.ug_dict = ug_dict;
        this.segmentation = segmentation;
        this.probEstimator = probEstimator;
        this.confirmation = confirmation;
        
        /* external data and software */
        this.traFile = traFile;
        this.oovFile = oovFile;
        this.refFile = refFile;
        this.oovCandidatesFile = oovCandidatesFile;
        this.engVocabFile = engVocabFile;
        this.oovAlignedFile = oovAlignedFile;
	}






	@Override
	public void run() {
		/* phase 2 and 3 */
		int ctr = 0;

		try {
			/* read translation with oov sentence by sentence */
			BufferedReader ft = new BufferedReader(new FileReader(traFile));
			/* read oov sentence by sentence */
			BufferedReader fo = new BufferedReader(new FileReader(oovFile));
			/* write translation with oov also translated sentence by sentence */
			//PrintWriter f = new PrintWriter(new FileWriter(resFile));
            //Writer f = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(resFile), "utf-8"));
            FileWriter f = new FileWriter(new File(resFile), false);
            
			String l_tra, l_oov;
			while (((l_tra = ft.readLine()) != null) && ((l_oov = fo.readLine()) != null)) {
			
				if (ctr >= ctr_lo && ctr <= ctr_up) {
                    /*compute tra_tok, unescaping html not happening*/
					l_tra = l_tra.replace("\n", "");
					String[] tra_tok = l_tra.split(" ");
					//for (int i=0; i<tra_tok.length; i++) {
					//	tra_tok[i] = StringEscapeUtils.unescapeHtml4(tra_tok[i]);
					//}
                   
                    /*compute oov_pos*/
                    l_oov = l_oov.replace("\n", "");
					String[] oov_tok = l_oov.split(" ");
					ArrayList<Integer> oov_pos = new ArrayList<Integer>();
					for (String oov_word: oov_tok) {
						if(oov_word != null && !oov_word.trim().isEmpty() && Arrays.asList(tra_tok).contains(oov_word) && !Pattern.matches("^[a-zA-Z0-9_]*$", oov_word.replaceAll("\\p{P}", ""))) {
							int oov_idx = Arrays.asList(tra_tok).indexOf(oov_word);
							oov_pos.add(oov_idx);
						}
					}
					Collections.sort(oov_pos);
				                   
                    /*compute context, excluding punctuation*/
                    Set<String> functionWords = new HashSet<String>(Arrays.asList("is", "was", "are", "were", "be", "the", "an", "a", "and", "or"));
					ArrayList<Integer> context = new ArrayList<Integer>();
					for (int i=0; i< tra_tok.length; i++) {
						if (!oov_pos.contains(i) && !Pattern.matches("\\p{Punct}", tra_tok[i]) && !functionWords.contains(tra_tok[i].toLowerCase())) {
							context.add(i);
						}
					}
                    
                    
                    /* the actual PMI-based oov translation happens here */
					System.out.println("[ID " + this.getId() + "] "+"Processing sentence "+String.valueOf(ctr)+"...");
					System.out.println("[ID " + this.getId() + "] "+"sentence: "+Arrays.toString(tra_tok));
					System.out.println("[ID " + this.getId() + "] "+"oov: "+oov_pos.toString());
					System.out.println("[ID " + this.getId() + "] "+"context: "+context.toString());

					String tra_wo_oov;
					if (!context.isEmpty() && !oov_pos.isEmpty()) {
						tra_wo_oov = String.join(" ", translate_oov(tra_tok, oov_pos, context));
					} else {
						tra_wo_oov = String.join(" ", tra_tok);
					}

					System.out.println("[ID " + this.getId() + "] "+"Processed sentence: "+tra_wo_oov);
					System.out.println("[ID " + this.getId() + "] "+"----------------");

					f.write(tra_wo_oov+"\n");
                    /*write the result to file as soon as it's available*/
					f.flush();
				}
				ctr++;
			}
			f.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}






	/* translate_oov replaces all oov words with translated words */
	/*
 	* tra_tok: tokenized translation
 	* oov_pos: oov positions
 	* context: context word positions
 	*/ 
	public String[] translate_oov (
		String[] tra_tok, 
		List<Integer> oov_pos, 
		List<Integer> context
    ) {
		/*list of words as context, the real context*/
		List<String> context_words = new ArrayList<String>();
        /*set of words as context, only used for extracting oov candidates*/
        Set<String> context_words_set = new HashSet<String>();
		for (int i : context) {
			context_words.add(tra_tok[i]);
            context_words_set.add(tra_tok[i]);
		}
        
        /*list of words as oov, the real oov*/
        List<String> oov_words = new ArrayList<String>();
        /*set of words as oov, only used for extracting oov candidates*/
        Set<String> oov_words_set = new HashSet<String>();
        for (int i : oov_pos) {
            oov_words.add(tra_tok[i]);
            oov_words_set.add(tra_tok[i]);
        }
        
        
        /*dictionary, key: oov, value: {candidate: score with context}*/
		Map<String, Map<String, Double>> oov_candidates = null;
        Map<String, Double> oov_candidates_eng_vocab = null;
		if (candidateSource.equals("ug_dict")) {
			oov_candidates = get_oov_candidates(oov_words_set, addAlignedOov);
		} else if (candidateSource.equals("eng_vocab")) {
			oov_candidates_eng_vocab = get_oov_candidates(oov_words_set, context_words_set);
		}

        
		/*dictionary, key: oov, value: elected candidate*/
		Map<String, String> oov_elect = new HashMap<String, String>();
		/*set, remaining untranslated oovs*/
		Set<String> oov_remaining = new HashSet<String>();
		for (int i : oov_pos) {
			oov_remaining.add(tra_tok[i]);
		}

        
        /* for whole-vocabulary candidate searching, one loop */
        if (candidateSource.equals("eng_vocab")) {
            /*for oov word that is also candidate, select it, and also remove it from candidates*/
            for (String oov : oov_words_set) {
                if (oov_candidates_eng_vocab.containsKey(oov)) {
                    oov_elect.put(oov, oov);
                    context_words.add(oov);
                    context_words_set.add(oov);
                    
                    oov_remaining.remove(oov);
                    oov_candidates_eng_vocab.remove(oov);
                    
                    System.out.println("[ID " + this.getId() + "] "+"oov: "+oov);
                    System.out.println("[ID " + this.getId() + "] "+"translation: "+oov);
                }
            }
            
            
            /* calculate the PMI between each word in oov_candidates with the contexts*/
            if (!oov_remaining.isEmpty()){
                double best_similarity = -1;
                String best_candidate = null;
                for (String eng : oov_candidates_eng_vocab.keySet()) {
                    /*similarity between candidate and context words*/
                    double similarity = get_similarity(eng, context_words);
                    oov_candidates_eng_vocab.put(eng, similarity);
                    if (similarity > best_similarity) {
                        best_similarity = similarity;
                        best_candidate = eng;
                    }
                }
                String elect_pre = best_candidate;
        
                
                for (Iterator<String> i = oov_remaining.iterator(); i.hasNext();) {
                    String oov = i.next();
                    System.out.println("[ID " + this.getId() + "] "+"oov: "+oov);
                    System.out.println("[ID " + this.getId() + "] "+"translation: "+elect_pre);
                    
                    oov_elect.put(oov, elect_pre);
                    context_words.add(elect_pre);
                    context_words_set.add(elect_pre);
                    
                    i.remove();
                    oov_candidates_eng_vocab.remove(elect_pre);
                    
                    if (oov_remaining.isEmpty()) break;
                    
                    best_similarity = -1;
                    best_candidate = null;
                    
                    final String elect_pre_tmp = elect_pre;
                    List<String> elect_pre_list = new ArrayList<String>(){{add(elect_pre_tmp);}};
                    for (String eng : oov_candidates_eng_vocab.keySet()) {
                        /*similarity between candidate and last elected word*/
                        double similarity = get_similarity(eng, elect_pre_list);
                        similarity = similarity + oov_candidates_eng_vocab.get(eng);
                        if (similarity > best_similarity) {
                            best_similarity = similarity;
                            best_candidate = eng;
                        }
                    }
                    elect_pre = best_candidate;
                }
            }
            
        /* for oov-specific candidate searching, two loops */
        } else if (candidateSource.equals("ug_dict")) {
            double best_similarity = -1;
            String best_oov = null;
            String best_candidate = null;

            /*--------translate the first oovs--------*/

            for (String oov : oov_candidates.keySet()) {
                /* oov_w_candidate stands for oov with candidates */
                System.out.println("[ID " + this.getId() + "] "+"Finding the best translation for "+oov+" under context "+context_words.toString()+"...");
                if (!oov_candidates.get(oov).containsKey(oov)) {
                    for (String candidate : oov_candidates.get(oov).keySet()) {   
                        //System.out.println("Computing similarity between ["+candidate+"] and "+context_words.toString()+"...");
                        double similarity = get_similarity(candidate, context_words);
                        oov_candidates.get(oov).put(candidate, similarity);
                        //System.out.println("Similarity: "+similarity);
                        if (similarity > best_similarity) {
                            best_similarity = similarity;
                            best_oov = oov;
                            best_candidate = candidate;
                        }
                    }
                }
                else {
                    oov_remaining.remove(oov);
                    oov_elect.put(oov, oov);
                }
            }

            String elect_pre = null;

            if (best_oov != null) {
                oov_remaining.remove(best_oov);
                oov_elect.put(best_oov, best_candidate);
                elect_pre = best_candidate;
                System.out.println("[ID " + this.getId() + "] "+"oov: "+best_oov);
                System.out.println("[ID " + this.getId() + "] "+"translation: "+best_candidate);
            }

            /*--------translate the rest of the oovs--------*/

            while (!oov_remaining.isEmpty()) {
                best_similarity = -1;
                best_oov = null;
                best_candidate = null;

                //Set<String> contexts_all = new HashSet<String>(){{addAll(context_words);addAll(oov_elect.values());}};
                List<String> contexts_all = new ArrayList<String>();
                for (String oov : oov_elect.keySet()) {
                    contexts_all.add(oov_elect.get(oov));
                }
                for (String context_word: context_words) {
                    contexts_all.add(context_word);
                }

                for (String oov : oov_remaining) {
                    System.out.println("[ID " + this.getId() + "] "+"Finding the best translation for "+oov+" under context "+contexts_all.toString()+"...");
                    final String elect_pre_tmp = elect_pre;
                    List<String> elect_pre_list = new ArrayList<String>(){{add(elect_pre_tmp);}};
                    for (String candidate : oov_candidates.get(oov).keySet()) {
                        //System.out.println("Computing similarity between ["+candidate+"] and "+contexts_all.toString()+"...");
                        double candidate_elect_similarity = get_similarity(candidate, elect_pre_list);
                        double candidate_context_similarity = candidate_elect_similarity + oov_candidates.get(oov).get(candidate);
                        oov_candidates.get(oov).put(candidate, candidate_context_similarity);
                        //System.out.println("Similarity: "+String.valueOf(candidate_context_similarity));
                        if (candidate_context_similarity > best_similarity) {
                            best_similarity = candidate_context_similarity;
                            best_oov = oov;
                            best_candidate = candidate;
                        }
                    }
                }
                oov_remaining.remove(best_oov);
                oov_elect.put(best_oov, best_candidate);
                elect_pre = best_candidate;
                System.out.println("[ID " + this.getId() + "] "+"oov: "+best_oov);
                System.out.println("[ID " + this.getId() + "] "+"translation: "+best_candidate);
            }
        }
		
        
		/* store the result */
		String[] tra_tok_wo_oov = new String[tra_tok.length];
		for (int pos=0; pos<tra_tok.length; pos++) {
			if (oov_pos.contains(pos)) {
				tra_tok_wo_oov[pos] = oov_elect.get(tra_tok[pos]);
			} else {
				tra_tok_wo_oov[pos] = tra_tok[pos];
			}
		}
        
		return tra_tok_wo_oov;
	}






	/* for eng_vocab:
     * guarantees that the candidate list is not empty
     * setting a score to be 10000 is to make sure that certain candidate is selected, thus avoiding unnecessary computation
     */
	public Map<String, Double> get_oov_candidates( 
        Set<String> oov_words_set, 
        Set<String> context_words_set
    ) {
		Map<String, Double> oov_candidates = new HashMap<String, Double>();
        for (String eng : eng_vocab) {
            /*don't allow context words to be in oov_candidates*/
            if (!context_words_set.contains(eng)) {
                if (!oov_words_set.contains(eng)) {
                    oov_candidates.put(eng, 0.0);
                } else {
                    /*if the oov word itself is in eng_vocab, set its score highest so it will get selected*/
                    oov_candidates.put(eng, 10000.0);
                }
            }
        }
        
		return oov_candidates;
	}






	/* for ug_dict:
     * guarantees that the candidate list is not empty
     * setting a score to be 10000 is to make sure that certain candidate is selected, thus avoiding unnecessary computation
     */
	public Map<String, Map<String, Double>> get_oov_candidates(
        Set<String> oov_words_set,
        boolean addAlignedOov 
    ) {
		Map<String, Map<String, Double>> oov_candidates = new HashMap<String, Map<String, Double>>();
        
		for (String oov_word : oov_words_set) {
			if (!ug_dict.containsKey(oov_word)) {
                /* if the oov word doesn't have any candidate, just use it for translation as it is; this also includes the scenario where the oov word itself is actually an english word, which ug_dict doesn't include */
				oov_candidates.put(oov_word, new HashMap<String, Double>(){{put(oov_word, 10000.0);}});
			} else {
				oov_candidates.put(oov_word, new HashMap<String, Double>(){{putAll(ug_dict.get(oov_word));}});
			}
		}
        
        /* add oov translation to the candidate list */
        if (addAlignedOov) {
            try {
                BufferedReader foa = new BufferedReader(new FileReader(oovAlignedFile));
                String l_oa;
                while ((l_oa = foa.readLine()) != null) {
                    l_oa = l_oa.replace("\n", "");
                    String[] oov_aligned_tok = l_oa.split("\t");
                    String oov_aligned = oov_aligned_tok[0];
                    /*only consider roov words in this sentence*/
                    if (oov_words_set.contains(oov_aligned)) {
                        /*NOT only consider the oov word that exists in oov_candidates*/
                        if (oov_candidates.containsKey(oov_aligned)) {
                            /*remove the existing candidate that is the same as the oov word*/
                            if (oov_candidates.get(oov_aligned).containsKey(oov_aligned)) {
                                oov_candidates.put(oov_aligned, new HashMap<String, Double>());
                                for (int i=1;i<oov_aligned_tok.length;i++) {
                                    String candidate = oov_aligned_tok[i];
                                    oov_candidates.get(oov_aligned).put(candidate, 0.0);
                                }
                            } else {
                                for (int i=1;i<oov_aligned_tok.length;i++) {
                                    String candidate = oov_aligned_tok[i];
                                    if (!oov_candidates.get(oov_aligned).containsKey(candidate)) {
                                        oov_candidates.get(oov_aligned).put(candidate, 0.0);
                                    }
                                }
                            }
                        } else {
                            oov_candidates.put(oov_aligned, new HashMap<String, Double>());
                            for (int i=1;i<oov_aligned_tok.length;i++) {
                                String candidate = oov_aligned_tok[i];
                                oov_candidates.get(oov_aligned).put(candidate, 0.0);
                            }
                        }
                    }
                }
            } catch (IOException e) {
			e.printStackTrace();
            }
        }
        
		return oov_candidates;
	}






	/* similarity between a word a list of words 
    * CAUTION: make sure word and word_set are properly handled and do not contain any non-english words so as to avoid unnecessary computation
    */
	public double get_similarity(String word, List<String> word_list) 
    {
        /* form the word_mat */
		String[][] word_mat = new String[word_list.size()][];
		for (int i=0; i<word_list.size(); i++) {
			word_mat[i] = new String[2];
			word_mat[i][0] = word;
			word_mat[i][1] = word_list.get(i);
		}

		/* the slowest part */
		double[][] pointwiseMutualInformation = getPointwiseMutualInformation(word_mat);
		
		double similarity = 0;
		for (int i=0; i<pointwiseMutualInformation.length; i++) {
			similarity = similarity + pointwiseMutualInformation[i][0];
		}
		return similarity;
	}






	/* phase 2 and 3 */
	public double[][] getPointwiseMutualInformation(String[][] wordsets) 
    {
		/* phase 2: create subsets from the topic words */
		SegmentationDefinition definitions[] = new SegmentationDefinition[wordsets.length];
		for (int i = 0; i < definitions.length; i++) {
			definitions[i] = segmentation.getSubsetDefinition(wordsets[i].length);
		}
		
		/* the slowest part */
		//tStart = System.currentTimeMillis();
		SubsetProbabilities probabilities[] = probEstimator.getProbabilities(wordsets, definitions);
		definitions = null;
		//tEnd = System.currentTimeMillis();
		//tDelta = tEnd - tStart;
		//elapsedSeconds = tDelta / 1000.0;
		//System.out.println("elapsed seconds: "+String.valueOf(elapsedSeconds)+"s");


		/* phase 3: confirmation measure - computing log-ratio measure (PMI) */
		double[][] pointwiseMutualInformationVector = new double[probabilities.length][];

		for (int i = 0; i < probabilities.length; i++) {
			double[] pointwiseMutualInformation = confirmation.calculateConfirmationValues(probabilities[i]);
			pointwiseMutualInformationVector[i] = new double[pointwiseMutualInformation.length];
			for (int j = 0; j < pointwiseMutualInformation.length; j++) {
                /*throttle pmi value to be non-negative*/
				if (pointwiseMutualInformation[j] < 0) {
					pointwiseMutualInformation[j] = 0;
				}
				pointwiseMutualInformationVector[i][j] = pointwiseMutualInformation[j];
			}
		}
	
		return pointwiseMutualInformationVector;
	}


}
