import sys 
import os
import math

dir_path = os.path.dirname(os.path.realpath(__file__))+"/"
sys.path.insert(0, dir_path+'../../')
from config import *
from utils import *
from oov_candidates_preprocessing import *
ocp = oov_candidates_preprocessing()

assert("ngram" in tmp_dir)
if not os.path.exists(tmp_dir):
    os.makedirs(tmp_dir)

from data_preprocessing import *

if not os.path.exists(train_ref_file):
    import shutil
    print("existing training data:")
    print("/home/ec2-user/kklab/Projects/lrlp/experiment_2017.08.08.il5-eng.y2r1.v1/translation/train/ref.eng.train.y2r1.v1")
    print("--------")
    print("/home/ec2-user/kklab/Projects/lrlp/experiment_2017.08.08.il6-eng.y2r1.v1/translation/train/ref.eng.train.y2r1.v1")
    print("--------")
    train_ref_file_old = input("train_ref_file_old: ")
    shutil.copy(train_ref_file_old, train_ref_file)

ngram = 4
lm_name = str(ngram)+"gram"
restrict_vocab = True

train_final = os.path.join(tmp_dir, os.path.basename(train_ref_file)+".final_"+lm_name)

if not os.path.exists(train_final):
    # set unk for non domain data
    train_non_domain_all_restrict_vocab = os.path.join(tmp_dir, os.path.basename(train_non_domain_all)+"_restrict_vocab")
    if not os.path.exists(train_non_domain_all_restrict_vocab):
        lexicon_xml = ocp.get_lexicon_xml_path()
        oov_candidates_all = ocp.get_oov_candidates_from_extracted(oov_candidates_dir, lexicon_xml)
        eng_vocab = ocp.get_eng_vocab(eng_vocab_file)
        vocab_set = build_vocab2(train_ref_file, dev_ref_file, unseq_ref_file, test_1best_file, test_oov_file, oov_candidates_all, eng_vocab)
        set_unk(train_non_domain_all, train_non_domain_all_restrict_vocab, vocab_set)
    else:
        print("UNK has been set in: "+train_non_domain_all_restrict_vocab)
    
    if restrict_vocab:
        train_non_domain_all = train_non_domain_all_restrict_vocab
    
    # data selection
    lm_in_domain_path = os.path.join(tmp_dir, "lm_"+lm_name+"_in_domain")
    lm_in_domain = train_lm(ngram, train_ref_file, lm_in_domain_path)
    print("----------------")
    
    train_non_domain_sub = os.path.join(tmp_dir, os.path.basename(train_non_domain_all)+".subset")
    lm_non_domain_subset_path = os.path.join(tmp_dir, "lm_"+lm_name+"_non_domain_subset_1")
    total_num_non_domain_sub = 500000
    random_sample(train_non_domain_all, total_num_non_domain_sub, train_non_domain_sub)
    lm_non_domain_subset = train_lm(ngram, train_non_domain_sub, lm_non_domain_subset_path)
    print("----------------")
    
    train_non_domain = train_non_domain_sub+".sample"
    lm_non_domain_path = os.path.join(tmp_dir, "lm_"+lm_name+"_non_domain")
    total_num_in_domain = get_file_length(train_ref_file)
    total_num_non_domain = random_sample(train_non_domain_sub, total_num_in_domain, train_non_domain)
    print("total_num_in_domain: "+str(total_num_in_domain))
    print("total_num_non_domain: "+str(total_num_non_domain))
    lm_non_domain = train_lm(ngram, train_non_domain, lm_non_domain_path)
    print("----------------")
    
    denominator = 2.0
    cutoff_num_non_domain = total_num_non_domain_sub/denominator
    score_sent = []
    with open(train_non_domain_sub) as f:
        for n, line in enumerate(f):
            sent = line.rstrip()
            cross_entropy_in_domain = get_cross_entropy(lm_in_domain, sent)
            cross_entropy_non_domain = get_cross_entropy(lm_non_domain, sent)
            cross_entropy_diff = -(cross_entropy_in_domain - cross_entropy_non_domain)
            if len(score_sent) < cutoff_num_non_domain:
                heapq.heappush(score_sent, (cross_entropy_diff, sent))
            else:
                spilled = heapq.heappushpop(score_sent, (cross_entropy_diff, sent))
    print("1/"+str(denominator)+" of the of non-domain training data has been loaded to heap.")
    print("----------------")
    
    perp = math.inf
    perp_subset, _, _, _ = get_perplexity(lm_non_domain_subset_path, dev_ref_file)
    print("perplexity of the language model on in-domain dev data trained on "+str(total_num_non_domain_sub)+" sentences selected from non-domain training data: "+str(perp_subset))
    while (perp_subset <= perp) and (denominator <= 2**10):
        train_non_domain_subset = train_non_domain_sub+"."+str(int(denominator))
        lm_non_domain_subset_path = os.path.join(tmp_dir, "lm_"+lm_name+"_non_domain_subset_"+str(denominator))
        with open(train_non_domain_subset, 'w') as fw:
            for pair in score_sent:
                fw.write(pair[1]+'\n')
        print("1/"+str(denominator)+" of the non-domain training data fetched.")
        lm_non_domain_subset = train_lm(ngram, train_non_domain_subset, lm_non_domain_subset_path)
        print("----------------")
    
        perp_w_oov, _, _, _ = get_perplexity(lm_non_domain_subset_path, dev_ref_file)
        perp = perp_subset
        perp_subset = perp_w_oov
        print("perplexity of in-domain dev data from the language model trained on "+str(cutoff_num_non_domain)+" sentences selected from non-domain training data: "+str(perp_w_oov))
        print("----------------")
    
        ctr = 0
        cutoff_num_non_domain = cutoff_num_non_domain/2
        while ctr < cutoff_num_non_domain:
            spilled = heapq.heappop(score_sent)
            ctr += 1
        print(str(ctr)+" sentences sampled as a subset.")
    
        denominator *= 2
    
    denominator_final = int(denominator/4)
    train_non_domain_subset_final = train_non_domain_sub+"."+str(denominator_final)
    print("----------------")
    
    merge_files([train_non_domain_subset_final, train_ref_file], train_final)
    print("train_final created at: "+train_final)
else:
    print("train_final exists at: "+train_final)
