import subprocess as sp
import os
import sys
import matplotlib.pyplot as plt
import xml.etree.ElementTree as et
import heapq
from random import shuffle

sys.path.insert(0, '/home/ec2-user/kklab/Projects/lrlp/scripts/oov_translate')
from config import *
from utils import *


def adapt_candidate_list_for_dclm(in_file_xml,\
                                  in_file_processed,\
                                  out_file_dclm):
    tree = et.parse(in_file_xml)
    root = tree.getroot()
    
    doc_ctr = 0
    sent_ctr = 0
    with open(out_file_dclm, 'w') as fw, open(in_file_processed) as f:
        for doc in root.findall('DOCUMENT'):
            for seg in doc.findall('SEGMENT'):
                line = f.readline().strip().lower()
                fw.write(line+"\n")
                sent_ctr += 1
            fw.write('=\n')
            doc_ctr += 1
    print("read from: "+in_file_processed)
    print("written to: "+out_file_dclm)
    print("number of sentences: "+str(sent_ctr))
    print("number of documents: "+str(doc_ctr))
    return sent_ctr, doc_ctr


def adapt_in_domain_for_dclm(in_file_xml, \
                             in_file_processed, \
                             out_file_dclm, \
                             num_concat, \
                             rid_func, \
                             stem_text):
    '''
    (1) add document boundaries, 
    (2) remove function words and punctuations,
    (3) lemmatize/stem the text
    params:
        in_file_xml: input file in the original xml format
        in_file_processed: input file with tokenized and recased sentences
        out_file_dclm: output file with document boundaries
        num_concat: number of sentences concatenated together
        rid_func: whether to remove function words
        stem_text: whether to stem the text
    return:
        sent_ctr: number of sentences
        doc_ctr: number of articles
    '''
    tree = et.parse(in_file_xml)
    root = tree.getroot()
    
    doc_ctr = 0
    sent_ctr = 0
    with open(out_file_dclm, 'w') as fw, open(in_file_processed) as f:
        for doc in root.findall('DOCUMENT'):
            concat_ctr = 0
            sent_concat = []
            for seg in doc.findall('SEGMENT'):
                l_tra = f.readline()
                tra_tok = l_tra.strip().split(' ')
                ### strip function words and punctuation
                if rid_func:
                    tra_tok_prune = [tok for tok in tra_tok \
                                     if tok.lower() not in function_words \
                                     and tok not in string.punctuation]
                else:
                    tra_tok_prune = list(tra_tok)
                    
                ### lemmatize each word
                if stem_text:
                    tra_tok_prune = [lmtzr.lemmatize(tok) for tok in tra_tok_prune]
                
                ### lower case
                tra_tok_prune = [tok.lower() for tok in tra_tok_prune]
                    
                l_res = ' '.join(tra_tok_prune)
                if l_res != '':
                    sent_concat.append(l_res)
                    concat_ctr += 1
                    if concat_ctr == num_concat:
                        fw.write(' '.join(sent_concat)+'\n')
                        sent_concat = []
                        concat_ctr = 0
                    sent_ctr += 1
                ### make sure the reference and hypothesis have the same number of sentences
                else:
                    sent_concat.append(l_res)
                    concat_ctr += 1
                    if concat_ctr == num_concat:
                        fw.write(' '.join(sent_concat)+'\n')
                        sent_concat = []
                        concat_ctr = 0
                    sent_ctr += 1
                        
            ### document boundary recognized by DCLM
            fw.write('=\n')
            doc_ctr += 1

    print("read from: "+in_file_processed)
    print("written to: "+out_file_dclm)
    print("number of sentences: "+str(sent_ctr))
    print("number of documents: "+str(doc_ctr))
    return sent_ctr, doc_ctr


def adapt_non_domain_for_dclm(in_file_raw, \
                              out_file_dclm, \
                              num_concat, \
                              rid_func, \
                              stem_text):
    '''
    (1) add document boundaries, 
    (2) remove function words and punctuations,
    (3) lemmatize/stem the text
    params:
        in_file_raw: input file in raw format
        out_file_dclm: output file with document boundaries
        num_concat: number of sentences concatenated together
        rid_func: whether to remove function words
        stem_text: whether to stem the text
    return:
        sent_ctr: number of sentences
        doc_ctr: number of articles
    '''
    doc_ctr = 0
    sent_ctr = 0
    
    concat_ctr = 0
    sent_concat = []
    with open(out_file_dclm, 'w') as fw, open(in_file_raw) as f:
        for line in f:
            if len(line.split('=')) == 3 and line.startswith(' = '):
                ### marks the start of a document
                ### otherwise," = = " marks paragraphs, 
                ### " = = = " marks sub paragraphs
                if doc_ctr != 0:
                    fw.write('=\n')
                doc_ctr += 1
                concat_ctr = 0
                sent_concat = []
                continue
            if line.strip() == '' or len(line.split('=')) > 3:
                continue
            sents = line.strip().split(' . ')
            for sent in sents:
                tra_tok = sent.strip().split(' ')
                ### strip function words and punctuation
                if rid_func:
                    tra_tok_prune = [tok for tok in tra_tok \
                                     if tok.lower() not in function_words \
                                     and tok not in string.punctuation]
                else:
                    tra_tok_prune = list(tra_tok)
                
                ### lower case
                tra_tok_prune = [tok.lower() for tok in tra_tok_prune]
                
                ### lemmatize each word
                if stem_text:
                    tra_tok_prune = [lmtzr.lemmatize(tok) for tok in tra_tok_prune]
                    
                l_res = ' '.join(tra_tok_prune)
                if l_res != '':
                    sent_concat.append(l_res)
                    concat_ctr += 1
                    if concat_ctr == num_concat:
                        fw.write(' '.join(sent_concat)+'\n')
                        sent_concat = []
                        concat_ctr = 0
                    sent_ctr += 1
    print("read from: "+in_file_raw)
    print("written to: "+out_file_dclm)
    print("number of sentences: "+str(sent_ctr))
    print("number of documents: "+str(doc_ctr))
    return sent_ctr, doc_ctr

################################################################

def get_seed_for_data_selection2(dev_ref, test_nbest, proc_seed_for_lm_training_selection):
    # parse test nbest
    test_nbest_text = tmp_dir+"text_"+test_nbest.split('/')[-1]
    test_nbest_idx = tmp_dir+"idx_"+test_nbest.split('/')[-1]
    if not os.path.exists(test_nbest_text) or os.stat(test_nbest_text).st_size==0:
        nbest_parser(test_nbest, test_nbest_text, test_nbest_idx)
        print("test_nbest_text created at: "+test_nbest_text+"\n")
    else:
        print("test_nbest_text exists at: "+test_nbest_text+"\n")
        
    # merge test nbest and dev ref
    seed_for_lm_training_selection = tmp_dir+"seed."+t+".dev_test."+yrv
    if not os.path.exists(seed_for_lm_training_selection) or os.stat(seed_for_lm_training_selection).st_size==0:
        merge_files([dev_ref, test_nbest_text], seed_for_lm_training_selection)
        print("seed_for_lm_training_selection created at: "+seed_for_lm_training_selection+"\n")
    else:
        print("seed_for_lm_training_selection exists at: "+seed_for_lm_training_selection+"\n")
    
    # lemmatize, rid func and lowercase the merged file
    #proc_seed_for_lm_training_selection = tmp_dir+"stemmed_seed."+t+".dev_test."+yrv
    if not os.path.exists(proc_seed_for_lm_training_selection) or os.stat(proc_seed_for_lm_training_selection).st_size==0:
        lemmatize_and_rid_func_and_lowercase(seed_for_lm_training_selection, proc_seed_for_lm_training_selection)
        print("proc_seed_for_lm_training_selection created at: "+proc_seed_for_lm_training_selection+"\n")
    else:
        print("proc_seed_for_lm_training_selection exists at: "+proc_seed_for_lm_training_selection+"\n")
    
    
def get_seed_for_data_selection(dev_in_domain_xml, dev_in_domain, test_in_domain_xml, test_in_domain_hyp):
    # -------- hyperparameters for adaptation of raw texts for data selection --------
    ### number of sentences concatenated
    num_concat = 1
    ### whether to strip out function words and punctuations
    rid_func = True
    ### whether to lemmatize the text
    stem_text = True

    ### ----------------
    ### don't train an LM on the in-domain data (word order messed up)
    ### do train an LM on domain-relevant data (selected from WikiText-103)

    ### in order to select domain-relevant data from WikiText-103, 
    ### in-domain data =
    ### {bounded in-domain dev data (reference)} + 
    ### {bounded in-domain test data (hypothesis)}

    ### for each document in the in-domain data and non-domain data (WikiText-103)
    ### strip out function words and lemmatize the text
    ### ----------------

    in_domain_data = tmp_dir+\
                    train_in_domain.split('/')[-1]+\
                    "_combined"+\
                    "_"+str(num_concat)+\
                    "_"+str(rid_func)+\
                    "_"+str(stem_text)

    if not os.path.exists(in_domain_data):
        # -------- input: in-domain dev xml (reference): dev_in_domain_xml --------
        # -------- output: in-domain dev documents (reference): dev_in_domain_adapt --------
        print("----------------")

        dev_in_domain_adapt = tmp_dir+\
                              dev_in_domain.split('/')[-1]+\
                              "_"+str(num_concat)+\
                              "_"+str(rid_func)+\
                              "_"+str(stem_text)
        print("dev_in_domain_adapt: "+dev_in_domain_adapt+"\n")


        #if not os.path.exists(dev_in_domain_adapt):    
        sent_ctr_dev_in_domain, doc_ctr_dev_in_domain = \
        adapt_in_domain_for_dclm(dev_in_domain_xml, \
                               dev_in_domain, \
                               dev_in_domain_adapt, \
                               num_concat, \
                               rid_func, \
                               stem_text)
        print("\n")


        # -------- input: in-domain test xml (hypothesis): test_in_domain_xml --------
        # -------- output: in-domain test documents (hypothesis): test_tra_file_adapt -------- 
        print("----------------")

        test_tra_file_adapt = tmp_dir+\
                         test_in_domain_hyp.split('/')[-1]+\
                         "_"+str(num_concat)+\
                         "_"+str(rid_func)+\
                         "_"+str(stem_text)
        print("test_tra_file_adapt: "+test_tra_file_adapt+"\n")


        #if not os.path.exists(test_tra_file_adapt):
        sent_ctr_test_tra_file, doc_ctr_test_tra_file = \
        adapt_in_domain_for_dclm(test_in_domain_xml, \
                                 test_in_domain_hyp, \
                                 test_tra_file_adapt, \
                                 num_concat, \
                                 rid_func, \
                                 stem_text)
        print("\n")

        # -------- input: in-domain dev documents (reference): dev_in_domain_adapt --------
        # -------- input: in-domain test documents (hypothesis): test_tra_file_adapt --------
        # -------- output: in-domain seed documents: in_domain_data --------
        print("----------------")

        sp.run(["touch", in_domain_data])
        sp.run(["rm", in_domain_data])
        sp.run(["touch", in_domain_data])
        sp.run(["cat", test_tra_file_adapt, ">>", in_domain_data])
        sp.run(["cat", dev_in_domain_adapt, ">>", in_domain_data])
        print("test_tra_file_adapt: "+test_tra_file_adapt+"\n")
        print("dev_in_domain_adapt: "+dev_in_domain_adapt+"\n")
        print("in_domain_data: "+in_domain_data+"\n")
    else:
        print("in-domain combined set exists.")
    return in_domain_data


def get_target_for_data_selection2(in_file_raw, out_file, out_file_proc):

    lmtzr = WordNetLemmatizer()
    lemmatize = lru_cache(maxsize=50000)(lmtzr.lemmatize)
    #lmtzr_cache = dict()
    
    doc_ctr = 0
    sent_ctr = 0
    
    if not os.path.exists(out_file_proc) or os.stat(out_file_proc).st_size==0:
        with open(in_file_raw) as f, open(out_file, 'w') as fw, open(out_file_proc, 'w') as fp:
            for line in f:
                # start of an article/document
                if len(line.split('=')) == 3 and line.startswith(' = '):

                    if doc_ctr != 0:
                        fw.write('=\n')
                        fp.write('=\n')
                    doc_ctr += 1
                    if doc_ctr%1000 == 0:
                        print(str(doc_ctr)+" documents processed.")
                    continue
                # start of a paragraph or a sub paragraph
                if line.strip() == '' or len(line.split('=')) > 3:
                    continue
                # body of an article/document
                sents = line.strip().split(' . ')
                for sent in sents:
                    tra_tok = sent.strip().split(' ')
                    if tra_tok[-1] == ".":
                        tra_tok = tra_tok[:-1]
                    
                    tra_tok_proc = [lemmatize(tok.lower()) \
                                    for tok in tra_tok \
                                    if tok.lower() not in function_words \
                                    and tok not in punctuations]
                    
                    fw.write(" ".join(tra_tok)+" .\n")
                    fp.write(" ".join(tra_tok_proc)+" .\n")

                    sent_ctr += 1
        print(str(sent_ctr)+" sentences processed.\n")
        print(str(doc_ctr)+" documents processed.\n")
        print("out_file created at: "+out_file+"\n")
        print("out_file_proc created at: "+out_file_proc+"\n")
    else:
        print("out_file exists at: "+out_file+"\n")
        print("out_file_proc exists at: "+out_file_proc+"\n")



def get_target_for_data_selection(wiki_dump_train):
    # -------- hyperparameters for adaptation of raw texts for data selection --------
    ### number of sentences concatenated
    num_concat = 1
    ### whether to strip out function words and punctuations
    rid_func = True
    ### whether to lemmatize the text
    stem_text = True
    
    # -------- input: non-domain data (WikiText-103): wiki_dump_train --------
    # -------- output: non-domain documents: train_non_domain_adapt --------
    print("----------------")

    train_non_domain_adapt = wiki_dump_dir+\
                             wiki_dump_train.split('/')[-1]+\
                             "_"+str(num_concat)+\
                             "_"+str(rid_func)+\
                             "_"+str(stem_text)
    #print("train_non_domain_adapt: "+train_non_domain_adapt+"\n")


    if not os.path.exists(train_non_domain_adapt) or os.stat(train_non_domain_adapt).st_size==0:
        sent_ctr_train_non_domain, doc_ctr_train_non_domain = \
        adapt_non_domain_for_dclm(wiki_dump_train, \
                                train_non_domain_adapt, \
                                num_concat, \
                                rid_func, \
                                stem_text)
        print("proc_target_for_lm_training_selection created at: "+train_non_domain_adapt+"\n")
    else:
        print("proc_target_for_lm_training_selection exists at: "+train_non_domain_adapt+"\n")
    return train_non_domain_adapt



def select_doc_with_high_jaccard_idx(proc_seed_in_domain, doc_non_domain, proc_doc_non_domain, num_of_doc_to_select, dclm_vocab, selected_non_domain_doc_dir, selected_non_domain_doc_with_unk_dir):
    '''
    params:
        selected_non_domain_doc_dir
        selected_non_domain_doc_with_unk_dir
        proc_seed_in_domain
        doc_non_domain
        proc_doc_non_domain
    return:
        selected_non_domain_doc_dir
    '''
    proc_seed_in_domain_wordcount = int(sp.run(["wc", "-w", proc_seed_in_domain], stdout=sp.PIPE).stdout.decode('utf-8').split(' ')[0])
       
    if os.listdir(selected_non_domain_doc_dir) == []:
        # form an in-domain word set
        in_domain_tok = set()
        with open(proc_seed_in_domain) as f:
            for line in f:
                l = line.strip().split(' ')
                for tok in l:
                    in_domain_tok.add(tok)
        
        jaccard_idx_rank = []
        doc_cache = ''
        proc_doc_non_domain_wordcount = 0
        intersection_wordcount = 0
        doc_ctr = 0
        sent_ctr = 0
        
        # rank documents by jaccard index
        with open(doc_non_domain) as fnp, open(proc_doc_non_domain) as fp:
            for line in fp:
                line_np = fnp.readline()
                
                if line.strip() != "=":
                    doc_cache+=line_np
                    
                    l = line.split(' ')
                    for tok in l:
                        proc_doc_non_domain_wordcount += 1
                        if tok in in_domain_tok:
                            intersection_wordcount += 1
                else:
                    assert(line_np.strip() == "=")
                    if sent_ctr == 0:
                        sent_ctr += 1
                        continue
                    
                    jaccard_idx = intersection_wordcount*1.0/(proc_seed_in_domain_wordcount+proc_doc_non_domain_wordcount-intersection_wordcount)
                    if len(jaccard_idx_rank) < num_of_doc_to_select:
                        heapq.heappush(jaccard_idx_rank, (-jaccard_idx, doc_cache))
                    else:
                        heapq.heappushpop(jaccard_idx_rank, (-jaccard_idx, doc_cache))
                    doc_cache = ''
                    proc_doc_non_domaon_wordcount = 0
                    intersection_word_count = 0
                    doc_ctr += 1
                    sent_ctr += 1
                    
                    if doc_ctr % 1000 == 0:
                        print(str(doc_ctr)+" documents ranked.")
            
            # handle the last document
            if doc_cache != '':
                jaccard_idx = intersection_wordcount*1.0/(proc_seed_in_domain_wordcount+proc_doc_non_domain_wordcount-intersection_wordcount)
                if len(jaccard_idx_rank) < num_of_doc_to_select:
                    heapq.heappush(jaccard_idx_rank, (-jaccard_idx, doc_cache))
                else:
                    heapq.heappushpop(jaccard_idx_rank, (-jaccard_idx, doc_cache))
        
        # pop out documents with highest jaccard indices
        doc_ptr = 0
        while jaccard_idx_rank != []:
            item = heapq.heappop(jaccard_idx_rank)
            jaccard_idx = -item[0]
            doc = item[1]
            # writing to file is the slowest part
            with open(selected_non_domain_doc_dir+str(doc_ptr), 'w') as fw, open(selected_non_domain_doc_with_unk_dir+str(doc_ptr), 'w') as fu:
                fw.write(doc)
                fu.write(set_unk_for_string(doc, dclm_vocab))
            doc_ptr += 1
            
            if doc_ptr % 1000 == 0:
                print(str(doc_ptr)+" documents written to file.")
        
        print("selected_non_domain_doc_dir created at: "+selected_non_domain_doc_dir+"\n")
        print("selected_non_domain_doc_with_unk_dir created at: "+selected_non_domain_doc_with_unk_dir+"\n")
    else:
        print("selected_non_domain_doc_dir exists at: "+selected_non_domain_doc_dir+"\n")
        print("selected_non_domain_doc_with_unk_dir exists at: "+selected_non_domain_doc_with_unk_dir+"\n")

        
def select_doc_with_high_jaccard_idx2(proc_seed_in_domain, doc_non_domain, proc_doc_non_domain, num_of_doc_to_select, train_ref_file, dev_ref_file, unseq_ref_file, test_1best_file, test_oov_file, oov_candidates_all, eng_vocab, selected_non_domain_doc_file, selected_non_domain_doc_with_unk_file):
    '''
    params:
        proc_seed_in_domain
        doc_non_domain
        proc_doc_non_domain
        num_of_doc_to_select
        dclm_vocab
        selected_non_domain_doc_file
        selected_non_domain_doc_with_unk_file
    '''
    proc_seed_in_domain_wordcount = int(sp.run(["wc", "-w", proc_seed_in_domain], stdout=sp.PIPE).stdout.decode('utf-8').split(' ')[0])
       
    if not os.path.exists(selected_non_domain_doc_with_unk_file):
        # get dclm_vocab
        dclm_vocab = build_vocab2(
            train_ref_file, \
            dev_ref_file, \
            unseq_ref_file, \
            test_1best_file, \
            test_oov_file, \
            oov_candidates_all, \
            eng_vocab)
        
        # form an in-domain word set
        in_domain_tok = set()
        with open(proc_seed_in_domain) as f:
            for line in f:
                l = line.strip().split(' ')
                for tok in l:
                    in_domain_tok.add(tok)
        
        jaccard_idx_rank = []
        doc_cache = ''
        proc_doc_non_domain_wordcount = 0
        intersection_wordcount = 0
        doc_ctr = 0
        sent_ctr = 0
        
        # rank documents by jaccard index
        with open(doc_non_domain) as fnp, open(proc_doc_non_domain) as fp:
            for line in fp:
                line_np = fnp.readline()
                
                if line.strip() != "=":
                    doc_cache+=line_np
                    
                    l = line.split(' ')
                    for tok in l:
                        proc_doc_non_domain_wordcount += 1
                        if tok in in_domain_tok:
                            intersection_wordcount += 1
                else:
                    assert(line_np.strip() == "=")
                    if sent_ctr == 0:
                        sent_ctr += 1
                        continue
                    
                    jaccard_idx = intersection_wordcount*1.0/(proc_seed_in_domain_wordcount+proc_doc_non_domain_wordcount-intersection_wordcount)
                    if len(jaccard_idx_rank) < num_of_doc_to_select:
                        heapq.heappush(jaccard_idx_rank, (-jaccard_idx, doc_cache))
                    else:
                        heapq.heappushpop(jaccard_idx_rank, (-jaccard_idx, doc_cache))
                    doc_cache = ''
                    proc_doc_non_domaon_wordcount = 0
                    intersection_word_count = 0
                    doc_ctr += 1
                    sent_ctr += 1
                    
                    if doc_ctr % 1000 == 0:
                        print(str(doc_ctr)+" documents ranked.")
            
            # handle the last document
            if doc_cache != '':
                jaccard_idx = intersection_wordcount*1.0/(proc_seed_in_domain_wordcount+proc_doc_non_domain_wordcount-intersection_wordcount)
                if len(jaccard_idx_rank) < num_of_doc_to_select:
                    heapq.heappush(jaccard_idx_rank, (-jaccard_idx, doc_cache))
                else:
                    heapq.heappushpop(jaccard_idx_rank, (-jaccard_idx, doc_cache))
        
        # pop out documents with highest jaccard indices
        doc_ptr = 0
        with open(selected_non_domain_doc_file, 'w') as fw, open(selected_non_domain_doc_with_unk_file, 'w') as fwu:
            while jaccard_idx_rank != []:
                item = heapq.heappop(jaccard_idx_rank)
                jaccard_idx = -item[0]
                doc = item[1]
                
                # writing to file is the slowest part
                fw.write(doc)
                fw.write("=\n")
                fwu.write(set_unk_for_string(doc, dclm_vocab))
                fwu.write("=\n")
                doc_ptr += 1

                if doc_ptr % 1000 == 0:
                    print(str(doc_ptr)+" documents written to file.")
        
        print("selected_non_domain_doc_file created at: "+selected_non_domain_doc_file+"\n")
        print("selected_non_domain_doc_with_unk_file created at: "+selected_non_domain_doc_with_unk_file+"\n")
    else:
        print("selected_non_domain_doc_file exists at: "+selected_non_domain_doc_file+"\n")
        print("selected_non_domain_doc_with_unk_file exists at: "+selected_non_domain_doc_with_unk_file+"\n")
        

def split_doc_into_files(in_domain_training_file_list, in_domain_doc_dir):
    
    if os.listdir(in_domain_doc_dir) == []:
        sent_ctr = 0
        doc_ctr = 0
        for in_domain_training_file in in_domain_training_file_list:
            if os.path.exists(in_domain_training_file):
                fw = open(in_domain_doc_dir+str(doc_ctr), 'w')
                with open(in_domain_training_file) as f:
                    for line in f:
                        if line.strip()!='=':
                            fw.write(line)
                            sent_ctr += 1
                        else:
                            if sent_ctr == 0:
                                continue
                            fw.close()
                            doc_ctr += 1
                            sent_ctr = 0
                            
                            if doc_ctr % 1000 == 0:
                                print(str(doc_ctr)+" documents written to file.")
                            fw = open(in_domain_doc_dir+str(doc_ctr), 'w')
                fw.close()
                if sent_ctr == 0:
                    os.remove(in_domain_doc_dir+str(doc_ctr))
        print("in_domain_doc_dir created at: "+in_domain_doc_dir+"\n")
    else:
        print("in_domain_doc_dir exists at: "+in_domain_doc_dir+"\n")


def combine_selected_non_domain_and_in_domain_doc(selected_non_domain_doc_dir, in_domain_doc_dir, combined_selected_non_domain_and_in_domain_doc):
    
    if not os.path.exists(combined_selected_non_domain_and_in_domain_doc) or os.stat(combined_selected_non_domain_and_in_domain_doc).st_size==0:
        selected_non_domain_docs = os.listdir(selected_non_domain_doc_dir)
        selected_non_domain_docs_num = len(selected_non_domain_docs)
        in_domain_docs = os.listdir(in_domain_doc_dir)
        in_domain_docs_num = len(in_domain_docs)
        doc_order = list(range(selected_non_domain_docs_num+in_domain_docs_num))
        random.shuffle(doc_order)
        
        # randomize the document order
        file_list = []
        for idx in doc_order:
            file_name = selected_non_domain_doc_dir+str(idx)
            if idx >= selected_non_domain_docs_num:
                file_name = in_domain_doc_dir+str(idx-selected_non_domain_docs_num)
            file_list.append(file_name)
    
        merge_files_with_boundary(file_list, combined_selected_non_domain_and_in_domain_doc)
        
        print("combined_selected_non_domain_and_in_domain_doc created at: "+combined_selected_non_domain_and_in_domain_doc+"\n")
    else:
        print("combined_selected_non_domain_and_in_domain_doc exists at: "+combined_selected_non_domain_and_in_domain_doc+"\n")
    
    
def establish_document_rank(in_domain_data, train_non_domain_adapt):
    # -------- establish ranking by Jaccard index --------
    # -------- input: in-domain seed documents: in_domain_data --------
    # -------- input: non-domain documents: train_non_domain_adapt --------
    # -------- output: jaccard index ranks of non-domain documents: jaccard_idx_rank_file --------
    print("----------------")

    word_c_in_domain = sp.run(["wc", "-w", in_domain_data], stdout=sp.PIPE).stdout.decode('utf-8')
    ### number of words in the in_domain data
    word_c_in_domain = int(word_c_in_domain.split(' ')[0])
    print("word_c_in_domain: "+str(word_c_in_domain)+"\n")
    
    jaccard_idx_rank_file = tmp_dir+"jaccard_idx_rank"

    if not os.path.exists(jaccard_idx_rank_file) or os.stat(jaccard_idx_rank_file).st_size==0:
        ### put in domain data in a set for computing intersection later
        in_domain_tok = set()
        with open(in_domain_data) as f:
            for line in f:
                l = line.strip().split(' ')
                for tok in l:
                    in_domain_tok.add(tok)

        ### for each pair of documents in the in-domain data and WikiText-103, 
        ### compute the Jaccard index (normalized count of words in the intersection)
        jaccard_idx_rank = []
        word_c_non_domain = 0
        intersection_c = 0
        file_ctr = 0

        with open(train_non_domain_adapt) as f:
            for line in f:
                ### not document boundary
                if line.strip() != "=":
                    l = line.split(' ')
                    for tok in l:
                        word_c_non_domain += 1
                        if tok in in_domain_tok:
                            intersection_c += 1
                ### document boundary
                else:
                    if file_ctr % 2000 == 0:
                        print(str(file_ctr)+" articles processed.")

                    jaccard_idx = intersection_c*1.0/(word_c_in_domain+word_c_non_domain-intersection_c)
                    ### since heapq is a min heap and we need documents with large jaccard_idx, push negative jaccard_idx into the heap
                    heapq.heappush(jaccard_idx_rank, (-jaccard_idx, file_ctr))
                    word_c_non_domain = 0
                    intersection_c = 0
                    file_ctr += 1

        with open(jaccard_idx_rank_file, 'w') as fw:
            while jaccard_idx_rank != []:
                item = heapq.heappop(jaccard_idx_rank)
                jaccard_idx = -item[0]
                file_num = item[1]
                fw.write(str(jaccard_idx)+" "+str(file_num)+"\n")
        print("document ranking by Jaccard index created at: "+jaccard_idx_rank_file+"\n")
    else:
        print("document ranking by Jaccard index exists at: "+jaccard_idx_rank_file+"\n")
    return jaccard_idx_rank_file


def select_document_index_with_high_rank(jaccard_idx_rank_file):
    # -------- select documents with high ranking --------
    # -------- input: jaccard index ranks of non-domain documents: jaccard_idx_rank_file --------
    # -------- output: indices of selected non-domain documents: train_non_domain_select
    print("----------------")

    jaccard_idx_sorted = []
    train_non_domain_select = set()
    ctr = 0
    with open(jaccard_idx_rank_file) as f:
        for line in f:
            jaccard_idx = float(line.strip().split(' ')[0])
            file_num = int(line.strip().split(' ')[1])
            jaccard_idx_sorted.append(jaccard_idx)
            train_non_domain_select.add(file_num)

            if ctr > 10000 or jaccard_idx < 0.15:
                break
            ctr += 1
    plt.plot(jaccard_idx_sorted)
    plt.xlabel("documents")
    plt.ylabel("Jaccard index")
    print(str(len(train_non_domain_select))+" document selected")
    return train_non_domain_select


def merge_selected_documents_with_in_domain_documents(train_in_domain_xml, train_in_domain, wiki_dump_train, train_non_domain_select):
    ### number of sentences concatenated
    num_concat = 1
    ### whether to strip out function words and punctuations
    rid_func = False
    ### whether to lemmatize the text
    stem_text = False

    # -------- prepare training data for training DCLM --------
    # -------- input: in-domain train xml (reference): train_in_domain_xml --------
    # -------- output: in-domain train documents (reference): train_in_domain_adapt --------
    print("----------------")

    train_in_domain_adapt = tmp_dir+\
                            train_in_domain.split('/')[-1]+\
                            "_"+str(num_concat)+\
                            "_"+str(rid_func)+\
                            "_"+str(stem_text)
    print("train_in_domain_adapt: "+train_in_domain_adapt+"\n")


    if not os.path.exists(train_in_domain_adapt):
        sent_ctr_train_in_domain, doc_ctr_train_in_domain = \
        adapt_in_domain_for_dclm(train_in_domain_xml, \
                               train_in_domain, \
                               train_in_domain_adapt, \
                               num_concat, \
                               rid_func, \
                               stem_text)
        print("\n")


    # -------- input: non-domain data (WikiText-103): wiki_dump_train --------
    # -------- output: non-domain documents (new config, w/o vocab pruning): train_non_domain_adapt --------
    print("----------------")

    train_non_domain_adapt = wiki_dump_dir+\
                             wiki_dump_train.split('/')[-1]+\
                             "_"+str(num_concat)+\
                             "_"+str(rid_func)+\
                             "_"+str(stem_text)
    print("train_non_domain_adapt: "+train_non_domain_adapt+"\n")


    if not os.path.exists(train_non_domain_adapt):
        sent_ctr_train_non_domain, doc_ctr_train_non_domain = \
        adapt_non_domain_for_dclm(wiki_dump_train, \
                                train_non_domain_adapt, \
                                num_concat, \
                                rid_func, \
                                stem_text)
        print("\n")
    else:
        print("train_non_domain_adapt exists at: "+train_non_domain_adapt)


    # -------- input: in-domain train documents (reference): train_in_domain_adapt --------
    # -------- input: non-domain documents (new config, w/o vocab pruning): train_non_domain_adapt --------
    # -------- input: indices of selected non-domain documents: train_non_domain_select --------
    # -------- output: combined training documents: train_final
    print("----------------")

    train_final = tmp_dir+"train_final"

    if not os.path.exists(train_final):
        train_non_domain_adapt_select = tmp_dir+"train_non_domain_adapt_select"
        print("train_non_domain_adapt_select: "+train_non_domain_adapt_select)

        file_ctr = 0

        vocab_set = build_vocab()

        with open(train_non_domain_adapt) as f, \
        open(train_non_domain_adapt_select, 'w') as fw:
            for line in f:
                ### not document boundary
                if line.strip() != "=":
                    ### only use the selected documents
                    if file_ctr in train_non_domain_select:
                        l = line.strip().split(' ')
                        for i in range(len(l)):
                            if l[i] not in vocab_set:
                                l[i] = "UNK"
                        fw.write(' '.join(l)+"\n")
                ### document boundary
                else:
                    if file_ctr % 2000 == 0:
                        print(str(file_ctr)+" articles processed.")
                    if file_ctr in train_non_domain_select:
                        fw.write("=\n")

                    file_ctr += 1


        merge_files_with_boundary([train_in_domain_adapt, train_non_domain_adapt_select], train_final)
        print("train_final: "+train_final)
    else:
        print("train_final exists at: "+train_final)
    return train_final                    



def get_dev_documents(dev_in_domain_xml, dev_in_domain):
    ### number of sentences concatenated
    num_concat = 1
    ### whether to strip out function words and punctuations
    rid_func = False
    ### whether to lemmatize the text
    stem_text = False

    # -------- prepare dev data for training DCLM --------
    # -------- input: in-domain dev xml (reference): dev_in_domain_xml --------
    # -------- output: in-domain dev documents (reference): dev_in_domain_adapt --------
    print("----------------")

    dev_in_domain_adapt = tmp_dir+\
                          dev_in_domain.split('/')[-1]+\
                          "_"+str(num_concat)+\
                          "_"+str(rid_func)+\
                          "_"+str(stem_text)
    print("dev_in_domain_adapt: "+dev_in_domain_adapt+"\n")

    if not os.path.exists(dev_in_domain_adapt):    
        sent_ctr_dev_in_domain, doc_ctr_dev_in_domain = \
        adapt_in_domain_for_dclm(dev_in_domain_xml, \
                               dev_in_domain, \
                               dev_in_domain_adapt, \
                               num_concat, \
                               rid_func, \
                               stem_text)
        print("\n")
    return dev_in_domain_adapt


def get_test_documents(in_domain_xml, in_domain_hyp):
    ### number of sentences concatenated
    num_concat = 1
    ### whether to strip out function words and punctuations
    rid_func = False
    ### whether to lemmatize the text
    stem_text = False
    
    # -------- prepare test data (tra_file) --------
    # -------- input: in-domain dev raw (hypothesis): dev_in_domain_hyp --------
    # -------- output: in-domain dev documents (hypothesis): dev_tra_file_adapt --------
    print("----------------")

    tra_file_adapt = tmp_dir+\
                     in_domain_hyp.split('/')[-1]+\
                     "_"+str(num_concat)+\
                     "_"+str(rid_func)+\
                     "_"+str(stem_text)
    print("dev_tra_file_adapt: "+tra_file_adapt+"\n")

    sent_ctr_tra_file, doc_ctr_tra_file = \
    adapt_in_domain_for_dclm(in_domain_xml, \
                             in_domain_hyp, \
                             tra_file_adapt, \
                             num_concat, \
                             rid_func, \
                             stem_text)
    print("\n")
    return tra_file_adapt





def train_charlm(combined_selected_non_domain_and_in_domain_doc, dev_ref_file, \
                 charlm_num_layer, charlm_input_dim, charlm_hidden_dim, \
                 charlm_model_file, charlm_dict_file, charlm_ppl_file, charlm_log_file):
    lr = 0.1

    model_dir = tmp_dir+"models/"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    gpu_id = 3
    
    forward_mem = 9192
    backward_mem = 1024
    param_mem = 512

    cmd0 = "cd "+os.getcwd()+"; "
    cmd0 += "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"+boost_dir+"; "
    cmd0 += "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"+dynet_dir+"; "

    cmd = cmd0 + "./charlm "+\
    "--dynet-gpu-ids "+str(gpu_id)+" "\
    "--dynet-mem "+",".join([str(forward_mem),str(backward_mem),str(param_mem)])+" "+\
    "train "+\
    combined_selected_non_domain_and_in_domain_doc+" "+\
    dev_ref_file+" "+\
    str(charlm_num_layer)+" "+\
    str(charlm_input_dim)+" "+\
    str(charlm_hidden_dim)+" "+\
    charlm_model_file+" "+\
    charlm_dict_file+" "+\
    charlm_ppl_file+" "+\
    charlm_log_file+" "+\
    str(lr)
    
    print(cmd+"\n")

    
def train_dclm(combined_selected_non_domain_with_unk_and_in_domain_doc, dev_ref_file, model_type, \
               dclm_num_layer, dclm_input_dim, dclm_hidden_dim, dclm_align_dim, dclm_len_thresh, \
               dclm_model_file, dclm_dict_file, dclm_ppl_file, dclm_log_file):
    lr = 0.1
    
    model_dir = tmp_dir+"models/"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    gpu_id = 3 
    
    forward_mem = 9192
    backward_mem = 1024
    param_mem = 512

    cmd0 = "cd "+os.getcwd()+"; "
    cmd0 += "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"+boost_dir+"; "
    cmd0 += "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"+dynet_dir+"; "

    cmd = cmd0 + "./dclm "+\
    "--dynet-gpu-ids "+str(gpu_id)+" "\
    "--dynet-mem "+",".join([str(forward_mem),str(backward_mem),str(param_mem)])+" "+\
    "train "+\
    combined_selected_non_domain_with_unk_and_in_domain_doc+" "+\
    dev_ref_file+" "+\
    model_type+" "+\
    str(dclm_num_layer)+" "+\
    str(dclm_input_dim)+" "+\
    str(dclm_hidden_dim)+" "+\
    str(dclm_align_dim)+" "+\
    str(dclm_len_thresh)+" "+\
    dclm_model_file+" "+\
    dclm_dict_file+" "+\
    dclm_ppl_file+" "+\
    dclm_log_file+" "+\
    str(lr)

    print(cmd+"\n")