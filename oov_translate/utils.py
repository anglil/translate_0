import numpy as np
import string
import sys 
import os.path
import subprocess
import copy
import html
import psutil
import threading
import time
import re
import random
import xml.etree.ElementTree as et
import nltk
import gzip
import shutil
from functools import lru_cache

dir_path = os.path.dirname(os.path.realpath(__file__))+"/"
sys.path.insert(0, dir_path)
from config import *

nltk.data.path.append(nltk_data_dir)
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


# -------- functions --------

def sh_realtime(script):
    import subprocess
    import sys
    p = subprocess.Popen(script, shell=True, stderr=subprocess.PIPE)
    
    while True:
        out = p.stderr.read(1)
        out = out.decode('utf-8')
        if out == '' and p.poll() != None:
            break
        if out != '':
            sys.stdout.write(out)
            sys.stdout.flush()


def sh(script, stdin=None):
    """Returns (stdout, stderr), raises error on non-zero return code"""
    import subprocess
    # Note: by using a list here (['bash', ...]) you avoid quoting issues, as the 
    # arguments are passed in exactly this order (spaces, quotes, and newlines won't
    # cause problems):
    proc = subprocess.Popen(['bash', '-c', script],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        stdin=subprocess.PIPE)
    stdout, stderr = proc.communicate()
    if proc.returncode:
        raise ScriptException(proc.returncode, stdout, stderr, script)
    return stdout.decode('utf-8'), stderr.decode('utf-8')



class ScriptException(Exception):
    def __init__(self, returncode, stdout, stderr, script):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        super(ScriptException, self).__init__(stdout.decode('utf-8')+stderr.decode('utf-8'))
        #Exception.__init__()


def update_param(key, value):
    '''
    obsolete
    '''
    if key == "dataset":
        dataset = value
    elif key == "s":
        s = value
    else:
        raise Exception("unsupported key: "+key)
        
    lines = []
    with open("config.py") as f:
        for line in f:
            l = line.strip().split('=')
            if l[0].strip() == "dataset":
                l[1] = "\""+dataset+"\""
                lines.append("=".join(l)+"\n")
            elif l[0].strip() == "s":
                l[1] = "\""+s+"\""
                lines.append("=".join(l)+"\n")
            else:
                lines.append(line)
    with open("config.py", 'w') as fw:
        for line in lines:
            fw.write(line)
        

### candidate list from an external source
#ug_dict = None
#if os.path.exists(oov_candidates_file):
#    ug_dict = get_ug_dict(oov_candidates_file, 0)

#oov_candidates_all = get_oov_candidates_from_external_source(oov_candidates_dir)
#eng_vocab = get_eng_vocab(eng_vocab_file)

# ----------------

def build_vocab3(ref_1best_files, candidate_list_files):
    vocab_set = set()
    print("start building vocabulary...")
    for ff in ref_1best_files:
        with open(ff) as f:
            for line in f:
                l = line.strip()
                if l != '=':
                    for tok in l.split(' '):
                        vocab_set.add(tok)
    for ff in candidate_list_files:
        with open(ff) as f:
            for line in f:
                if line != "\n" and line != "=\n":
                    oov_pos_candidates = line.split(" ")
                    for pos_candidates in oov_pos_candidates:
                        candidates = set(pos_candidates.split(":")[1].split(","))
                        vocab_set |= candidates
    print("vocabulary built!")
    return vocab_set

def build_vocab2(train_ref_file, dev_ref_file, unseq_ref_file, test_1best_file, test_oov_file, oov_candidates_all, eng_vocab):
    vocab_set = set()
    print("start building vocabulary...")
    with open(train_ref_file) as f:
        for line in f:
            l = line.strip()
            if l != '=':
                for tok in l.split(' '):
                    vocab_set.add(tok)
    print("train_ref_file included.")
    with open(dev_ref_file) as f:
        for line in f:
            l = line.strip()
            if l != '=':
                for tok in l.split(' '):
                    vocab_set.add(tok)
    print("dev_ref_file included.")
    if os.path.exists(unseq_ref_file):
        with open(unseq_ref_file) as f:
            for line in f:
                l = line.strip()
                if l != '=':
                    for tok in l.split(' '):
                        vocab_set.add(tok)
        print("unseq_ref_file included.")
    with open(test_1best_file) as fo, open(test_oov_file) as fv:
        for line in fo:
            line_oov = fv.readline()
            l = line.strip()
            l_oov = line_oov.strip()
            if l != '=':
                l = l.split(' ')
                l_oov = [(int(item) if item != '' else None) for item in l_oov.split(' ')]
                for item in range(len(l)):
                    if item not in l_oov:
                        vocab_set.add(l[item])
            else:
                assert(l_oov == '=')
    print("test_1best_file included.")
    for oov_word in oov_candidates_all:
        vocab_set |= oov_candidates_all[oov_word]
    print("oov_candidates_all included.")
    vocab_set |= eng_vocab
    print("eng_vocab included.\n")
    return vocab_set

    
def build_vocab():
    vocab_set = set()
    print("start building the vocabulary...")
    with open(train_in_domain) as f:
        for line in f:
            l = line.strip().split(' ')
            for tok in l:
                vocab_set.add(tok.lower())
    print("in-domain training ref included.")
    with open(dev_in_domain) as f:
        for line in f:
            l = line.strip().split(' ')
            for tok in l:
                vocab_set.add(tok.lower())
    print("in-domain dev ref included.")
    with open(test_in_domain_hyp) as ft, open(test_in_domain_oov) as fo:
        for l_tra in ft:
            l_oov = fo.readline()
            
            tra_tok, oov_pos, context = get_context_oov_pos(l_tra, l_oov)
            for i in range(len(tra_tok)):
                if i not in oov_pos:
                    vocab_set.add(tra_tok[i].lower())
    print("in-domain test hyp included.")
    vocab_set |= eng_vocab
    print("eng_vocab included.")
    dev_ug_dict = get_ug_dict(dev_oov_candidates_file, 0)
    test_ug_dict = get_ug_dict(test_oov_candidates_file, 0)
    for oov_word in dev_ug_dict:
        for candidate in dev_ug_dict[oov_word]:
            vocab_set.add(candidate.lower())
    for oov_word in test_ug_dict:
        for candidate in test_ug_dict[oov_word]:
            vocab_set.add(candidate.lower())
    print("candidate list included.")
    print("vocab size: "+str(len(vocab_set)))
    return vocab_set



def set_unk(train_non_domain_all0, train_non_domain_all_restrict_vocab, vocab_set):
    '''
    used in n-gram lm for setting UNK
    params:
        train_non_domain_all0: input file
        train_non_domain_all_restrict_vocab: output file with UNK set
        vocab_set: vocabulary
    '''
    with open(train_non_domain_all0) as f, open(train_non_domain_all_restrict_vocab, 'w') as fw:
        ctr = 0
        for line in f:
            l = line.strip().split(' ')
            for i in range(len(l)):
                if l[i].lower() not in vocab_set:
                    l[i] = "UNK"
            fw.write(' '.join(l)+'\n')
            
            if ctr % 2000000 == 0:
                print(str(ctr)+" sentences processed.")
            ctr += 1
    print("UNK set in: "+train_non_domain_all_restrict_vocab)
        

def set_unk_for_string(in_string, vocab_set):
    '''
    used in dclm for setting UNK
    params:
        in_string: a document, with '\n', and delimited by ' '
        vocab_set: vocabulary
    '''
    in_string_lines = in_string.split('\n')
    in_string_lines_new = []
    for line in in_string_lines:
        l = line.split(' ')
        l_new = [(item if item in vocab_set or item == '' else 'UNK') for item in l]
        in_string_lines_new.append(' '.join(l_new))
    return '\n'.join(in_string_lines_new)
 

def parse_oov_candidates(l_candidate):
    '''
    guarantee that there is no empty candidate
    '''
    candidate_map = dict()
    item_candidate = l_candidate.split(" ")
    for item in item_candidate:
        pair = item.split(":")
        oov_pos = int(pair[0])
        candidates = pair[1].split(",")
        candidate_map[oov_pos] = list()
        for candidate in candidates:
            candidate_map[oov_pos].append(candidate)
    return candidate_map


def get_oov_candidates(ug_dict, oov_words_set): 
    '''
    return a dictionary of oov words as keys with a dictionary of candidate translations as values
    '''
    oov_candidates = {}
    for oov_word in oov_words_set:
        if oov_word not in ug_dict:
            oov_candidates[oov_word]={oov_word:10000}
        else:
            oov_candidates[oov_word]=ug_dict[oov_word]
    return oov_candidates



def get_oov_candidates_eng_vocab(eng_vocab, oov_words_set, context_words_set):
    '''
    return a dictionary of candidates as keys and scores as values
    '''
    oov_candidates = {}
    for eng in eng_vocab:
        if eng not in context_words_set:
            if eng not in oov_words_set:
                oov_candidates[eng] = 0
            else:
                oov_candidates[eng] = 10000
    return oov_candidates
                

def get_oov_candidates_all(candidate_source, add_aligned_oov, ug_dict, eng_vocab, oov_words_set, context_words_set):
    '''
    an emsemble of get_oov_candidates and get_oov_candidates_eng_vocab
    '''
    if candidate_source is "ug_dict":
        oov_candidates = get_oov_candidates(ug_dict, oov_words_set)
        if add_aligned_oov:
            with open(oov_aligned_file) as f:
                for line in f:
                    line = line.strip().split('\t')
                    oov = line[0]
                    candidates = line[1:]
                    if oov in oov_words_set:
                        add_ref_to_oov_candidates(oov_candidates, oov, candidates)
    elif candidate_source is "eng_vocab":
        oov_candidates = get_oov_candidates_eng_vocab(eng_vocab, oov_words_set, context_words_set)
    return oov_candidates
    

def add_ref_to_oov_candidates(oov_candidates, oov_aligned, candidates_aligned):
    '''
    add aligned oov and its candidates to oov_candidates,
    used only when *ug_dict* is applied
    params:
        oov_candidates: {oov:{candidate:score}}
        oov_aligned: oov
        candidates_aligned: list or set of candidate words
    return:
        None
    '''
    if oov_aligned in oov_candidates:
        ### remove the existing candidate that is the same as the oov word
        if oov_aligned in oov_candidates[oov_aligned]:
            oov_candidates[oov_aligned] = {c:0 for c in candidates_aligned}
        ### add the candidates to the existing candidates of the oov word
        else:
            for candidate in candidates_aligned:
                if candidate not in oov_candidates[oov_aligned]:
                    ### initial candidate score: 0
                    oov_candidates[oov_aligned][candidate] = 0
    else:
        ### initial candidate score: 0
        oov_candidates[oov_aligned] = {c:0 for c in candidates_aligned}
        

def get_context_oov_pos(l_tra, l_oov):
    '''
    parse oov and context from raw texts
    params:
        l_tra: line_translation
        l_oov: line_oov
    return:
        tra_tok: tokenized translation
        oov_pos: positions of oov words
        context: positions of context words
    '''
    ### extract tra_tok
    tra_tok = l_tra.strip().split(' ')
    ###  unescaping html symbols
    #for i in range(len(tra_tok)):
    #    tra_tok[i] = html.unescape(tra_tok[i])
    
    ### extract oov_pos
    oov_tok = l_oov.strip().split(' ')
    oov_pos = set()
    oov_not_in_tra = set()
    translator = str.maketrans('', '', string.punctuation)
    for oov_word in oov_tok:
        condition1 = (oov_word.translate(translator) != '')
        condition2 = (oov_word in tra_tok)
        condition_non_roman = (s not in roman_ab) and (not re.match("^[a-zA-Z0-9_]*$", oov_word.translate(translator)))
        condition_roman = (s in roman_ab) and (oov_word.translate(translator).lower() not in eng_vocab)
        ### confirmed english words are excluded from oov words
        if condition1 and condition2 and (condition_non_roman or condition_roman):
            oov_idx = [i for i, x in enumerate(tra_tok) if x.lower() == oov_word.lower()]
            for idx in oov_idx:
                oov_pos.add(idx)
        else:
            oov_not_in_tra.add(oov_word)
    oov_pos = sorted(list(oov_pos))
    
    ### extract context, excluding punctuations and function words
    #function_words = set(["is", "was", "are", "were", "be", "the", "an", "a", "and", "or"])
    context = [i for i in range(len(tra_tok)) if i not in oov_pos and tra_tok[i] not in string.punctuation and tra_tok[i].lower() not in function_words]
    
    return tra_tok, oov_pos, context


def get_num_hyp(oov_candidates, tra_tok, oov_pos):
    '''
    param: 
        oov_candidates:
        tra_tok:
        oov_pos:
    return:
        num_hyp: number of hypotheses
    '''
    num_hyp = 1
    for pos in oov_pos:
        tok = tra_tok[pos]
        if tok in oov_candidates and oov_candidates[tok] != {}:
            num_hyp *= len(oov_candidates[tok])   
            #print("num_hyp: "+str(num_hyp))
    return num_hyp


def get_file_length(f):
    '''
    param:
        f: path to file
    return:
        l: number of sentences in file
    '''
    stdout, _ = sh("wc -l "+f)
    l = int(stdout.strip().split(' ')[0])
    return l


def nbest_parser(n_best_file_raw, n_best_file_clean, n_best_file_idx, lower_case=False):
    '''
    parse moses -n-best-list output; only parse the index l[0] and the text l[1]
    param:
        n_best_file_raw: -n-best-list output
        n_best_file_clean: n-best sentences in one file
        lower_case: whether to lower case the sentences
    return:
        sent_idx: list of sentence indices
    '''
    ctr = 0
    sent_idx = []
    with open(n_best_file_raw) as f, open(n_best_file_clean, 'w') as fw, open(n_best_file_idx, 'w') as fidx:
        for line in f:
            if line.strip() != "=":
                l = line.split(" ||| ")
                sent_idx.append(int(l[0]))
                fidx.write(l[0]+"\n")
                if lower_case:
                    fw.write(l[1].lower()+"\n")
                else:
                    fw.write(l[1]+"\n")

                ctr += 1
                
                if ctr%1000==0:
                    print(str(ctr)+" nbest parsed.")
            else:
                fw.write("=\n")
                fidx.write("=\n")
    print(str(ctr)+" sentences written in file.")
    return sent_idx

            

def lemmatize_word(lmtzr, lmtzr_cache, word):
    '''
    obsolete
    '''
    if word in lmtzr_cache:
        return lmtzr_cache[word]
    else:
        lemmatized_word = lmtzr.lemmatize(word)
        lmtzr_cache[word] = lemmatized_word
        return lemmatized_word

    
def lemmatize_and_rid_func_and_lowercase(in_file, out_file):
    '''
    stem words and get rid of function words, for data selection
    '''
    lmtzr = WordNetLemmatizer()
    lemmatize = lru_cache(maxsize=50000)(lmtzr.lemmatize)
    #lmtzr_cache = dict()
    
    in_file_len = get_file_length(in_file)
    ctr = 0
    with open(in_file) as f, open(out_file, 'w') as fw:
        for line in f:
            if line.strip() == "=":
                fw.write("=\n")
            else:
                l = line.strip().split(' ')
                l_new = [lemmatize(tok.lower()) for tok in l \
                        if tok.lower() not in function_words \
                        and tok not in punctuations]
                fw.write(' '.join(l_new)+"\n")
                
                ctr += 1
            if ctr%1000 == 0:
                print(str(ctr)+"/"+str(in_file_len)+" sentences processed.")


def merge_files(in_file_list, out_file):
    '''
    merge in_file_1 and in_file_2 into out_file, in order
    params:
        in_file_list: list of input files
        out_file: output file
    return:
        None
    '''
    with open(out_file, 'w') as fout:
        for in_file in in_file_list:
            with open(in_file) as fin:
                for line in fin:
                    fout.write(line)


def merge_files_with_boundary(in_file_list, out_file, throw_long_sent):
    '''
    merge in_file_1 and in_file_2 into out_file, in order, with document boundary
    params:
        in_file_list: list of input files
        out_file: output file
    '''
    doc_ctr = 0
    with open(out_file, 'w') as fout:
        for in_file in in_file_list:
            with open(in_file) as fin:
                for line in fin:
                    if throw_long_sent:
                        sent_len = len(line.split(' '))
                        if sent_len < 100:
                            fout.write(line)
                        #else:
                        #    print("A sentence with length "+str(sent_len)+" thrown.")
                    else:
                        fout.write(line)
            fout.write("=\n")
            doc_ctr += 1
            if doc_ctr%1000 == 0:
                print(str(doc_ctr)+" documents merged.")
    print("merged file created at: "+out_file)
        
        
def multithread_routine(method_params, res_file, cached_res, tmp_dir, thread_constructor):
    '''
    params:
        method_params: list of hyperparameters for a method (not including res_file)
        res_file: path to file that stores the final result
        cached_res: list of resources generated in advance to be shared across threads
        tmp_dir: directory in which to store intermediate results generated by threads
        thread_constructor: functional argument as the thread constructor
    return:
        None
    '''
    thread_num = int(get_thread_num()/2) ### use half of all cores
    print("Number of threads: "+str(thread_num))
    total_num = get_file_length(tra_file)
    print("Number of sentences: "+str(total_num))
    bins = get_bin_by_thread(thread_num, total_num)
    
    res_list = list()
    thread_list = list()
    
    for i in range(thread_num):
        ctr_lo = bins[i][0]
        ctr_up = bins[i][1]
        res_file_tmp = tmp_dir+"_".join([str(param) for param in method_params])+"_"+str(ctr_lo)+"_"+str(ctr_up)
        
        ### concatenate tuples to feed into the thread constructor
        params = ()
        for item in method_params:
            params += (item,)
        params += (res_file_tmp,)
        params += (ctr_lo,)
        params += (ctr_up,)
        for item in cached_res:
            params += (item,)
        t = thread_constructor(*params)
        
        res_list.append(res_file_tmp)
        thread_list.append(t)
    
    for i in range(thread_num):
        thread_list[i].start()
    
    for i in range(thread_num):
        thread_list[i].join()
    print("All threads finished.")
    
    merge_files(res_list, res_file)
    print(str(len(res_list))+" result files merged.")
    

def random_sample(full_file, k, sample_file):
    '''
    sample k lines randomly from full_file and write to sample_file
    param:
        full_file: path to file to sample sentences from
        k: number of samples
        sample_file: path to file of sampled sentences
    return:
        total_num: total number of sentences in full_file. If this function was called before, then don't return anything
    '''
    ### sample sentences from file
    sample = []
    total_num = 0
    with open(full_file) as f:
        for n, line in enumerate(f):
            if n < k:
                sample.append(line.rstrip())
            else:
                r = random.randint(0, n)
                if r < k:
                    sample[r] = line.rstrip()
            total_num = n
    print("Sampled "+str(k)+" sentences from "+full_file)

    ### write sentences into file
    with open(sample_file, 'w') as fw:
        for sent in sample:
            fw.write(sent+'\n')
    print("Wrote "+str(k)+" sentences to "+sample_file)
        
    return total_num

def add_to_dict_set(key, val, d) :
    if key in d:
        if isinstance(val, set):
            d[key] |= val
        else:
            d[key].add(val)
    else:
        if isinstance(val, set):
            d[key] = val
        else:
            d[key] = {val}
    return d

def get_all_sentences(tra_tok, oov_candidates):
    '''
    recursively get all sentences with all oov candidates,
    used only when ug_dict is applied
    params:
        tra_tok: tokenized translation
        oov_candidates: {oov:{candidate:score}}
    return:
        list of sentences
    '''
    sentences = None
    if tra_tok[0] in oov_candidates and tra_tok[0] not in oov_candidates[tra_tok[0]]:
        sentences = list(oov_candidates[tra_tok[0]].keys())
    else:
        sentences = [tra_tok[0]]

    if len(tra_tok) > 1:
        all_sentences = []
        sentences_after = get_all_sentences(tra_tok[1:], oov_candidates)
        for sent in sentences:
            for sent_after in sentences_after:
                all_sentences.append(sent+' '+sent_after)
                #print(psutil.virtual_memory().percent)
                #print(psutil.cpu_percent())
        
        return all_sentences
    else:
        #print(sentences)
        return sentences  

    
def write_all_sentences(ctr, \
                        pre_hyp, \
                        tra_tok, \
                        oov_pos, \
                        oov_candidates, \
                        lazy_dir):
    '''
    params:
        ctr: set to 0
        pre_hyp: set to []
        tra_tok: 
        oov_pos: 
        oov_candidates: eng_vocab style: {candidate:score}
        lazy_dir: directory in which to write all candidate sentences, indexed by ctr
    '''
    if len(tra_tok) == 1:
        if len(pre_hyp) in oov_pos:
            candidates = list(oov_candidates.keys())
            for candidate in candidates:
                hyp = ' '.join(pre_hyp+[candidate])
                with open(lazy_dir+str(ctr), 'w') as fw:
                    fw.write(hyp)
                ctr += 1
            return ctr
        else:
            hyp = ' '.join(pre_hyp+tra_tok)
            with open(lazy_dir+str(ctr), 'w') as fw:
                fw.write(hyp)
            ctr += 1
            return ctr
    else:
        if len(pre_hyp) in oov_pos:
            candidates = list(oov_candidates.keys())

            for candidate in candidates:
                ctr = write_all_sentences(ctr, \
                                    pre_hyp+[candidate], \
                                    tra_tok[1:], \
                                    oov_pos, \
                                    oov_candidates, \
                                    lazy_dir)
            return ctr
        else:
            return write_all_sentences(ctr, \
                         pre_hyp+[tra_tok[0]], \
                         tra_tok[1:], \
                         oov_pos, \
                         oov_candidates, \
                         lazy_dir)


        
def get_thread_num():
    '''
    optimize for the number of threads
    '''
    thread, _ = sh("lscpu | grep 'Thread(s) per core'")
    thread = int(thread.strip().split(':')[1])
    core, _ = sh("lscpu | grep 'Core(s) per socket'")
    core = int(core.strip().split(':')[1])
    socket, _ = sh("lscpu | grep 'Socket(s)'")
    socket = int(socket.strip().split(':')[1])
    return thread * core * socket



def get_bin_by_thread(thread_num, total_num):
    '''
    bucket total_num of instances into bins for parallelization
    params:
        thread_num: number of threads the system supports at most
        total_num: total number of instances to parallelize
    return: 
        bins: arrays of boundaries: [[ctr_lo, ctr_up]...]
    '''
    bins = []
    bin_base_size = int(total_num/thread_num)
    bin_residual = total_num%thread_num
    ctr = 0
    while ctr < total_num:
        ctr_lo = ctr
        if bin_residual > 0:
            ctr_up = ctr + bin_base_size
            bin_residual -= 1
        else:
            ctr_up = ctr + bin_base_size - 1
        bins.append([ctr_lo, ctr_up])
        ctr = ctr_up + 1
    return bins


def write_translation_to_xml(xml_in_file, onebest_file, xml_out_file):
    '''
    params:
        xml_in_file: input
        onebest_file: input
        xml_out_file: output
    '''
    tree = et.parse(xml_in_file)
    root = tree.getroot()
    f = open(onebest_file)

    ctr = 0
    for doc in root.findall('DOCUMENT'):
        for seg in doc.findall('SEGMENT'):
            line = f.readline()
            l = line.strip().split(' ')
            
            onebest_cost = 1e9
            onebest_cost_str = ""
            
            src = seg.find('SOURCE')
            nbest = src.find('NBEST')
            
            for hyp in nbest.findall('HYP'):
                ### get cost of this hypothesis
                cost_str = hyp.attrib["cost"]
                cost = float(cost_str)
                
                ### check if this hypothesis is one best
                if cost < onebest_cost:
                    onebest_cost = cost
                    onebest_cost_str = cost_str
                
            for hyp in nbest.findall('HYP'):
                if hyp.attrib["cost"] == onebest_cost_str:
                    align = hyp.find('ALIGNMENT')
                    target_tok = align.find('TOKENIZED_TARGET')
                    assert(len(l)==len(target_tok.findall('TOKEN')))
                    #if len(l) != len(target_tok.findall('TOKEN')):
                    #    print("new: "+str(l))
                    #    print("old: "+str([tok.text for tok in target_tok.findall('TOKEN')]))
                    tok_ctr = 0
                    for tok in target_tok.findall('TOKEN'):
                        tok.text = l[tok_ctr]
                        tok_ctr += 1
                    
            ctr += 1
        line = f.readline()
        #if line.strip() != "=":
        #    print(line)
        assert(line.strip()=="=")
    
    tree.write(xml_out_file, encoding="UTF-8")
    print("xml file created at: "+xml_out_file)
    
    with open(xml_out_file, 'rb') as fi, gzip.open(xml_out_file+".gz", 'wb') as fo:
        shutil.copyfileobj(fi, fo)
    print("file compressed at: "+xml_out_file+".gz")
