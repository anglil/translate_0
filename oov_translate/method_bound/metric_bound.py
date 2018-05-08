import numpy as np
import html # python 3.4
import os.path
import string
from utils import *
import sys


moses_dir = "/home/ec2-user/kklab/src/mosesdecoder/"
### ./fast_align -i text.fr-en -d -o -v > forward.align
fast_align = moses_dir+"fast_align/build/fast_align"
### multi-bleu.perl -lc ref < hyp (lc stands for lowercase)
multi_bleu = moses_dir+"scripts/generic/multi-bleu.perl"
### ./sentence-bleu ref < hyp (result is from 0 to 1)
sent_bleu = moses_dir+"bin/sentence-bleu"
### meteor
meteor_bin = moses_dir+"meteor-1.5/meteor-1.5.jar"







def get_best_hyp_bleu(all_hyp, ref_sent, hyp_file, ref_file):
    '''
    pick the best hyp out of a bunch of hyp based on sentence level bleu score
    '''
    with open(hyp_file, 'w') as fw_hyp, open(ref_file, 'w') as fw_ref:
        for hyp in all_hyp:
            fw_hyp.write(hyp+'\n')
            fw_ref.write(ref_sent+'\n')
    stdout, stderr = sh(sent_bleu+" "+ref_file+" < "+hyp_file)
    bleu_scores = stdout.strip().split('\n')
    bleu_scores = [float(item) for item in bleu_scores]
    best_bleu_idx = bleu_scores.index(max(bleu_scores))
    best_hyp = all_hyp[best_bleu_idx]
    return best_hyp



def get_best_hyp_meteor(all_hyp, ref_sent, hyp_file, ref_file):
    '''
    pick the best hyp out of a bunch of hyp based on meteor score (alignment based)
    all_hyp: list
    '''
    with open(hyp_file, 'w') as fw_hyp, open(ref_file, 'w') as fw_ref:
        for hyp in all_hyp:
            fw_hyp.write(hyp+'\n')
            fw_ref.write(ref_sent+'\n')
    stdout, stderr = sh("java -Xmx2G -jar "+meteor_bin+" "+hyp_file+" "+ref_file+" -norm -noPunct | grep 'Segment'")
    meteor_scores = stdout.strip().split('\n')
    meteor_scores = [float(line.split('\t')[1]) for line in meteor_scores]
    best_meteor_idx = meteor_scores.index(max(meteor_scores))
    best_hyp = all_hyp[best_meteor_idx]
    return best_hyp



def get_best_long_trans(tra_tok, oov_candidates, l_ref, metric_dir, metric):
    ''' 
    for sentences with lots of oovs, we translate oovs one by one
    '''
    ### ./sentence-bleu ref < hyp (result is from 0 to 1)
    tra_tok_new = list(tra_tok)
    for i in range(len(tra_tok)):
        if tra_tok[i] in oov_candidates and tra_tok[i] not in oov_candidates[tra_tok[i]]:
            candidates = list(oov_candidates[tra_tok[i]].keys())
            hyp_file = metric_dir+"hyp_long"
            ref_file = metric_dir+"ref_long"
            all_hyp = []
            for candidate in candidates:
                sent = list(tra_tok_new)
                sent[i] = candidate
                all_hyp.append(' '.join(sent))
            if metric == "bleu":
                best_candidate = get_best_hyp_bleu(all_hyp, l_ref, hyp_file, ref_file)
            elif metric == "meteor":
                best_candidate = get_best_hyp_meteor(all_hyp, l_ref, hyp_file, ref_file)
            tra_tok_new[i] = candidates[all_hyp.index(best_candidate)]

    return ' '.join(tra_tok_new)



def align_oov(oov_pos, tra_tok, ref_tok, pairs):
    '''
    given alignment with the reference, output the translation without oov and mapping from oov words to english words
    res: translated sentence by the alignment
    oov_trans: the translation for the oov words
    '''
    from random import randint
    ### source language positions
    src_lang = []
    ### target language positions
    tgt_lang = []
    for pair in pairs:
        lr = pair.split('-')
        src_lang.append(int(lr[0]))
        tgt_lang.append(int(lr[1]))
    
    ### oov translation {oov word: set({translations})}
    oov_trans = {}

    res = list(tra_tok)
    for i in range(len(tra_tok)):
        ### identify the oov word
        if i in oov_pos:
            ### replace the oov word with the aligned word from reference
            if i in src_lang:
                res[i] = ref_tok[tgt_lang[src_lang.index(i)]]
            ### replace the oov word not aligned with any reference word, with a word in reference that doesn't appear in the translation result
            else:
                idx_ref = 0
                while idx_ref < len(ref_tok) and ref_tok[idx_ref] in res:
                    idx_ref += 1
                if idx_ref != len(ref_tok):
                    res[i] = ref_tok[idx_ref]
                #else:
                #    res[i] = ref_tok[randint(0,len(ref_tok)-1)]

            if tra_tok[i] not in oov_trans:
                oov_trans[tra_tok[i]] = {res[i]}
            else:
                oov_trans[tra_tok[i]].add(res[i])

    return res, oov_trans

            

### get the maximum gain on bleu ir meteor
### method: 
### 1. align
### 2. lattice (sentence level bleu, or meteor)
### 3. lattice-align
### meteor:
### 1. bleu
### 2. meteor
def get_best_metric(dataset, method, metric, tra, oov, oov_candidates_file, eng_vocab_file, res_file, metric_dir, oov_aligned_file):

    ug_dict = get_ug_dict(oov_candidates_file, 0)
    eng_vocab = get_eng_vocab(eng_vocab_file)

    ### maximum possible gain 
    ### reference based
    if method == "align":
        triple_pipe = metric_dir+"triple_pipe"
        forward_align = metric_dir+"forward_align"

        with open(tra) as ft, open(ref) as fr, open(triple_pipe, 'w') as fp:
            for l_tra in ft:
                l_ref = fr.readline().strip()
                fp.write(l_tra.strip())
                fp.write(" ||| ")
                fp.write(l_ref)
                fp.write('\n')
        ### source: translation with oov
        ### target: reference
        sh(fast_align+" -i "+triple_pipe+" -d -o -v > "+forward_align)

        ctr = 0
        with open(tra) as ft, open(oov) as fo, open(ref) as fr, open(forward_align) as ff, open(res_file, 'w') as fres:
            for l_tra in ft:
                l_oov = fo.readline()
                l_ref = fr.readline().strip()
                
                if ctr >= 0:
                    ### html unescaping happens
                    tra_tok, oov_pos, context = get_context_oov_pos(l_tra, l_oov)

                    ref_tok = l_ref.split(' ')
                    l_align = ff.readline().strip()                
                    pairs = l_align.split(' ')
                    
                    ### prevening english words from being aligned to reference
                    oov_pos = [pos for pos in oov_pos if tra_tok[pos] not in eng_vocab]

                    res, _ = align_oov(oov_pos, tra_tok, ref_tok, pairs)

                    res = ' '.join(res)
                    print(res)
                    #print("--------")
                    fres.write(res+'\n')

                ctr += 1
    
    ### oov candidate word list based, with reference words added to the candidate word list
    elif method == "lattice-align":
        triple_pipe = metric_dir+"triple_pipe_mix"
        forward_align = metric_dir+"forward_align_mix"

        with open(tra) as ft, open(ref) as fr, open(triple_pipe, 'w') as fp:
            for l_tra in ft:
                l_ref = fr.readline().strip()
                fp.write(l_tra.strip())
                fp.write(" ||| ")
                fp.write(l_ref)
                fp.write('\n')
        ### source: translation with oov
        ### target: reference
        sh(fast_align+" -i "+triple_pipe+" -d -o -v > "+forward_align)
        
        ctr = 0
        with open(tra) as ft, open(oov) as fo, open(ref) as fr, open(res_file, 'w') as fb, open(forward_align) as ff, open(oov_aligned_file, 'w') as foa:
            for l_tra in ft:
                l_oov = fo.readline()
                l_ref = fr.readline().strip()
                
                ref_tok = l_ref.split(' ')
                l_align = ff.readline().strip()                
                pairs = l_align.split(' ')
                
                if ctr >= 0:
                    ### html unescaping happens
                    tra_tok, oov_pos, context = get_context_oov_pos(l_tra, l_oov)
                    
                    ### prevening english words from being aligned to reference
                    oov_pos = [pos for pos in oov_pos if tra_tok[pos] not in eng_vocab]
                    
                    ### {oov:{candidate}}
                    _, oov_trans = align_oov(oov_pos, tra_tok, ref_tok, pairs)
                    
                    ### {oov:{candidate:score}}
                    oov_words_set = set([tra_tok[i] for i in oov_pos])
                    oov_candidates = get_oov_candidates(ug_dict, oov_words_set)
                    
                    ### merge the oracle oov candidates and the actual oov candidates
                    for oov in oov_trans:
                        ### write oov translation to file for other programs to use
                        foa.write(oov+'\t'+'\t'.join(list(oov_trans[oov]))+'\n')
                        if oov in oov_candidates:
                            for candidate in oov_trans[oov]:
                                if candidate not in oov_candidates[oov]:
                                    oov_candidates[oov][candidate] = 0
                        else:
                            oov_candidates[oov] = {c:0 for c in oov_trans[oov]}
                     
                    ### for test set: sentence 316 is super long
                    ### for dev set: sentence 305, 452 is super long
                    s = [305, 452, 686]
                    if dataset == "dev":
                        s = [305, 452]
                    elif dataset == "test":
                        s = [316]

                    ### consider all hypotheses all at once
                    if ctr not in s:
                        ### recursively get all possible combination
                        all_sentences = get_all_sentences(tra_tok, oov_candidates)
                        ref_file = metric_dir+"ref_mix_"+str(ctr)
                        hyp_file = metric_dir+"hyp_mix_"+str(ctr)
                        if metric == "bleu":
                            best_trans = get_best_hyp_bleu(all_sentences, l_ref, hyp_file, ref_file)
                        elif metric == "meteor":
                            best_trans = get_best_hyp_meteor(all_sentences, l_ref, hyp_file, ref_file)
                    ### too many hypotheses to consider all at once, so decode the oovs one by one
                    else:
                        best_trans = get_best_long_trans(tra_tok, oov_candidates, l_ref, metric_dir, metric)

                    print(best_trans)
                    #print("--------")
                    fb.write(best_trans+'\n')
                
                ctr += 1                   
        
    ### oov candidate word list based
    elif method == "lattice":
        ctr = 0
        with open(tra) as ft, open(oov) as fo, open(ref) as fr, open(res_file, 'w') as fb:
            for l_tra in ft:
                l_oov = fo.readline()
                l_ref = fr.readline().strip()
                
                if ctr >= 0:
                    ###
                    # tra_tok: tokenized translation with oov, with html unescaped
                    # oov_pos: oov word posistions
                    # context: context word positions
                    ###
                    ### html unescaping happens
                    tra_tok, oov_pos, context = get_context_oov_pos(l_tra, l_oov)
                    oov_words_set = set([tra_tok[i] for i in oov_pos])
                    oov_candidates = get_oov_candidates(ug_dict, oov_words_set)

                    
                    ### for test set: sentence 316 is super long
                    ### for dev set: sentence 305, 452 is super long
                    s = [305, 452, 686]
                    if dataset == "dev":
                        s = [305, 452]
                    elif dataset == "test":
                        s = [316]

                    if ctr not in s:
                        ### recursively get all possible combination
                        all_sentences = get_all_sentences(tra_tok, oov_candidates)
                        ref_file = metric_dir+"ref_"+str(ctr)
                        hyp_file = metric_dir+"hyp_"+str(ctr)
                        if metric == "bleu":
                            best_trans = get_best_hyp_bleu(all_sentences, l_ref, hyp_file, ref_file)
                        elif metric == "meteor":
                            best_trans = get_best_hyp_meteor(all_sentences, l_ref, hyp_file, ref_file)
                    else:
                        #best_trans = ' '.join(tra_tok)                
                        #best_trans = l_ref
                        best_trans = get_best_long_trans(tra_tok, oov_candidates, l_ref, metric_dir, metric)

                    print(best_trans)
                    fb.write(best_trans+'\n')                

                ctr += 1
            
            



if __name__ == "__main__":
    
    dataset = sys.argv[1] # "test" or "dev"
    metric = sys.argv[2] # "meteor" or "bleu"; need to choose only when method == "lattice" or method == "lattice-align"
    method = sys.argv[3] # "lattice" or "align" or "lattice-align"
      
    ### som, amh, yor
    s = "yor"
    ### eng
    t = "eng"

    y = "y2"
    r = "r1"
    v = "v1"

    exp_dir = "/home/ec2-user/kklab/Projects/lrlp/experiment_elisa."+s+"-"+t+"."+y+r+"."+v+"/"
    data_dir = "/home/ec2-user/kklab/data/ELISA/evals/"+y+"/elisa."+s+".package."+y+r+"."+v+"/"

    # -------- read --------
    ### translation with oov
    tra = exp_dir+"translation/"+dataset+"/elisa."+s+"-"+t+"."+dataset+"."+y+r+"."+v+".translated."+t

    ### oov words
    oov = exp_dir+"translation/"+dataset+"/oov_"+dataset

    ### the oov word dictionary provided by some other program
    ### TODO: generalize
    oov_candidates_file = "/home/ec2-user/kklab/data/lorelei/for-angli/uig.elisa."+dataset+".combined.output"
    
    ### english vocabulary
    eng_vocab_file = "/home/ec2-user/kklab/data/google-10000-english/20k.txt"

    ### reference translation (html not unescaped)
    ref = data_dir+dataset+"/elisa."+s+"-"+t+"."+dataset+"."+y+r+"."+v+".true."+t
        
    # -------- write --------
    res_file = exp_dir+"translation/"+dataset+"/best_"+metric+"_"+method
    
    oov_aligned_file = exp_dir+"translation/"+dataset+"/oov_aligned"
    
    metric_dir = exp_dir+"translation/"+dataset+"/"+metric+"/"
    if not os.path.exists(metric_dir):
        os.makedirs(metric_dir)
    
    ### method:
    # 1. lattice
    # 2. align
    # 3. lattice-align
    ### metric (applied only when method == lattice, because if method == aligned, then metric isn't used.):
    # 1. meteor
    # 2. bleu
    ### result file
    
    get_best_metric(dataset, method, metric, tra, oov, oov_candidates_file, eng_vocab_file, res_file, metric_dir, oov_aligned_file)
