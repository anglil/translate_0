import sys 
import os

dir_path = os.path.dirname(os.path.realpath(__file__))+"/"
sys.path.insert(0, dir_path+'../../../oov_translate')
from config import *
from utils import *
from oov_candidates_preprocessing import *

#def get_best_hyp_lm(lm, \
#                    pre_hyp, \
#                    tra_tok, \
#                    oov_candidates, \
#                    best_score, \
#                    best_hyp):
#    '''
#    score sentences in a dfs fashion, such that space complexity is O(1)
#    for ug_dict only
#    params:
#        lm: language model in kenLM format
#        pre_hyp: tokenized hyp with oov translated preceding tra_tok
#        tra_tok: list of tokens with oov un-translated
#        oov_candidates: {oov:{candidate:score}}
#        best_score: highest log probability
#        best_hyp: hypothesis with highest log probability, in string
#    return:
#        best_score: highest log probability updated
#        hyp: oov translated hyp in string format
#    '''
#    if len(tra_tok) == 1:
#        tok = tra_tok[0]
#        if tok in oov_candidates and tok not in oov_candidates[tok]:
#            candidates = list(oov_candidates[tok].keys())
#            for candidate in candidates:
#                hyp = ' '.join(pre_hyp+[candidate])
#                score = lm.score(hyp)
#                if score > best_score:
#                    best_score = score
#                    best_hyp = hyp
#                #print(score)
#                #print(hyp)
#            return best_score, best_hyp
#        else:
#            hyp = ' '.join(pre_hyp+tra_tok)
#            score = lm.score(hyp)
#            if score > best_score:
#                best_score = score
#                best_hyp = hyp
#            #print(score)
#            #print(hyp)
#            return best_score, best_hyp
#    else:
#        tok = tra_tok[0]
#        if tok in oov_candidates and tok not in oov_candidates[tok]:
#            candidates = list(oov_candidates[tok].keys())
#            for candidate in candidates:
#                score, hyp = get_best_hyp_lm(lm, pre_hyp+[candidate], tra_tok[1:], oov_candidates, best_score, best_hyp)
#                if score > best_score:
#                    best_score = score
#                    best_hyp = hyp
#            return best_score, best_hyp
#        else:
#            return get_best_hyp_lm(lm, pre_hyp+[tok], tra_tok[1:], oov_candidates, best_score, best_hyp)


#def get_best_hyp_lattice_lm(lm_final_path, tra_tok, oov_pos, oov_candidates, candidate_source, lazy_dir):
#    '''
#    translate oov words by forming a word lattice and decoding using beam search
#    params:
#        lm_final_path: path to lm in kenLM format, not binary
#        tra_tok: list of tokens
#        oov_pos: list of oov positions in tra_tok
#        oov_candidates: {oov:{candidate:score}}
#        lazy_dir: direcotry in which to put all hypergraph files
#    return:
#        best translation in string format
#    '''
#    lazy_file = lazy_dir + '0'
#
#    total_vertex_count = 2
#    total_edge_count = 2
#
#    lazy_text = []
#    lazy_text.append('1')
#    lazy_text.append('<s> |||')
#    vertex_idx = 0
#    for pos in range(len(tra_tok)):
#        if pos not in oov_pos:
#            lazy_text.append('1')
#            lazy_text.append('['+str(vertex_idx)+'] '+tra_tok[pos]+' |||')
#            total_vertex_count += 1
#            total_edge_count += 1
#            vertex_idx += 1
#        else:
#            if candidate_source == "ug_dict":
#                candidates = oov_candidates[tra_tok[pos]]
#            elif candidate_source == "eng_vocab":
#                candidates = oov_candidates.keys()
#            candidate_num = len(candidates)
#            lazy_text.append(str(candidate_num))
#            for candidate in candidates:         
#                lazy_text.append('['+str(vertex_idx)+'] '+candidate+' |||')     
#            total_vertex_count += 1
#            total_edge_count += candidate_num
#            vertex_idx += 1
#    lazy_text.append('1')
#    lazy_text.append('['+str(vertex_idx)+'] </s> |||')
#
#    ### the tokens <s> and </s> should appear explicitly
#    with open(lazy_file, 'w') as flazy:
#        flazy.write('{} {}'.format(total_vertex_count, total_edge_count)+'\n')
#        for line in lazy_text:
#            flazy.write(line+'\n')
#
#    beam_size = 100
#
#    #sh("vim -c wq "+lazy_file)
#    stdout, stderr = sh(hypergraph_dec+\
#                        " -i "+lazy_dir+\
#                        " -l "+lm_final_path+".binary"+\
#                        " -K "+str(beam_size)+\
#                        " -W LanguageModel=1.0 LanguageModel_OOV=0 WordPenalty=0")
#
#    return stdout.split('\n')[0].split('|||')[1].strip()

def write_lattice(lattice_dir, onebest_tok, candidate_map):
    lattice_file = os.path.join(lattice_dir, '0')
    total_vertex_count = 2
    total_edge_count = 2
    lazy_text = []
    lazy_text.append('1')
    lazy_text.append('<s> |||')
    vertex_idx = 0
    for pos in range(len(onebest_tok)):
        if pos not in candidate_map:
            lazy_text.append('1')
            lazy_text.append('['+str(vertex_idx)+'] '+onebest_tok[pos]+' |||')
            total_vertex_count += 1
            total_edge_count += 1
            vertex_idx += 1
        else:
            candidates = candidate_map[pos]
            candidate_num = len(candidates)
            lazy_text.append(str(candidate_num))
            for candidate in candidates:
                lazy_text.append('['+str(vertex_idx)+'] '+candidate+' |||')
            total_vertex_count += 1
            total_edge_count += candidate_num
            vertex_idx += 1
    lazy_text.append('1')
    lazy_text.append('['+str(vertex_idx)+'] </s> |||')

    with open(lattice_file, 'w') as fw:
        fw.write('{} {}'.format(total_vertex_count, total_edge_count)+'\n')
        for line in lazy_text:
            fw.write(line+'\n')


def decode_lattice(lattice_dir, bin_lm_file, beam_size):
    cmd = hypergraph_dec+\
          " -i "+lattice_dir+\
          " -l "+bin_lm_file+\
          " -K "+str(beam_size)+\
          " -W LanguageModel=1.0 LanguageModel_OOV=0 WordPenalty=0"
    stdout, stderr = sh(cmd)
    return stdout.split('\n')[0].split('|||')[1].strip()


def rescore_lattice(onebest_file, candidate_list_file, tmp_lattice_dir, bin_lm_file, beam_size, res_file):
    ctr = 0
    with open(onebest_file) as fo, open(candidate_list_file) as fc, open(res_file, "w") as fr:
        for l_onebest in fo:
            l_onebest = l_onebest.strip()
            l_candidate = fc.readline().strip()
            if l_candidate != "" and l_candidate != "=":
                candidate_map = parse_oov_candidates(l_candidate)
                onebest_tok = l_onebest.split(' ')
                lattice_dir = os.path.join(tmp_lattice_dir, os.path.basename(onebest_file)+"_"+str(ctr))
                if not os.path.exists(lattice_dir):
                    os.makedirs(lattice_dir)
                write_lattice(lattice_dir, onebest_tok, candidate_map)
                trans = decode_lattice(lattice_dir, bin_lm_file, beam_size)
                fr.write(trans+"\n")
            elif l_candidate == "=":
                fr.write("=\n")
            else:
                fr.write(l_onebest+"\n")
            ctr += 1
            if ctr%100 == 0:
                print(str(ctr)+" sentences processed.")

#class LM (threading.Thread):
#    def __init__(self, \
#                 candidate_source, \
#                 add_aligned_oov, \
#                 language_model, \
#                 res_file, \
#                 ctr_lo, \
#                 ctr_up, \
#                 eng_vocab, \
#                 ug_dict, \
#                 lm_final, \
#                 lm_final_path):
#        '''
#        params:
#            method params: list
#            res_file
#            ctr_lo
#            ctr_up
#            cached resource: list
#        return:
#            an LM instance
#        '''
#        
#        threading.Thread.__init__(self)
#        
#        ### method params
#        self.candidate_source = candidate_source
#        self.add_aligned_oov = add_aligned_oov
#        self.language_model = language_model
#        
#        ### one thread writes to one temporary file, later to be merged
#        self.res_file = res_file 
#        
#        ### lower and upper bounds of instance indices
#        self.ctr_lo = ctr_lo
#        self.ctr_up = ctr_up
#        
#        ### established, cached resources, passed as arguments from outside
#        self.eng_vocab = eng_vocab
#        self.ug_dict = ug_dict
#        self.lm_final = lm_final
#        self.lm_final_path = lm_final_path
#        
#    def run(self):
#        ctr = 0
#        with open(tra_file) as ft, \
#        open(oov_file) as fo, \
#        open(self.res_file, 'w') as fres:
#            for l_tra in ft:
#                l_oov = fo.readline()
#
#                if ctr >= self.ctr_lo and ctr <= self.ctr_up:
#                    ###
#                    # tra_tok: tokenized translation with oov, with html unescaped
#                    # oov_pos: oov word posistions
#                    # context: context word positions
#                    ###
#                    tra_tok, oov_pos, context = get_context_oov_pos(l_tra, l_oov)
#                    oov_words_set = set([tra_tok[i] for i in oov_pos])
#                    context_words_set = set([tra_tok[i] for i in context])
#
#                    ### get oov candidates
#                    oov_candidates = get_oov_candidates_all(self.candidate_source, \
#                                                            self.add_aligned_oov, \
#                                                            self.ug_dict, \
#                                                            self.eng_vocab, \
#                                                            oov_words_set, \
#                                                            context_words_set)
#
#                    ### translate
#                    if self.candidate_source == "ug_dict":
#                        num_hyp = get_num_hyp(oov_candidates, tra_tok, oov_pos)
#                        
#                        if num_hyp <= 50**2.8:
#                            _, best_trans = get_best_hyp_lm(self.lm_final, \
#                                                            [], \
#                                                            tra_tok, \
#                                                            oov_candidates, \
#                                                            -math.inf, \
#                                                            ' '.join(tra_tok))
#                            #all_sentences = get_all_sentences(tra_tok, oov_candidates)
#                            #best_score, best_trans = get_best_hyp_bylm(lm_final, all_sentences)
#                        else:
#                            lazy_dir = tmp_dir+candidate_source+"_"+str(add_aligned_oov)+"_"+language_model+"/"
#                            if not os.path.exists(lazy_dir):
#                                os.makedirs(lazy_dir)
#                            best_trans = get_best_hyp_lattice_lm(self.lm_final_path, \
#                                                                 tra_tok, \
#                                                                 oov_pos, \
#                                                                 oov_candidates, \
#                                                                 self.candidate_source, \
#                                                                 lazy_dir)
##                             best_trans = get_best_hyp_est_lm(self.lm_final, \
##                                                              tra_tok, \
##                                                              oov_words_set, \
##                                                              oov_candidates, \
##                                                              self.candidate_source)
#
#                    elif candidate_source == "eng_vocab":
#                        lazy_dir = tmp_dir+candidate_source+"_"+str(add_aligned_oov)+"_"+language_model+"_"+str(ctr)+"/"
#                        if not os.path.exists(lazy_dir):
#                            os.makedirs(lazy_dir)
#                        best_trans = get_best_hyp_lattice_lm(self.lm_final_path, \
#                                                             tra_tok, \
#                                                             oov_pos, \
#                                                             oov_candidates, \
#                                                             self.candidate_source, \
#                                                             lazy_dir)
##                         num_hyp = len(oov_candidates)*len(oov_pos)
##                         best_trans = get_best_hyp_est_lm(self.lm_final, \
##                                                          tra_tok, \
##                                                          oov_words_set, \
##                                                          oov_candidates, \
##                                                          self.candidate_source)
#                    print(ctr)
#                    print(best_trans)
#                    fres.write(best_trans+'\n')
#
#                ctr += 1

if __name__ == "__main__":
    ocp = oov_candidates_preprocessing()

    ngram = 4 
    lm_name = str(ngram)+"gram"
    lm_final_path_bin = os.path.join(tmp_dir, "lm_"+lm_name+"_final.binary")
    beam_size = 100

    tmp_lattice_dir = os.path.join(tmp_dir, "lattices")
    if not os.path.exists(tmp_lattice_dir):
        os.makedirs(tmp_lattice_dir)

    # dev, test, domain, eval
    for dataset in {"dev", "test", "domain", "eval"}:
        # extracted, eng_vocab, extracted_eng_vocab
        for candidate_source in {"extracted_eng_vocab"}:
            # oov file prep
            data_in_domain_xml, _, onebest_file, _, _, candidate_list_file = ocp.init(dataset, candidate_source)
            res_attr = "_".join([lm_name, candidate_source])
            res_file = exp_dir+"translation/"+dataset+"/"+".".join([res_attr,t,dataset,yrv])
            print('--------')
            #multithread_routine([candidate_source, add_aligned_oov, language_model],res_file, [eng_vocab, ug_dict, lm_final, lm_final_path], tmp_dir, LM)
            rescore_lattice(
                onebest_file, \
                candidate_list_file, \
                tmp_lattice_dir, \
                lm_final_path_bin, \
                beam_size, \
                res_file)
            print('--------')
            #if os.path.exists(res_file):# and dataset == "eval":
            #    print("res_file for "+dataset+" exists at: "+res_file)
            #    res_file_xml = corpus_dir+".".join([dataset_name+"-uw-oov", candidate_source, st, dataset, yrv, "xml"])
            #    write_translation_to_xml(data_in_domain_xml, res_file, res_file_xml)
            #    print("xml written to: "+res_file_xml)
            #    print('--------\n')

